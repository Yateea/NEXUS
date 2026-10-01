"""
NEXUS - Generation des dimensions synthetiques (section 17 du cahier des charges).

Usage (depuis la racine du projet, venv active, PostgreSQL demarre) :
    python pipeline/ingestion/generate_dimensions.py

Ecrit les fichiers Parquet dans data/generated/dimensions/ puis les charge
dans PostgreSQL. Le script peut etre relance : les dimensions sont videes
(TRUNCATE ... CASCADE) avant rechargement.
"""
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_URL = "postgresql+psycopg2://{u}:{p}@{h}:{port}/{d}".format(
    u=os.getenv("POSTGRES_USER", "nexus_user"),
    p=os.getenv("POSTGRES_PASSWORD", "nexus_pass"),
    h=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    d=os.getenv("POSTGRES_DB", "nexus_db"),
)

OUT_DIR = ROOT / "data" / "generated" / "dimensions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- dim_date
def make_dim_date(start=date(2023, 1, 1), end=date(2025, 12, 31)):
    # Jours feries marocains fixes (les fetes religieuses mobiles ne sont pas incluses)
    fixed_holidays = {(1, 1), (1, 11), (5, 1), (7, 30), (8, 14), (8, 20), (8, 21), (11, 6), (11, 18)}
    rows = []
    d = start
    while d <= end:
        iso = d.isocalendar()
        rows.append({
            "date_id": int(d.strftime("%Y%m%d")),
            "date": d,
            "day": d.day,
            "day_of_week": d.isoweekday(),
            "week": iso[1],
            "month": d.month,
            "quarter": (d.month - 1) // 3 + 1,
            "year": d.year,
            "is_weekend": d.isoweekday() >= 6,
            "is_holiday": (d.month, d.day) in fixed_holidays,
        })
        d += timedelta(days=1)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_site
def make_dim_site():
    return pl.DataFrame({
        "site_id": ["SITE_001", "SITE_002", "SITE_003"],
        "site_name": ["NEXUS Casablanca", "NEXUS Tanger", "NEXUS Kenitra"],
        "country": ["Morocco"] * 3,
        "city": ["Casablanca", "Tanger", "Kenitra"],
        "latitude": [33.5731, 35.7595, 34.2610],
        "longitude": [-7.5898, -5.8340, -6.5802],
        "production_capacity": [500000.0, 320000.0, 280000.0],
        "warehouse_capacity": [400000.0, 250000.0, 200000.0],
    })


# ---------------------------------------------------------------- dim_warehouse
def make_dim_warehouse(sites):
    # 8 entrepots : 3 Casablanca, 3 Tanger, 2 Kenitra
    site_ids = ["SITE_001"] * 3 + ["SITE_002"] * 3 + ["SITE_003"] * 2
    coords = {r["site_id"]: (r["latitude"], r["longitude"]) for r in sites.to_dicts()}
    types = ["Raw Materials", "Finished Goods", "Distribution", "Raw Materials",
             "Finished Goods", "Distribution", "Finished Goods", "Distribution"]
    rows = []
    for i, sid in enumerate(site_ids, start=1):
        lat, lon = coords[sid]
        rows.append({
            "warehouse_id": "WH_{:03d}".format(i),
            "site_id": sid,
            "warehouse_type": types[i - 1],
            "capacity_units": float(rng.integers(30000, 90000)),
            "latitude": round(lat + float(rng.normal(0, 0.02)), 6),
            "longitude": round(lon + float(rng.normal(0, 0.02)), 6),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_production_line
def make_dim_production_line():
    # 25 lignes : 9 Casablanca, 8 Tanger, 8 Kenitra
    site_ids = ["SITE_001"] * 9 + ["SITE_002"] * 8 + ["SITE_003"] * 8
    ptypes = ["Assembly", "Machining", "Packaging", "Injection", "Welding"]
    rows = []
    for i, sid in enumerate(site_ids, start=1):
        rows.append({
            "line_id": "LINE_{:03d}".format(i),
            "site_id": sid,
            "line_name": "Line {:02d}".format(i),
            "production_type": ptypes[(i - 1) % len(ptypes)],
            "capacity_per_hour": float(rng.integers(150, 900)),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_machine
def make_dim_machine(lines):
    line_rows = lines.to_dicts()
    mtypes = ["CNC", "Press", "Robot Arm", "Conveyor", "Injection Molder", "Welder", "Packer"]
    makers = ["Siemens", "ABB", "Fanuc", "KUKA", "Bosch Rexroth", "Mitsubishi Electric"]
    rows = []
    for i in range(1, 121):
        line = line_rows[(i - 1) % len(line_rows)]
        install = date(2015, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 9)))
        rows.append({
            "machine_id": "M-{:03d}".format(i),
            "site_id": line["site_id"],
            "line_id": line["line_id"],
            "machine_type": mtypes[int(rng.integers(0, len(mtypes)))],
            "installation_date": install,
            "manufacturer": makers[int(rng.integers(0, len(makers)))],
            "nominal_capacity": float(rng.integers(50, 400)),
            "maintenance_interval": int(rng.choice([500, 750, 1000, 1500])),  # en heures de fonctionnement
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_vehicle
def make_dim_vehicle():
    vtypes = [("Van", 1200, 9.5), ("Light Truck", 3500, 14.0), ("Truck", 12000, 26.0), ("Semi-Trailer", 24000, 33.0)]
    fuels = ["Diesel", "Diesel", "Diesel", "Gasoline", "Electric"]
    rows = []
    for i in range(1, 201):
        vt, cap, cons = vtypes[int(rng.integers(0, len(vtypes)))]
        fuel = fuels[int(rng.integers(0, len(fuels)))]
        rows.append({
            "vehicle_id": "V-{:03d}".format(i),
            "vehicle_type": vt,
            "capacity_kg": float(cap),
            "fuel_type": fuel,
            "consumption_per_100km": round(cons * float(rng.uniform(0.9, 1.15)), 2) if fuel != "Electric" else 0.0,
            "home_site_id": "SITE_00{}".format(int(rng.integers(1, 4))),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_supplier
def make_dim_supplier():
    countries = [("Morocco", ["Casablanca", "Tanger", "Fes", "Agadir"]),
                 ("Spain", ["Madrid", "Barcelona", "Valencia"]),
                 ("France", ["Lyon", "Toulouse", "Paris"]),
                 ("Turkey", ["Istanbul", "Bursa"]),
                 ("China", ["Shenzhen", "Shanghai", "Ningbo"]),
                 ("Germany", ["Stuttgart", "Munich"])]
    weights = np.array([0.30, 0.20, 0.15, 0.10, 0.15, 0.10])
    lead_by_country = {"Morocco": 5, "Spain": 9, "France": 10, "Turkey": 14, "China": 35, "Germany": 12}
    terms = ["Net 30", "Net 45", "Net 60", "Prepaid"]
    rows = []
    for i in range(1, 301):
        idx = int(rng.choice(len(countries), p=weights))
        country, cities = countries[idx]
        base = lead_by_country[country]
        rows.append({
            "supplier_id": "SUP-{:03d}".format(i),
            "supplier_name": "Supplier {:03d} {}".format(i, country),
            "country": country,
            "city": cities[int(rng.integers(0, len(cities)))],
            "lead_time_days": max(2, int(rng.normal(base, base * 0.2))),
            "payment_terms": terms[int(rng.integers(0, len(terms)))],
            "reliability_score": round(float(np.clip(rng.normal(85, 9), 45, 100)), 2),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_product
def make_dim_product(n=5000):
    cats = ["CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06", "CAT_07", "CAT_08"]
    unit_cost = np.round(rng.lognormal(mean=3.2, sigma=0.8, size=n), 2)
    margin = rng.uniform(1.15, 1.75, size=n)
    selling = np.round(unit_cost * margin, 2)
    lead = rng.integers(3, 45, size=n)
    daily_demand_guess = rng.uniform(5, 120, size=n)
    safety = np.round(daily_demand_guess * rng.uniform(2, 6, size=n), 0)
    reorder = np.round(safety + daily_demand_guess * lead, 0)
    rows = []
    for i in range(n):
        cat = cats[int(rng.integers(0, len(cats)))]
        rows.append({
            "product_id": "P-{:04d}".format(i + 1),
            "product_name": "Product {:04d}".format(i + 1),
            "category_id": cat,
            "subcategory_id": "{}_{:02d}".format(cat, int(rng.integers(1, 6))),
            "unit_cost": float(unit_cost[i]),
            "selling_price": float(selling[i]),
            "weight": round(float(rng.uniform(0.05, 25.0)), 3),
            "volume": round(float(rng.uniform(0.001, 0.8)), 3),
            "safety_stock": float(safety[i]),
            "reorder_point": float(reorder[i]),
            "lead_time_days": int(lead[i]),
            "production_time_minutes": round(float(rng.uniform(2, 90)), 2),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- dim_customer
def make_dim_customer(n=50000):
    segments = ["Retail", "Wholesale", "Distributor", "Industrial", "Online"]
    seg_p = [0.35, 0.15, 0.15, 0.15, 0.20]
    geo = {
        "Morocco": {"Casablanca-Settat": ["Casablanca", "El Jadida", "Mohammedia"],
                    "Rabat-Sale-Kenitra": ["Rabat", "Kenitra", "Sale"],
                    "Tanger-Tetouan": ["Tanger", "Tetouan"],
                    "Fes-Meknes": ["Fes", "Meknes"],
                    "Marrakech-Safi": ["Marrakech", "Safi"],
                    "Souss-Massa": ["Agadir"]},
        "France": {"Ile-de-France": ["Paris"], "Auvergne-Rhone-Alpes": ["Lyon"]},
        "Spain": {"Catalonia": ["Barcelona"], "Madrid": ["Madrid"]},
    }
    country_p = [0.85, 0.09, 0.06]
    countries = list(geo.keys())
    rows = []
    for i in range(n):
        country = countries[int(rng.choice(len(countries), p=country_p))]
        regions = list(geo[country].keys())
        region = regions[int(rng.integers(0, len(regions)))]
        cities = geo[country][region]
        rows.append({
            "customer_id": "C-{:06d}".format(i + 1),
            "customer_segment": segments[int(rng.choice(len(segments), p=seg_p))],
            "country": country,
            "city": cities[int(rng.integers(0, len(cities)))],
            "region": region,
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    print("Generation des dimensions NEXUS Manufacturing...")
    dim_date = make_dim_date()
    dim_site = make_dim_site()
    dim_warehouse = make_dim_warehouse(dim_site)
    dim_line = make_dim_production_line()
    dim_machine = make_dim_machine(dim_line)
    dim_vehicle = make_dim_vehicle()
    dim_supplier = make_dim_supplier()
    dim_product = make_dim_product()
    dim_customer = make_dim_customer()

    # Ordre = ordre de chargement (respecte les cles etrangeres)
    tables = [
        ("dim_date", dim_date),
        ("dim_site", dim_site),
        ("dim_warehouse", dim_warehouse),
        ("dim_production_line", dim_line),
        ("dim_machine", dim_machine),
        ("dim_vehicle", dim_vehicle),
        ("dim_supplier", dim_supplier),
        ("dim_product", dim_product),
        ("dim_customer", dim_customer),
    ]

    for name, df in tables:
        df.write_parquet(OUT_DIR / (name + ".parquet"))
        print("  {:<22} {:>7} lignes -> parquet".format(name, df.height))

    print("Chargement dans PostgreSQL...")
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        conn.execute(text(
            "TRUNCATE dim_customer, dim_product, dim_supplier, dim_vehicle, dim_machine, "
            "dim_production_line, dim_warehouse, dim_site, dim_date RESTART IDENTITY CASCADE"
        ))

    for name, df in tables:
        df.write_database(table_name=name, connection=DB_URL, if_table_exists="append", engine="sqlalchemy")
        print("  {:<22} charge".format(name))

    print("Termine.")


if __name__ == "__main__":
    main()
