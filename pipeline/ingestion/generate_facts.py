"""
NEXUS - Generation des tables de faits synthetiques (section 17 du cahier des charges).

Prerequis : generate_dimensions.py deja execute (lit les Parquet des dimensions).

Usage (depuis la racine du projet, venv active) :
    python pipeline/ingestion/generate_facts.py                  # palier 100K (scale 0.01)
    python pipeline/ingestion/generate_facts.py --scale 0.1      # palier 1M
    python pipeline/ingestion/generate_facts.py --scale 1        # volumes cibles complets (50M capteurs...)
    python pipeline/ingestion/generate_facts.py --no-db          # Parquet uniquement

Les donnees sont generees par chunks (1M lignes) : ecrites en Parquet dans
data/generated/facts/<table>/ puis chargees dans PostgreSQL via COPY.
Le script est relancable : les tables de faits generees sont videes avant chargement.

Anomalies volontairement injectees (pour tester detection + dashboards) :
  - Energie : +22-30 % sur SITE_001 du 10 au 14/09/2025, a production stable
  - Machines M-003, M-031, M-042 : degradation progressive (temperature, vibration)
    a partir du 01/11/2025, plus defauts et arrets a partir du 01/09/2025
  - Fournisseurs peu fiables : retards plus frequents (fact_purchase)
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DIM_DIR = ROOT / "data" / "generated" / "dimensions"
OUT_DIR = ROOT / "data" / "generated" / "facts"
M5_PATTERNS_FILE = ROOT / "data" / "external" / "m5" / "m5_patterns.json"


def load_m5_patterns():
    """Charge les patterns reels M5 (section 18) s'ils ont ete extraits, sinon None
    (dans ce cas la saisonnalite synthetique par defaut est utilisee)."""
    if not M5_PATTERNS_FILE.exists():
        print("  [info] m5_patterns.json introuvable -> saisonnalite synthetique par defaut utilisee pour fact_sales")
        return None
    with open(M5_PATTERNS_FILE) as f:
        p = json.load(f)
    print("  [info] Patterns M5 charges depuis {} (calibrage reel de fact_sales)".format(M5_PATTERNS_FILE))
    return p


NASA_PATTERNS_FILE = ROOT / "data" / "external" / "nasa" / "nasa_patterns.json"


def load_nasa_patterns():
    """Charge la courbe de degradation reelle NASA C-MAPSS (section 18) si extraite."""
    if not NASA_PATTERNS_FILE.exists():
        print("  [info] nasa_patterns.json introuvable -> rampe de degradation lineaire par defaut utilisee")
        return None
    with open(NASA_PATTERNS_FILE) as f:
        p = json.load(f)
    print("  [info] Courbe de degradation NASA chargee depuis {} (calibrage reel de fact_machine_sensor)".format(NASA_PATTERNS_FILE))
    return p


TETOUAN_PATTERNS_FILE = ROOT / "data" / "external" / "energy" / "tetouan_patterns.json"


def load_tetouan_patterns():
    """Charge le profil de consommation reel UCI Tetouan (section 18) si extrait."""
    if not TETOUAN_PATTERNS_FILE.exists():
        print("  [info] tetouan_patterns.json introuvable -> profil energetique sinusoidal par defaut utilise")
        return None
    with open(TETOUAN_PATTERNS_FILE) as f:
        p = json.load(f)
    print("  [info] Profil energetique Tetouan charge depuis {} (calibrage reel de fact_energy)".format(TETOUAN_PATTERNS_FILE))
    return p


NYC_PATTERNS_FILE = ROOT / "data" / "external" / "transport" / "nyc_tlc_patterns.json"


def load_nyc_patterns():
    """Charge les patterns reels NYC TLC (section 18) si extraits."""
    if not NYC_PATTERNS_FILE.exists():
        print("  [info] nyc_tlc_patterns.json introuvable -> bruit de trajet par defaut utilise")
        return None
    with open(NYC_PATTERNS_FILE) as f:
        p = json.load(f)
    print("  [info] Patterns NYC TLC charges depuis {} (calibrage reel de fact_transport)".format(NYC_PATTERNS_FILE))
    return p

SEED = 42
HOT_MACHINES = ["M-003", "M-031", "M-042"]

# Volumes cibles (section 17) ; fact_maintenance a un volume fixe.
TARGETS = {
    "fact_sales": 10_000_000,
    "fact_inventory": 20_000_000,
    "fact_purchase": 2_000_000,
    "fact_production": 5_000_000,
    "fact_energy": 20_000_000,
    "fact_machine_sensor": 50_000_000,
    "fact_transport": 5_000_000,
}

DESTINATIONS = [
    ("Casablanca", 20), ("Rabat", 90), ("Kenitra", 130), ("El Jadida", 100),
    ("Marrakech", 240), ("Fes", 300), ("Meknes", 250), ("Tanger", 340),
    ("Tetouan", 320), ("Agadir", 500), ("Safi", 250), ("Oujda", 600),
]


# ------------------------------------------------------------------ contexte
def to_date_id(d64):
    """datetime64[D] -> entier YYYYMMDD (vectorise)."""
    d = d64.astype("datetime64[D]")
    years = d.astype("datetime64[Y]").astype(int) + 1970
    months = d.astype("datetime64[M]").astype(int) % 12 + 1
    days = (d - d.astype("datetime64[M]").astype("datetime64[D]")).astype(int) + 1
    return years * 10000 + months * 100 + days


class Ctx:
    def __init__(self):
        rs = np.random.default_rng(SEED)

        dd = pl.read_parquet(DIM_DIR / "dim_date.parquet")
        self.date_ids = dd["date_id"].to_numpy()
        self.dates = dd["date"].to_numpy().astype("datetime64[D]")
        self.dow = dd["day_of_week"].to_numpy()
        self.month = dd["month"].to_numpy()
        self.is_holiday = dd["is_holiday"].to_numpy()
        doy = (self.dates - self.dates.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
        self.doy = doy
        year = self.dates.astype("datetime64[Y]").astype(int) + 1970
        self.year = year

        def probs(weekend_factor, amp, growth):
            w = 1 + amp * np.sin(2 * np.pi * (doy - 80) / 365.25)
            w = w * (1 + growth * (year - 2023))
            w = np.where(self.dow >= 6, w * weekend_factor, w)
            return w / w.sum()

        m5 = load_m5_patterns()
        if m5 is not None:
            dow_mult = np.array([m5["dow_multiplier"][str(int(d))] for d in self.dow])
            month_mult = np.array([m5["month_multiplier"][str(int(m))] for m in self.month])
            growth_factor = (1 + m5["yearly_growth_rate"]) ** (year - year.min())
            event_mult = np.where(self.is_holiday, m5["event_uplift_ratio"], 1.0)
            w = dow_mult * month_mult * growth_factor * event_mult
            self.p_sales = w / w.sum()
        else:
            self.p_sales = probs(1.10, 0.20, 0.08)

        self.p_prod = probs(0.35, 0.10, 0.03)
        self.p_transport = probs(0.50, 0.12, 0.05)

        prod = pl.read_parquet(DIM_DIR / "dim_product.parquet")
        self.prod_ids = prod["product_id"].to_numpy()
        self.sell = prod["selling_price"].to_numpy()
        self.cost = prod["unit_cost"].to_numpy()
        self.reorder = prod["reorder_point"].to_numpy()
        pop = rs.pareto(1.2, len(self.prod_ids)) + 1.0
        self.prod_pop = pop / pop.sum()

        self.cust_ids = pl.read_parquet(DIM_DIR / "dim_customer.parquet")["customer_id"].to_numpy()

        mach = pl.read_parquet(DIM_DIR / "dim_machine.parquet")
        self.mach_ids = mach["machine_id"].to_numpy()
        self.mach_site = mach["site_id"].to_numpy()
        self.mach_line = mach["line_id"].to_numpy()
        self.mach_cap = mach["nominal_capacity"].to_numpy()
        self.mach_interval = mach["maintenance_interval"].to_numpy()
        self.mach_energy_base = np.round(self.mach_cap * 0.25, 2)  # kW nominal
        self.mach_hours0 = rs.uniform(2000, 40000, len(self.mach_ids))
        self.hot = np.isin(self.mach_ids, HOT_MACHINES)

        sup = pl.read_parquet(DIM_DIR / "dim_supplier.parquet")
        self.sup_ids = sup["supplier_id"].to_numpy()
        self.sup_lead = sup["lead_time_days"].to_numpy()
        self.sup_rel = sup["reliability_score"].to_numpy()

        self.wh_ids = pl.read_parquet(DIM_DIR / "dim_warehouse.parquet")["warehouse_id"].to_numpy()
        wh = pl.read_parquet(DIM_DIR / "dim_warehouse.parquet")
        self.wh_by_site = {s: wh.filter(pl.col("site_id") == s)["warehouse_id"].to_numpy()
                           for s in wh["site_id"].unique().to_list()}

        veh = pl.read_parquet(DIM_DIR / "dim_vehicle.parquet")
        self.veh_ids = veh["vehicle_id"].to_numpy()
        self.veh_site = veh["home_site_id"].to_numpy()
        self.veh_cons = veh["consumption_per_100km"].to_numpy()

        self.purchase_idx = np.where(self.dates <= np.datetime64("2025-11-30"))[0]
        self.deg_start_idx = int(np.where(self.date_ids == 20251101)[0][0])

        nasa = load_nasa_patterns()
        if nasa is not None:
            self.nasa_curve_x = np.array(nasa["degradation_curve"]["life_fraction"])
            self.nasa_curve_y = np.array(nasa["degradation_curve"]["degradation_index"])
        else:
            self.nasa_curve_x = None
            self.nasa_curve_y = None

        tetouan = load_tetouan_patterns()
        if tetouan is not None:
            self.tetouan_hourly = np.array([tetouan["hourly_multiplier"][str(h)] for h in range(24)])
            self.tetouan_dow = np.array([tetouan["dow_multiplier"][str(d)] for d in range(1, 8)])
        else:
            self.tetouan_hourly = None
            self.tetouan_dow = None

        nyc = load_nyc_patterns()
        if nyc is not None:
            self.nyc_hourly_speed = np.array([nyc["hourly_speed_multiplier"][str(h)] for h in range(24)])
            self.nyc_noise_mu = nyc["duration_noise_lognormal_mu"]
            self.nyc_noise_sigma = nyc["duration_noise_lognormal_sigma"]
        else:
            self.nyc_hourly_speed = None
            self.nyc_noise_mu = None
            self.nyc_noise_sigma = None


def make_timestamps(ctx, di, rng):
    n = len(di)
    secs = rng.integers(0, 86400, n)
    ts = ctx.dates[di].astype("datetime64[s]") + secs.astype("timedelta64[s]")
    return ts.astype("datetime64[us]"), secs // 3600


# ------------------------------------------------------------------ generateurs
def gen_sales(n, rng, ctx):
    di = rng.choice(len(ctx.date_ids), n, p=ctx.p_sales)
    pi = rng.choice(len(ctx.prod_ids), n, p=ctx.prod_pop)
    ci = rng.integers(0, len(ctx.cust_ids), n)
    site = rng.choice(np.array(["SITE_001", "SITE_002", "SITE_003"]), n, p=[0.5, 0.3, 0.2])
    qty = (1 + rng.poisson(3.0, n)).astype(float)
    promo = rng.random(n) < 0.12
    rate = np.where(promo, rng.choice(np.array([0.05, 0.10, 0.15, 0.20]), n), 0.0)
    price = ctx.sell[pi]
    gross = qty * price
    discount = np.round(gross * rate, 2)
    return pl.DataFrame({
        "date_id": ctx.date_ids[di],
        "product_id": ctx.prod_ids[pi],
        "customer_id": ctx.cust_ids[ci],
        "site_id": site,
        "quantity": qty,
        "unit_price": price,
        "discount": discount,
        "revenue": np.round(gross - discount, 2),
    })


def gen_inventory(n, rng, ctx):
    di = rng.integers(0, len(ctx.date_ids), n)
    pi = rng.integers(0, len(ctx.prod_ids), n)
    wi = rng.integers(0, len(ctx.wh_ids), n)
    rp = ctx.reorder[pi]
    opening = np.round(rp * rng.lognormal(0.1, 0.6, n))
    inbound = np.round(np.where(rng.random(n) < 0.15, rp * rng.uniform(0.5, 2.0, n), 0.0))
    produced = np.round(np.where(rng.random(n) < 0.25, rp * rng.uniform(0.2, 1.2, n), 0.0))
    sales_q = np.round(rp * rng.uniform(0.02, 0.25, n) * rng.lognormal(0, 0.3, n))
    outbound = np.round(rp * rng.uniform(0.0, 0.10, n))
    avail = opening + inbound + produced
    outbound = np.minimum(outbound, avail)
    sales_q = np.minimum(sales_q, avail - outbound)
    closing = avail - outbound - sales_q  # closing = opening + inbound + production - sales - outbound
    return pl.DataFrame({
        "date_id": ctx.date_ids[di],
        "product_id": ctx.prod_ids[pi],
        "warehouse_id": ctx.wh_ids[wi],
        "opening_stock": opening,
        "inbound_quantity": inbound,
        "production_quantity": produced,
        "sales_quantity": sales_q,
        "outbound_quantity": outbound,
        "closing_stock": closing,
        "stock_value": np.round(closing * ctx.cost[pi], 2),
    })


def gen_purchase(n, rng, ctx):
    oi = rng.choice(ctx.purchase_idx, n)
    si = rng.integers(0, len(ctx.sup_ids), n)
    pi = rng.integers(0, len(ctx.prod_ids), n)
    order_date = ctx.dates[oi]
    expected = order_date + ctx.sup_lead[si].astype("timedelta64[D]")
    p_delay = np.clip((100 - ctx.sup_rel[si]) / 100 * 2.0, 0.02, 0.8)
    delayed = rng.random(n) < p_delay
    delay = np.where(delayed, 1 + rng.geometric(0.3, n), 0)
    actual = expected + delay.astype("timedelta64[D]")
    maxd = ctx.dates[-1]
    expected = np.minimum(expected, maxd)
    actual = np.minimum(actual, maxd)
    delay_days = (actual - expected).astype(int)
    qty = np.round(rng.lognormal(5.5, 0.7, n))
    price = np.round(ctx.cost[pi] * rng.uniform(0.90, 1.05, n), 2)
    order_id = to_date_id(order_date)
    return pl.DataFrame({
        "date_id": order_id,
        "supplier_id": ctx.sup_ids[si],
        "product_id": ctx.prod_ids[pi],
        "order_date_id": order_id,
        "expected_date_id": to_date_id(expected),
        "actual_date_id": to_date_id(actual),
        "quantity": qty,
        "unit_price": price,
        "total_cost": np.round(qty * price, 2),
        "is_delayed": delay_days > 0,
        "delay_days": delay_days,
    })


def gen_production(n, rng, ctx):
    di = rng.choice(len(ctx.date_ids), n, p=ctx.p_prod)
    mi = rng.integers(0, len(ctx.mach_ids), n)
    pi = rng.integers(0, len(ctx.prod_ids), n)
    planned = np.round(ctx.mach_cap[mi] * rng.uniform(2, 8, n))
    hot = ctx.hot[mi] & (ctx.date_ids[di] >= 20250901)
    ach = np.clip(rng.normal(0.94, 0.06, n), 0.5, 1.05)
    ach = np.where(hot, ach * rng.uniform(0.75, 0.92, n), ach)
    actual = np.round(planned * ach)
    defect = rng.binomial(actual.astype(np.int64), np.where(hot, 0.06, 0.02))
    downtime = np.round(rng.exponential(np.where(hot, 70.0, 22.0), n), 2)
    return pl.DataFrame({
        "date_id": ctx.date_ids[di],
        "site_id": ctx.mach_site[mi],
        "line_id": ctx.mach_line[mi],
        "machine_id": ctx.mach_ids[mi],
        "product_id": ctx.prod_ids[pi],
        "planned_quantity": planned,
        "actual_quantity": actual,
        "defect_quantity": defect.astype(float),
        "production_time_minutes": np.round(rng.uniform(300, 480, n), 2),
        "downtime_minutes": downtime,
    })


def gen_energy(n, rng, ctx):
    di = rng.choice(len(ctx.date_ids), n, p=ctx.p_prod)
    mi = rng.integers(0, len(ctx.mach_ids), n)
    ts, hour = make_timestamps(ctx, di, rng)
    if ctx.tetouan_hourly is not None:
        dow_idx = ctx.dow[di] - 1  # dow 1..7 -> index 0..6
        load = ctx.tetouan_hourly[hour] * ctx.tetouan_dow[dow_idx]
    else:
        load = 0.55 + 0.35 * np.clip(np.sin(np.pi * (hour - 5) / 16), 0, 1)
    power = ctx.mach_energy_base[mi] * load * rng.normal(1.0, 0.06, n)
    did = ctx.date_ids[di]
    site = ctx.mach_site[mi]
    spike = (did >= 20250910) & (did <= 20250914) & (site == "SITE_001")
    power = np.where(spike, power * rng.uniform(1.22, 1.30, n), power)
    power = np.where(ctx.hot[mi] & (did >= 20251101), power * 1.12, power)
    temp = 20 + 8 * np.sin(2 * np.pi * (ctx.doy[di] - 110) / 365.25) + 4 * np.sin(np.pi * (hour - 8) / 12) + rng.normal(0, 1.5, n)
    hum = np.clip(65 - 0.8 * (temp - 20) + rng.normal(0, 6, n), 25, 95)
    return pl.DataFrame({
        "timestamp": ts,
        "site_id": site,
        "line_id": ctx.mach_line[mi],
        "machine_id": ctx.mach_ids[mi],
        "energy_kwh": np.round(power, 4),  # lecture horaire : kWh = kW x 1h
        "power_kw": np.round(power, 4),
        "temperature": np.round(temp, 2),
        "humidity": np.round(hum, 2),
    })


def gen_sensor(n, rng, ctx):
    di = rng.integers(0, len(ctx.date_ids), n)
    mi = rng.integers(0, len(ctx.mach_ids), n)
    ts, hour = make_timestamps(ctx, di, rng)
    hours = ctx.mach_hours0[mi] + di * 14.0 + hour * 0.6
    life_fraction = np.clip((di - ctx.deg_start_idx) / 60.0, 0, 1)
    if ctx.nasa_curve_x is not None:
        ramp = np.where(ctx.hot[mi], np.interp(life_fraction, ctx.nasa_curve_x, ctx.nasa_curve_y), 0.0)
    else:
        ramp = np.where(ctx.hot[mi], life_fraction, 0.0)
    temperature = 68 + rng.normal(0, 2.2, n) + 18 * ramp + 0.00015 * hours
    vibration = 2.1 + rng.normal(0, 0.25, n) + 2.4 * ramp + 0.00003 * hours
    pressure = 6.0 + rng.normal(0, 0.3, n) - 0.6 * ramp
    rpm = 1500 + rng.normal(0, 40, n) - 60 * ramp
    power = ctx.mach_energy_base[mi] * rng.uniform(0.6, 0.9, n) * (1 + 0.15 * ramp)
    return pl.DataFrame({
        "timestamp": ts,
        "machine_id": ctx.mach_ids[mi],
        "temperature": np.round(temperature, 3),
        "vibration": np.round(vibration, 3),
        "pressure": np.round(pressure, 3),
        "rpm": np.round(rpm, 2),
        "power": np.round(power, 3),
        "operating_hours": np.round(hours, 2),
    })


def gen_transport(n, rng, ctx):
    di = rng.choice(len(ctx.date_ids), n, p=ctx.p_transport)
    vi = rng.integers(0, len(ctx.veh_ids), n)
    ts, hour = make_timestamps(ctx, di, rng)
    wh = np.empty(n, dtype=object)
    for site, ids in ctx.wh_by_site.items():
        mask = ctx.veh_site[vi] == site
        if mask.any():
            wh[mask] = rng.choice(ids, int(mask.sum()))
    dest_i = rng.integers(0, len(DESTINATIONS), n)
    dest = np.array([d[0] for d in DESTINATIONS])[dest_i]
    base = np.array([d[1] for d in DESTINATIONS], dtype=float)[dest_i]
    dist = np.round(base * rng.uniform(0.8, 1.3, n), 1)

    base_speed_kmh = 55.0
    if ctx.nyc_hourly_speed is not None:
        effective_speed = base_speed_kmh * ctx.nyc_hourly_speed[hour]
    else:
        effective_speed = base_speed_kmh
    planned = np.round(dist / effective_speed * 60 + 25, 1)

    if ctx.nyc_noise_mu is not None:
        actual = np.round(planned * np.exp(rng.normal(ctx.nyc_noise_mu, ctx.nyc_noise_sigma, n)), 1)
    else:
        actual = np.round(planned * np.exp(rng.normal(0.04, 0.18, n)), 1)
    delay = np.round(np.maximum(0, actual - planned), 1)
    failed = rng.random(n) < 0.004
    status = np.where(failed, "failed", np.where(delay > 15, "delayed", "on_time"))
    fuel = np.round(dist * ctx.veh_cons[vi] / 100 * rng.uniform(0.95, 1.10, n), 2)
    return pl.DataFrame({
        "date_id": ctx.date_ids[di],
        "vehicle_id": ctx.veh_ids[vi],
        "warehouse_id": wh.astype(str),
        "destination": dest,
        "distance_km": dist,
        "planned_duration": planned,
        "actual_duration": actual,
        "fuel_consumption": fuel,
        "delivery_status": status,
        "delay_minutes": np.where(failed, 120.0, delay),
    })


def gen_maintenance(rng, ctx, per_machine=12):
    mi = np.repeat(np.arange(len(ctx.mach_ids)), per_machine)
    n = len(mi)
    di = rng.integers(0, len(ctx.date_ids), n)
    corrective = rng.random(n) < 0.18
    duration = np.where(corrective, rng.uniform(4, 30, n), rng.uniform(1, 4, n))
    cost = np.where(corrective, rng.lognormal(7.6, 0.6, n), rng.lognormal(6.2, 0.4, n))
    return pl.DataFrame({
        "machine_id": ctx.mach_ids[mi],
        "date_id": ctx.date_ids[di],
        "maintenance_type": np.where(corrective, "corrective", "preventive"),
        "duration_hours": np.round(duration, 2),
        "cost": np.round(cost, 2),
        "failure_flag": corrective,
    })


GENERATORS = [
    ("fact_sales", gen_sales),
    ("fact_inventory", gen_inventory),
    ("fact_purchase", gen_purchase),
    ("fact_production", gen_production),
    ("fact_energy", gen_energy),
    ("fact_machine_sensor", gen_sensor),
    ("fact_transport", gen_transport),
]


# ------------------------------------------------------------------ chargement
def connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "nexus_user"),
        password=os.getenv("POSTGRES_PASSWORD", "nexus_pass"),
        dbname=os.getenv("POSTGRES_DB", "nexus_db"),
    )


def copy_df(conn, table, df):
    buf = io.BytesIO()
    df.write_csv(buf, datetime_format="%Y-%m-%d %H:%M:%S")
    buf.seek(0)
    cols = ",".join(df.columns)
    with conn.cursor() as cur:
        cur.copy_expert("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)".format(table, cols), buf)
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.01, help="fraction des volumes cibles (0.01 = 100K ventes)")
    parser.add_argument("--chunk", type=int, default=1_000_000)
    parser.add_argument("--no-db", action="store_true", help="ecrire les Parquet sans charger PostgreSQL")
    args = parser.parse_args()

    ctx = Ctx()
    conn = None
    if not args.no_db:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("TRUNCATE fact_sales, fact_inventory, fact_purchase, fact_production, fact_energy, "
                        "fact_maintenance, fact_transport, fact_machine_sensor RESTART IDENTITY")
        conn.commit()

    print("Generation des faits (scale={}) ...".format(args.scale))
    t_all = time.time()

    for t_idx, (table, gen) in enumerate(GENERATORS):
        total = max(1000, int(TARGETS[table] * args.scale))
        out = OUT_DIR / table
        out.mkdir(parents=True, exist_ok=True)
        done, k, t0 = 0, 0, time.time()
        while done < total:
            m = min(args.chunk, total - done)
            rng = np.random.default_rng(SEED + 10_000 * (t_idx + 1) + k)
            df = gen(m, rng, ctx)
            df.write_parquet(out / "part_{:04d}.parquet".format(k))
            if conn is not None:
                copy_df(conn, table, df)
            done += m
            k += 1
        print("  {:<20} {:>12,} lignes  ({:.1f}s)".format(table, total, time.time() - t0))

    mt = gen_maintenance(np.random.default_rng(SEED + 99), ctx)
    (OUT_DIR / "fact_maintenance").mkdir(parents=True, exist_ok=True)
    mt.write_parquet(OUT_DIR / "fact_maintenance" / "part_0000.parquet")
    if conn is not None:
        copy_df(conn, "fact_maintenance", mt)
    print("  {:<20} {:>12,} lignes".format("fact_maintenance", mt.height))

    if conn is not None:
        with conn.cursor() as cur:
            cur.execute("ANALYZE")
        conn.commit()
        conn.close()
    print("Termine en {:.1f}s.".format(time.time() - t_all))


if __name__ == "__main__":
    main()
