"""
NEXUS - Validation des donnees generees (Data Quality, section 27).

Usage (depuis C:\\Users\\user\\NEXUS) :
    python pipeline\\validation\\validate_facts.py

Niveaux : PASS / INFO / WARN / FAIL. Code de sortie 1 s'il y a au moins un FAIL.
Rapport JSON ecrit dans data\\validation_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("data/generated")
EXT = Path("data/external")
c = pl.col
REPORT = []


def log(level, name, detail=""):
    REPORT.append({"level": level, "name": name, "detail": detail})
    print("[%-4s] %s  %s" % (level, name, detail))


def collect(lf):
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def missing(vals, ref):
    known = set(ref.to_list())
    return [v for v in vals.drop_nulls().unique().to_list() if v not in known]


def d(col):
    """date_id YYYYMMDD (int) -> Date"""
    return c(col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)


# ---------------------------------------------------------------- chargement
dims = {p.stem: pl.read_parquet(p) for p in sorted((ROOT / "dimensions").glob("*.parquet"))}
facts = {
    p.name: pl.scan_parquet(str(p / "*.parquet"))
    for p in sorted((ROOT / "facts").iterdir())
    if p.is_dir()
}

PK = {
    "dim_customer": "customer_id",
    "dim_date": "date_id",
    "dim_machine": "machine_id",
    "dim_product": "product_id",
    "dim_production_line": "line_id",
    "dim_site": "site_id",
    "dim_supplier": "supplier_id",
    "dim_vehicle": "vehicle_id",
    "dim_warehouse": "warehouse_id",
}


# ---------------------------------------------------------------- dimensions
def check_dimensions():
    print("\n== 1. Dimensions ==")
    for t, k in PK.items():
        if t not in dims:
            log("FAIL", t, "fichier manquant")
            continue
        df = dims[t]
        dup = df.height - df[k].n_unique()
        nul = df[k].null_count()
        log("PASS" if dup == 0 and nul == 0 else "FAIL", "%s.%s unique" % (t, k),
            "%d lignes, doublons=%d, nulls=%d" % (df.height, dup, nul))

    dd = dims["dim_date"]
    recomputed = (dd["date"].dt.year().cast(pl.Int64) * 10000
                  + dd["date"].dt.month().cast(pl.Int64) * 100
                  + dd["date"].dt.day().cast(pl.Int64))
    log("PASS" if bool((dd["date_id"] == recomputed).all()) else "FAIL",
        "dim_date.date_id = YYYYMMDD")
    gaps = int((dd["date"].diff().drop_nulls().dt.total_days() != 1).sum())
    log("PASS" if gaps == 0 else "FAIL", "dim_date continue (1 jour/ligne)",
        "%s -> %s, trous=%d" % (dd["date"].min(), dd["date"].max(), gaps))

    links = [
        ("dim_machine", "site_id", "dim_site"),
        ("dim_machine", "line_id", "dim_production_line"),
        ("dim_production_line", "site_id", "dim_site"),
        ("dim_warehouse", "site_id", "dim_site"),
        ("dim_vehicle", "home_site_id", "dim_site"),
    ]
    for t, col, ref in links:
        bad = missing(dims[t][col], dims[ref][PK[ref]])
        log("PASS" if not bad else "FAIL", "%s.%s -> %s" % (t, col, ref),
            "" if not bad else "orphelins: %s" % bad[:5])

    j = dims["dim_machine"].join(
        dims["dim_production_line"].select("line_id", c("site_id").alias("line_site")),
        on="line_id", how="left")
    n = int((j["site_id"] != j["line_site"]).sum())
    log("PASS" if n == 0 else "FAIL", "dim_machine.site_id = site de sa ligne", "ecarts=%d" % n)


# ---------------------------------------------------------------- integrite referentielle
FKS = {
    "fact_sales": {"date_id": "dim_date", "product_id": "dim_product",
                   "customer_id": "dim_customer", "site_id": "dim_site"},
    "fact_inventory": {"date_id": "dim_date", "product_id": "dim_product",
                       "warehouse_id": "dim_warehouse"},
    "fact_purchase": {"date_id": "dim_date", "order_date_id": "dim_date",
                      "expected_date_id": "dim_date", "actual_date_id": "dim_date",
                      "supplier_id": "dim_supplier", "product_id": "dim_product"},
    "fact_production": {"date_id": "dim_date", "site_id": "dim_site", "line_id": "dim_production_line",
                        "machine_id": "dim_machine", "product_id": "dim_product"},
    "fact_energy": {"site_id": "dim_site", "line_id": "dim_production_line", "machine_id": "dim_machine"},
    "fact_machine_sensor": {"machine_id": "dim_machine"},
    "fact_maintenance": {"machine_id": "dim_machine", "date_id": "dim_date"},
    "fact_transport": {"date_id": "dim_date", "vehicle_id": "dim_vehicle", "warehouse_id": "dim_warehouse"},
}


def check_foreign_keys():
    print("\n== 2. Integrite referentielle ==")
    for f, cols in FKS.items():
        for col, dim in cols.items():
            vals = collect(facts[f].select(c(col).unique()))[col]
            bad = missing(vals, dims[dim][PK[dim]])
            log("PASS" if not bad else "FAIL", "%s.%s -> %s" % (f, col, dim),
                "" if not bad else "%d valeurs orphelines, ex: %s" % (len(bad), bad[:3]))

    dm = dims["dim_machine"].lazy().select("machine_id", c("site_id").alias("m_site"),
                                           c("line_id").alias("m_line"))
    for t in ("fact_production", "fact_energy"):
        q = facts[t].join(dm, on="machine_id", how="left").select(
            ((c("site_id") != c("m_site")) | (c("line_id") != c("m_line")))
            .fill_null(False).cast(pl.Int64).sum())
        n = collect(q)[0, 0]
        log("PASS" if n == 0 else "FAIL", "%s: machine/ligne/site coherents avec dim_machine" % t,
            "lignes incoherentes=%d" % n)


# ---------------------------------------------------------------- regles metier
def run_rules(table, rules):
    lf = facts[table]
    n = collect(lf.select(pl.len()))[0, 0]
    aggs = [bad.fill_null(False).cast(pl.Int64).sum().alias(name) for name, bad, _, _ in rules]
    res = collect(lf.select(aggs)).row(0, named=True)
    nulls = collect(lf.select(pl.all().null_count())).row(0, named=True)
    tot = sum(nulls.values())
    log("PASS" if tot == 0 else "FAIL", "%s: valeurs nulles" % table,
        "%s lignes, nulls=%d" % (format(n, ","), tot) + ("" if tot == 0 else " %s" % {k: v for k, v in nulls.items() if v}))
    for name, _, sev, note in rules:
        cnt = res[name]
        level = "PASS" if cnt == 0 else sev
        log(level, "%s.%s" % (table, name), "%d / %s (%.3f%%) %s" % (cnt, format(n, ","), 100.0 * cnt / n, note if cnt else ""))


def check_business_rules():
    print("\n== 3. Regles metier ==")

    run_rules("fact_sales", [
        ("quantity<=0", c("quantity") <= 0, "FAIL", ""),
        ("unit_price<=0", c("unit_price") <= 0, "FAIL", ""),
        ("discount hors [0, qty*prix] (montant)",
         (c("discount") < 0) | (c("discount") > c("quantity") * c("unit_price") + 0.01), "FAIL", ""),
        ("revenue != qty*prix - discount",
         (c("revenue") - (c("quantity") * c("unit_price") - c("discount"))).abs() > 0.02, "FAIL", ""),
        ("quantity non entiere", c("quantity") != c("quantity").floor(), "WARN", "-> Int32 en Silver impossible"),
    ])

    run_rules("fact_inventory", [
        ("stock negatif",
         (c("opening_stock") < 0) | (c("closing_stock") < 0) | (c("inbound_quantity") < 0)
         | (c("production_quantity") < 0) | (c("sales_quantity") < 0) | (c("outbound_quantity") < 0),
         "FAIL", ""),
        ("closing != opening+inbound+prod-sales-outbound",
         (c("closing_stock") - (c("opening_stock") + c("inbound_quantity") + c("production_quantity")
                                - c("sales_quantity") - c("outbound_quantity"))).abs() > 0.01,
         "FAIL", "(stock clippe a 0 ?)"),
        ("stock_value<0", c("stock_value") < 0, "FAIL", ""),
    ])

    inv = facts["fact_inventory"].join(
        dims["dim_product"].lazy().select("product_id", "unit_cost", "selling_price"),
        on="product_id", how="left")
    r = collect(inv.select(
        ((c("stock_value") - c("closing_stock") * c("unit_cost")).abs() > 0.01 * c("stock_value").abs() + 0.05)
        .cast(pl.Int64).sum().alias("vs_cost"),
        ((c("stock_value") - c("closing_stock") * c("selling_price")).abs() > 0.01 * c("stock_value").abs() + 0.05)
        .cast(pl.Int64).sum().alias("vs_price"),
        pl.len().alias("n"))).row(0, named=True)
    log("PASS" if min(r["vs_cost"], r["vs_price"]) == 0 else "WARN",
        "fact_inventory.stock_value = closing * unit_cost (ou selling_price)",
        "ecarts vs unit_cost=%d, vs selling_price=%d sur %s" % (r["vs_cost"], r["vs_price"], format(r["n"], ",")))

    dup = collect(facts["fact_inventory"].select(pl.len()))[0, 0] - collect(
        facts["fact_inventory"].select("date_id", "product_id", "warehouse_id").unique().select(pl.len()))[0, 0]
    log("PASS" if dup == 0 else "WARN", "fact_inventory: cle (date, produit, entrepot) unique",
        "doublons=%d (a dedoublonner ou agreger en Silver)" % dup)

    run_rules("fact_purchase", [
        ("quantity<=0", c("quantity") <= 0, "FAIL", ""),
        ("unit_price<=0", c("unit_price") <= 0, "FAIL", ""),
        ("total_cost != qty*prix",
         (c("total_cost") - c("quantity") * c("unit_price")).abs() > 0.02 + 1e-4 * c("total_cost").abs(),
         "WARN", ""),
        ("is_delayed incoherent avec delay_days", c("is_delayed") != (c("delay_days") > 0), "FAIL", ""),
        ("delay_days<0", c("delay_days") < 0, "WARN", ""),
        ("delay_days != actual-expected",
         c("delay_days") != (d("actual_date_id") - d("expected_date_id")).dt.total_days().clip(lower_bound=0),
         "WARN", ""),
        ("expected < order", d("expected_date_id") < d("order_date_id"), "FAIL", ""),
        ("actual < order", d("actual_date_id") < d("order_date_id"), "FAIL", ""),
    ])

    run_rules("fact_production", [
        ("quantites negatives",
         (c("planned_quantity") < 0) | (c("actual_quantity") < 0) | (c("defect_quantity") < 0), "FAIL", ""),
        ("defect > actual", c("defect_quantity") > c("actual_quantity"), "FAIL", ""),
        ("downtime<0 ou temps<0", (c("downtime_minutes") < 0) | (c("production_time_minutes") < 0), "FAIL", ""),
        ("temps+arret > 1440 min", c("production_time_minutes") + c("downtime_minutes") > 1440, "WARN", ""),
        ("actual > planned", c("actual_quantity") > c("planned_quantity"), "INFO", "(normal si faible)"),
    ])
    s = collect(facts["fact_production"].select(
        (c("actual_quantity").sum() / c("planned_quantity").sum()).alias("ach"),
        (c("defect_quantity").sum() / c("actual_quantity").sum()).alias("dr"))).row(0, named=True)
    log("INFO", "fact_production KPIs", "achievement=%.1f%%  defect_rate=%.2f%%" % (100 * s["ach"], 100 * s["dr"]))

    run_rules("fact_energy", [
        ("energy<0 ou power<0", (c("energy_kwh") < 0) | (c("power_kw") < 0), "FAIL", ""),
        ("humidity hors [0,100]", (c("humidity") < 0) | (c("humidity") > 100), "FAIL", ""),
        ("temperature hors [-10,60]", (c("temperature") < -10) | (c("temperature") > 60), "WARN", ""),
        ("energy_kwh == power_kw", c("energy_kwh") == c("power_kw"), "INFO",
         "(choix documente: lecture horaire, kWh = kW x 1h)"),
    ])

    run_rules("fact_machine_sensor", [
        ("mesures negatives",
         (c("vibration") < 0) | (c("pressure") < 0) | (c("rpm") < 0) | (c("power") < 0)
         | (c("operating_hours") < 0), "FAIL", ""),
        ("temperature hors [-20,200]", (c("temperature") < -20) | (c("temperature") > 200), "WARN", ""),
    ])

    run_rules("fact_maintenance", [
        ("duration<=0", c("duration_hours") <= 0, "FAIL", ""),
        ("cost<0", c("cost") < 0, "FAIL", ""),
    ])
    m = collect(facts["fact_maintenance"].group_by("maintenance_type").agg(
        pl.len().alias("n"), c("failure_flag").mean().alias("failure_rate")).sort("n", descending=True))
    for row in m.to_dicts():
        log("INFO", "fact_maintenance type=%s" % row["maintenance_type"],
            "n=%d  failure_flag=%.0f%%" % (row["n"], 100 * row["failure_rate"]))

    run_rules("fact_transport", [
        ("distance/durees <= 0",
         (c("distance_km") <= 0) | (c("planned_duration") <= 0) | (c("actual_duration") <= 0), "FAIL", ""),
        ("fuel<0", c("fuel_consumption") < 0, "FAIL", ""),
        ("delay != actual-planned", (c("delay_minutes") - (c("actual_duration") - c("planned_duration"))).abs() > 1,
         "INFO", "(delay peut etre clippe a 0 ou en autre unite)"),
        ("vitesse implausible (<5 ou >130 km/h)",
         (c("distance_km") / (c("actual_duration") / 60.0) > 130) | (c("distance_km") / (c("actual_duration") / 60.0) < 5),
         "WARN", "(duree supposee en minutes)"),
    ])
    st = collect(facts["fact_transport"].group_by("delivery_status").agg(
        pl.len().alias("n"), c("delay_minutes").min().alias("min"),
        c("delay_minutes").mean().alias("avg"), c("delay_minutes").max().alias("max")).sort("n", descending=True))
    total = sum(r["n"] for r in st.to_dicts())
    for r in st.to_dicts():
        log("INFO", "fact_transport status=%s" % r["delivery_status"],
            "%.1f%%  delay min/avg/max = %.0f / %.0f / %.0f" % (100.0 * r["n"] / total, r["min"], r["avg"], r["max"]))
    ontime = sum(r["n"] for r in st.to_dicts() if "time" in str(r["delivery_status"]).lower()) / total
    log("PASS" if 0.75 <= ontime <= 0.97 else "WARN", "fact_transport taux de livraisons a l'heure",
        "%.1f%% (cible realiste : 75-97%%)" % (100 * ontime))


# ---------------------------------------------------------------- fidelite aux patterns reels
def find_hourly(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "hour" in k.lower() and hasattr(v, "__len__") and len(v) == 24:
                return [v[str(h)] for h in range(24)] if isinstance(v, dict) else list(v)
        for v in obj.values():
            r = find_hourly(v)
            if r:
                return r
    return None


def check_patterns():
    print("\n== 4. Fidelite aux patterns reels ==")
    path = EXT / "energy" / "tetouan_patterns.json"
    prof = collect(facts["fact_energy"].group_by(c("timestamp").dt.hour().alias("h"))
                   .agg(c("energy_kwh").mean().alias("m")).sort("h"))
    gen = prof["m"].to_numpy()
    gen = gen / gen.mean()
    log("INFO", "fact_energy profil genere", "pic a %02dh, creux a %02dh" % (int(gen.argmax()), int(gen.argmin())))
    try:
        ref = find_hourly(json.loads(path.read_text(encoding="utf-8")))
        if ref is None:
            log("WARN", "tetouan_patterns.json", "profil horaire introuvable - cles: %s" %
                list(json.loads(path.read_text(encoding="utf-8")).keys()))
        else:
            ref = np.array(ref, dtype=float)
            corr = float(np.corrcoef(gen, ref)[0, 1])
            log("PASS" if corr > 0.8 else "WARN", "fact_energy vs profil Tetouan",
                "correlation=%.3f (pic reel %02dh)" % (corr, int(ref.argmax())))
    except FileNotFoundError:
        log("WARN", "tetouan_patterns.json", "fichier introuvable: %s" % path)

    cols = facts["fact_transport"].collect_schema().names()
    if "departure_time" not in cols:
        log("WARN", "fact_transport heure de depart absente",
            "effet heure de pointe NYC non verifiable et inutilisable pour ML-04 -> ajouter departure_time")
    else:
        h = collect(facts["fact_transport"].group_by(c("departure_time").dt.hour().alias("h")).agg(
            (c("delay_minutes") > 15).mean().alias("p_delayed")).sort("h"))
        rush = float(h.filter(c("h").is_between(8, 18))["p_delayed"].mean())
        night = float(h.filter((c("h") <= 6) | (c("h") >= 21))["p_delayed"].mean())
        log("PASS" if rush > 1.5 * night else "WARN", "fact_transport: effet heure de pointe sur les retards",
            "retards en journee (8-18h)=%.1f%% vs nuit=%.1f%%" % (100 * rush, 100 * night))
    s = collect(facts["fact_transport"].select(
        (c("actual_duration") / c("planned_duration")).log().std().alias("sigma"),
        (c("distance_km") / (c("actual_duration") / 60.0)).median().alias("speed"))).row(0, named=True)
    log("INFO", "fact_transport bruit/vitesse", "sigma ln(reel/prevu)=%.3f (NYC=0.497), vitesse mediane=%.1f km/h"
        % (s["sigma"], s["speed"]))

    t = collect(facts["fact_machine_sensor"].select(c("timestamp").min().alias("a"), c("timestamp").max().alias("b"),
                                                   c("machine_id").n_unique().alias("m"))).row(0, named=True)
    log("INFO", "fact_machine_sensor", "%s -> %s, %d machines" % (t["a"], t["b"], t["m"]))


def check_degradation():
    print("\n== 5. Signal de panne (ML-03) ==")
    ev = facts["fact_maintenance"].filter(c("failure_flag")).select(
        "machine_id", d("date_id").alias("fail_date"))
    s = facts["fact_machine_sensor"].select(
        "machine_id", c("timestamp").dt.date().alias("day"), "vibration", "temperature")
    j = s.join(ev, on="machine_id").with_columns(
        (c("fail_date") - c("day")).dt.total_days().alias("dtf"))
    r = collect(j.select(
        c("vibration").filter(c("dtf").is_between(0, 30)).mean().alias("vib_pre"),
        c("vibration").filter(c("dtf").is_between(90, 180)).mean().alias("vib_base"),
        c("temperature").filter(c("dtf").is_between(0, 30)).mean().alias("t_pre"),
        c("temperature").filter(c("dtf").is_between(90, 180)).mean().alias("t_base"))).row(0, named=True)
    if any(v is None for v in r.values()):
        log("WARN", "capteurs avant pannes corrective", "pas assez de donnees pour comparer")
        return
    ok = r["vib_pre"] > 1.15 * r["vib_base"] and r["t_pre"] > r["t_base"] + 2
    log("PASS" if ok else "FAIL", "capteurs en derive avant les pannes corrective",
        "vibration %.2f (30j avant) vs %.2f (90-180j avant) | temperature %.1f vs %.1f"
        % (r["vib_pre"], r["vib_base"], r["t_pre"], r["t_base"]))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    check_dimensions()
    check_foreign_keys()
    check_business_rules()
    check_patterns()
    check_degradation()

    counts = {k: sum(1 for r in REPORT if r["level"] == k) for k in ("PASS", "INFO", "WARN", "FAIL")}
    print("\n== Resume == %s" % counts)
    Path("data").mkdir(exist_ok=True)
    Path("data/validation_report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.exit(1 if counts["FAIL"] else 0)