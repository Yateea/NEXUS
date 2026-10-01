"""
NEXUS - Extraction des patterns M5 Forecasting (section 18 du cahier des charges).

Regle d'or (section 18) : on ne joint JAMAIS les donnees M5 directement dans NEXUS.
On extrait des PATTERNS STATISTIQUES (saisonnalite, tendance, effet promo/evenement)
et on les stocke dans un fichier JSON, qui servira ensuite a recalibrer le
generateur synthetique NEXUS (generate_facts.py) sur des bases realistes.

Prerequis : calendar.csv, sales_train_validation.csv places dans data/external/m5/

Usage (depuis la racine du projet, venv active) :
    python pipeline/ingestion/extract_m5_patterns.py

Sortie : data/external/m5/m5_patterns.json
"""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
M5_DIR = ROOT / "data" / "external" / "m5"
OUT_FILE = M5_DIR / "m5_patterns.json"

# Convention M5 : wday 1=Samedi, 2=Dimanche, 3=Lundi ... 7=Vendredi
M5_WDAY_TO_ISO = {1: 6, 2: 7, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}  # -> jour ISO (1=Lundi..7=Dimanche)


def main():
    cal_path = M5_DIR / "calendar.csv"
    sales_path = M5_DIR / "sales_train_validation.csv"
    if not cal_path.exists() or not sales_path.exists():
        raise FileNotFoundError(
            "calendar.csv et/ou sales_train_validation.csv introuvables dans {}".format(M5_DIR)
        )

    print("Lecture du calendrier...")
    calendar = pl.read_csv(cal_path)
    calendar = calendar.with_columns([
        pl.col("wday").replace_strict(M5_WDAY_TO_ISO).alias("iso_dow"),
        (pl.col("event_name_1").is_not_null() & (pl.col("event_name_1") != "")).alias("has_event"),
        ((pl.col("snap_CA") == 1) | (pl.col("snap_TX") == 1) | (pl.col("snap_WI") == 1)).alias("has_snap"),
    ])
    d_to_meta = calendar.select(["d", "iso_dow", "month", "year", "has_event", "has_snap"])

    print("Identification des colonnes de jours (d_1, d_2, ...) presentes dans les ventes...")
    header = pl.scan_csv(sales_path, n_rows=0).collect_schema().names()
    day_cols = [c for c in header if c.startswith("d_")]
    print("  {} colonnes de jours detectees".format(len(day_cols)))

    lazy_sales = pl.scan_csv(sales_path)

    # ---------------------------------------------------------- total quotidien (somme verticale)
    print("Calcul du total de ventes par jour (pattern calendrier)...")
    daily_totals_row = lazy_sales.select([pl.col(c).sum().alias(c) for c in day_cols]).collect()
    daily = daily_totals_row.transpose(include_header=True, header_name="d", column_names=["total_sales"])
    daily = daily.join(d_to_meta, on="d", how="left").drop_nulls(subset=["iso_dow"])

    overall_mean = daily["total_sales"].mean()

    dow_pattern = (
        daily.group_by("iso_dow").agg(pl.col("total_sales").mean().alias("mean_sales"))
        .sort("iso_dow")
        .with_columns((pl.col("mean_sales") / overall_mean).alias("multiplier"))
    )
    dow_multiplier = {int(r["iso_dow"]): round(float(r["multiplier"]), 4) for r in dow_pattern.to_dicts()}

    month_pattern = (
        daily.group_by("month").agg(pl.col("total_sales").mean().alias("mean_sales"))
        .sort("month")
        .with_columns((pl.col("mean_sales") / overall_mean).alias("multiplier"))
    )
    month_multiplier = {int(r["month"]): round(float(r["multiplier"]), 4) for r in month_pattern.to_dicts()}

    year_pattern = (
        daily.group_by("year").agg(pl.col("total_sales").mean().alias("mean_sales")).sort("year")
    )
    yr = year_pattern.to_dicts()
    growth_rates = []
    for i in range(1, len(yr)):
        prev, cur = yr[i - 1]["mean_sales"], yr[i]["mean_sales"]
        if prev:
            growth_rates.append((cur - prev) / prev)
    yearly_growth_rate = round(float(sum(growth_rates) / len(growth_rates)), 4) if growth_rates else 0.0

    event_mean = daily.filter(pl.col("has_event"))["total_sales"].mean()
    no_event_mean = daily.filter(~pl.col("has_event"))["total_sales"].mean()
    event_uplift_ratio = round(float(event_mean / no_event_mean), 4) if no_event_mean else 1.0

    snap_mean = daily.filter(pl.col("has_snap"))["total_sales"].mean()
    no_snap_mean = daily.filter(~pl.col("has_snap"))["total_sales"].mean()
    snap_uplift_ratio = round(float(snap_mean / no_snap_mean), 4) if no_snap_mean else 1.0

    # ---------------------------------------------------------- part par categorie (somme horizontale)
    print("Calcul de la repartition des ventes par categorie...")
    cat_totals = (
        lazy_sales
        .with_columns(pl.sum_horizontal(day_cols).alias("total_item_sales"))
        .group_by("cat_id")
        .agg(pl.col("total_item_sales").sum())
        .collect()
    )
    total_all = cat_totals["total_item_sales"].sum()
    category_share = {
        r["cat_id"]: round(float(r["total_item_sales"] / total_all), 4)
        for r in cat_totals.to_dicts()
    }

    patterns = {
        "source": "M5 Forecasting Accuracy (Kaggle/Walmart)",
        "n_days_analyzed": daily.height,
        "dow_multiplier": dow_multiplier,       # {1: Lundi ... 7: Dimanche}
        "month_multiplier": month_multiplier,   # {1: Janvier ... 12: Decembre}
        "yearly_growth_rate": yearly_growth_rate,
        "event_uplift_ratio": event_uplift_ratio,   # ventes moyennes les jours d'evenement vs normal
        "snap_uplift_ratio": snap_uplift_ratio,     # proxy promo/aide alimentaire
        "category_share": category_share,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    print("\nPatterns extraits :")
    print(json.dumps(patterns, indent=2, ensure_ascii=False))
    print("\nEcrit dans {}".format(OUT_FILE))


if __name__ == "__main__":
    main()
