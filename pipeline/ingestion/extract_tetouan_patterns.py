"""
NEXUS - Extraction des patterns de consommation energetique UCI Tetouan (section 18).

Regle d'or (section 18) : pas de jointure directe Tetouan -> NEXUS. On extrait
la FORME du profil de consommation (courbe horaire, effet week-end, sensibilite
a la temperature) pour remplacer la courbe sinusoidale inventee dans
generate_facts.py par un profil calibre sur une vraie ville.

Methode :
  1. Chargement du CSV (colonnes : DateTime, Temperature, Humidity, Wind Speed,
     general diffuse flows, diffuse flows, Zone 1/2/3 Power Consumption).
  2. Consommation totale = somme des 3 zones.
  3. Profil horaire (24 valeurs, normalise autour de 1.0) : a quelle heure la
     ville consomme le plus/moins.
  4. Effet jour de semaine vs week-end.
  5. Sensibilite a la temperature : regression simple pour estimer de combien
     varie la consommation par degre C (effet climatisation/chauffage).

Usage (depuis la racine du projet, venv active) :
    python pipeline/ingestion/extract_tetouan_patterns.py

Sortie : data/external/energy/tetouan_patterns.json
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
ENERGY_DIR = ROOT / "data" / "external" / "energy"
OUT_FILE = ENERGY_DIR / "tetouan_patterns.json"


def find_csv():
    candidates = list(ENERGY_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Aucun fichier .csv trouve dans {}".format(ENERGY_DIR))
    # prend le plus gros fichier (evite d'attraper un patterns.json ou autre par erreur)
    return max(candidates, key=lambda p: p.stat().st_size)


def main():
    csv_path = find_csv()
    print("Lecture de {}...".format(csv_path.name))

    df = pl.read_csv(csv_path, try_parse_dates=False)
    df.columns = [c.strip() for c in df.columns]

    zone_cols = [c for c in df.columns if "Zone" in c and "Power" in c]
    print("  {} lignes, colonnes zones detectees : {}".format(df.height, zone_cols))

    df = df.with_columns(pl.sum_horizontal(zone_cols).alias("total_consumption"))

    dt_col = "DateTime" if "DateTime" in df.columns else df.columns[0]
    df = df.with_columns(
        pl.col(dt_col).str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M", strict=False).alias("ts")
    )
    if df["ts"].null_count() == df.height:
        # format de date alternatif (jour/mois/annee ou iso)
        df = df.with_columns(pl.col(dt_col).str.strptime(pl.Datetime, strict=False).alias("ts"))

    df = df.with_columns([
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.weekday().alias("dow"),  # 1=Lundi .. 7=Dimanche (convention polars)
        pl.col("ts").dt.month().alias("month"),
    ])

    overall_mean = df["total_consumption"].mean()

    print("Calcul du profil horaire...")
    hourly = (
        df.group_by("hour").agg(pl.col("total_consumption").mean().alias("mean_c"))
        .sort("hour")
        .with_columns((pl.col("mean_c") / overall_mean).alias("multiplier"))
    )
    hourly_multiplier = {int(r["hour"]): round(float(r["multiplier"]), 4) for r in hourly.to_dicts()}

    print("Calcul de l'effet jour de semaine / week-end...")
    dow_pattern = (
        df.group_by("dow").agg(pl.col("total_consumption").mean().alias("mean_c"))
        .sort("dow")
        .with_columns((pl.col("mean_c") / overall_mean).alias("multiplier"))
    )
    dow_multiplier = {int(r["dow"]): round(float(r["multiplier"]), 4) for r in dow_pattern.to_dicts()}

    print("Estimation de la sensibilite a la temperature...")
    temp_col = "Temperature"
    valid = df.select([temp_col, "total_consumption"]).drop_nulls()
    x = valid[temp_col].to_numpy()
    y = valid["total_consumption"].to_numpy()
    # regression lineaire simple (moindres carres) : y = a*x + b
    a, b = np.polyfit(x, y, 1)
    temp_sensitivity_pct_per_degree = round(float(a / overall_mean * 100), 3)
    correlation = float(np.corrcoef(x, y)[0, 1])

    patterns = {
        "source": "UCI Power Consumption of Tetouan City",
        "n_records": df.height,
        "hourly_multiplier": hourly_multiplier,
        "dow_multiplier": dow_multiplier,
        "temperature_correlation": round(correlation, 4),
        "temp_sensitivity_pct_per_degree": temp_sensitivity_pct_per_degree,
        "mean_temperature": round(float(x.mean()), 2),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    print("\nProfil horaire de consommation (1.0 = moyenne journaliere) :")
    for h in range(24):
        m = hourly_multiplier.get(h, 1.0)
        bar = "#" * int(m * 30)
        print("  {:02d}h  x{:.3f}  {}".format(h, m, bar))

    print("\nEffet jour de semaine (1=Lundi ... 7=Dimanche) :")
    for d in range(1, 8):
        print("  jour {} : x{:.3f}".format(d, dow_multiplier.get(d, 1.0)))

    print("\nCorrelation temperature/consommation : {:+.3f}".format(correlation))
    print("Sensibilite : {:+.3f}% de consommation par degre C".format(temp_sensitivity_pct_per_degree))
    print("Ecrit dans {}".format(OUT_FILE))


if __name__ == "__main__":
    main()
