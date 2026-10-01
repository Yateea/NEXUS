"""
NEXUS - Extraction des patterns NYC TLC (section 18 du cahier des charges).

Regle d'un : pas de jointure directe NYC TLC -> NEXUS, et pas de transfert
litteral des vitesses (un taxi a Manhattan et un camion de livraison sur
autoroute marocaine n'ont pas la meme vitesse absolue). On extrait 2 patterns
RELATIFS, transposables a n'importe quel contexte de transport routier :

  1. L'effet heures de pointe sur la vitesse (multiplicateur relatif par heure
     de la journee) : le ralentissement aux heures de pointe est un phenomene
     universel, meme si les vitesses absolues different.
  2. La variabilite reelle des temps de trajet (bruit multiplicatif log-normal
     entre duree prevue et duree reelle) : capture le niveau d'alea du trafic
     reel, plus fiable qu'une variance inventee a la main.

La vitesse de base (55 km/h) et le temps fixe de chargement (25 min) restent
des hypotheses NEXUS propres au contexte logistique marocain (autoroute inter-
villes), pas des valeurs importees de New York.

Usage (depuis la racine du projet, venv active) :
    python pipeline/ingestion/extract_nyc_tlc_patterns.py

Sortie : data/external/transport/nyc_tlc_patterns.json
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
NYC_DIR = ROOT / "data" / "external" / "transport" / "yellow_taxi"
OUT_FILE = ROOT / "data" / "external" / "transport" / "nyc_tlc_patterns.json"

MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 180
MIN_DISTANCE_MI = 0.1
MAX_DISTANCE_MI = 100
MIN_SPEED_MPH = 1
MAX_SPEED_MPH = 80


def main():
    files = sorted(NYC_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Aucun fichier .parquet trouve dans {}".format(NYC_DIR))
    print("Lecture de {} fichier(s) : {}".format(len(files), [f.name for f in files]))

    lazy = pl.scan_parquet([str(f) for f in files])
    cols = lazy.collect_schema().names()
    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in cols else cols[1]
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in cols else cols[2]

    df = (
        lazy.select([pickup_col, dropoff_col, "trip_distance"])
        .with_columns([
            ((pl.col(dropoff_col) - pl.col(pickup_col)).dt.total_seconds() / 60.0).alias("duration_min"),
            pl.col(pickup_col).dt.hour().alias("hour"),
        ])
        .filter(
            (pl.col("duration_min") >= MIN_DURATION_MIN) & (pl.col("duration_min") <= MAX_DURATION_MIN) &
            (pl.col("trip_distance") >= MIN_DISTANCE_MI) & (pl.col("trip_distance") <= MAX_DISTANCE_MI)
        )
        .with_columns((pl.col("trip_distance") / (pl.col("duration_min") / 60.0)).alias("speed_mph"))
        .filter((pl.col("speed_mph") >= MIN_SPEED_MPH) & (pl.col("speed_mph") <= MAX_SPEED_MPH))
        .collect()
    )
    print("  {:,} trajets valides apres nettoyage (GPS/valeurs aberrantes exclues)".format(df.height))

    print("Calcul de la vitesse mediane de reference...")
    ref_speed = df["speed_mph"].median()

    print("Calcul de l'effet heure de pointe (vitesse relative par heure)...")
    by_hour = (
        df.group_by("hour").agg(pl.col("speed_mph").median().alias("median_speed"))
        .sort("hour")
        .with_columns((pl.col("median_speed") / ref_speed).alias("multiplier"))
    )
    hourly_speed_multiplier = {int(r["hour"]): round(float(r["multiplier"]), 4) for r in by_hour.to_dicts()}

    print("Calcul de la variabilite reelle des temps de trajet (bruit log-normal)...")
    df = df.with_columns(
        (pl.col("duration_min") / (pl.col("trip_distance") / ref_speed * 60.0)).alias("ratio_actual_expected")
    )
    log_ratio = np.log(df["ratio_actual_expected"].to_numpy())
    log_ratio = log_ratio[np.isfinite(log_ratio)]
    noise_mu = float(np.mean(log_ratio))
    noise_sigma = float(np.std(log_ratio))

    patterns = {
        "source": "NYC TLC Yellow Taxi Trip Records",
        "n_trips_analyzed": df.height,
        "reference_speed_mph": round(float(ref_speed), 2),
        "hourly_speed_multiplier": hourly_speed_multiplier,
        "duration_noise_lognormal_mu": round(noise_mu, 4),
        "duration_noise_lognormal_sigma": round(noise_sigma, 4),
        "note": "hourly_speed_multiplier et le bruit log-normal sont des patterns RELATIFS, "
                "applicables a la vitesse de base NEXUS (55 km/h autoroute) -- "
                "la vitesse absolue NYC (trafic urbain) n'est pas transferee telle quelle.",
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    print("\nEffet heure de pointe (1.0 = vitesse mediane globale) :")
    for h in range(24):
        m = hourly_speed_multiplier.get(h, 1.0)
        bar = "#" * int(m * 30)
        tag = " <-- ralentissement" if m < 0.85 else ""
        print("  {:02d}h  x{:.3f}  {}{}".format(h, m, bar, tag))

    print("\nVitesse mediane de reference : {:.1f} mph".format(ref_speed))
    print("Bruit reel (log-normal) : mu={:+.4f}  sigma={:.4f}".format(noise_mu, noise_sigma))
    print("Ecrit dans {}".format(OUT_FILE))


if __name__ == "__main__":
    main()
