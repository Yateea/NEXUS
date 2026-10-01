"""
NEXUS - Extraction des patterns de degradation NASA C-MAPSS (section 18).

Regle d'or (section 18) : pas de jointure directe NASA -> NEXUS. On extrait la
FORME de la courbe de degradation (comment un capteur evolue a mesure qu'une
machine approche de la panne) et on la stocke, pour remplacer la rampe lineaire
simpliste actuellement utilisee dans generate_facts.py par une courbe realiste
(la degradation reelle accelere generalement en fin de vie, elle n'est pas lineaire).

Methode :
  1. Chargement de train_FD001.txt (colonnes : unit_number, cycle, 3 settings,
     21 capteurs - section 16).
  2. Pour chaque machine (unit), on calcule sa duree de vie totale et sa
     progression relative (life_fraction = cycle / duree_de_vie, de 0 a 1).
  3. Detection automatique des capteurs "informatifs" (ceux dont la valeur est
     correlee a life_fraction) et rejet des capteurs plats (bruit constant),
     sans supposer a l'avance lesquels le sont (section 16 : "ne pas inventer
     que sensor_2 = temperature").
  4. Pour les capteurs informatifs, normalisation par machine (min-max), puis
     moyenne par tranche de life_fraction (20 tranches) -> courbe moyenne de
     degradation (a quel rythme la degradation s'accelere).
  5. Sauvegarde de cette courbe + de la distribution des durees de vie dans
     data/external/nasa/nasa_patterns.json.

Usage (depuis la racine du projet, venv active) :
    python pipeline/ingestion/extract_nasa_patterns.py
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
NASA_DIR = ROOT / "data" / "external" / "nasa" / "C-MAPSS"
OUT_FILE = ROOT / "data" / "external" / "nasa" / "nasa_patterns.json"

N_BINS = 20
CORR_THRESHOLD = 0.5  # |correlation( capteur, life_fraction )| minimale pour etre juge "informatif"


def load_cmapss(path):
    cols = ["unit", "cycle", "op1", "op2", "op3"] + ["sensor_{:02d}".format(i) for i in range(1, 22)]
    df = pl.read_csv(
        path, separator=" ", has_header=False, new_columns=cols,
        infer_schema_length=10000,
    )
    # Le format C-MAPSS a des espaces multiples -> colonnes fantomes (null) en fin de ligne
    df = df.select([c for c in df.columns if df[c].null_count() < df.height])
    return df


def main():
    train_path = NASA_DIR / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError("train_FD001.txt introuvable dans {}".format(NASA_DIR))

    print("Lecture de train_FD001.txt...")
    df = load_cmapss(train_path)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    print("  {} lignes, {} unites (machines), {} capteurs".format(
        df.height, df["unit"].n_unique(), len(sensor_cols)))

    print("Calcul de la progression de vie (life_fraction) par machine...")
    life = df.group_by("unit").agg(pl.col("cycle").max().alias("max_cycle"))
    df = df.join(life, on="unit").with_columns(
        (pl.col("cycle") / pl.col("max_cycle")).alias("life_fraction")
    )

    print("Detection des capteurs informatifs (correles a la progression de vie)...")
    correlations = {}
    for s in sensor_cols:
        if df[s].std() is None or df[s].std() < 1e-9:
            continue
        c = df.select(pl.corr("life_fraction", s)).item()
        if c is not None and not np.isnan(c):
            correlations[s] = float(c)

    informative = [s for s, c in correlations.items() if abs(c) >= CORR_THRESHOLD]
    if not informative:
        informative = sorted(correlations, key=lambda s: -abs(correlations[s]))[:5]
    print("  Capteurs informatifs retenus : {}".format(informative))
    for s in informative:
        print("    {} : correlation = {:+.3f}".format(s, correlations[s]))

    print("Normalisation par machine et calcul de la courbe moyenne de degradation...")
    df = df.with_columns([
        ((pl.col(s) - pl.col(s).min().over("unit")) /
         (pl.col(s).max().over("unit") - pl.col(s).min().over("unit") + 1e-9)).alias(s + "_norm")
        for s in informative
    ])
    # sens de degradation uniforme (min-max ci-dessus suffit : 0=etat initial, 1=etat le plus extreme observe)

    combined = df.select(
        ["life_fraction"] + [s + "_norm" for s in informative]
    ).with_columns(
        pl.mean_horizontal([s + "_norm" for s in informative]).alias("degradation_index")
    )

    bin_idx = (pl.col("life_fraction") * N_BINS).floor().clip(0, N_BINS - 1).cast(pl.Int32)
    combined = combined.with_columns(bin_idx.alias("bin_idx"))
    curve = (
        combined.group_by("bin_idx")
        .agg(pl.col("degradation_index").mean().alias("mean_degradation"))
        .sort("bin_idx")
        .with_columns(((pl.col("bin_idx") + 0.5) / N_BINS).alias("bin"))
    )

    # re-normalise la courbe finale entre 0 (debut de vie) et 1 (fin de vie, juste avant panne)
    dmin, dmax = curve["mean_degradation"].min(), curve["mean_degradation"].max()
    curve = curve.with_columns(
        ((pl.col("mean_degradation") - dmin) / (dmax - dmin + 1e-9)).alias("degradation_normalized")
    )

    life_stats = df.select("unit", "max_cycle").unique()
    patterns = {
        "source": "NASA C-MAPSS FD001 (Turbofan Engine Degradation Simulation)",
        "n_units": df["unit"].n_unique(),
        "informative_sensors": informative,
        "sensor_correlations": {s: round(correlations[s], 4) for s in informative},
        "mean_life_cycles": round(float(life_stats["max_cycle"].mean()), 1),
        "std_life_cycles": round(float(life_stats["max_cycle"].std()), 1),
        "degradation_curve": {
            "life_fraction": [float(x) for x in curve["bin"].to_list()],
            "degradation_index": [round(float(x), 4) for x in curve["degradation_normalized"].to_list()],
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    print("\nCourbe de degradation moyenne (0=machine neuve, 1=juste avant panne) :")
    for lf, di in zip(patterns["degradation_curve"]["life_fraction"], patterns["degradation_curve"]["degradation_index"]):
        bar = "#" * int(di * 40)
        print("  life_fraction={:.2f}  degradation={:.3f}  {}".format(lf, di, bar))

    print("\nDuree de vie moyenne des machines : {:.0f} cycles (ecart-type {:.0f})".format(
        patterns["mean_life_cycles"], patterns["std_life_cycles"]))
    print("Ecrit dans {}".format(OUT_FILE))


if __name__ == "__main__":
    main()
