"""
NEXUS - ML-04 Delay Prediction (section 11 du cahier des charges).

Objectif : predire si une livraison risque d'etre en retard, et de combien.

Approche :
  - 2 modeles sur les memes features :
      1. Classifieur (HistGradientBoostingClassifier) : probabilite de retard
         (delivery_status == 'delayed', i.e. delay > 15 min).
      2. Regresseur (HistGradientBoostingRegressor) : retard attendu en minutes.
  - Features : distance, heure de depart (calibree sur le vrai pattern NYC TLC,
    section 18), jour de la semaine, type/consommation du vehicule, moyenne
    historique de retard sur la destination (calculee UNIQUEMENT sur le train,
    pour eviter toute fuite de donnees vers le test).
  - Validation TEMPORELLE (section 31) : train = avant le cutoff, test = les
    derniers jours.
  - Evaluation : Precision/Recall/F1 pour le classifieur, MAE/RMSE pour le
    regresseur (section 30).
  - Les predictions sont ecrites dans fact_delay_prediction, avec le retard
    reellement observe a cote pour validation/Power BI.

Prerequis : migrations 03_add_transport_departure_time.sql et
04_add_delay_prediction.sql deja appliquees.

Usage (depuis la racine du projet, venv active) :
    python ml/delay_prediction/train_delay_prediction.py
"""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL_NAME = "HistGradientBoosting-v1"
DELAY_THRESHOLD_MIN = 15


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "nexus_user"),
        password=os.getenv("POSTGRES_PASSWORD", "nexus_pass"),
        dbname=os.getenv("POSTGRES_DB", "nexus_db"),
    )


def fetch_df(conn, query, params=None):
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pl.DataFrame(rows, schema=cols, orient="row")


# ------------------------------------------------------------------ donnees
def load_data(conn):
    print("Chargement des livraisons (fact_transport x dim_vehicle)...")
    df = fetch_df(conn, """
        SELECT t.transport_id, t.date_id, t.vehicle_id, t.destination,
               t.departure_timestamp, t.distance_km::float AS distance_km,
               t.planned_duration::float AS planned_duration,
               t.delay_minutes::float AS delay_minutes,
               t.delivery_status,
               v.vehicle_type, v.fuel_type, v.consumption_per_100km::float AS consumption_per_100km
        FROM fact_transport t
        JOIN dim_vehicle v ON v.vehicle_id = t.vehicle_id
        WHERE t.delivery_status != 'failed'
        ORDER BY t.departure_timestamp
    """)
    print("  {:,} livraisons chargees".format(df.height))
    return df


def build_features(df):
    print("Feature engineering (heure, jour, historique route)...")
    df = df.with_columns([
        pl.col("departure_timestamp").dt.hour().alias("departure_hour"),
        pl.col("departure_timestamp").dt.weekday().alias("day_of_week"),
        (pl.col("delay_minutes") > DELAY_THRESHOLD_MIN).cast(pl.Int8).alias("label_delayed"),
    ])

    # encodage categoriel simple (peu de modalites)
    for col in ["vehicle_type", "fuel_type", "destination"]:
        cats = df[col].unique().sort().to_list()
        cat_map = {c: i for i, c in enumerate(cats)}
        df = df.with_columns(pl.col(col).replace_strict(cat_map, default=-1).alias(col + "_code"))

    return df


NUMERIC_FEATURES = [
    "distance_km", "planned_duration", "departure_hour", "day_of_week",
    "consumption_per_100km",
]
CATEGORICAL_FEATURES = ["vehicle_type_code", "fuel_type_code", "destination_code"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["destination_historical_delay"]
CATEGORICAL_IDX = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]


def add_route_history(train, test):
    """Moyenne historique de retard par destination, calculee UNIQUEMENT sur le train
    (evite toute fuite de donnees du futur vers le passe ou du test vers le train)."""
    route_avg = train.group_by("destination").agg(pl.col("delay_minutes").mean().alias("destination_historical_delay"))
    global_avg = train["delay_minutes"].mean()
    train = train.join(route_avg, on="destination", how="left")
    test = test.join(route_avg, on="destination", how="left").with_columns(
        pl.col("destination_historical_delay").fill_null(global_avg)
    )
    return train, test


# ------------------------------------------------------------------ train / eval
def temporal_split(df, test_days):
    cutoff_dates = df.select(pl.col("date_id").unique().sort()).to_series()
    cutoff = cutoff_dates[-test_days] if len(cutoff_dates) > test_days else cutoff_dates[0]
    train = df.filter(pl.col("date_id") < cutoff)
    test = df.filter(pl.col("date_id") >= cutoff)
    return train, test, cutoff


def train_classifier(train, test):
    X_train, y_train = train.select(FEATURES).to_numpy(), train["label_delayed"].to_numpy()
    X_test, y_test = test.select(FEATURES).to_numpy(), test["label_delayed"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6, random_state=42,
        categorical_features=CATEGORICAL_IDX,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print("  [Classifieur] entrainement {:.1f}s  Precision={:.3f}  Recall={:.3f}  F1={:.3f}  ({} retards reels / {})".format(
        time.time() - t0, precision, recall, f1, int(y_test.sum()), len(y_test)))
    return model, {"precision": precision, "recall": recall, "f1": f1}


def train_regressor(train, test):
    X_train, y_train = train.select(FEATURES).to_numpy(), train["delay_minutes"].to_numpy()
    X_test, y_test = test.select(FEATURES).to_numpy(), test["delay_minutes"].to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_depth=6, random_state=42,
        categorical_features=CATEGORICAL_IDX,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print("  [Regresseur]  entrainement {:.1f}s  MAE={:.2f} min  RMSE={:.2f} min".format(
        time.time() - t0, mae, rmse))
    return model, {"mae": mae, "rmse": rmse}


# ------------------------------------------------------------------ ecriture
def write_predictions(conn, test, clf, reg):
    print("Generation des predictions sur le jeu de test...")
    X_test = test.select(FEATURES).to_numpy()
    proba = clf.predict_proba(X_test)[:, 1]
    expected_delay = np.clip(reg.predict(X_test), 0, None)

    out = test.select(["transport_id", "delay_minutes"]).rename({"delay_minutes": "actual_delay_minutes"}).with_columns([
        pl.Series("delay_probability", np.round(proba, 4)),
        pl.Series("expected_delay_minutes", np.round(expected_delay, 2)),
        pl.lit(MODEL_NAME).alias("model_name"),
    ])

    with conn.cursor() as cur:
        cur.execute("TRUNCATE fact_delay_prediction RESTART IDENTITY")
    conn.commit()

    buf = io.BytesIO()
    out.write_csv(buf)
    buf.seek(0)
    cols = ",".join(out.columns)
    with conn.cursor() as cur:
        cur.copy_expert("COPY fact_delay_prediction ({}) FROM STDIN WITH (FORMAT csv, HEADER true)".format(cols), buf)
    conn.commit()
    print("  {:,} predictions ecrites dans fact_delay_prediction".format(out.height))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()

    t0 = time.time()
    conn = get_conn()

    df = load_data(conn)
    df = build_features(df)
    train, test, cutoff = temporal_split(df, args.test_days)
    train, test = add_route_history(train, test)
    print("  train: {:,} livraisons  |  test: {:,} livraisons  (cutoff={})".format(train.height, test.height, cutoff))
    print("  taux de retard (train): {:.1f}%  (test): {:.1f}%".format(
        100 * train["label_delayed"].mean(), 100 * test["label_delayed"].mean()))

    clf, clf_metrics = train_classifier(train, test)
    reg, reg_metrics = train_regressor(train, test)

    preds = write_predictions(conn, test, clf, reg)

    print("\nExemple de predictions (10 premieres livraisons du test) :")
    print(preds.head(10))

    conn.close()
    print("\nTermine en {:.1f}s.".format(time.time() - t0))
    print("Classifieur : Precision={:.3f}  Recall={:.3f}  F1={:.3f}".format(
        clf_metrics["precision"], clf_metrics["recall"], clf_metrics["f1"]))
    print("Regresseur  : MAE={:.2f} min  RMSE={:.2f} min".format(reg_metrics["mae"], reg_metrics["rmse"]))


if __name__ == "__main__":
    main()
