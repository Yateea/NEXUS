"""
NEXUS - ML-02 Anomaly Detection (section 11 du cahier des charges).

Detecte automatiquement : consommation energetique inhabituelle, degradation
capteur machine, baisse de production / hausse des defauts, retards de
livraison inhabituels. Un seul algorithme (Isolation Forest) applique a
4 domaines, comme recommande dans le cahier des charges.

Principe :
  - Pour chaque domaine, on normalise les features (z-score, par entite quand
    pertinent : par machine, par site) pour que l'algorithme compare des
    valeurs comparables (un site plus gros ne doit pas sembler "anormal"
    juste parce qu'il consomme plus dans l'absolu).
  - IsolationForest(contamination=0.02) : ~2% des points les plus atypiques
    sont marques anomalies.
  - Parmi les anomalies detectees, la severite (LOW/MEDIUM/HIGH/CRITICAL) est
    attribuee par quantile du score d'anomalie (les plus atypiques = CRITICAL).
  - Tout est ecrit dans fact_anomaly (table unique, section 15), consultable
    par l'Anomaly Center (Power BI, section 20) et /api/anomalies (FastAPI).

Usage (depuis la racine du projet, venv active) :
    python ml/anomaly_detection/train_anomaly_detection.py
    python ml/anomaly_detection/train_anomaly_detection.py --contamination 0.03
"""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL_NAME = "IsolationForest-v1"


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


# ------------------------------------------------------------------ coeur : detection generique
def zscore_by_group(df, features, group_col):
    """Normalise chaque feature (moyenne/ecart-type) au sein de chaque groupe (ex: par machine)."""
    out = df.copy()
    for f in features:
        g = out.groupby(group_col)[f]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan).fillna(1.0)
        out[f + "_z"] = (out[f] - mean) / std
    return out


def detect(df_pd, features, group_col=None, contamination=0.02):
    """
    Renvoie (is_anomaly: bool array, anomaly_score: float array [plus haut = plus anormal]).
    """
    if group_col:
        df_pd = zscore_by_group(df_pd, features, group_col)
        X = df_pd[[f + "_z" for f in features]].to_numpy()
    else:
        X = df_pd[features].to_numpy()
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    model = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X)
    pred = model.predict(X)                 # -1 = anomalie, 1 = normal
    raw_score = model.decision_function(X)  # + haut = + normal
    anomaly_score = -raw_score              # + haut = + anormal (plus intuitif)
    return pred == -1, anomaly_score


def assign_severity(scores):
    """severite par quantile parmi les points deja identifies comme anomalies."""
    q90, q70, q40 = np.quantile(scores, [0.90, 0.70, 0.40])
    sev = np.where(scores >= q90, "CRITICAL",
          np.where(scores >= q70, "HIGH",
          np.where(scores >= q40, "MEDIUM", "LOW")))
    return sev


# ------------------------------------------------------------------ domaine : ENERGY
def process_energy(conn, contamination):
    print("Domaine ENERGY...")
    df = fetch_df(conn, """
        SELECT energy_id, timestamp, site_id, machine_id, energy_kwh::float AS energy_kwh, power_kw::float AS power_kw, temperature::float AS temperature
        FROM fact_energy
    """).to_pandas()
    if df.empty:
        return pd.DataFrame()

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    features = ["energy_kwh", "power_kw", "temperature", "hour", "dow"]

    is_anom, score = detect(df, features, group_col="machine_id", contamination=contamination)
    sub = df[is_anom].copy()
    sub["anomaly_score"] = score[is_anom]
    sub["severity"] = assign_severity(sub["anomaly_score"])
    baseline = df.groupby("machine_id")["energy_kwh"].transform("mean")
    sub["expected_value"] = baseline[is_anom].values

    return pd.DataFrame({
        "timestamp": sub["timestamp"],
        "domain": "energy",
        "entity_id": sub["machine_id"],
        "metric": "energy_kwh",
        "actual_value": sub["energy_kwh"],
        "expected_value": sub["expected_value"],
        "anomaly_score": sub["anomaly_score"],
        "severity": sub["severity"],
        "model": MODEL_NAME,
    })


# ------------------------------------------------------------------ domaine : MACHINE (capteurs)
def process_machine(conn, contamination):
    print("Domaine MACHINE (capteurs)...")
    df = fetch_df(conn, """
        SELECT sensor_reading_id, timestamp, machine_id, temperature::float AS temperature, vibration::float AS vibration, pressure::float AS pressure, rpm::float AS rpm, power::float AS power
        FROM fact_machine_sensor
    """).to_pandas()
    if df.empty:
        return pd.DataFrame()

    features = ["temperature", "vibration", "pressure", "rpm", "power"]
    is_anom, score = detect(df, features, group_col="machine_id", contamination=contamination)
    sub = df[is_anom].copy()
    sub["anomaly_score"] = score[is_anom]
    sub["severity"] = assign_severity(sub["anomaly_score"])
    baseline = df.groupby("machine_id")["temperature"].transform("mean")
    sub["expected_value"] = baseline[is_anom].values

    return pd.DataFrame({
        "timestamp": sub["timestamp"],
        "domain": "machine",
        "entity_id": sub["machine_id"],
        "metric": "temperature_vibration_pattern",
        "actual_value": sub["temperature"],
        "expected_value": sub["expected_value"],
        "anomaly_score": sub["anomaly_score"],
        "severity": sub["severity"],
        "model": MODEL_NAME,
    })


# ------------------------------------------------------------------ domaine : PRODUCTION
def process_production(conn, contamination):
    print("Domaine PRODUCTION...")
    df = fetch_df(conn, """
        SELECT production_id, date_id, machine_id, planned_quantity::float AS planned_quantity, actual_quantity::float AS actual_quantity,
               defect_quantity::float AS defect_quantity, downtime_minutes::float AS downtime_minutes
        FROM fact_production
    """).to_pandas()
    if df.empty:
        return pd.DataFrame()

    df["achievement_rate"] = df["actual_quantity"] / df["planned_quantity"].replace(0, np.nan)
    df["defect_rate"] = df["defect_quantity"] / df["actual_quantity"].replace(0, np.nan)
    df = df.fillna(0)
    features = ["achievement_rate", "defect_rate", "downtime_minutes"]

    is_anom, score = detect(df, features, group_col=None, contamination=contamination)
    sub = df[is_anom].copy()
    sub["anomaly_score"] = score[is_anom]
    sub["severity"] = assign_severity(sub["anomaly_score"])

    ts = pd.to_datetime(sub["date_id"].astype(str), format="%Y%m%d")
    return pd.DataFrame({
        "timestamp": ts,
        "domain": "production",
        "entity_id": sub["machine_id"],
        "metric": "defect_rate",
        "actual_value": sub["defect_rate"],
        "expected_value": df["defect_rate"].mean(),
        "anomaly_score": sub["anomaly_score"],
        "severity": sub["severity"],
        "model": MODEL_NAME,
    })


# ------------------------------------------------------------------ domaine : TRANSPORT
def process_transport(conn, contamination):
    print("Domaine TRANSPORT...")
    df = fetch_df(conn, """
        SELECT transport_id, date_id, vehicle_id, distance_km::float AS distance_km, planned_duration::float AS planned_duration,
               actual_duration::float AS actual_duration, delay_minutes::float AS delay_minutes
        FROM fact_transport
    """).to_pandas()
    if df.empty:
        return pd.DataFrame()

    features = ["distance_km", "planned_duration", "delay_minutes"]
    is_anom, score = detect(df, features, group_col=None, contamination=contamination)
    sub = df[is_anom].copy()
    sub["anomaly_score"] = score[is_anom]
    sub["severity"] = assign_severity(sub["anomaly_score"])

    ts = pd.to_datetime(sub["date_id"].astype(str), format="%Y%m%d")
    return pd.DataFrame({
        "timestamp": ts,
        "domain": "transport",
        "entity_id": sub["vehicle_id"],
        "metric": "delay_minutes",
        "actual_value": sub["delay_minutes"],
        "expected_value": df["delay_minutes"].mean(),
        "anomaly_score": sub["anomaly_score"],
        "severity": sub["severity"],
        "model": MODEL_NAME,
    })


# ------------------------------------------------------------------ ecriture
def write_anomalies(conn, df):
    if df.empty:
        print("Aucune anomalie a ecrire.")
        return
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fact_anomaly RESTART IDENTITY")
    conn.commit()

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    cols = ",".join(df.columns)
    with conn.cursor() as cur:
        cur.copy_expert("COPY fact_anomaly ({}) FROM STDIN WITH (FORMAT csv, HEADER true)".format(cols), buf)
    conn.commit()
    print("{:,} anomalies ecrites dans fact_anomaly".format(len(df)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contamination", type=float, default=0.02, help="proportion attendue d'anomalies (0.02 = 2%%)")
    args = parser.parse_args()

    t0 = time.time()
    conn = get_conn()

    parts = [
        process_energy(conn, args.contamination),
        process_machine(conn, args.contamination),
        process_production(conn, args.contamination),
        process_transport(conn, args.contamination),
    ]
    all_anomalies = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    print("\nRepartition par domaine :")
    print(all_anomalies.groupby("domain").size().to_string())
    print("\nRepartition par severite :")
    print(all_anomalies.groupby("severity").size().to_string())

    write_anomalies(conn, all_anomalies)
    conn.close()
    print("\nTermine en {:.1f}s.".format(time.time() - t0))


if __name__ == "__main__":
    main()
