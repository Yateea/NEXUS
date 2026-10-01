"""
NEXUS - ML-01 Demand Forecasting (section 11 du cahier des charges).

Objectif : prevoir la demande (quantite vendue) par produit a +1, +2, +3, +7 jours.

Approche :
  - Agregation quotidienne des ventes par produit (fact_sales), completee avec les
    jours sans vente (quantite = 0), pour obtenir une vraie serie temporelle continue.
  - Features : lags (J-1, J-7), moyennes mobiles (7j, 28j), calendrier (jour de semaine,
    mois, weekend, ferie - dim_date), tendance temporelle, attributs produit (categorie,
    prix, cout).
  - 1 seul modele global (HistGradientBoostingRegressor) sur tous les produits, plus
    2 modeles quantile (10e/90e percentile) pour l'intervalle de confiance.
  - Validation TEMPORELLE (pas de split aleatoire) : train = tout sauf les N derniers
    jours, test = les N derniers jours (section 31 du cahier des charges).
  - Previsions ecrites dans fact_demand_forecast (+1, +2, +3, +7 jours a partir de la
    derniere date disponible), pour tous les produits ou un sous-ensemble (--top-n).

Usage (depuis la racine du projet, venv active, PostgreSQL demarre) :
    python ml/forecasting/train_demand_forecast.py                     # tous les produits
    python ml/forecasting/train_demand_forecast.py --top-n 300         # 300 produits les + vendus (rapide)
    python ml/forecasting/train_demand_forecast.py --skip-intervals    # sans bornes de confiance (plus rapide)
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

HORIZONS = [1, 2, 3, 7]
MODEL_NAME = "HistGradientBoostingRegressor-v1"


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


# ------------------------------------------------------------------ data
def load_data(conn, top_n):
    print("Chargement des donnees depuis PostgreSQL...")

    if top_n:
        products = fetch_df(conn, """
            SELECT product_id FROM fact_sales
            GROUP BY product_id ORDER BY SUM(quantity) DESC LIMIT %s
        """, (top_n,))
    else:
        products = fetch_df(conn, "SELECT product_id FROM dim_product")
    product_ids = products["product_id"].to_list()
    print("  {} produits retenus".format(len(product_ids)))

    daily_sales = fetch_df(conn, """
        SELECT date_id, product_id, SUM(quantity)::float AS quantity
        FROM fact_sales
        WHERE product_id = ANY(%s)
        GROUP BY date_id, product_id
    """, (product_ids,))

    dim_date = fetch_df(conn, """
        SELECT date_id, date, day_of_week, month, quarter, year, is_weekend, is_holiday
        FROM dim_date ORDER BY date_id
    """)

    dim_product = fetch_df(conn, """
        SELECT product_id, category_id, unit_cost, selling_price, safety_stock, lead_time_days
        FROM dim_product WHERE product_id = ANY(%s)
    """, (product_ids,))

    return product_ids, daily_sales, dim_date, dim_product


def build_feature_frame(product_ids, daily_sales, dim_date, dim_product):
    print("Construction de la grille produit x date (jours sans vente = 0)...")
    grid = pl.DataFrame({"product_id": product_ids}).join(
        dim_date.select("date_id"), how="cross"
    )
    df = (
        grid.join(daily_sales, on=["product_id", "date_id"], how="left")
        .with_columns(pl.col("quantity").fill_null(0.0))
        .join(dim_date, on="date_id", how="left")
        .join(dim_product, on="product_id", how="left")
        .sort(["product_id", "date_id"])
    )

    print("Feature engineering (lags, moyennes mobiles, tendance)...")
    trend = dim_date.sort("date_id").with_row_index("trend_idx")
    df = df.join(trend.select(["date_id", "trend_idx"]), on="date_id", how="left")

    df = df.with_columns([
        pl.col("quantity").shift(1).over("product_id").alias("lag_1"),
        pl.col("quantity").shift(7).over("product_id").alias("lag_7"),
        pl.col("quantity").shift(1).rolling_mean(window_size=7).over("product_id").alias("rolling_7d_demand"),
        pl.col("quantity").shift(1).rolling_mean(window_size=28).over("product_id").alias("rolling_28d_demand"),
        pl.col("is_weekend").cast(pl.Int8),
        pl.col("is_holiday").cast(pl.Int8),
    ])

    # category_id -> code numerique (categorielle pour HistGradientBoostingRegressor)
    cats = df["category_id"].unique().sort().to_list()
    cat_map = {c: i for i, c in enumerate(cats)}
    df = df.with_columns(
        pl.col("category_id").replace_strict(cat_map, default=-1).alias("category_code")
    )

    # les premieres lignes de chaque produit n'ont pas de lag_28 -> on les ecarte de l'entrainement
    df = df.drop_nulls(subset=["lag_1", "lag_7", "rolling_7d_demand", "rolling_28d_demand"])
    return df, cat_map


FEATURES = [
    "lag_1", "lag_7", "rolling_7d_demand", "rolling_28d_demand", "trend_idx",
    "day_of_week", "month", "quarter", "is_weekend", "is_holiday",
    "unit_cost", "selling_price", "safety_stock", "lead_time_days", "category_code",
]
CATEGORICAL_IDX = [FEATURES.index("category_code")]


# ------------------------------------------------------------------ train / eval
def train_eval(df, test_days):
    max_date = df["date_id"].max()
    cutoff_dates = df.select(pl.col("date_id").unique().sort()).to_series()
    cutoff = cutoff_dates[-test_days] if len(cutoff_dates) > test_days else cutoff_dates[0]

    train = df.filter(pl.col("date_id") < cutoff)
    test = df.filter(pl.col("date_id") >= cutoff)
    print("  train: {:,} lignes  |  test: {:,} lignes  (cutoff={})".format(train.height, test.height, cutoff))

    X_train, y_train = train.select(FEATURES).to_numpy(), train["quantity"].to_numpy()
    X_test, y_test = test.select(FEATURES).to_numpy(), test["quantity"].to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_depth=8,
        categorical_features=CATEGORICAL_IDX, random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print("  entrainement : {:.1f}s".format(time.time() - t0))

    pred = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    nonzero = y_test > 0
    mape = np.mean(np.abs((y_test[nonzero] - pred[nonzero]) / y_test[nonzero])) * 100 if nonzero.sum() else float("nan")
    print("  MAE={:.3f}  RMSE={:.3f}  MAPE={:.1f}%  (sur {} points de test avec ventes>0)".format(
        mae, rmse, mape, int(nonzero.sum())))

    return model, {"mae": mae, "rmse": rmse, "mape": mape}


def train_quantile(df, alpha, test_days):
    max_date = df["date_id"].max()
    cutoff_dates = df.select(pl.col("date_id").unique().sort()).to_series()
    cutoff = cutoff_dates[-test_days] if len(cutoff_dates) > test_days else cutoff_dates[0]
    train = df.filter(pl.col("date_id") < cutoff)
    X_train, y_train = train.select(FEATURES).to_numpy(), train["quantity"].to_numpy()
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha, max_iter=150, learning_rate=0.08, max_depth=6,
        categorical_features=CATEGORICAL_IDX, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------ forecast recursif
def build_future_calendar(dim_date):
    """Genere les jours suivant la derniere date du calendrier (absents de dim_date)."""
    import datetime as dt
    last_date = dim_date.sort("date_id")["date"][-1]
    if isinstance(last_date, str):
        last_date = dt.date.fromisoformat(last_date)
    rows = []
    for h in range(1, max(HORIZONS) + 1):
        d = last_date + dt.timedelta(days=h)
        iso = d.isocalendar()
        rows.append({
            "date_id": int(d.strftime("%Y%m%d")),
            "date": d,
            "day_of_week": d.isoweekday(),
            "month": d.month,
            "quarter": (d.month - 1) // 3 + 1,
            "year": d.year,
            "is_weekend": d.isoweekday() >= 6,
            "is_holiday": False,  # simplifie : pas de feries connus au-dela de l'historique
        })
    return pl.DataFrame(rows)


def forecast_future(model, model_lo, model_hi, df, dim_date, dim_product, cat_map):
    print("Generation des previsions a +1/+2/+3/+7 jours...")
    last_date_id = df["date_id"].max()

    hist = (
        df.sort(["product_id", "date_id"])
        .group_by("product_id", maintain_order=True)
        .tail(28)
        .select(["product_id", "date_id", "quantity"])
    )

    future_dates = build_future_calendar(dim_date)

    # le trend continue au-dela de l'historique (indices suivants)
    trend_map = dict(zip(dim_date["date_id"].to_list(), range(dim_date.height)))
    base_trend = dim_date.height
    for i, fid in enumerate(future_dates["date_id"].to_list()):
        trend_map[fid] = base_trend + i
    prod_info = {r["product_id"]: r for r in dim_product.to_dicts()}

    results = []
    series = {pid: g.sort("date_id")["quantity"].to_list()
              for pid, g in hist.group_by("product_id")}
    dates_series = {pid: g.sort("date_id")["date_id"].to_list()
                    for pid, g in hist.group_by("product_id")}

    future_rows = future_dates.to_dicts()

    for pid in df["product_id"].unique().to_list():
        q_hist = series.get(pid, [0.0] * 28)
        if len(q_hist) < 28:
            q_hist = [0.0] * (28 - len(q_hist)) + q_hist
        info = prod_info.get(pid, {})
        cat_code = cat_map.get(info.get("category_id"), -1)

        for h_idx, frow in enumerate(future_rows, start=1):
            lag_1 = q_hist[-1]
            lag_7 = q_hist[-7]
            roll7 = float(np.mean(q_hist[-7:]))
            roll28 = float(np.mean(q_hist[-28:]))
            feat = np.array([[
                lag_1, lag_7, roll7, roll28, trend_map.get(frow["date_id"], 0),
                frow["day_of_week"], frow["month"], frow["quarter"],
                int(frow["is_weekend"]), int(frow["is_holiday"]),
                info.get("unit_cost", 0) or 0, info.get("selling_price", 0) or 0,
                info.get("safety_stock", 0) or 0, info.get("lead_time_days", 0) or 0,
                cat_code,
            ]])
            pred = max(0.0, float(model.predict(feat)[0]))
            q_hist = q_hist[1:] + [pred]

            if h_idx in HORIZONS:
                if model_lo is not None:
                    lo = max(0.0, float(model_lo.predict(feat)[0]))
                    hi = max(lo, float(model_hi.predict(feat)[0]))
                else:
                    lo, hi = None, None
                results.append({
                    "date_id": last_date_id,
                    "product_id": pid,
                    "forecast_date": frow["date"],
                    "forecast_quantity": round(pred, 2),
                    "lower_bound": round(lo, 2) if lo is not None else None,
                    "upper_bound": round(hi, 2) if hi is not None else None,
                    "model_name": MODEL_NAME,
                })

    return pl.DataFrame(results)


# ------------------------------------------------------------------ load to DB
def write_forecasts(conn, forecasts):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fact_demand_forecast RESTART IDENTITY")
    conn.commit()

    buf = io.BytesIO()
    forecasts.write_csv(buf)
    buf.seek(0)
    cols = ",".join(forecasts.columns)
    with conn.cursor() as cur:
        cur.copy_expert("COPY fact_demand_forecast ({}) FROM STDIN WITH (FORMAT csv, HEADER true)".format(cols), buf)
    conn.commit()
    print("  {:,} previsions ecrites dans fact_demand_forecast".format(forecasts.height))


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=0, help="0 = tous les produits, sinon les N plus vendus")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--skip-intervals", action="store_true", help="ne pas entrainer les modeles quantile (plus rapide)")
    args = parser.parse_args()

    t_all = time.time()
    conn = get_conn()

    product_ids, daily_sales, dim_date, dim_product = load_data(conn, args.top_n)
    df, cat_map = build_feature_frame(product_ids, daily_sales, dim_date, dim_product)

    print("\nEntrainement du modele principal...")
    model, metrics = train_eval(df, args.test_days)

    model_lo = model_hi = None
    if not args.skip_intervals:
        print("\nEntrainement des modeles d'intervalle (10e / 90e percentile)...")
        model_lo = train_quantile(df, 0.10, args.test_days)
        model_hi = train_quantile(df, 0.90, args.test_days)

    forecasts = forecast_future(model, model_lo, model_hi, df, dim_date, dim_product, cat_map)
    write_forecasts(conn, forecasts)

    conn.close()
    print("\nTermine en {:.1f}s.".format(time.time() - t_all))
    print("Metriques finales : MAE={:.3f}  RMSE={:.3f}  MAPE={:.1f}%".format(
        metrics["mae"], metrics["rmse"], metrics["mape"]))


if __name__ == "__main__":
    main()
