from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db

app = FastAPI(
    title="NEXUS API",
    description="Intelligent Operations & Decision Platform - Backend API",
    version="0.1.0",
)

# CORS ouvert pour le dev local (React tourne sur un autre port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "NEXUS API is running"}


@app.get("/api/products")
def get_products(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT * FROM dim_product ORDER BY product_id LIMIT :limit"), {"limit": limit}).mappings().all()
    return list(rows)


@app.get("/api/inventory")
def get_inventory(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT i.*, p.product_name
            FROM fact_inventory i
            JOIN dim_product p ON p.product_id = i.product_id
            ORDER BY i.date_id DESC
            LIMIT :limit
        """),
        {"limit": limit},
    ).mappings().all()
    return list(rows)


@app.get("/api/production")
def get_production(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT pr.*, p.product_name
            FROM fact_production pr
            JOIN dim_product p ON p.product_id = pr.product_id
            ORDER BY pr.date_id DESC
            LIMIT :limit
        """),
        {"limit": limit},
    ).mappings().all()
    return list(rows)


@app.get("/api/machines")
def get_machines(db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT * FROM dim_machine ORDER BY machine_id")).mappings().all()
    return list(rows)


@app.get("/api/energy")
def get_energy(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(
        text("SELECT * FROM fact_energy ORDER BY timestamp DESC LIMIT :limit"),
        {"limit": limit},
    ).mappings().all()
    return list(rows)


@app.get("/api/transport")
def get_transport(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(
        text("SELECT * FROM fact_transport ORDER BY date_id DESC LIMIT :limit"),
        {"limit": limit},
    ).mappings().all()
    return list(rows)


@app.get("/api/anomalies")
def get_anomalies(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(
        text("SELECT * FROM fact_anomaly ORDER BY timestamp DESC LIMIT :limit"),
        {"limit": limit},
    ).mappings().all()
    return list(rows)


@app.get("/api/forecast/{product_id}")
def get_forecast(product_id: str, db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT * FROM fact_demand_forecast
            WHERE product_id = :product_id
            ORDER BY forecast_date
        """),
        {"product_id": product_id},
    ).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"Aucune prevision trouvee pour le produit {product_id}. Le modele ML-01 n'a peut-etre pas encore ete execute.")
    return list(rows)


@app.get("/api/machine-risk/{machine_id}")
def get_machine_risk(machine_id: str, db: Session = Depends(get_db)):
    machine = db.execute(
        text("SELECT * FROM dim_machine WHERE machine_id = :machine_id"),
        {"machine_id": machine_id},
    ).mappings().first()
    if not machine:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} introuvable")

    last_readings = db.execute(
        text("""
            SELECT * FROM fact_machine_sensor
            WHERE machine_id = :machine_id
            ORDER BY timestamp DESC
            LIMIT 1
        """),
        {"machine_id": machine_id},
    ).mappings().first()

    # Placeholder : le vrai score viendra du modele ML-03 (section 11).
    # Pour l'instant, on renvoie la derniere mesure brute sans risque calcule.
    return {
        "machine": dict(machine),
        "last_sensor_reading": dict(last_readings) if last_readings else None,
        "failure_risk": None,
        "note": "Le modele ML-03 (Predictive Maintenance) n'est pas encore branche sur cet endpoint.",
    }


@app.post("/api/simulation")
def run_simulation(params: dict):
    """
    Squelette du What-If Simulator (section 23).
    Recoit des parametres (demand_change_pct, capacity_change_pct, etc.)
    et renverra a terme des impacts calcules par les modeles.
    Pour l'instant : validation des parametres recus uniquement.
    """
    allowed_keys = {
        "demand_change_pct",
        "production_capacity_change_pct",
        "supplier_lead_time_extra_days",
        "energy_cost_change_pct",
        "transport_cost_change_pct",
    }
    unknown = set(params.keys()) - allowed_keys
    if unknown:
        raise HTTPException(status_code=400, detail=f"Parametres inconnus : {unknown}")

    return {
        "received_parameters": params,
        "estimated_impact": None,
        "note": "Logique de simulation a implementer (section 23) une fois les modeles ML branches.",
    }