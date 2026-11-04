from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sqlite3
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.ml_engine import AeroModels, FEATURES

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT.parent / "data" / "aeroguard.db"
DB_PATH.parent.mkdir(exist_ok=True)

app = FastAPI(
    title="AeroGuard AI API",
    description="Educational aircraft engine health monitoring demo using synthetic data.",
    version="1.0.0"
)
models = AeroModels()
models.train()
# Load NASA trained model
NASA_MODEL_PATH = ROOT.parent / "models" / "nasa_fd001_tuned.joblib"

nasa_bundle = joblib.load(NASA_MODEL_PATH)

nasa_model = nasa_bundle["model"]
nasa_features = nasa_bundle["features"]

print("NASA model loaded successfully!")

class SensorReading(BaseModel):
    aircraft_id: str = Field(default="AG-204", min_length=2, max_length=30)
    fan_inlet_temp: float = Field(ge=-20, le=100)
    core_speed: float = Field(gt=0, le=20000)
    vibration: float = Field(ge=0, le=20)
    oil_pressure: float = Field(ge=0, le=150)
    exhaust_temp: float = Field(ge=0, le=1200)
    fuel_flow: float = Field(ge=0, le=10)
    operating_cycles: int = Field(ge=0, le=10000)
class NASAEngineReading(BaseModel):
    engine_id: str = "NASA-FD001"

    cycle: float = Field(gt=0)

    settings: list[float] = Field(
        min_length=3,
        max_length=3
    )

    sensors: list[float] = Field(
        min_length=21,
        max_length=21
    )    

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            aircraft_id TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            rul_cycles INTEGER NOT NULL,
            priority TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """)

init_db()

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AeroGuard AI", "model": "trained",
            "disclaimer": "Synthetic educational prototype; not for operational aviation decisions."}

@app.get("/api/metrics")
def metrics():
    return models.metrics

@app.post("/api/predict")
def predict(reading: SensorReading):
    payload = reading.model_dump()
    result = models.predict(payload)
    import json
    with connect() as con:
        cur = con.execute(
            "INSERT INTO predictions(created_at, aircraft_id, risk_level, rul_cycles, priority, payload) VALUES(?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), reading.aircraft_id,
             result["risk_level"], result["estimated_rul_cycles"],
             result["maintenance_priority"], json.dumps(payload))
        )
        prediction_id = cur.lastrowid
    return {"prediction_id": prediction_id, "aircraft_id": reading.aircraft_id, **result}

@app.post("/api/nasa/predict")
def nasa_predict(reading: NASAEngineReading):

    # Prepare input features
    input_values = {
        "cycle": reading.cycle
    }

    # Add 3 settings
    for i, value in enumerate(reading.settings, start=1):
        input_values[f"setting_{i}"] = value

    # Add 21 sensors
    for i, value in enumerate(reading.sensors, start=1):
        input_values[f"sensor_{i}"] = value

    # Arrange features in training order
    input_df = pd.DataFrame(
        [[input_values[col] for col in nasa_features]],
        columns=nasa_features
    )

    # Predict Remaining Useful Life
    predicted_rul = nasa_model.predict(input_df)[0]

    predicted_rul = max(0, round(float(predicted_rul), 2))

    return {
        "engine_id": reading.engine_id,
        "predicted_rul_cycles": predicted_rul,
        "model": "NASA FD001 Random Forest",
        "message": "NASA engine RUL prediction successful"
    }


@app.get("/api/history")
def history(limit: int = 20):
    limit = min(max(limit, 1), 100)
    with connect() as con:
        rows = con.execute(
            "SELECT id, created_at, aircraft_id, risk_level, rul_cycles, priority "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

@app.delete("/api/history")
def clear_history():
    with connect() as con:
        con.execute("DELETE FROM predictions")
    return {"message": "Prediction history cleared"}

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
