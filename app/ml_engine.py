"""Synthetic educational ML engine for AeroGuard AI.

The models are trained on generated sensor data so the project runs without
external datasets. Replace this generator with validated operational data
before doing any serious research. This is not an aviation safety tool.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = [
    "fan_inlet_temp", "core_speed", "vibration", "oil_pressure",
    "exhaust_temp", "fuel_flow", "operating_cycles"
]

class AeroModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier = None
        self.regressor = None
        self.metrics = {}

    def _make_training_data(self, n: int = 7000):
        rng = np.random.default_rng(self.random_state)
        cycles = rng.integers(10, 500, n)
        wear = np.clip(cycles / 550 + rng.normal(0, .13, n), 0, 1.15)
        fan_temp = 24 + 22 * wear + rng.normal(0, 3.0, n)
        core_speed = 9700 - 900 * wear + rng.normal(0, 170, n)
        vibration = .8 + 3.2 * wear + rng.normal(0, .42, n)
        oil_pressure = 58 - 22 * wear + rng.normal(0, 3.5, n)
        exhaust_temp = 520 + 180 * wear + rng.normal(0, 18, n)
        fuel_flow = 1.9 + .65 * wear + rng.normal(0, .12, n)

        X = pd.DataFrame({
            "fan_inlet_temp": fan_temp,
            "core_speed": core_speed,
            "vibration": vibration,
            "oil_pressure": oil_pressure,
            "exhaust_temp": exhaust_temp,
            "fuel_flow": fuel_flow,
            "operating_cycles": cycles
        })
        # Synthetic teaching labels. They are not real engineering limits.
        risk_score = (
            0.22 * ((vibration - 1.0) / 3.5) +
            0.20 * ((exhaust_temp - 520) / 200) +
            0.18 * ((58 - oil_pressure) / 28) +
            0.16 * ((9700 - core_speed) / 1100) +
            0.12 * ((fuel_flow - 1.9) / .8) +
            0.12 * wear + rng.normal(0, .055, n)
        )
        y_class = np.where(risk_score > .52, 2, np.where(risk_score > .32, 1, 0))
        rul = np.clip(520 - cycles - 220 * wear + rng.normal(0, 28, n), 5, 520)
        return X, y_class, rul

    def train(self):
        X, y, rul = self._make_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=self.random_state, stratify=y
        )
        _, _, r_train, r_test = train_test_split(
            X, rul, test_size=.2, random_state=self.random_state
        )
        self.classifier = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=180, max_depth=12, min_samples_leaf=3,
                class_weight="balanced", random_state=self.random_state, n_jobs=-1
            ))
        ])
        self.regressor = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=180, max_depth=14, min_samples_leaf=3,
                random_state=self.random_state, n_jobs=-1
            ))
        ])
        self.classifier.fit(X_train, y_train)
        self.regressor.fit(X_train, r_train)
        self.metrics = {
            "risk_model_accuracy": round(float(accuracy_score(y_test, self.classifier.predict(X_test))), 3),
            "rul_model_mae_cycles": round(float(mean_absolute_error(r_test, self.regressor.predict(X_test))), 1),
            "training_rows": int(len(X)),
            "features": len(FEATURES),
            "data_type": "synthetic demonstration data"
        }

    def predict(self, payload: dict):
        row = pd.DataFrame([{f: float(payload[f]) for f in FEATURES}], columns=FEATURES)
        class_id = int(self.classifier.predict(row)[0])
        probs = self.classifier.predict_proba(row)[0]
        rul = max(0, int(round(float(self.regressor.predict(row)[0]))))
        labels = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        # Rank global model feature importances as a transparent, global explanation.
        model = self.classifier.named_steps["model"]
        importances = model.feature_importances_
        explanation = sorted(
            [{"feature": f.replace("_", " ").title(), "importance": round(float(v), 3)}
             for f, v in zip(FEATURES, importances)],
            key=lambda x: x["importance"], reverse=True
        )
        return {
            "risk_level": labels[class_id],
            "risk_probabilities": {
                "low": round(float(probs[0]), 3),
                "moderate": round(float(probs[1]), 3),
                "high": round(float(probs[2]), 3)
            },
            "estimated_rul_cycles": rul,
            "maintenance_priority": "URGENT REVIEW" if class_id == 2 else (
                "SCHEDULE REVIEW" if class_id == 1 else "ROUTINE MONITORING"
            ),
            "explanation": explanation[:5],
            "model_note": "Feature importance is global model-level importance, not a causal explanation."
        }
