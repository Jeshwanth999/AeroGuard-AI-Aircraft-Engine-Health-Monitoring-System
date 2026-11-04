# AeroGuard AI — Aircraft Engine Health Intelligence

A full-stack aerospace + machine-learning portfolio project. It accepts engine sensor readings, predicts a synthetic risk category, estimates remaining useful life (RUL) in cycles, shows global model feature importance, and stores prediction history.

> **Important:** This is an educational prototype trained on generated synthetic data. It is not validated against real aircraft data, certified, or suitable for flight operations, maintenance decisions, or safety-critical use. Sensor ranges and generated labels are demonstration assumptions only.

## Features

- **Frontend:** Responsive HTML, CSS, JavaScript dashboard with Chart.js.
- **Backend:** FastAPI REST API with request validation.
- **ML:** Random Forest classifier for LOW / MODERATE / HIGH risk and Random Forest regressor for estimated RUL.
- **Explainability:** Displays global feature importance. This is not a causal or per-prediction explanation.
- **Database:** SQLite prediction history.
- **Model metrics:** Validation accuracy and RUL MAE displayed on the dashboard.
- **API docs:** Interactive Swagger docs at `/docs`.

## Architecture

```text
Browser dashboard (HTML/CSS/JS + Chart.js)
                 |
                 | HTTP / JSON
                 v
          FastAPI REST API
           /      |       \
          /       |        \
   ML Engine   SQLite DB   Metrics
      |
      +-- RandomForest risk classifier
      +-- RandomForest RUL regressor
      |
      v
 Synthetic training data (generated in Python)
```

## Project structure

```text
AeroGuard_AI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ml_engine.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── data/
├── tests/
│   └── test_api.py
├── requirements.txt
├── run_windows.bat
├── run_mac_linux.sh
└── README.md
```

## Run locally

### 1. Install Python
Use Python 3.10–3.12. Confirm:

```bash
python --version
```

### 2. Open terminal in the project folder

Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Open the application

- Dashboard: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/health

The ML models train automatically when the backend starts. The first startup may take a few seconds.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check API status |
| GET | `/api/metrics` | Get model validation metrics |
| POST | `/api/predict` | Predict risk and RUL |
| GET | `/api/history` | Read recent prediction history |
| DELETE | `/api/history` | Clear prediction history |

### Example prediction JSON

```json
{
  "aircraft_id": "AG-204",
  "fan_inlet_temp": 36,
  "core_speed": 9300,
  "vibration": 1.8,
  "oil_pressure": 50,
  "exhaust_temp": 590,
  "fuel_flow": 2.1,
  "operating_cycles": 180
}
```

## ML workflow

1. Generate a reproducible synthetic sensor dataset.
2. Create risk labels and RUL targets using transparent teaching formulas.
3. Split into training and validation sets.
4. Train Random Forest classification and regression models.
5. Calculate validation accuracy and mean absolute error.
6. Accept new readings through the API and return predictions.
7. Persist prediction metadata in SQLite.

The displayed model metrics describe performance on a synthetic validation split only. They should not be represented as real-world aircraft-engine accuracy.

## Next-level improvements

1. Replace generated data with a properly licensed NASA C-MAPSS dataset.
2. Compare Random Forest, XGBoost, and sequence models.
3. Add time-series trend charts and sensor drift detection.
4. Add user authentication and role-based access.
5. Add Docker deployment and CI tests.
6. Add SHAP explanations after validating the model and explanation setup.
7. Track model versions, dataset lineage, and drift.

## Resume description

**AeroGuard AI — Explainable Aircraft Engine Health Monitoring System**

Built a full-stack predictive-maintenance prototype using Python, FastAPI, scikit-learn, SQLite, and JavaScript. Developed Random Forest classification and regression models to estimate synthetic engine risk levels and remaining useful life from sensor readings. Implemented REST APIs, an interactive monitoring dashboard, feature-importance visualization, and prediction-history tracking.

## Suggested interview explanation

"I built AeroGuard AI as an aerospace predictive-maintenance learning project. The frontend collects engine sensor readings and sends them to a FastAPI backend. The backend validates the payload and passes it to two trained Random Forest models: one classifies the risk level, while the other estimates remaining useful life in cycles. The API returns probabilities, RUL, and global feature importance, and saves the result in SQLite. The current model uses synthetic data, so I would validate and retrain it on an appropriate real dataset before making any operational claims."
