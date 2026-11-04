
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "nasa" / "test_FD001.txt"
RUL_PATH = BASE_DIR / "data" / "nasa" / "RUL_FD001.txt"

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
df = df.iloc[:, :26]
df.columns = columns

df = df.groupby("unit").tail(1).sort_values("unit")

actual_rul = pd.read_csv(RUL_PATH, sep=r"\s+", header=None).iloc[:, 0]

features = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X_test = df[features]

models = {
    "Original": "nasa_fd001_rul.joblib",
    "Tuned": "nasa_fd001_tuned.joblib"
}

for name, filename in models.items():

    bundle = joblib.load(BASE_DIR / "models" / filename)

    model = bundle["model"]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(actual_rul, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual_rul, predictions)
    )

    print("\n", name, "MODEL")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
