
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Dataset path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "nasa" / "train_FD001.txt"

print("Dataset path:", DATA_PATH)
print("File exists:", DATA_PATH.exists())

# 2. Create column names
columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# 3. Load NASA dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None
)

df = df.iloc[:, :26]
df.columns = columns

# 4. Calculate Remaining Useful Life (RUL)
max_cycles = df.groupby("unit")["cycle"].max()
df["RUL"] = df["unit"].map(max_cycles) - df["cycle"]

# Common C-MAPSS RUL cap
df["RUL"] = df["RUL"].clip(upper=125)

# 5. Select input features
feature_columns = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X = df[feature_columns]
y = df["RUL"]
groups = df["unit"]

# 6. Split data by engine unit
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

# 7. Train Random Forest model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 8. Evaluate model
predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("NASA FD001 Model Training Completed!")
print("-------------------------------------")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# 9. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    "models/nasa_fd001_rul.joblib"
)

print("Model saved successfully!")
