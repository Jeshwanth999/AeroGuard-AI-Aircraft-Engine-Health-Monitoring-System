
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "nasa" / "train_FD001.txt"

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
df = df.iloc[:, :26]
df.columns = columns

# Calculate RUL
max_cycles = df.groupby("unit")["cycle"].max()

df["RUL"] = df["unit"].map(max_cycles) - df["cycle"]
df["RUL"] = df["RUL"].clip(upper=125)

feature_columns = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X = df[feature_columns]
y = df["RUL"]
groups = df["unit"]

# Group-based train validation split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

# Hyperparameter combinations
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

print("Starting Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, predictions)
)

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print("\nMODEL PERFORMANCE")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save tuned model
MODEL_PATH = BASE_DIR / "models" / "nasa_fd001_tuned.joblib"

joblib.dump(
    {
        "model": best_model,
        "features": feature_columns
    },
    MODEL_PATH
)

print("\nTuned model saved successfully!")
