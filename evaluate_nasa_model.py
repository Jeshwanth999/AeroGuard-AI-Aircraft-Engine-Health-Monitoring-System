
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# File paths
TEST_PATH = BASE_DIR / "data" / "nasa" / "test_FD001.txt"
RUL_PATH = BASE_DIR / "data" / "nasa" / "RUL_FD001.txt"
MODEL_PATH = BASE_DIR / "models" / "nasa_fd001_rul.joblib"

# Column names
columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load test dataset
test_df = pd.read_csv(TEST_PATH, sep=r"\s+", header=None)
test_df = test_df.iloc[:, :26]
test_df.columns = columns

# Load actual RUL labels
actual_rul = pd.read_csv(RUL_PATH, sep=r"\s+", header=None).iloc[:, 0].values

# Get final cycle record for every engine
final_rows = test_df.groupby("unit").tail(1).copy()
final_rows = final_rows.sort_values("unit")

# Create features
feature_columns = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

X_test = final_rows[feature_columns]

# Load trained model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

# Predict RUL
predicted_rul = model.predict(X_test)

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Engine_ID": final_rows["unit"].values,
    "Actual_RUL": actual_rul,
    "Predicted_RUL": np.round(predicted_rul, 2)
})

comparison["Error"] = (
    comparison["Actual_RUL"] - comparison["Predicted_RUL"]
)

# Metrics
mae = mean_absolute_error(actual_rul, predicted_rul)
rmse = np.sqrt(mean_squared_error(actual_rul, predicted_rul))

print("\nNASA FD001 MODEL EVALUATION")
print("-" * 45)

print("Total Test Engines:", len(comparison))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

print("\nFirst 10 Engine Predictions:")
print(comparison.head(10).to_string(index=False))

# Save results
comparison.to_csv(
    BASE_DIR / "nasa_evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")
