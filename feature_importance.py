
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "nasa_fd001_rul.joblib"

# Load trained model
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]

# Get feature importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10).to_string(index=False))

# Plot feature importance
top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("NASA Sensors")
plt.title("NASA FD001 - Feature Importance")
plt.tight_layout()

plt.savefig(BASE_DIR / "feature_importance.png")
plt.show()

print("\nChart saved as feature_importance.png")

