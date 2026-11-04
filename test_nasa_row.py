import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "data" / "nasa" / "train_FD001.txt"

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(file_path, sep=r"\s+", header=None)
df = df.iloc[:, :26]
df.columns = columns

# Select engine 1, cycle 100
row = df[(df["unit"] == 1) & (df["cycle"] == 100)]

print("\nNASA Actual Dataset Reading:")
print(row.to_string(index=False))