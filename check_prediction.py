import pickle
import pandas as pd
import holidays
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "footfall_prediction_model.json"
DATA_PATH = BASE_DIR / "hourlyfootfall_till_current_date1.csv"

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

# Sort
group_cols = ["store_id", "gate_id"]

df = df.sort_values(
    ["store_id", "gate_id", "date"]
).reset_index(drop=True)

# Calendar features
india_holidays = holidays.India()

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.weekday
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

df["holiday"] = df["date"].dt.date.isin(
    india_holidays
).astype(int)

# Group
grouped = df.groupby(
    group_cols
)["total_footfall"]

# Lags
df["lag1"] = grouped.shift(1)
df["lag7"] = grouped.shift(7)
df["lag30"] = grouped.shift(30)

# Rolling
df["rolling7"] = (
    df.groupby(group_cols)["total_footfall"]
    .transform(
        lambda x:
            x.shift(1)
            .rolling(7, min_periods=1)
            .mean()
    )
)

df["rolling30"] = (
    df.groupby(group_cols)["total_footfall"]
    .transform(
        lambda x:
            x.shift(1)
            .rolling(30, min_periods=1)
            .mean()
    )
)

# Find target row
target = df[
    (df["date"] == "2026-06-26") &
    (df["store_id"] == 532) &
    (df["gate_id"] == 0)
]

print("\nTARGET ROW")
print("=" * 60)

print(
    target[
        [
            "date",
            "store_id",
            "gate_id",
            "total_footfall",
            "lag1",
            "lag7",
            "lag30",
            "rolling7",
            "rolling30"
        ]
    ].to_string(index=False)
)

# Model features
features = [
    "store_id",
    "gate_id",
    "year",
    "month",
    "day",
    "weekday",
    "week",
    "quarter",
    "is_weekend",
    "holiday",
    "lag1",
    "lag7",
    "lag30",
    "rolling7",
    "rolling30"
]

# Prediction
X = target[features]

prediction = model.predict(X)

print("\nMODEL PREDICTION")
print("=" * 60)

print("Predicted:", prediction[0])
print("Actual:", target["total_footfall"].iloc[0])