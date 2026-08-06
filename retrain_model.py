import pandas as pd
import pickle
import holidays

from xgboost import XGBRegressor


errors = pd.read_csv("error_log.csv")

average_error = errors["error"].tail(7).mean()

THRESHOLD = 20

if average_error <= THRESHOLD:

    print("No Retraining Needed")

    exit()

print("Loading dataset...")

# Load dataset
df = pd.read_csv("hourlyfootfall_till_current_date1.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Holiday Feature
# -----------------------------
india_holidays = holidays.India()

df["holiday"] = df["date"].isin(india_holidays).astype(int)

# -----------------------------
# Date Features
# -----------------------------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.weekday
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

# -----------------------------
# Sort data
# -----------------------------
df = df.sort_values(["store_id", "gate_id", "date"])

# -----------------------------
# Lag Features
# -----------------------------
df["lag1"] = df.groupby(
    ["store_id", "gate_id"]
)["total_footfall"].shift(1)

df["lag7"] = df.groupby(
    ["store_id", "gate_id"]
)["total_footfall"].shift(7)

df["lag30"] = df.groupby(
    ["store_id", "gate_id"]
)["total_footfall"].shift(30)

# -----------------------------
# Rolling Features
# -----------------------------
df["rolling7"] = (
    df.groupby(["store_id", "gate_id"])["total_footfall"]
      .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["rolling30"] = (
    df.groupby(["store_id", "gate_id"])["total_footfall"]
      .transform(lambda x: x.shift(1).rolling(30).mean())
)

# -----------------------------
# Remove missing rows
# -----------------------------
df = df.dropna()

# -----------------------------
# Features
# -----------------------------
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

X = df[features]

y = df["total_footfall"]

print("Training model...")

model = XGBRegressor(

    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42

)

model.fit(X, y)

print("Saving model...")

pickle.dump(
    model,
    open("footfall_prediction_model.pkl", "wb")
)

print("Model retrained successfully!")