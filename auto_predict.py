import pandas as pd
import pickle
import holidays
from datetime import date
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "footfall_prediction_model.pkl"
DATA_PATH = BASE_DIR / "hourlyfootfall_till_current_date1.csv"
PREDICTION_LOG = BASE_DIR / "prediction_log.csv"


# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# -----------------------------
# LOAD DATA
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


# -----------------------------
# HOLIDAYS
# -----------------------------

india_holidays = holidays.India()


# -----------------------------
# PREDICTION DATE
# -----------------------------

selected_date = pd.Timestamp(date.today())

print("Prediction date:", selected_date.date())


# -----------------------------
# GET STORE + GATE COMBINATIONS
# -----------------------------

store_gate = (
    df[["store_id", "gate_id"]]
    .drop_duplicates()
)


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


all_predictions = []


# -----------------------------
# PREDICT FOR EVERY STORE/GATE
# -----------------------------

for _, row in store_gate.iterrows():

    store_id = row["store_id"]
    gate_id = row["gate_id"]

    history = df[
        (df["store_id"] == store_id) &
        (df["gate_id"] == gate_id)
    ].sort_values("date")


    if history.empty:
        continue


    lag1 = history["total_footfall"].iloc[-1]

    lag7 = (
        history["total_footfall"].iloc[-7]
        if len(history) >= 7
        else lag1
    )

    lag30 = (
        history["total_footfall"].iloc[-30]
        if len(history) >= 30
        else lag1
    )

    rolling7 = history[
        "total_footfall"
    ].tail(min(7, len(history))).mean()

    rolling30 = history[
        "total_footfall"
    ].tail(min(30, len(history))).mean()


    weekday = selected_date.weekday()

    input_data = pd.DataFrame({

        "store_id": [store_id],

        "gate_id": [gate_id],

        "year": [selected_date.year],

        "month": [selected_date.month],

        "day": [selected_date.day],

        "weekday": [weekday],

        "week": [
            int(selected_date.isocalendar().week)
        ],

        "quarter": [selected_date.quarter],

        "is_weekend": [
            1 if weekday >= 5 else 0
        ],

        "holiday": [
            1 if selected_date in india_holidays
            else 0
        ],

        "lag1": [lag1],

        "lag7": [lag7],

        "lag30": [lag30],

        "rolling7": [rolling7],

        "rolling30": [rolling30]
    })


    prediction = float(
        model.predict(
            input_data[features]
        )[0]
    )


    # Don't allow negative prediction
    prediction = max(0, prediction)


    all_predictions.append({

        "date": selected_date.strftime(
            "%Y-%m-%d"
        ),

        "store_id": store_id,

        "gate_id": gate_id,

        "predicted": round(
            prediction,
            2
        )
    })


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

new_predictions = pd.DataFrame(
    all_predictions
)


# -----------------------------
# SAVE / UPDATE PREDICTION LOG
# -----------------------------

if os.path.exists(PREDICTION_LOG):

    old = pd.read_csv(
        PREDICTION_LOG
    )

    combined = pd.concat(
        [old, new_predictions],
        ignore_index=True
    )

    # Remove duplicate predictions
    combined = combined.drop_duplicates(
        subset=[
            "date",
            "store_id",
            "gate_id"
        ],
        keep="last"
    )

else:

    combined = new_predictions


combined.to_csv(
    PREDICTION_LOG,
    index=False
)


print(
    f"✅ {len(new_predictions)} predictions saved."
)

print(
    f"📁 {PREDICTION_LOG}"
)