import pandas as pd
import numpy as np
import holidays
from datetime import date
from pathlib import Path
import os
import xgboost as xgb


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "footfall_prediction_model.json"
DATA_PATH = BASE_DIR / "hourlyfootfall_till_current_date1.csv"
PREDICTION_LOG = BASE_DIR / "prediction_log.csv"


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = xgb.XGBRegressor()

model.load_model(str(MODEL_PATH))

print("Model loaded successfully.")

print()
print("Model features:")
print(model.get_booster().feature_names)

features = model.get_booster().feature_names

print()
print("Number of features:", len(features))


# =========================================================
# LOAD DATA
# =========================================================

print()
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(
    subset=[
        "date",
        "store_id",
        "gate_id",
        "total_footfall"
    ]
).copy()

df = df.sort_values(
    ["store_id", "gate_id", "date"]
).reset_index(drop=True)

print("Rows:", len(df))

print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)


# =========================================================
# HOLIDAYS
# =========================================================

india_holidays = holidays.India()


# =========================================================
# PREDICTION DATE
# =========================================================

selected_date = pd.Timestamp(date.today())

print()
print("=" * 60)
print("PREDICTION DATE")
print("=" * 60)

print("Prediction date:", selected_date.date())


# =========================================================
# STORE + GATE COMBINATIONS
# =========================================================

store_gate = (
    df[
        [
            "store_id",
            "gate_id"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "store_id",
            "gate_id"
        ]
    )
    .reset_index(drop=True)
)

print()
print(
    "Store/gate combinations:",
    len(store_gate)
)


# =========================================================
# CREATE CALENDAR FEATURES
# =========================================================

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

df["weekday"] = df["date"].dt.weekday

df["week"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["quarter"] = df["date"].dt.quarter

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

df["holiday"] = (
    df["date"]
    .dt.date
    .isin(india_holidays)
    .astype(int)
)


# =========================================================
# CYCLICAL FEATURES
# =========================================================

df["weekday_sin"] = np.sin(
    2 * np.pi * df["weekday"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi * df["weekday"] / 7
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# =========================================================
# EXTRA CALENDAR FEATURES
# =========================================================

df["is_month_start"] = (
    df["date"].dt.is_month_start.astype(int)
)

df["is_month_end"] = (
    df["date"].dt.is_month_end.astype(int)
)

df["is_quarter_start"] = (
    df["date"].dt.is_quarter_start.astype(int)
)

df["is_quarter_end"] = (
    df["date"].dt.is_quarter_end.astype(int)
)


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

all_predictions = []

failed = 0

print()
print("=" * 60)
print("STARTING PREDICTIONS")
print("=" * 60)


for index, row in store_gate.iterrows():

    store_id = row["store_id"]
    gate_id = row["gate_id"]


    # -----------------------------------------------------
    # HISTORY FOR THIS STORE + GATE
    # -----------------------------------------------------

    history = df[
        (df["store_id"] == store_id) &
        (df["gate_id"] == gate_id)
    ].sort_values("date").copy()


    if history.empty:
        continue


    footfall = history[
        "total_footfall"
    ].astype(float)


    # -----------------------------------------------------
    # LAGS
    # -----------------------------------------------------

    lag1 = (
        footfall.iloc[-1]
    )

    lag7 = (
        footfall.iloc[-7]
        if len(footfall) >= 7
        else footfall.iloc[-1]
    )

    lag14 = (
        footfall.iloc[-14]
        if len(footfall) >= 14
        else footfall.iloc[-1]
    )

    lag21 = (
        footfall.iloc[-21]
        if len(footfall) >= 21
        else footfall.iloc[-1]
    )

    lag28 = (
        footfall.iloc[-28]
        if len(footfall) >= 28
        else footfall.iloc[-1]
    )

    lag30 = (
        footfall.iloc[-30]
        if len(footfall) >= 30
        else footfall.iloc[-1]
    )


    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    rolling7 = (
        footfall
        .tail(min(7, len(footfall)))
        .mean()
    )

    rolling14 = (
        footfall
        .tail(min(14, len(footfall)))
        .mean()
    )

    rolling30 = (
        footfall
        .tail(min(30, len(footfall)))
        .mean()
    )


    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    trend = (
        footfall.iloc[-1]
        -
        footfall.iloc[-min(7, len(footfall))]
    )


    # -----------------------------------------------------
    # STORE MEAN
    # -----------------------------------------------------

    store_history = df[
        df["store_id"] == store_id
    ]

    store_mean = (
        store_history[
            "total_footfall"
        ].mean()
    )


    # -----------------------------------------------------
    # GATE MEAN
    # -----------------------------------------------------

    gate_history = df[
        df["gate_id"] == gate_id
    ]

    gate_mean = (
        gate_history[
            "total_footfall"
        ].mean()
    )


    # -----------------------------------------------------
    # STORE + WEEKDAY MEAN
    # -----------------------------------------------------

    prediction_weekday = (
        selected_date.weekday()
    )

    store_weekday_history = df[
        (df["store_id"] == store_id) &
        (df["weekday"] == prediction_weekday)
    ]

    if len(store_weekday_history) > 0:

        store_weekday_mean = (
            store_weekday_history[
                "total_footfall"
            ].mean()
        )

    else:

        store_weekday_mean = store_mean


    # -----------------------------------------------------
    # GATE + WEEKDAY MEAN
    # -----------------------------------------------------

    gate_weekday_history = df[
        (df["gate_id"] == gate_id) &
        (df["weekday"] == prediction_weekday)
    ]

    if len(gate_weekday_history) > 0:

        gate_weekday_mean = (
            gate_weekday_history[
                "total_footfall"
            ].mean()
        )

    else:

        gate_weekday_mean = gate_mean


    # -----------------------------------------------------
    # ZERO FEATURES
    # -----------------------------------------------------

    zero_lag1 = int(
        lag1 == 0
    )

    zero_lag7 = int(
        lag7 == 0
    )

    recent7 = footfall.tail(
        min(7, len(footfall))
    )

    recent30 = footfall.tail(
        min(30, len(footfall))
    )

    zero_count7 = int(
        (recent7 == 0).sum()
    )

    zero_count30 = int(
        (recent30 == 0).sum()
    )


    # -----------------------------------------------------
    # CALENDAR FOR PREDICTION DATE
    # -----------------------------------------------------

    year = selected_date.year

    month = selected_date.month

    day = selected_date.day

    weekday = selected_date.weekday()

    week = int(
        selected_date.isocalendar().week
    )

    quarter = selected_date.quarter

    is_weekend = int(
        weekday >= 5
    )

    holiday = int(
        selected_date.date()
        in india_holidays
    )


    weekday_sin = np.sin(
        2 * np.pi * weekday / 7
    )

    weekday_cos = np.cos(
        2 * np.pi * weekday / 7
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )


    is_month_start = int(
        selected_date.is_month_start
    )

    is_month_end = int(
        selected_date.is_month_end
    )

    is_quarter_start = int(
        selected_date.is_quarter_start
    )

    is_quarter_end = int(
        selected_date.is_quarter_end
    )


    # -----------------------------------------------------
    # CREATE INPUT ROW
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "store_id": [store_id],

        "gate_id": [gate_id],

        "year": [year],

        "month": [month],

        "day": [day],

        "weekday": [weekday],

        "week": [week],

        "quarter": [quarter],

        "is_weekend": [is_weekend],

        "holiday": [holiday],

        "weekday_sin": [weekday_sin],

        "weekday_cos": [weekday_cos],

        "month_sin": [month_sin],

        "month_cos": [month_cos],

        "lag1": [lag1],

        "lag7": [lag7],

        "lag14": [lag14],

        "lag21": [lag21],

        "lag28": [lag28],

        "lag30": [lag30],

        "rolling7": [rolling7],

        "rolling14": [rolling14],

        "rolling30": [rolling30],

        "trend": [trend],

        "store_mean": [store_mean],

        "gate_mean": [gate_mean],

        "store_weekday_mean": [
            store_weekday_mean
        ],

        "gate_weekday_mean": [
            gate_weekday_mean
        ],

        "is_month_start": [
            is_month_start
        ],

        "is_month_end": [
            is_month_end
        ],

        "is_quarter_start": [
            is_quarter_start
        ],

        "is_quarter_end": [
            is_quarter_end
        ],

    })


    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in input_data.columns
    ]

    if missing:

        print(
            "Missing features:",
            missing
        )

        failed += 1

        continue


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    try:

        prediction = float(
            model.predict(
                input_data[features]
            )[0]
        )

    except Exception as e:

        print(
            f"Prediction failed for "
            f"{store_id}-{gate_id}: {e}"
        )

        failed += 1

        continue


    # -----------------------------------------------------
    # NO NEGATIVE FOOTFALL
    # -----------------------------------------------------

    prediction = max(
        0,
        prediction
    )


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    all_predictions.append({

        "date":
            selected_date.strftime(
                "%Y-%m-%d"
            ),

        "store_id":
            int(store_id),

        "gate_id":
            int(gate_id),

        "predicted":
            round(
                prediction,
                2
            )

    })


    if (index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(store_gate)}"
        )


# =========================================================
# CREATE DATAFRAME
# =========================================================

new_predictions = pd.DataFrame(
    all_predictions
)


print()
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print(
    "New predictions:",
    len(new_predictions)
)

print(
    "Failed:",
    failed
)


# =========================================================
# SAVE / UPDATE LOG
# =========================================================

if not new_predictions.empty:

    if PREDICTION_LOG.exists():

        old = pd.read_csv(
            PREDICTION_LOG
        )

        combined = pd.concat(
            [
                old,
                new_predictions
            ],
            ignore_index=True
        )

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "date",
                    "store_id",
                    "gate_id"
                ],
                keep="last"
            )
        )

    else:

        combined = new_predictions


    combined.to_csv(
        PREDICTION_LOG,
        index=False
    )


    print()
    print(
        "Prediction log saved:"
    )

    print(
        PREDICTION_LOG
    )

else:

    print()
    print(
        "No predictions were generated."
    )


# =========================================================
# SHOW SAMPLE
# =========================================================

if not new_predictions.empty:

    print()
    print(
        new_predictions.head(20)
    )