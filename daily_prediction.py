import os
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import holidays

from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# SUPABASE
# ---------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY are not configured."
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

DATA_PATH = BASE_DIR / "hourlyfootfall_till_current_date1.csv"

STAGE1_MODEL_PATH = BASE_DIR / "stage1_low_classifier.pkl"
LOW_MODEL_PATH = BASE_DIR / "low_footfall_model.pkl"
NORMAL_MODEL_PATH = BASE_DIR / "normal_footfall_model.pkl"

PREDICTION_CONFIG_PATH = BASE_DIR / "prediction_config.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

FEATURES = [
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
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "lag1",
    "lag7",
    "lag14",
    "lag21",
    "lag28",
    "lag30",
    "rolling7",
    "rolling14",
    "rolling30",
    "trend",
    "store_mean",
    "gate_mean",
    "store_weekday_mean",
    "gate_weekday_mean",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
]


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

print("Loading models...")

with open(STAGE1_MODEL_PATH, "rb") as f:
    stage1_model = pickle.load(f)

with open(LOW_MODEL_PATH, "rb") as f:
    low_model = pickle.load(f)

with open(NORMAL_MODEL_PATH, "rb") as f:
    normal_model = pickle.load(f)

print("Models loaded successfully.")


# ---------------------------------------------------------
# HOLIDAYS
# ---------------------------------------------------------

india_holidays = holidays.India()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading historical data...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "date",
        "store_id",
        "gate_id",
        "total_footfall"
    ]
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["store_id"] = pd.to_numeric(
    df["store_id"],
    errors="coerce"
)

df["gate_id"] = pd.to_numeric(
    df["gate_id"],
    errors="coerce"
)

df["total_footfall"] = pd.to_numeric(
    df["total_footfall"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "date",
        "store_id",
        "gate_id",
        "total_footfall"
    ]
)

df = df.drop_duplicates(
    subset=[
        "date",
        "store_id",
        "gate_id"
    ],
    keep="last"
)

df = df.sort_values(
    [
        "store_id",
        "gate_id",
        "date"
    ]
).reset_index(drop=True)

print(
    "Historical rows:",
    len(df)
)


# ---------------------------------------------------------
# CREATE FEATURES
# ---------------------------------------------------------

def create_features(data):

    data = data.copy()

    data = data.sort_values(
        [
            "store_id",
            "gate_id",
            "date"
        ]
    ).reset_index(drop=True)

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["weekday"] = data["date"].dt.weekday

    data["week"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["quarter"] = data["date"].dt.quarter

    data["is_weekend"] = (
        data["weekday"] >= 5
    ).astype(int)

    data["holiday"] = (
        data["date"]
        .dt.date
        .isin(india_holidays)
        .astype(int)
    )

    data["weekday_sin"] = np.sin(
        2 * np.pi * data["weekday"] / 7
    )

    data["weekday_cos"] = np.cos(
        2 * np.pi * data["weekday"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * (data["month"] - 1) / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * (data["month"] - 1) / 12
    )

    group = data.groupby(
        [
            "store_id",
            "gate_id"
        ],
        sort=False
    )["total_footfall"]

    data["lag1"] = group.shift(1)
    data["lag7"] = group.shift(7)
    data["lag14"] = group.shift(14)
    data["lag21"] = group.shift(21)
    data["lag28"] = group.shift(28)
    data["lag30"] = group.shift(30)

    data["rolling7"] = group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    data["rolling14"] = group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    data["rolling30"] = group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=1
        )
        .mean()
    )

    data["trend"] = (
        data["rolling7"]
        /
        data["rolling30"].replace(
            0,
            np.nan
        )
    )

    data["_past"] = group.shift(1)

    data["store_mean"] = (
        data.groupby("store_id")["_past"]
        .transform(
            lambda x:
            x.expanding().mean()
        )
    )

    data["gate_mean"] = (
        data.groupby("gate_id")["_past"]
        .transform(
            lambda x:
            x.expanding().mean()
        )
    )

    data["store_weekday_mean"] = (
        data.groupby(
            [
                "store_id",
                "weekday"
            ]
        )["_past"]
        .transform(
            lambda x:
            x.expanding().mean()
        )
    )

    data["gate_weekday_mean"] = (
        data.groupby(
            [
                "gate_id",
                "weekday"
            ]
        )["_past"]
        .transform(
            lambda x:
            x.expanding().mean()
        )
    )

    data["is_month_start"] = (
        data["date"]
        .dt.is_month_start
        .astype(int)
    )

    data["is_month_end"] = (
        data["date"]
        .dt.is_month_end
        .astype(int)
    )

    data["is_quarter_start"] = (
        data["date"]
        .dt.is_quarter_start
        .astype(int)
    )

    data["is_quarter_end"] = (
        data["date"]
        .dt.is_quarter_end
        .astype(int)
    )

    data = data.drop(
        columns=["_past"]
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data


# ---------------------------------------------------------
# TARGET TRANSFORMATION
# ---------------------------------------------------------

def convert_prediction(raw_prediction):

    raw_prediction = float(
        raw_prediction
    )

    raw_prediction = np.clip(
        raw_prediction,
        -20,
        20
    )

    prediction = np.expm1(
        raw_prediction
    )

    return max(
        0.0,
        float(prediction)
    )


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

def predict_for_store_gate(
    history,
    store,
    gate,
    target_date
):

    target_date = pd.Timestamp(
        target_date
    )

    history = history[
        (history["store_id"] == store)
        &
        (history["gate_id"] == gate)
        &
        (history["date"] < target_date)
    ].copy()

    if history.empty:
        print(
            f"No history for store={store}, gate={gate}"
        )
        return None

    future_row = pd.DataFrame({
        "date": [target_date],
        "store_id": [store],
        "gate_id": [gate],
        "total_footfall": [np.nan]
    })

    temp = pd.concat(
        [
            history,
            future_row
        ],
        ignore_index=True
    )

    features = create_features(
        temp
    )

    row = features[
        features["date"] == target_date
    ].tail(1)

    if row.empty:
        return None

    X = row[FEATURES].copy()

    fallback_columns = [
        "lag1",
        "lag7",
        "lag14",
        "lag21",
        "lag28",
        "lag30",
        "rolling7",
        "rolling14",
        "rolling30",
        "store_mean",
        "gate_mean",
        "store_weekday_mean",
        "gate_weekday_mean"
    ]

    for col in fallback_columns:

        if pd.isna(X.iloc[0][col]):

            X.loc[
                X.index[0],
                col
            ] = history[
                "total_footfall"
            ].mean()

    if pd.isna(
        X.iloc[0]["trend"]
    ):
        X.loc[
            X.index[0],
            "trend"
        ] = 1.0

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    stage = int(
        stage1_model.predict(X)[0]
    )

    if stage == 1:
        model = low_model
        model_name = "Low-footfall model"
    else:
        model = normal_model
        model_name = "Normal-footfall model"

    raw_prediction = model.predict(X)[0]

    prediction = convert_prediction(
        raw_prediction
    )

    return {
        "date": target_date.strftime(
            "%Y-%m-%d"
        ),
        "store_id": int(store),
        "gate_id": int(gate),
        "predicted": round(
            prediction,
            2
        ),
        "model": model_name
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    target_date = (
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
        .normalize()
    )

    print(
        "Automatic prediction date:",
        target_date.date()
    )

    stores = sorted(
        df["store_id"]
        .dropna()
        .unique()
    )

    predictions = []

    for store in stores:

        gates = sorted(
            df[
                df["store_id"] == store
            ]["gate_id"]
            .dropna()
            .unique()
        )

        for gate in gates:

            print(
                f"Predicting store={store}, gate={gate}"
            )

            result = predict_for_store_gate(
                df,
                store,
                gate,
                target_date
            )

            if result:
                predictions.append(
                    result
                )

    if not predictions:
        raise RuntimeError(
            "No predictions were generated."
        )

    print(
        f"Generated {len(predictions)} predictions."
    )

    # -----------------------------------------------------
    # SAVE TO SUPABASE
    # -----------------------------------------------------

    supabase.table(
        "predictions"
    ).upsert(
        predictions,
        on_conflict=(
            "date,store_id,gate_id"
        )
    ).execute()

    print(
        "Predictions successfully saved to Supabase."
    )


if __name__ == "__main__":
    main()