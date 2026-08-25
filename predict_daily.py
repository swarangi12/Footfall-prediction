import os
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from supabase import create_client


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

print("=" * 60)
print("DAILY FOOTFALL PREDICTION")
print("=" * 60)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY"
)

if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL GitHub Secret is missing."
    )

if not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_KEY GitHub Secret is missing."
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("Supabase connection initialized.")


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "hourlyfootfall_till_current_date1.csv"
)

STAGE1_MODEL_PATH = os.path.join(
    BASE_DIR,
    "stage1_low_classifier.pkl"
)

LOW_MODEL_PATH = os.path.join(
    BASE_DIR,
    "low_footfall_model.pkl"
)

NORMAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "normal_footfall_model.pkl"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DATA_PATH,
    STAGE1_MODEL_PATH,
    LOW_MODEL_PATH,
    NORMAL_MODEL_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# MODEL FEATURES
# ============================================================

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
    "is_quarter_end"
]


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("Loading models...")

with open(
    STAGE1_MODEL_PATH,
    "rb"
) as f:

    stage1_model = pickle.load(f)


with open(
    LOW_MODEL_PATH,
    "rb"
) as f:

    low_model = pickle.load(f)


with open(
    NORMAL_MODEL_PATH,
    "rb"
) as f:

    normal_model = pickle.load(f)


print("Models loaded successfully.")


# ============================================================
# INDIA HOLIDAYS
# ============================================================

try:

    import holidays

    india_holidays = holidays.India()

    print(
        "India holidays loaded successfully."
    )

except Exception as e:

    print(
        "Holiday package unavailable:",
        e
    )

    india_holidays = {}


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading historical data...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Historical rows: {len(df):,}"
)


# ============================================================
# CLEAN DATA
# ============================================================

print("Cleaning data...")


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ------------------------------------------------------------
# IMPORTANT:
# Remove timezone information if present.
#
# This prevents:
#
# TypeError:
# Cannot compare tz-naive and tz-aware datetime-like objects
# ------------------------------------------------------------

try:

    if df["date"].dt.tz is not None:

        df["date"] = (
            df["date"]
            .dt
            .tz_localize(None)
        )

except Exception:

    pass


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


# ------------------------------------------------------------
# Remove duplicates
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=[
        "date",
        "store_id",
        "gate_id"
    ],
    keep="last"
)


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

df = df.sort_values(
    [
        "store_id",
        "gate_id",
        "date"
    ]
).reset_index(
    drop=True
)


print(
    f"Clean historical rows: {len(df):,}"
)


# ============================================================
# GET TODAY'S DATE IN INDIA
# ============================================================

india_now = datetime.now(
    ZoneInfo("Asia/Kolkata")
)

target_date = pd.Timestamp(
    india_now.date()
)

# target_date is intentionally timezone-naive
target_date = target_date.normalize()


print()
print(
    "India current time:",
    india_now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)

print(
    "Automatic prediction date:",
    target_date.strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(data):

    data = data.copy()


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )


    # Remove timezone if present

    try:

        if data["date"].dt.tz is not None:

            data["date"] = (
                data["date"]
                .dt
                .tz_localize(None)
            )

    except Exception:

        pass


    data = data.sort_values(
        [
            "store_id",
            "gate_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # CALENDAR FEATURES
    # ========================================================

    data["year"] = (
        data["date"].dt.year
    )

    data["month"] = (
        data["date"].dt.month
    )

    data["day"] = (
        data["date"].dt.day
    )

    data["weekday"] = (
        data["date"].dt.weekday
    )

    data["week"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["quarter"] = (
        data["date"].dt.quarter
    )

    data["is_weekend"] = (
        data["weekday"] >= 5
    ).astype(int)

    data["holiday"] = (
        data["date"]
        .dt.date
        .isin(india_holidays)
        .astype(int)
    )


    # ========================================================
    # CYCLICAL FEATURES
    # ========================================================

    data["weekday_sin"] = np.sin(
        2 * np.pi *
        data["weekday"] / 7
    )

    data["weekday_cos"] = np.cos(
        2 * np.pi *
        data["weekday"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi *
        (data["month"] - 1) / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi *
        (data["month"] - 1) / 12
    )


    # ========================================================
    # GROUP
    # ========================================================

    group = data.groupby(
        [
            "store_id",
            "gate_id"
        ],
        sort=False
    )["total_footfall"]


    # ========================================================
    # LAGS
    # ========================================================

    data["lag1"] = group.shift(1)

    data["lag7"] = group.shift(7)

    data["lag14"] = group.shift(14)

    data["lag21"] = group.shift(21)

    data["lag28"] = group.shift(28)

    data["lag30"] = group.shift(30)


    # ========================================================
    # ROLLING FEATURES
    # ========================================================

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


    # ========================================================
    # TREND
    # ========================================================

    data["trend"] = (
        data["rolling7"]
        /
        data["rolling30"].replace(
            0,
            np.nan
        )
    )


    # ========================================================
    # PAST VALUE
    # ========================================================

    data["_past"] = group.shift(1)


    # ========================================================
    # STORE MEAN
    # ========================================================

    data["store_mean"] = (
        data.groupby(
            "store_id"
        )["_past"]
        .transform(
            lambda x:
            x.expanding().mean()
        )
    )


    # ========================================================
    # GATE MEAN
    # ========================================================

    data["gate_mean"] = (
        data.groupby(
            "gate_id"
        )["_past"]
        .transform(
            lambda x:
            x.expanding().mean()
        )
    )


    # ========================================================
    # STORE + WEEKDAY MEAN
    # ========================================================

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


    # ========================================================
    # GATE + WEEKDAY MEAN
    # ========================================================

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


    # ========================================================
    # DATE FLAGS
    # ========================================================

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


    # ========================================================
    # CLEAN
    # ========================================================

    data = data.drop(
        columns=[
            "_past"
        ]
    )


    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    return data


# ============================================================
# PREDICT ONE STORE + GATE
# ============================================================

def predict_for_store_gate(
    working_data,
    store,
    gate,
    target_date
):

    print(
        f"Predicting store={int(store)}, "
        f"gate={int(gate)}"
    )


    # --------------------------------------------------------
    # IMPORTANT TIMEZONE FIX
    # --------------------------------------------------------

    target_date = pd.Timestamp(
        target_date
    )


    if target_date.tzinfo is not None:

        target_date = (
            target_date
            .tz_localize(None)
        )


    target_date = target_date.normalize()


    # --------------------------------------------------------
    # COPY DATA
    # --------------------------------------------------------

    working_data = working_data.copy()


    # --------------------------------------------------------
    # MAKE DATA DATE TIMEZONE-NAIVE
    # --------------------------------------------------------

    working_data["date"] = pd.to_datetime(
        working_data["date"],
        errors="coerce"
    )


    try:

        if working_data["date"].dt.tz is not None:

            working_data["date"] = (
                working_data["date"]
                .dt
                .tz_localize(None)
            )

    except Exception:

        pass


    # --------------------------------------------------------
    # FILTER STORE + GATE + HISTORY
    # --------------------------------------------------------

    history = working_data[
        (
            working_data["store_id"]
            == store
        )
        &
        (
            working_data["gate_id"]
            == gate
        )
        &
        (
            working_data["date"]
            < target_date
        )
    ].copy()


    history = history.sort_values(
        "date"
    )


    if history.empty:

        print(
            f"No historical data for "
            f"store={store}, gate={gate}"
        )

        return None


    # ========================================================
    # FUTURE ROW
    # ========================================================

    future_row = pd.DataFrame({

        "date": [
            target_date
        ],

        "store_id": [
            store
        ],

        "gate_id": [
            gate
        ],

        "total_footfall": [
            np.nan
        ]

    })


    # ========================================================
    # TEMP DATA
    # ========================================================

    temp = pd.concat(
        [
            history,
            future_row
        ],
        ignore_index=True
    )


    # ========================================================
    # CREATE FEATURES
    # ========================================================

    temp_features = create_features(
        temp
    )


    row = temp_features[
        temp_features["date"]
        == target_date
    ].tail(1)


    if row.empty:

        print(
            "Could not create feature row."
        )

        return None


    # ========================================================
    # CHECK FEATURES
    # ========================================================

    missing_features = [

        feature

        for feature in FEATURES

        if feature not in row.columns

    ]


    if missing_features:

        raise ValueError(
            "Missing features: "
            +
            ", ".join(
                missing_features
            )
        )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    X = row[
        FEATURES
    ].copy()


    # ========================================================
    # FALLBACK VALUES
    # ========================================================

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


    history_mean = history[
        "total_footfall"
    ].mean()


    history_last = history[
        "total_footfall"
    ].iloc[-1]


    for col in fallback_columns:

        if pd.isna(
            X.iloc[0][col]
        ):

            if col.startswith(
                "lag"
            ):

                value = history_last

            else:

                value = history_mean


            X.loc[
                X.index[0],
                col
            ] = value


    # ========================================================
    # TREND FALLBACK
    # ========================================================

    if pd.isna(
        X.iloc[0]["trend"]
    ):

        X.loc[
            X.index[0],
            "trend"
        ] = 1.0


    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    X = X.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    X = X.fillna(0)


    # ========================================================
    # STAGE 1
    # ========================================================

    stage_prediction = (
        stage1_model
        .predict(X)[0]
    )


    try:

        stage_prediction = int(
            stage_prediction
        )

    except Exception:

        stage_prediction = 0


    # ========================================================
    # SELECT MODEL
    # ========================================================

    if stage_prediction == 1:

        selected_model = (
            low_model
        )

        model_name = (
            "Low-footfall model"
        )

    else:

        selected_model = (
            normal_model
        )

        model_name = (
            "Normal-footfall model"
        )


    # ========================================================
    # RAW PREDICTION
    # ========================================================

    raw_prediction = (
        selected_model
        .predict(X)[0]
    )


    # ========================================================
    # LOG1P → REAL FOOTFALL
    # ========================================================

    raw_prediction = np.clip(
        raw_prediction,
        -20,
        20
    )


    prediction = np.expm1(
        raw_prediction
    )


    prediction = max(
        0.0,
        float(prediction)
    )


    print(
        f"Prediction = {prediction:.2f} "
        f"using {model_name}"
    )


    return {
        "date":
            target_date.strftime(
                "%Y-%m-%d"
            ),

        "store_id":
            int(store),

        "gate_id":
            int(gate),

        "predicted":
            round(
                prediction,
                2
            ),

        "model":
            model_name
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("STARTING AUTOMATIC PREDICTION")
    print("=" * 60)


    # ========================================================
    # INDIA DATE
    # ========================================================

    india_now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )


    target_date = pd.Timestamp(
        india_now.date()
    ).normalize()


    print(
        "India time:",
        india_now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    print(
        "Prediction date:",
        target_date.strftime(
            "%Y-%m-%d"
        )
    )


    # ========================================================
    # STORE/GATE COMBINATIONS
    # ========================================================

    combinations = (
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
        .reset_index(
            drop=True
        )
    )


    print()
    print(
        "Total store/gate combinations:",
        len(combinations)
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    results = []


    for _, combination in combinations.iterrows():

        store = combination[
            "store_id"
        ]

        gate = combination[
            "gate_id"
        ]


        try:

            result = predict_for_store_gate(
                df,
                store,
                gate,
                target_date
            )


            if result is not None:

                results.append(
                    result
                )


        except Exception as e:

            print(
                f"ERROR for "
                f"store={store}, "
                f"gate={gate}: {e}"
            )


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:

        raise RuntimeError(
            "No predictions were generated."
        )


    prediction_df = pd.DataFrame(
        results
    )


    print()
    print(
        "Predictions generated:",
        len(prediction_df)
    )


    print(
        prediction_df.head()
    )


    # ========================================================
    # SAVE TO SUPABASE
    # ========================================================

    print()
    print(
        "Saving predictions to Supabase..."
    )


    records = prediction_df.to_dict(
        orient="records"
    )


    try:

        response = (
            supabase
            .table(
                "prediction_log"
            )
            .upsert(
                records,
                on_conflict=(
                    "date,store_id,gate_id"
                )
            )
            .execute()
        )


        print(
            "Supabase save successful."
        )


        print(
            f"Saved {len(records)} predictions."
        )


    except Exception as e:

        print(
            "Supabase save failed:"
        )

        print(e)

        raise


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 60)
    print("DAILY PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()