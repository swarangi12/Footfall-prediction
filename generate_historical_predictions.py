import pandas as pd
import numpy as np
import holidays
import xgboost as xgb
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "footfall_prediction_model.json"
DATA_PATH = BASE_DIR / "hourlyfootfall_till_current_date1.csv"
OUTPUT_PATH = BASE_DIR / "prediction_log.csv"


# =========================================================
# SETTINGS
# =========================================================

END_DATE = pd.Timestamp("2026-07-23")

# Number of historical days to evaluate
HISTORICAL_DAYS = 365

# Number of rows sent to XGBoost at one time
PREDICTION_BATCH_SIZE = 10000


# =========================================================
# MODEL FEATURES
# =========================================================

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
    "rolling30",
    "zero_lag1",
    "zero_lag7",
    "zero_count7",
    "zero_count30"
]


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

print("Model loaded successfully.")

print()
print("Model expects features:")
print(model.feature_names_in_)

print()
print("Number of model features:")
print(len(model.feature_names_in_))


# =========================================================
# CHECK FEATURES
# =========================================================

model_features = list(model.feature_names_in_)

if model_features != features:

    print()
    print("WARNING: Feature order/name mismatch.")
    print()

    print("Expected by model:")
    print(model_features)

    print()

    print("Script features:")
    print(features)

    raise ValueError(
        "Model features do not match the features used by this script."
    )


# =========================================================
# LOAD DATA
# =========================================================

print()
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Original rows:", len(df))

print(
    "Original date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)


# =========================================================
# FILTER DATA
# =========================================================

df = df[
    df["date"] <= END_DATE
].copy()

df = df.sort_values(
    ["store_id", "gate_id", "date"]
).reset_index(drop=True)

print()
print(
    "Date range after filtering:",
    df["date"].min(),
    "to",
    df["date"].max()
)

print("Rows after filtering:", len(df))


# =========================================================
# REMOVE DUPLICATES
# =========================================================

df = (
    df
    .drop_duplicates(
        subset=[
            "date",
            "store_id",
            "gate_id"
        ],
        keep="last"
    )
    .copy()
)


# =========================================================
# HOLIDAYS
# =========================================================

india_holidays = holidays.India()


# =========================================================
# STORE/GATE COMBINATIONS
# =========================================================

combinations = (
    df[
        ["store_id", "gate_id"]
    ]
    .drop_duplicates()
    .sort_values(
        ["store_id", "gate_id"]
    )
    .reset_index(drop=True)
)

print()
print("=" * 60)
print("STORE/GATE INFORMATION")
print("=" * 60)

print(
    "Total store/gate combinations:",
    len(combinations)
)


# =========================================================
# FIND PERMANENTLY INACTIVE COMBINATIONS
# =========================================================

inactive = (
    df.groupby(
        ["store_id", "gate_id"]
    )["total_footfall"]
    .agg(
        total_records="count",
        max_footfall="max"
    )
)

inactive = inactive[
    inactive["max_footfall"] == 0
]

inactive_combinations = set(
    inactive.index.tolist()
)

print(
    "Permanently inactive combinations:",
    len(inactive_combinations)
)


# =========================================================
# TARGET DATES
# =========================================================

all_dates = sorted(
    df["date"].drop_duplicates()
)

if len(all_dates) > HISTORICAL_DAYS:

    target_dates = all_dates[
        -HISTORICAL_DAYS:
    ]

else:

    target_dates = all_dates


target_dates = pd.DatetimeIndex(target_dates)

print()
print("=" * 60)
print("HISTORICAL PREDICTION RANGE")
print("=" * 60)

print(
    "First prediction date:",
    target_dates.min()
)

print(
    "Last prediction date:",
    target_dates.max()
)

print(
    "Number of prediction dates:",
    len(target_dates)
)


# =========================================================
# CREATE FEATURES FOR ONE GROUP
# =========================================================

def create_group_features(group, target_dates, store_id, gate_id):

    group = (
        group
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .copy()
    )

    # -----------------------------------------------------
    # Keep only target dates that actually exist
    # -----------------------------------------------------

    target_set = set(target_dates)

    target_rows = group[
        group["date"].isin(target_set)
    ].copy()

    if target_rows.empty:
        return None

    # -----------------------------------------------------
    # Footfall values indexed by date
    # -----------------------------------------------------

    values = (
        group
        .set_index("date")["total_footfall"]
        .astype(float)
        .sort_index()
    )

    rows = []

    # -----------------------------------------------------
    # Process dates
    # -----------------------------------------------------

    for target_date, target_row in target_rows.iterrows():

        pass


# =========================================================
# FAST FEATURE CREATION
# =========================================================

print()
print("=" * 60)
print("PREPARING HISTORICAL FEATURES")
print("=" * 60)

feature_rows = []

processed_combinations = 0
skipped_inactive = 0
skipped_history = 0

# ---------------------------------------------------------
# Pre-group the dataframe.
#
# This is MUCH faster than repeatedly doing:
#
# df[(df["store_id"] == store_id) &
#    (df["gate_id"] == gate_id)]
# ---------------------------------------------------------

grouped = df.groupby(
    ["store_id", "gate_id"],
    sort=False
)

total_combinations = len(combinations)

for store_id, gate_id in combinations[
    ["store_id", "gate_id"]
].itertuples(index=False, name=None):

    combination = (
        int(store_id),
        int(gate_id)
    )

    # -----------------------------------------------------
    # Skip permanently inactive
    # -----------------------------------------------------

    if combination in inactive_combinations:

        skipped_inactive += 1
        continue

    # -----------------------------------------------------
    # Get group directly
    # -----------------------------------------------------

    try:
        group = grouped.get_group(
            combination
        )
    except KeyError:
        continue

    group = (
        group
        .sort_values("date")
        .copy()
    )

    # -----------------------------------------------------
    # Use arrays instead of repeatedly filtering pandas.
    # -----------------------------------------------------

    dates = (
        group["date"]
        .to_numpy()
    )

    footfall = (
        group["total_footfall"]
        .astype(float)
        .to_numpy()
    )

    date_to_index = {
        date: i
        for i, date in enumerate(dates)
    }

    # -----------------------------------------------------
    # Process only target dates
    # -----------------------------------------------------

    for target_date in target_dates:

        # -------------------------------------------------
        # Actual target row must exist
        # -------------------------------------------------

        target_index = date_to_index.get(
            target_date
        )

        if target_index is None:
            continue

        # -------------------------------------------------
        # Need at least 30 observations BEFORE target.
        #
        # Original code checks:
        # len(history) < 30
        # -------------------------------------------------

        if target_index < 30:

            skipped_history += 1
            continue

        # -------------------------------------------------
        # History ends immediately before target.
        # -------------------------------------------------

        history_values = footfall[
            target_index - 30:
            target_index
        ]

        # -------------------------------------------------
        # Calendar features
        # -------------------------------------------------

        row = {

            "store_id": int(store_id),

            "gate_id": int(gate_id),

            "year": target_date.year,

            "month": target_date.month,

            "day": target_date.day,

            "weekday": target_date.weekday(),

            "week": int(
                target_date.isocalendar().week
            ),

            "quarter": target_date.quarter,

            "is_weekend": int(
                target_date.weekday() >= 5
            ),

            "holiday": int(
                target_date.date()
                in india_holidays
            ),

            # -------------------------------------------------
            # Lag features
            # -------------------------------------------------

            "lag1": history_values[-1],

            "lag7": history_values[-7],

            "lag30": history_values[-30],

            # -------------------------------------------------
            # Rolling features
            # -------------------------------------------------

            "rolling7": np.mean(
                history_values[-7:]
            ),

            "rolling30": np.mean(
                history_values[-30:]
            ),

            # -------------------------------------------------
            # Zero features
            # -------------------------------------------------

            "zero_lag1": int(
                history_values[-1] == 0
            ),

            "zero_lag7": int(
                history_values[-7] == 0
            ),

            "zero_count7": int(
                np.sum(
                    history_values[-7:] == 0
                )
            ),

            "zero_count30": int(
                np.sum(
                    history_values[-30:] == 0
                )
            ),

            # -------------------------------------------------
            # Actual value
            # -------------------------------------------------

            "_actual": float(
                footfall[target_index]
            ),

            "_date": target_date
        }

        feature_rows.append(row)

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    processed_combinations += 1

    if processed_combinations % 25 == 0:

        print(
            "Processed combinations:",
            processed_combinations,
            "/",
            total_combinations
        )


# =========================================================
# CREATE FEATURE DATAFRAME
# =========================================================

print()
print("=" * 60)
print("FEATURE PREPARATION COMPLETE")
print("=" * 60)

feature_df = pd.DataFrame(
    feature_rows
)

print(
    "Rows prepared:",
    len(feature_df)
)

print(
    "Inactive combinations skipped:",
    skipped_inactive
)

print(
    "Short-history rows skipped:",
    skipped_history
)


# =========================================================
# STOP IF EMPTY
# =========================================================

if feature_df.empty:

    print()
    print(
        "NO PREDICTIONS WERE GENERATED."
    )

    raise SystemExit


# =========================================================
# FAST BATCH PREDICTION
# =========================================================

print()
print("=" * 60)
print("STARTING BATCH PREDICTION")
print("=" * 60)

# ---------------------------------------------------------
# Extract features once
# ---------------------------------------------------------

X = feature_df[
    features
].copy()

# ---------------------------------------------------------
# XGBoost predicts thousands of rows at once.
# This replaces hundreds of thousands of individual
# model.predict() calls.
# ---------------------------------------------------------

predictions = []

total_rows = len(X)

for start in range(
    0,
    total_rows,
    PREDICTION_BATCH_SIZE
):

    end = min(
        start + PREDICTION_BATCH_SIZE,
        total_rows
    )

    batch = X.iloc[
        start:end
    ]

    batch_predictions = (
        model.predict(batch)
    )

    predictions.extend(
        batch_predictions
    )

    print(
        "Predicted rows:",
        end,
        "/",
        total_rows
    )


# =========================================================
# ADD PREDICTIONS
# =========================================================

feature_df["predicted"] = np.maximum(
    0,
    np.asarray(predictions)
)

feature_df["actual"] = (
    feature_df["_actual"]
)

feature_df["absolute_error"] = (
    np.abs(
        feature_df["predicted"]
        -
        feature_df["actual"]
    )
)


# =========================================================
# SAFE ERROR %
# =========================================================

positive_actual = (
    feature_df["actual"] > 0
)

feature_df["error_percent"] = np.nan

feature_df.loc[
    positive_actual,
    "error_percent"
] = (
    feature_df.loc[
        positive_actual,
        "absolute_error"
    ]
    /
    feature_df.loc[
        positive_actual,
        "actual"
    ]
    *
    100
)


# =========================================================
# CREATE FINAL RESULT
# =========================================================

prediction_df = pd.DataFrame({

    "date":
        feature_df["_date"],

    "store_id":
        feature_df["store_id"],

    "gate_id":
        feature_df["gate_id"],

    "predicted":
        feature_df["predicted"],

    "actual":
        feature_df["actual"],

    "absolute_error":
        feature_df["absolute_error"],

    "error_percent":
        feature_df["error_percent"]
})


# =========================================================
# SORT
# =========================================================

prediction_df = (
    prediction_df
    .sort_values(
        [
            "date",
            "store_id",
            "gate_id"
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# CALCULATE MAE
# =========================================================

mae = (
    prediction_df[
        "absolute_error"
    ].mean()
)


# =========================================================
# CALCULATE RMSE
# =========================================================

rmse = np.sqrt(
    (
        prediction_df[
            "absolute_error"
        ] ** 2
    ).mean()
)


# =========================================================
# MAPE
# =========================================================

positive_actual = (
    prediction_df["actual"] > 0
)

if positive_actual.any():

    mape = (
        (
            prediction_df.loc[
                positive_actual,
                "absolute_error"
            ]
            /
            prediction_df.loc[
                positive_actual,
                "actual"
            ]
        )
        .mean()
        * 100
    )

else:

    mape = np.nan


# =========================================================
# ROUND VALUES
# =========================================================

prediction_df["predicted"] = (
    prediction_df["predicted"]
    .round(2)
)

prediction_df["actual"] = (
    prediction_df["actual"]
    .round(2)
)

prediction_df["absolute_error"] = (
    prediction_df["absolute_error"]
    .round(2)
)

prediction_df["error_percent"] = (
    prediction_df["error_percent"]
    .round(2)
)


# =========================================================
# SAVE
# =========================================================

prediction_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print(
    "Prediction rows:",
    len(prediction_df)
)

print(
    "Permanently inactive skipped:",
    skipped_inactive
)

print(
    "Short-history skipped:",
    skipped_history
)


print()
print("=" * 60)
print("FINAL METRICS")
print("=" * 60)

print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "MAPE (actual > 0 only):",
    round(mape, 4)
)


print()
print("=" * 60)
print("DATE RANGE")
print("=" * 60)

print(
    "First:",
    prediction_df["date"].min()
)

print(
    "Last:",
    prediction_df["date"].max()
)


print()
print("=" * 60)
print("SAMPLE RESULTS")
print("=" * 60)

print(
    prediction_df
    .head(20)
    .to_string(index=False)
)


print()
print("=" * 60)
print("SAVED")
print("=" * 60)

print(
    OUTPUT_PATH
)