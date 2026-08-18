# ================================================================
# FOOTFALL PREDICTION - TWO STAGE MODEL V3.1
# ================================================================
#
# V3.1 FIXES:
#
# 1. Leakage-safe historical statistics
#    - store_mean
#    - gate_mean
#    - store_weekday_mean
#    - gate_weekday_mean
#
# 2. Strict DATE-based train/test split
#    - No date appears in both train and test
#
# 3. Keeps the same 32 features
#
# 4. Keeps the same XGBoost architecture
#
# 5. Target transformation:
#       log1p(total_footfall)
#
# Stage 1:
#       LOW vs NORMAL/HIGH classifier
#
# Stage 2:
#       LOW footfall regressor
#       NORMAL/HIGH footfall regressor
#
# ================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG
# ================================================================

DATA_PATH = "hourlyfootfall_till_current_date1.csv"

MODEL_DIR = "models_v3_1"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# Bottom 25% of TRAINING footfall = LOW
LOW_FOOTFALL_PERCENTILE = 25


# ================================================================
# 32 FEATURES
# ================================================================

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


# ================================================================
# START
# ================================================================

print("=" * 75)
print("FOOTFALL TWO-STAGE MODEL V3.1")
print("=" * 75)

print("\nIMPORTANT FIXES:")
print("1. Leakage-safe historical statistics")
print("2. Strict date-based train/test split")
print("3. No overlapping train/test dates")
print("4. Same 32-feature architecture")
print("5. Same XGBoost hyperparameters")


print(f"\nNumber of features: {len(FEATURES)}")

print("\nFeatures:")

for i, feature in enumerate(FEATURES, 1):
    print(f"{i:2d}. {feature}")


# ================================================================
# CREATE MODEL DIRECTORY
# ================================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ================================================================
# LOAD DATA
# ================================================================

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"""
Dataset not found:

{DATA_PATH}

Put the CSV in the same folder as this script
or change DATA_PATH.
"""
    )


df = pd.read_csv(DATA_PATH)

print(
    f"\nDataset shape: {df.shape}"
)

print("\nColumns:")

print(
    df.columns.tolist()
)


# ================================================================
# CHECK REQUIRED COLUMNS
# ================================================================

required_columns = [

    "date",
    "gate_id",
    "store_id",
    "total_footfall"

]

missing_columns = [

    col
    for col in required_columns
    if col not in df.columns

]

if missing_columns:

    raise ValueError(

        "\nMissing required columns:\n"
        +
        "\n".join(
            missing_columns
        )

    )


# ================================================================
# PROCESS DATE
# ================================================================

print("\n" + "=" * 75)
print("PROCESSING DATE")
print("=" * 75)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=["date"]
).copy()


# ================================================================
# PROCESS TARGET
# ================================================================

print("\nTarget column: total_footfall")

df["total_footfall"] = pd.to_numeric(
    df["total_footfall"],
    errors="coerce"
)

df = df.dropna(
    subset=["total_footfall"]
).copy()


# Footfall cannot be negative
df["total_footfall"] = df[
    "total_footfall"
].clip(
    lower=0
)


# ================================================================
# REMOVE DUPLICATE STORE/GATE/DATE ROWS
# ================================================================

print("\nChecking duplicate store/gate/date rows...")

duplicate_count = df.duplicated(
    subset=[
        "store_id",
        "gate_id",
        "date"
    ]
).sum()

print(
    f"Duplicate rows: {duplicate_count:,}"
)

if duplicate_count > 0:

    print(
        "Removing duplicate rows..."
    )

    df = df.drop_duplicates(
        subset=[
            "store_id",
            "gate_id",
            "date"
        ],
        keep="last"
    ).copy()


# ================================================================
# SORT DATA
# ================================================================

print("\nSorting data...")

# IMPORTANT:
# For lag/rolling calculations, each store+gate
# must be chronological.

df = df.sort_values(
    [
        "store_id",
        "gate_id",
        "date"
    ]
).reset_index(
    drop=True
)


# ================================================================
# CALENDAR FEATURES
# ================================================================

print("\n" + "=" * 75)
print("CREATING CALENDAR FEATURES")
print("=" * 75)

df["year"] = (
    df["date"].dt.year
)

df["month"] = (
    df["date"].dt.month
)

df["day"] = (
    df["date"].dt.day
)

df["weekday"] = (
    df["date"].dt.weekday
)

df["week"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["quarter"] = (
    df["date"].dt.quarter
)

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

df["is_month_start"] = (
    df["date"]
    .dt.is_month_start
    .astype(int)
)

df["is_month_end"] = (
    df["date"]
    .dt.is_month_end
    .astype(int)
)

df["is_quarter_start"] = (
    df["date"]
    .dt.is_quarter_start
    .astype(int)
)

df["is_quarter_end"] = (
    df["date"]
    .dt.is_quarter_end
    .astype(int)
)


# ================================================================
# HOLIDAY
# ================================================================

print("\nCreating holiday feature...")

if "holiday" not in df.columns:

    try:

        import holidays

        years = df[
            "year"
        ].unique()

        india_holidays = holidays.India(
            years=years
        )

        df["holiday"] = (

            df["date"]
            .dt.date
            .map(
                lambda x:
                1
                if x in india_holidays
                else 0
            )

        )

        print(
            "India holiday calendar loaded."
        )

    except Exception as e:

        print(
            "Holiday package unavailable."
        )

        print(
            f"Reason: {e}"
        )

        df["holiday"] = 0

else:

    df["holiday"] = pd.to_numeric(
        df["holiday"],
        errors="coerce"
    ).fillna(0)


# ================================================================
# CYCLICAL FEATURES
# ================================================================

print("\nCreating cyclical features...")

df["weekday_sin"] = np.sin(
    2 *
    np.pi *
    df["weekday"] /
    7
)

df["weekday_cos"] = np.cos(
    2 *
    np.pi *
    df["weekday"] /
    7
)

df["month_sin"] = np.sin(
    2 *
    np.pi *
    (df["month"] - 1) /
    12
)

df["month_cos"] = np.cos(
    2 *
    np.pi *
    (df["month"] - 1) /
    12
)


# ================================================================
# LAG FEATURES
# ================================================================

print("\n" + "=" * 75)
print("CREATING LAG FEATURES")
print("=" * 75)

group_cols = [
    "store_id",
    "gate_id"
]

grouped = df.groupby(
    group_cols,
    sort=False
)["total_footfall"]


df["lag1"] = grouped.shift(1)

df["lag7"] = grouped.shift(7)

df["lag14"] = grouped.shift(14)

df["lag21"] = grouped.shift(21)

df["lag28"] = grouped.shift(28)

df["lag30"] = grouped.shift(30)


# ================================================================
# ROLLING FEATURES
# ================================================================

print("\nCreating rolling features...")

df["rolling7"] = grouped.transform(

    lambda x:

    x.shift(1)
    .rolling(
        7,
        min_periods=7
    )
    .mean()

)


df["rolling14"] = grouped.transform(

    lambda x:

    x.shift(1)
    .rolling(
        14,
        min_periods=14
    )
    .mean()

)


df["rolling30"] = grouped.transform(

    lambda x:

    x.shift(1)
    .rolling(
        30,
        min_periods=30
    )
    .mean()

)


# ================================================================
# TREND
# ================================================================

print("\nCreating trend feature...")

df["trend"] = (
    df["rolling7"]
    -
    df["rolling30"]
)


# ================================================================
# LEAKAGE-SAFE HISTORICAL MEAN FUNCTION
# ================================================================

print("\n" + "=" * 75)
print("CREATING LEAKAGE-SAFE HISTORICAL FEATURES")
print("=" * 75)


def historical_mean_by_date(
    data,
    group_columns,
    target_column
):
    """
    Calculate historical mean using ONLY dates
    strictly before the current date.

    This prevents future information from
    entering the feature.
    """

    temp = (

        data
        .groupby(
            group_columns + ["date"],
            as_index=False
        )[target_column]
        .agg(
            daily_sum="sum",
            daily_count="count"
        )

    )

    temp = temp.sort_values(
        group_columns + ["date"]
    ).reset_index(
        drop=True
    )


    # ------------------------------------------------------------
    # Cumulative sum BEFORE current date
    # ------------------------------------------------------------

    temp["cumulative_sum"] = (

        temp
        .groupby(
            group_columns
        )["daily_sum"]
        .cumsum()

        -

        temp["daily_sum"]

    )


    # ------------------------------------------------------------
    # Cumulative count BEFORE current date
    # ------------------------------------------------------------

    temp["cumulative_count"] = (

        temp
        .groupby(
            group_columns
        )["daily_count"]
        .cumsum()

        -

        temp["daily_count"]

    )


    # ------------------------------------------------------------
    # Historical mean
    # ------------------------------------------------------------

    temp["historical_mean"] = np.where(

        temp["cumulative_count"] > 0,

        temp["cumulative_sum"]
        /
        temp["cumulative_count"],

        np.nan

    )


    return temp[
        group_columns +
        [
            "date",
            "historical_mean"
        ]
    ]


# ================================================================
# STORE MEAN
# ================================================================

print(
    "Creating store historical mean..."
)

store_history = historical_mean_by_date(

    df,

    ["store_id"],

    "total_footfall"

)


df = df.merge(

    store_history.rename(

        columns={
            "historical_mean":
            "store_mean"
        }

    ),

    on=[
        "store_id",
        "date"
    ],

    how="left"

)


# ================================================================
# GATE MEAN
# ================================================================

print(
    "Creating gate historical mean..."
)

gate_history = historical_mean_by_date(

    df,

    ["gate_id"],

    "total_footfall"

)


df = df.merge(

    gate_history.rename(

        columns={
            "historical_mean":
            "gate_mean"
        }

    ),

    on=[
        "gate_id",
        "date"
    ],

    how="left"

)


# ================================================================
# STORE + WEEKDAY MEAN
# ================================================================

print(
    "Creating store-weekday historical mean..."
)

store_weekday_history = historical_mean_by_date(

    df,

    [
        "store_id",
        "weekday"
    ],

    "total_footfall"

)


df = df.merge(

    store_weekday_history.rename(

        columns={
            "historical_mean":
            "store_weekday_mean"
        }

    ),

    on=[
        "store_id",
        "weekday",
        "date"
    ],

    how="left"

)


# ================================================================
# GATE + WEEKDAY MEAN
# ================================================================

print(
    "Creating gate-weekday historical mean..."
)

gate_weekday_history = historical_mean_by_date(

    df,

    [
        "gate_id",
        "weekday"
    ],

    "total_footfall"

)


df = df.merge(

    gate_weekday_history.rename(

        columns={
            "historical_mean":
            "gate_weekday_mean"
        }

    ),

    on=[
        "gate_id",
        "weekday",
        "date"
    ],

    how="left"

)


# ================================================================
# CHECK FEATURES
# ================================================================

print("\n" + "=" * 75)
print("CHECKING 32 FEATURES")
print("=" * 75)

missing_features = [

    feature

    for feature in FEATURES

    if feature not in df.columns

]

if missing_features:

    print(
        "\nMissing features:"
    )

    for feature in missing_features:

        print(
            " -",
            feature
        )

    raise ValueError(
        "\nFeature creation failed."
    )


print(
    "\nAll 32 features successfully created."
)


# ================================================================
# CHECK FOR INFINITE VALUES
# ================================================================

df[FEATURES] = df[
    FEATURES
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ================================================================
# REMOVE INSUFFICIENT HISTORY
# ================================================================

history_features = [

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


print("\n" + "=" * 75)
print("REMOVING INSUFFICIENT HISTORY")
print("=" * 75)

print(
    f"Rows before: {len(df):,}"
)

df = df.dropna(
    subset=history_features
).copy()

print(
    f"Rows after : {len(df):,}"
)


# ================================================================
# SORT BY DATE
# ================================================================

df = df.sort_values(
    [
        "date",
        "store_id",
        "gate_id"
    ]
).reset_index(
    drop=True
)


# ================================================================
# DATE-BASED TRAIN / TEST SPLIT
# ================================================================

print("\n" + "=" * 75)
print("STRICT DATE-BASED TRAIN / TEST SPLIT")
print("=" * 75)


unique_dates = np.sort(
    df["date"]
    .dropna()
    .unique()
)


split_date_index = int(

    len(unique_dates)
    *
    (1 - TEST_SIZE)

)


split_date = unique_dates[
    split_date_index
]


train_df = df[
    df["date"] < split_date
].copy()


test_df = df[
    df["date"] >= split_date
].copy()


print(
    f"\nTotal unique dates: "
    f"{len(unique_dates):,}"
)

print(
    f"Split date: "
    f"{pd.Timestamp(split_date).date()}"
)

print(
    f"\nTotal rows : "
    f"{len(df):,}"
)

print(
    f"Train rows : "
    f"{len(train_df):,}"
)

print(
    f"Test rows  : "
    f"{len(test_df):,}"
)


print(
    "\nTraining period:"
)

print(
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)


print(
    "\nTesting period:"
)

print(
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# ================================================================
# DATE OVERLAP CHECK
# ================================================================

train_dates = set(
    train_df["date"]
)

test_dates = set(
    test_df["date"]
)

date_overlap = (
    train_dates
    .intersection(
        test_dates
    )
)


if len(date_overlap) > 0:

    raise ValueError(

        f"""
DATE LEAKAGE DETECTED!

Overlapping dates:
{len(date_overlap)}

Examples:
{list(sorted(date_overlap))[:10]}
"""

    )


print(
    "\n✓ No date overlap between train and test."
)


# ================================================================
# LOW FOOTFALL THRESHOLD
# ================================================================

print("\n" + "=" * 75)
print("CALCULATING LOW-FOOTFALL THRESHOLD")
print("=" * 75)


LOW_THRESHOLD = np.percentile(

    train_df[
        "total_footfall"
    ],

    LOW_FOOTFALL_PERCENTILE

)


print(
    f"\nPercentile: "
    f"{LOW_FOOTFALL_PERCENTILE}%"
)

print(
    f"Low-footfall threshold: "
    f"{LOW_THRESHOLD:.2f}"
)


# ================================================================
# STAGE 1 LABEL
# ================================================================

train_df["low_footfall"] = (

    train_df[
        "total_footfall"
    ]

    <=

    LOW_THRESHOLD

).astype(int)


test_df["low_footfall"] = (

    test_df[
        "total_footfall"
    ]

    <=

    LOW_THRESHOLD

).astype(int)


print(
    "\nTraining class distribution:"
)

print(
    train_df[
        "low_footfall"
    ].value_counts()
)


print(
    "\n0 = NORMAL/HIGH"
)

print(
    "1 = LOW"
)


# ================================================================
# X / Y
# ================================================================

X_train = train_df[
    FEATURES
].copy()


X_test = test_df[
    FEATURES
].copy()


y_train = train_df[
    "total_footfall"
].copy()


y_test = test_df[
    "total_footfall"
].copy()


# ================================================================
# FINAL FEATURE VALIDATION
# ================================================================

print("\n" + "=" * 75)
print("FINAL FEATURE VALIDATION")
print("=" * 75)


print(
    f"X_train shape: "
    f"{X_train.shape}"
)

print(
    f"X_test shape : "
    f"{X_test.shape}"
)


if X_train.isnull().any().any():

    raise ValueError(
        "NaN values found in X_train."
    )


if X_test.isnull().any().any():

    raise ValueError(
        "NaN values found in X_test."
    )


print(
    "\n✓ No NaN values in training/test features."
)


# ================================================================
# LOG TARGET
# ================================================================

y_train_log = np.log1p(
    y_train
)

y_test_log = np.log1p(
    y_test
)


# ================================================================
# STAGE 1 - LOW FOOTFALL CLASSIFIER
# ================================================================

print("\n" + "=" * 75)
print("STAGE 1 - LOW FOOTFALL CLASSIFIER")
print("=" * 75)


stage1_model = XGBClassifier(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=7,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


stage1_model.fit(

    X_train,

    train_df[
        "low_footfall"
    ]

)


# ================================================================
# STAGE 1 PREDICTION
# ================================================================

stage1_pred = stage1_model.predict(
    X_test
)


stage1_probability = (
    stage1_model.predict_proba(
        X_test
    )[:, 1]
)


# ================================================================
# STAGE 1 METRICS
# ================================================================

stage1_accuracy = accuracy_score(

    test_df[
        "low_footfall"
    ],

    stage1_pred

)


stage1_precision = precision_score(

    test_df[
        "low_footfall"
    ],

    stage1_pred,

    zero_division=0

)


stage1_recall = recall_score(

    test_df[
        "low_footfall"
    ],

    stage1_pred,

    zero_division=0

)


stage1_f1 = f1_score(

    test_df[
        "low_footfall"
    ],

    stage1_pred,

    zero_division=0

)


print(
    "\nStage 1 Results"
)

print(
    f"Accuracy : "
    f"{stage1_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{stage1_precision:.4f}"
)

print(
    f"Recall   : "
    f"{stage1_recall:.4f}"
)

print(
    f"F1 Score : "
    f"{stage1_f1:.4f}"
)


print(
    "\nConfusion Matrix:"
)

cm = confusion_matrix(

    test_df[
        "low_footfall"
    ],

    stage1_pred

)

print(cm)


# ================================================================
# TRAINING MASKS
# ================================================================

low_train_mask = (

    train_df[
        "low_footfall"
    ] == 1

)


normal_train_mask = (

    train_df[
        "low_footfall"
    ] == 0

)


X_train_low = X_train[
    low_train_mask
]


y_train_low = y_train_log[
    low_train_mask
]


X_train_normal = X_train[
    normal_train_mask
]


y_train_normal = y_train_log[
    normal_train_mask
]


print("\n" + "=" * 75)
print("STAGE 2 DATA")
print("=" * 75)


print(
    f"Low-footfall samples     : "
    f"{len(X_train_low):,}"
)


print(
    f"Normal/high-footfall     : "
    f"{len(X_train_normal):,}"
)


# ================================================================
# STAGE 2A - LOW FOOTFALL REGRESSOR
# ================================================================

print("\n" + "=" * 75)
print("STAGE 2A - LOW FOOTFALL REGRESSOR")
print("=" * 75)


low_model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


low_model.fit(

    X_train_low,

    y_train_low

)


# ================================================================
# STAGE 2B - NORMAL/HIGH REGRESSOR
# ================================================================

print("\n" + "=" * 75)
print("STAGE 2B - NORMAL/HIGH FOOTFALL REGRESSOR")
print("=" * 75)


normal_model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


normal_model.fit(

    X_train_normal,

    y_train_normal

)


# ================================================================
# FINAL PREDICTIONS
# ================================================================

print("\n" + "=" * 75)
print("GENERATING FINAL PREDICTIONS")
print("=" * 75)


final_log_prediction = np.zeros(
    len(X_test)
)


predicted_low_mask = (

    stage1_pred == 1

)


predicted_normal_mask = (

    stage1_pred == 0

)


# ================================================================
# LOW MODEL PREDICTION
# ================================================================

if predicted_low_mask.sum() > 0:

    X_low_test = X_test[
        predicted_low_mask
    ]

    low_prediction = (
        low_model.predict(
            X_low_test
        )
    )

    final_log_prediction[
        predicted_low_mask
    ] = low_prediction


# ================================================================
# NORMAL/HIGH MODEL PREDICTION
# ================================================================

if predicted_normal_mask.sum() > 0:

    X_normal_test = X_test[
        predicted_normal_mask
    ]

    normal_prediction = (
        normal_model.predict(
            X_normal_test
        )
    )

    final_log_prediction[
        predicted_normal_mask
    ] = normal_prediction


# ================================================================
# LOG -> ORIGINAL SCALE
# ================================================================

final_prediction = np.expm1(
    final_log_prediction
)


final_prediction = np.maximum(
    final_prediction,
    0
)


# ================================================================
# METRICS FUNCTION
# ================================================================

def calculate_mape(
    actual,
    predicted,
    minimum_actual=0
):

    mask = (
        actual >= minimum_actual
    )

    actual_filtered = np.asarray(
        actual
    )[mask]

    predicted_filtered = np.asarray(
        predicted
    )[mask]

    nonzero_mask = (
        actual_filtered != 0
    )

    actual_filtered = (
        actual_filtered[
            nonzero_mask
        ]
    )

    predicted_filtered = (
        predicted_filtered[
            nonzero_mask
        ]
    )

    if len(actual_filtered) == 0:

        return np.nan

    return np.mean(

        np.abs(
            (
                actual_filtered
                -
                predicted_filtered
            )
            /
            actual_filtered
        )

    ) * 100


def calculate_wape(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:

        return np.nan

    return (

        np.sum(
            np.abs(
                actual
                -
                predicted
            )
        )

        /

        denominator

    ) * 100


# ================================================================
# OVERALL METRICS
# ================================================================

mae = mean_absolute_error(

    y_test,

    final_prediction

)


rmse = np.sqrt(

    mean_squared_error(

        y_test,

        final_prediction

    )

)


r2 = r2_score(

    y_test,

    final_prediction

)


mape_all = calculate_mape(

    y_test,

    final_prediction,

    minimum_actual=0

)


mape_10 = calculate_mape(

    y_test,

    final_prediction,

    minimum_actual=10

)


mape_25 = calculate_mape(

    y_test,

    final_prediction,

    minimum_actual=25

)


mape_50 = calculate_mape(

    y_test,

    final_prediction,

    minimum_actual=50

)


wape = calculate_wape(

    y_test,

    final_prediction

)


print("\n" + "=" * 75)
print("OVERALL TWO-STAGE RESULTS")
print("=" * 75)


print(
    f"\nMAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"R² % : {r2 * 100:.2f}%"
)

print(
    f"MAPE : {mape_all:.4f}%"
)

print(
    f"MAPE >= 10 : {mape_10:.4f}%"
)

print(
    f"MAPE >= 25 : {mape_25:.4f}%"
)

print(
    f"MAPE >= 50 : {mape_50:.4f}%"
)

print(
    f"WAPE : {wape:.4f}%"
)


# ================================================================
# LOW ACTUAL GROUP
# ================================================================

actual_low_mask = (

    test_df[
        "low_footfall"
    ] == 1

)


if actual_low_mask.sum() > 1:

    low_actual = y_test[
        actual_low_mask
    ]

    low_pred = final_prediction[
        actual_low_mask.values
    ]

    low_mae = mean_absolute_error(

        low_actual,

        low_pred

    )

    low_rmse = np.sqrt(

        mean_squared_error(

            low_actual,

            low_pred

        )

    )

    low_r2 = r2_score(

        low_actual,

        low_pred

    )

    low_mape = calculate_mape(

        low_actual,

        low_pred,

        minimum_actual=1

    )

    low_wape = calculate_wape(

        low_actual,

        low_pred

    )


    print("\n" + "=" * 75)
    print("LOW-FOOTFALL PERFORMANCE")
    print("=" * 75)


    print(
        f"Samples: "
        f"{len(low_actual):,}"
    )

    print(
        f"MAE : "
        f"{low_mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{low_rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{low_r2:.4f}"
    )

    print(
        f"MAPE: "
        f"{low_mape:.4f}%"
    )

    print(
        f"WAPE: "
        f"{low_wape:.4f}%"
    )


# ================================================================
# NORMAL/HIGH ACTUAL GROUP
# ================================================================

actual_normal_mask = (

    test_df[
        "low_footfall"
    ] == 0

)


if actual_normal_mask.sum() > 1:

    normal_actual = y_test[
        actual_normal_mask
    ]

    normal_pred = final_prediction[
        actual_normal_mask.values
    ]

    normal_mae = mean_absolute_error(

        normal_actual,

        normal_pred

    )

    normal_rmse = np.sqrt(

        mean_squared_error(

            normal_actual,

            normal_pred

        )

    )

    normal_r2 = r2_score(

        normal_actual,

        normal_pred

    )

    normal_mape = calculate_mape(

        normal_actual,

        normal_pred,

        minimum_actual=10

    )

    normal_wape = calculate_wape(

        normal_actual,

        normal_pred

    )


    print("\n" + "=" * 75)
    print("NORMAL/HIGH PERFORMANCE")
    print("=" * 75)


    print(
        f"Samples: "
        f"{len(normal_actual):,}"
    )

    print(
        f"MAE : "
        f"{normal_mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{normal_rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{normal_r2:.4f}"
    )

    print(
        f"MAPE: "
        f"{normal_mape:.4f}%"
    )

    print(
        f"WAPE: "
        f"{normal_wape:.4f}%"
    )


# ================================================================
# CLASSIFICATION ROUTING ANALYSIS
# ================================================================

actual_low = (
    test_df[
        "low_footfall"
    ].values == 1
)

predicted_low = (
    stage1_pred == 1
)


actual_low_pred_low = (
    actual_low &
    predicted_low
)

actual_low_pred_normal = (
    actual_low &
    (~predicted_low)
)

actual_normal_pred_low = (
    (~actual_low) &
    predicted_low
)

actual_normal_pred_normal = (
    (~actual_low) &
    (~predicted_low)
)


print("\n" + "=" * 75)
print("STAGE 1 ROUTING ANALYSIS")
print("=" * 75)


print(
    f"\nActual LOW → LOW model:"
    f" {actual_low_pred_low.sum():,}"
)

print(
    f"Actual LOW → NORMAL model:"
    f" {actual_low_pred_normal.sum():,}"
)

print(
    f"Actual NORMAL → LOW model:"
    f" {actual_normal_pred_low.sum():,}"
)

print(
    f"Actual NORMAL → NORMAL model:"
    f" {actual_normal_pred_normal.sum():,}"
)


# ================================================================
# SAVE MODELS
# ================================================================

print("\n" + "=" * 75)
print("SAVING MODELS")
print("=" * 75)


# ------------------------------------------------
# Stage 1
# ------------------------------------------------

stage1_path = os.path.join(

    MODEL_DIR,

    "stage1_low_classifier.pkl"

)


with open(

    stage1_path,

    "wb"

) as f:

    pickle.dump(

        stage1_model,

        f

    )


# ------------------------------------------------
# Low model
# ------------------------------------------------

low_model_path = os.path.join(

    MODEL_DIR,

    "low_footfall_model.pkl"

)


with open(

    low_model_path,

    "wb"

) as f:

    pickle.dump(

        low_model,

        f

    )


# ------------------------------------------------
# Normal model
# ------------------------------------------------

normal_model_path = os.path.join(

    MODEL_DIR,

    "normal_footfall_model.pkl"

)


with open(

    normal_model_path,

    "wb"

) as f:

    pickle.dump(

        normal_model,

        f

    )


# ================================================================
# SAVE MODEL INFORMATION
# ================================================================

model_info = {

    "model_version":
        "v3.1_two_stage",

    "target":
        "total_footfall",

    "date_column":
        "date",

    "features":
        FEATURES,

    "number_of_features":
        len(FEATURES),

    "low_threshold":
        float(LOW_THRESHOLD),

    "low_percentile":
        LOW_FOOTFALL_PERCENTILE,

    "target_transform":
        "log1p",

    "classifier":
        "XGBClassifier",

    "low_regressor":
        "XGBRegressor",

    "normal_regressor":
        "XGBRegressor",

    "train_start":
        str(
            train_df["date"].min()
        ),

    "train_end":
        str(
            train_df["date"].max()
        ),

    "test_start":
        str(
            test_df["date"].min()
        ),

    "test_end":
        str(
            test_df["date"].max()
        ),

    "metrics": {

        "stage1_accuracy":
            float(
                stage1_accuracy
            ),

        "stage1_precision":
            float(
                stage1_precision
            ),

        "stage1_recall":
            float(
                stage1_recall
            ),

        "stage1_f1":
            float(
                stage1_f1
            ),

        "overall_mae":
            float(mae),

        "overall_rmse":
            float(rmse),

        "overall_r2":
            float(r2),

        "overall_mape":
            float(mape_all)
            if not np.isnan(mape_all)
            else None,

        "overall_wape":
            float(wape)
            if not np.isnan(wape)
            else None

    }

}


info_path = os.path.join(

    MODEL_DIR,

    "model_info.pkl"

)


with open(

    info_path,

    "wb"

) as f:

    pickle.dump(

        model_info,

        f

    )


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

results = test_df[

    [
        "date",
        "store_id",
        "gate_id",
        "total_footfall",
        "low_footfall"
    ]

].copy()


results["predicted_footfall"] = (
    final_prediction
)


results["absolute_error"] = (

    np.abs(

        results[
            "total_footfall"
        ]

        -

        results[
            "predicted_footfall"
        ]

    )

)


results["error_percent"] = np.where(

    results[
        "total_footfall"
    ] != 0,

    (

        results[
            "absolute_error"
        ]

        /

        results[
            "total_footfall"
        ]

    ) * 100,

    np.nan

)


results["stage1_probability_low"] = (
    stage1_probability
)


results["model_used"] = np.where(

    predicted_low_mask,

    "low_model",

    "normal_model"

)


# ================================================================
# SAVE RESULTS
# ================================================================

results_path = os.path.join(

    MODEL_DIR,

    "test_predictions_v3_1.csv"

)


results.to_csv(

    results_path,

    index=False

)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print(
    "\nCreating feature importance..."
)


importance_df = pd.DataFrame({

    "feature":
        FEATURES,

    "low_model_importance":
        low_model.feature_importances_,

    "normal_model_importance":
        normal_model.feature_importances_,

    "classifier_importance":
        stage1_model.feature_importances_

})


importance_df = (

    importance_df

    .sort_values(

        "normal_model_importance",

        ascending=False

    )

)


importance_path = os.path.join(

    MODEL_DIR,

    "feature_importance_v3_1.csv"

)


importance_df.to_csv(

    importance_path,

    index=False

)


# ================================================================
# SAVE CONFIG FOR STREAMLIT
# ================================================================

config = {

    "model_version":
        "v3.1",

    "features":
        FEATURES,

    "target":
        "total_footfall",

    "low_threshold":
        float(LOW_THRESHOLD),

    "target_transform":
        "log1p",

    "model_type":
        "two_stage",

    "stage1_model":
        stage1_path,

    "low_model":
        low_model_path,

    "normal_model":
        normal_model_path

}


config_path = os.path.join(

    MODEL_DIR,

    "prediction_config.pkl"

)


with open(

    config_path,

    "wb"

) as f:

    pickle.dump(

        config,

        f

    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETE - V3.1")
print("=" * 75)


print("\nDataset:")

print(
    f"Rows used: "
    f"{len(df):,}"
)

print(
    f"Features: "
    f"{len(FEATURES)}"
)

print(
    "Target: total_footfall"
)

print(
    f"Low threshold: "
    f"{LOW_THRESHOLD:.2f}"
)


print("\n" + "-" * 75)


print(
    f"Train period: "
    f"{train_df['date'].min().date()}"
    f" → "
    f"{train_df['date'].max().date()}"
)


print(
    f"Test period : "
    f"{test_df['date'].min().date()}"
    f" → "
    f"{test_df['date'].max().date()}"
)


print("\n" + "-" * 75)


print(
    f"Stage 1 Accuracy : "
    f"{stage1_accuracy * 100:.2f}%"
)


print(
    f"Stage 1 Precision: "
    f"{stage1_precision * 100:.2f}%"
)


print(
    f"Stage 1 Recall   : "
    f"{stage1_recall * 100:.2f}%"
)


print(
    f"Stage 1 F1       : "
    f"{stage1_f1:.4f}"
)


print(
    f"Overall MAE      : "
    f"{mae:.2f}"
)


print(
    f"Overall RMSE     : "
    f"{rmse:.2f}"
)


print(
    f"Overall R²       : "
    f"{r2:.4f}"
)


print(
    f"Overall R²       : "
    f"{r2 * 100:.2f}%"
)


print(
    f"Overall MAPE     : "
    f"{mape_all:.2f}%"
)


print(
    f"MAPE >= 10       : "
    f"{mape_10:.2f}%"
)


print(
    f"WAPE             : "
    f"{wape:.2f}%"
)


print("\n" + "-" * 75)

print("Saved files:")

print(
    f"1. {stage1_path}"
)

print(
    f"2. {low_model_path}"
)

print(
    f"3. {normal_model_path}"
)

print(
    f"4. {info_path}"
)

print(
    f"5. {results_path}"
)

print(
    f"6. {importance_path}"
)

print(
    f"7. {config_path}"
)


print("\n" + "=" * 75)
print("DONE")
print("=" * 75)