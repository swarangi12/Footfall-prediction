import pandas as pd
import numpy as np
import pickle
import holidays

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# SETTINGS
# =========================================================

DATA_FILE = "hourlyfootfall_till_current_date1.csv"
MODEL_FILE = "footfall_prediction_model.json"

VALIDATION_DAYS = 60


# =========================================================
# OPTIONAL RETRAINING CHECK
# =========================================================

try:
    errors = pd.read_csv("error_log.csv")

    if "error_percent" in errors.columns:

        recent = errors.loc[
            errors["actual"] > 0,
            "error_percent"
        ].dropna().tail(7)

        if len(recent) > 0:

            recent_mape = recent.mean()

            print(
                f"Recent 7-record MAPE: "
                f"{recent_mape:.2f}%"
            )

            AUTO_SKIP_IF_GOOD = False

            if AUTO_SKIP_IF_GOOD and recent_mape <= 10:

                print("No Retraining Needed")

                raise SystemExit

except FileNotFoundError:

    print(
        "error_log.csv not found. "
        "Continuing with retraining."
    )


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"]
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
    "Rows:",
    len(df)
)

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

df["holiday"] = (
    df["date"]
    .dt.date
    .isin(india_holidays)
    .astype(int)
)


# =========================================================
# CALENDAR FEATURES
# =========================================================

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


# =========================================================
# CYCLICAL TIME FEATURES
# =========================================================

# Weekday cycle

df["weekday_sin"] = np.sin(
    2 * np.pi *
    df["weekday"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi *
    df["weekday"] / 7
)


# Month cycle

df["month_sin"] = np.sin(
    2 * np.pi *
    (df["month"] - 1) / 12
)

df["month_cos"] = np.cos(
    2 * np.pi *
    (df["month"] - 1) / 12
)


# =========================================================
# GROUPED TIME-SERIES FEATURES
# =========================================================

group_cols = [
    "store_id",
    "gate_id"
]

grouped = (
    df.groupby(group_cols)["total_footfall"]
)


# =========================================================
# LAG FEATURES
# =========================================================

df["lag1"] = (
    grouped.shift(1)
)

df["lag7"] = (
    grouped.shift(7)
)

df["lag14"] = (
    grouped.shift(14)
)

df["lag21"] = (
    grouped.shift(21)
)

df["lag28"] = (
    grouped.shift(28)
)

df["lag30"] = (
    grouped.shift(30)
)


# =========================================================
# ROLLING FEATURES
# IMPORTANT:
# shift(1) prevents target leakage
# =========================================================

df["rolling7"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=7
        )
        .mean()
    )
)


df["rolling14"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            14,
            min_periods=14
        )
        .mean()
    )
)


df["rolling21"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            21,
            min_periods=21
        )
        .mean()
    )
)


df["rolling28"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            28,
            min_periods=28
        )
        .mean()
    )
)


df["rolling30"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=30
        )
        .mean()
    )
)


# =========================================================
# ZERO-FOOTFALL FEATURES
# =========================================================

df["zero_lag1"] = (
    df["lag1"] == 0
).astype(int)


df["zero_lag7"] = (
    df["lag7"] == 0
).astype(int)


df["zero_count7"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=7
        )
        .apply(
            lambda y:
            np.sum(y == 0),
            raw=True
        )
    )
)


df["zero_count30"] = (

    grouped
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=30
        )
        .apply(
            lambda y:
            np.sum(y == 0),
            raw=True
        )
    )
)


# =========================================================
# TREND FEATURES
# =========================================================

df["trend_7_vs_30"] = (

    df["rolling7"] /
    df["rolling30"].replace(
        0,
        np.nan
    )
)


df["trend_14_vs_28"] = (

    df["rolling14"] /
    df["rolling28"].replace(
        0,
        np.nan
    )
)


# =========================================================
# DAY-OF-WEEK HISTORICAL AVERAGE
# =========================================================

df["_past_footfall"] = (
    grouped.shift(1)
)


df["dow_mean"] = (

    df.groupby(
        [
            "store_id",
            "gate_id",
            "weekday"
        ]
    )["_past_footfall"]
    .transform(
        lambda x:
        x.expanding()
        .mean()
        .shift(1)
    )
)


# Fill missing DOW mean using
# broader rolling history.

df["dow_mean"] = (
    df["dow_mean"]
    .fillna(
        df["rolling30"]
    )
)


# =========================================================
# REMOVE TEMPORARY COLUMN
# =========================================================

df = df.drop(
    columns=[
        "_past_footfall"
    ]
)


# =========================================================
# FEATURES
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
    "rolling21",
    "rolling28",
    "rolling30",

    "zero_lag1",
    "zero_lag7",
    "zero_count7",
    "zero_count30",

    "trend_7_vs_30",
    "trend_14_vs_28",

    "dow_mean"
]


# =========================================================
# CLEAN TRAINING DATA
# =========================================================

df = df.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)

df = df.dropna(
    subset=
    features +
    ["total_footfall"]
).copy()


print()
print(
    "Training rows after "
    "feature creation:",
    len(df)
)

print(
    "Number of features:",
    len(features)
)


# =========================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# =========================================================

unique_dates = np.sort(
    df["date"].unique()
)

if len(unique_dates) <= VALIDATION_DAYS:

    raise ValueError(
        "Not enough dates for "
        "the requested validation period."
    )


validation_dates = (
    unique_dates[
        -VALIDATION_DAYS:
    ]
)

validation_start = (
    validation_dates[0]
)


train_df = df[
    df["date"] <
    validation_start
].copy()


valid_df = df[
    df["date"] >=
    validation_start
].copy()


print()
print("=" * 60)
print("CHRONOLOGICAL VALIDATION")
print("=" * 60)

print(
    "Training:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print(
    "Validation:",
    valid_df["date"].min(),
    "to",
    valid_df["date"].max()
)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(valid_df)
)


# =========================================================
# TARGET
# =========================================================
#
# IMPORTANT CHANGE:
#
# Instead of training directly on:
#
#     total_footfall
#
# we train on:
#
#     log(1 + total_footfall)
#
# This reduces the dominance of very large
# footfall values.
#
# Predictions are converted back using:
#
#     expm1(prediction)
#
# =========================================================

X_train = (
    train_df[features]
)

X_valid = (
    valid_df[features]
)


# Keep the ORIGINAL values
# for evaluation.

y_train_actual = (
    train_df[
        "total_footfall"
    ].to_numpy()
)

y_valid_actual = (
    valid_df[
        "total_footfall"
    ].to_numpy()
)


# Log-transform only
# the training target.

y_train = np.log1p(
    y_train_actual
)

y_valid_log = np.log1p(
    y_valid_actual
)


print()
print("=" * 60)
print("TARGET TRANSFORMATION")
print("=" * 60)

print(
    "Training target: "
    "log1p(total_footfall)"
)

print(
    "Validation metrics: "
    "calculated on original footfall"
)


# =========================================================
# TRAIN MODEL
# =========================================================

print()
print("=" * 60)
print("TRAINING LOG-TARGET XGBOOST MODEL")
print("=" * 60)


model = XGBRegressor(

    n_estimators=1200,

    learning_rate=0.025,

    max_depth=7,

    min_child_weight=5,

    subsample=0.85,

    colsample_bytree=0.85,

    gamma=0.05,

    reg_alpha=0.10,

    reg_lambda=2.0,

    objective="reg:squarederror",

    eval_metric="mae",

    tree_method="hist",

    random_state=42,

    n_jobs=-1
)


model.fit(

    X_train,

    y_train,

    eval_set=[
        (
            X_valid,
            y_valid_log
        )
    ],

    verbose=False
)


# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

# Model predicts log(1 + footfall)

valid_pred_log = (
    model.predict(
        X_valid
    )
)


# Convert back to
# original footfall scale.

valid_pred = np.expm1(
    valid_pred_log
)


# Footfall cannot be negative.

valid_pred = np.maximum(
    valid_pred,
    0
)


# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_valid_actual,
    valid_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_valid_actual,
        valid_pred
    )
)


# MAPE only where
# actual > 0.

positive = (
    y_valid_actual > 0
)


mape = (

    np.mean(

        np.abs(

            (
                y_valid_actual[
                    positive
                ]
                -
                valid_pred[
                    positive
                ]
            )
            /
            y_valid_actual[
                positive
            ]

        )

    )

    * 100
)


# =========================================================
# MAPE BY FOOTFALL SIZE
# =========================================================

def calculate_mape(
    actual,
    predicted,
    minimum_actual=1
):

    mask = (
        actual >=
        minimum_actual
    )

    if mask.sum() == 0:
        return np.nan

    return (

        np.mean(

            np.abs(

                (
                    actual[mask]
                    -
                    predicted[mask]
                )
                /
                actual[mask]

            )

        )
        * 100

    )


mape_1_5 = calculate_mape(
    y_valid_actual,
    valid_pred,
    1
)

mask_1_5 = (
    (y_valid_actual >= 1)
    &
    (y_valid_actual <= 5)
)

if mask_1_5.sum() > 0:

    mape_1_5 = (

        np.mean(

            np.abs(

                (
                    y_valid_actual[
                        mask_1_5
                    ]
                    -
                    valid_pred[
                        mask_1_5
                    ]
                )
                /
                y_valid_actual[
                    mask_1_5
                ]

            )

        )
        * 100

    )
else:

    mape_1_5 = np.nan


mask_1_10 = (
    (y_valid_actual >= 1)
    &
    (y_valid_actual <= 10)
)

if mask_1_10.sum() > 0:

    mape_1_10 = (

        np.mean(

            np.abs(

                (
                    y_valid_actual[
                        mask_1_10
                    ]
                    -
                    valid_pred[
                        mask_1_10
                    ]
                )
                /
                y_valid_actual[
                    mask_1_10
                ]

            )

        )
        * 100

    )

else:

    mape_1_10 = np.nan


mape_ge_5 = calculate_mape(
    y_valid_actual,
    valid_pred,
    5
)

mape_ge_10 = calculate_mape(
    y_valid_actual,
    valid_pred,
    10
)

mape_ge_25 = calculate_mape(
    y_valid_actual,
    valid_pred,
    25
)

mape_ge_50 = calculate_mape(
    y_valid_actual,
    valid_pred,
    50
)

mape_ge_100 = calculate_mape(
    y_valid_actual,
    valid_pred,
    100
)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("=" * 60)
print("VALIDATION RESULTS - LOG TARGET")
print("=" * 60)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAPE : {mape:.4f}%"
)

print(
    f"MAPE-based score : "
    f"{max(0, 100 - mape):.2f}%"
)


print()
print("-" * 60)
print("MAPE BY ACTUAL FOOTFALL")
print("-" * 60)

print(
    f"MAPE actual 1-5   : "
    f"{mape_1_5:.4f}%"
)

print(
    f"MAPE actual 1-10  : "
    f"{mape_1_10:.4f}%"
)

print(
    f"MAPE actual >= 5  : "
    f"{mape_ge_5:.4f}%"
)

print(
    f"MAPE actual >= 10 : "
    f"{mape_ge_10:.4f}%"
)

print(
    f"MAPE actual >= 25 : "
    f"{mape_ge_25:.4f}%"
)

print(
    f"MAPE actual >= 50 : "
    f"{mape_ge_50:.4f}%"
)

print(
    f"MAPE actual >= 100: "
    f"{mape_ge_100:.4f}%"
)


# =========================================================
# LOW FOOTFALL MAE
# =========================================================

low_5 = (
    (y_valid_actual >= 1)
    &
    (y_valid_actual <= 5)
)

low_10 = (
    (y_valid_actual >= 1)
    &
    (y_valid_actual <= 10)
)


if low_5.sum() > 0:

    low5_mae = mean_absolute_error(
        y_valid_actual[low_5],
        valid_pred[low_5]
    )

else:

    low5_mae = np.nan


if low_10.sum() > 0:

    low10_mae = mean_absolute_error(
        y_valid_actual[low_10],
        valid_pred[low_10]
    )

else:

    low10_mae = np.nan


print()
print("-" * 60)
print("LOW FOOTFALL PERFORMANCE")
print("-" * 60)

print(
    f"MAE actual 1-5  : "
    f"{low5_mae:.4f}"
)

print(
    f"MAE actual 1-10 : "
    f"{low10_mae:.4f}"
)


# =========================================================
# COMPARISON WITH BASELINE
# =========================================================

BASELINE_MAPE = 23.0862

improvement = (
    BASELINE_MAPE -
    mape
)

relative_improvement = (

    improvement /
    BASELINE_MAPE
) * 100


print()
print("=" * 60)
print("COMPARISON WITH BASELINE")
print("=" * 60)

print(
    f"Baseline MAPE : "
    f"{BASELINE_MAPE:.4f}%"
)

print(
    f"New MAPE      : "
    f"{mape:.4f}%"
)

print(
    f"Improvement   : "
    f"{improvement:.4f} "
    f"percentage points"
)

print(
    f"Relative      : "
    f"{relative_improvement:.2f}%"
)


if mape < BASELINE_MAPE:

    print()
    print(
        "SUCCESS: New model "
        "improved over the baseline."
    )

else:

    print()
    print(
        "RESULT: New model did NOT "
        "beat the 23.0862% baseline."
    )

    print(
        "Keep the previous model "
        "as the better model."
    )


# =========================================================
# SAVE MODEL
# =========================================================

print()
print("=" * 60)
print("SAVING MODEL")
print("=" * 60)


with open(
    MODEL_FILE,
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print(
    f"Model saved to: "
    f"{MODEL_FILE}"
)


# =========================================================
# SAVE FEATURE LIST
# =========================================================

with open(
    "model_features.pkl",
    "wb"
) as f:

    pickle.dump(
        features,
        f
    )


print(
    "Feature list saved to: "
    "model_features.pkl"
)


# =========================================================
# SAVE VALIDATION PREDICTIONS
# =========================================================

validation_output = valid_df[
    [
        "date",
        "store_id",
        "gate_id",
        "total_footfall"
    ]
].copy()


validation_output[
    "predicted"
] = valid_pred


validation_output[
    "absolute_error"
] = np.abs(

    validation_output[
        "total_footfall"
    ]
    -
    validation_output[
        "predicted"
    ]

)


validation_output[
    "error_percent"
] = np.where(

    validation_output[
        "total_footfall"
    ] > 0,

    validation_output[
        "absolute_error"
    ]
    /
    validation_output[
        "total_footfall"
    ]
    * 100,

    np.nan
)


validation_output = (
    validation_output
    .rename(
        columns={
            "total_footfall":
            "actual"
        }
    )
)


validation_output.to_csv(
    "validation_predictions_log.csv",
    index=False
)


print(
    "Validation predictions saved to: "
    "validation_predictions_log.csv"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({

    "feature": features,

    "importance":
        model.feature_importances_

})


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print()
print("=" * 60)
print("TOP 20 FEATURES")
print("=" * 60)

print(
    importance
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print()
print("=" * 60)
print("LOG-TARGET MODEL TRAINING COMPLETE")
print("=" * 60)

print()
print(
    "Baseline MAPE : "
    f"{BASELINE_MAPE:.4f}%"
)

print(
    "New MAPE      : "
    f"{mape:.4f}%"
)

print()

if mape < BASELINE_MAPE:

    print(
        "The log-target model is "
        "better than the baseline."
    )

    print(
        "Use this model for the next "
        "backtesting step."
    )

else:

    print(
        "The baseline model remains "
        "better."
    )

    print(
        "Do NOT replace your baseline "
        "model with this one."
    )