# ================================================================
# FOOTFALL PREDICTION V3.1 - COMPLETE ANALYSIS
# ================================================================
#
# This script ONLY analyzes the already-generated V3.1 predictions.
#
# It does NOT:
#   - load the training dataset
#   - create holiday features
#   - create lag features
#   - train models
#
# Input:
#   models_v3_1/test_predictions_v3_1.csv
#
# Output:
#   analysis_v3_1/
#
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG
# ================================================================

MODEL_DIR = "models_v3_1"

ANALYSIS_DIR = "analysis_v3_1"

PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "test_predictions_v3_1.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    MODEL_DIR,
    "feature_importance_v3_1.csv"
)

MODEL_INFO_FILE = os.path.join(
    MODEL_DIR,
    "model_info.pkl"
)


# ================================================================
# CREATE OUTPUT DIRECTORY
# ================================================================

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


print("=" * 75)
print("FOOTFALL PREDICTION V3.1 COMPLETE ANALYSIS")
print("=" * 75)


# ================================================================
# CHECK INPUT FILES
# ================================================================

print("\n")
print("=" * 75)
print("CHECKING INPUT FILES")
print("=" * 75)

if not os.path.exists(PREDICTION_FILE):

    raise FileNotFoundError(
        f"\nPrediction file not found:\n"
        f"{PREDICTION_FILE}\n\n"
        f"First run the V3.1 training script."
    )

print(
    f"\nPrediction file found:\n"
    f"{PREDICTION_FILE}"
)


# ================================================================
# LOAD PREDICTIONS
# ================================================================

print("\n")
print("=" * 75)
print("LOADING V3.1 PREDICTIONS")
print("=" * 75)

df = pd.read_csv(
    PREDICTION_FILE
)

print(
    f"\nRows loaded: {len(df):,}"
)

print("\nColumns:")
print(df.columns.tolist())


# ================================================================
# REQUIRED COLUMNS
# ================================================================

required_columns = [
    "date",
    "store_id",
    "gate_id",
    "total_footfall",
    "predicted_footfall"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing)
    )


# ================================================================
# DATE
# ================================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=["date"]
).copy()


# ================================================================
# NUMERIC COLUMNS
# ================================================================

numeric_columns = [
    "store_id",
    "gate_id",
    "total_footfall",
    "predicted_footfall"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "total_footfall",
        "predicted_footfall"
    ]
).copy()


# ================================================================
# ERROR FEATURES
# ================================================================

print("\n")
print("=" * 75)
print("CREATING ERROR FEATURES")
print("=" * 75)


df["error"] = (
    df["predicted_footfall"]
    -
    df["total_footfall"]
)


df["absolute_error"] = (
    np.abs(
        df["error"]
    )
)


df["squared_error"] = (
    df["error"] ** 2
)


# Actual - predicted
# Positive = underprediction
# Negative = overprediction

df["underprediction_amount"] = np.maximum(
    df["total_footfall"]
    -
    df["predicted_footfall"],
    0
)


df["overprediction_amount"] = np.maximum(
    df["predicted_footfall"]
    -
    df["total_footfall"],
    0
)


df["error_percent_calculated"] = np.where(

    df["total_footfall"] != 0,

    (
        df["absolute_error"]
        /
        np.abs(
            df["total_footfall"]
        )
    ) * 100,

    np.nan
)


# Use existing error_percent if available
# otherwise use calculated value

if "error_percent" in df.columns:

    df["error_percent"] = pd.to_numeric(
        df["error_percent"],
        errors="coerce"
    )

else:

    df["error_percent"] = (
        df["error_percent_calculated"]
    )


# ================================================================
# PREDICTION DIRECTION
# ================================================================

df["prediction_direction"] = np.select(

    [
        df["error"] > 0,
        df["error"] < 0,
        df["error"] == 0
    ],

    [
        "OVERPREDICTED",
        "UNDERPREDICTED",
        "EXACT"
    ],

    default="EXACT"
)


# ================================================================
# FOOTFALL CATEGORY
# ================================================================

if "low_footfall" in df.columns:

    df["low_footfall"] = pd.to_numeric(
        df["low_footfall"],
        errors="coerce"
    )

    df["footfall_category"] = np.where(

        df["low_footfall"] == 1,

        "LOW",

        "NORMAL/HIGH"
    )

else:

    # If low_footfall isn't present,
    # classify using the training threshold
    # from model_info if possible.

    try:

        import pickle

        with open(
            MODEL_INFO_FILE,
            "rb"
        ) as f:

            model_info = pickle.load(f)

        threshold = float(
            model_info["low_threshold"]
        )

        df["footfall_category"] = np.where(

            df["total_footfall"] <= threshold,

            "LOW",

            "NORMAL/HIGH"
        )

    except Exception:

        df["footfall_category"] = "UNKNOWN"


# ================================================================
# BASIC SUMMARY
# ================================================================

print("\n")
print("=" * 75)
print("DATA SUMMARY")
print("=" * 75)

print(
    f"\nRows: {len(df):,}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)

print(
    f"Stores: "
    f"{df['store_id'].nunique()}"
)

print(
    f"Gates: "
    f"{df['gate_id'].nunique()}"
)


# ================================================================
# METRIC FUNCTION
# ================================================================

def calculate_metrics(data):

    actual = data["total_footfall"]

    predicted = data["predicted_footfall"]

    if len(data) == 0:

        return {
            "samples": 0,
            "actual_mean": np.nan,
            "predicted_mean": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE": np.nan,
            "WAPE": np.nan,
            "bias": np.nan,
            "median_absolute_error": np.nan,
            "max_absolute_error": np.nan
        }

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    try:

        r2 = r2_score(
            actual,
            predicted
        )

    except Exception:

        r2 = np.nan


    valid_mape = (
        actual != 0
    )

    if valid_mape.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    actual[valid_mape]
                    -
                    predicted[valid_mape]
                )
                /
                actual[valid_mape]
            )
        ) * 100

    else:

        mape = np.nan


    denominator = np.sum(
        np.abs(actual)
    )

    if denominator != 0:

        wape = (
            np.sum(
                np.abs(
                    actual - predicted
                )
            )
            /
            denominator
        ) * 100

    else:

        wape = np.nan


    return {

        "samples":
            len(data),

        "actual_mean":
            actual.mean(),

        "predicted_mean":
            predicted.mean(),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

        "MAPE":
            mape,

        "WAPE":
            wape,

        "bias":
            (
                predicted - actual
            ).mean(),

        "median_absolute_error":
            median_absolute_error(
                actual,
                predicted
            ),

        "max_absolute_error":
            np.max(
                np.abs(
                    actual - predicted
                )
            )
    }


# ================================================================
# OVERALL PERFORMANCE
# ================================================================

print("\n")
print("=" * 75)
print("OVERALL PERFORMANCE")
print("=" * 75)

overall = calculate_metrics(
    df
)

for key, value in overall.items():

    if key == "samples":

        print(
            f"{key:<25}: "
            f"{value:,}"
        )

    else:

        print(
            f"{key:<25}: "
            f"{value:.4f}"
        )


# ================================================================
# SAVE OVERALL SUMMARY
# ================================================================

overall_df = pd.DataFrame(
    [overall]
)

overall_df.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "overall_performance.csv"
    ),
    index=False
)


# ================================================================
# 1. STORE-WISE PERFORMANCE
# ================================================================

print("\n")
print("=" * 75)
print("1. STORE-WISE PERFORMANCE")
print("=" * 75)


store_rows = []

for store_id, group in df.groupby(
    "store_id"
):

    metrics = calculate_metrics(
        group
    )

    metrics["store_id"] = store_id

    store_rows.append(
        metrics
    )


store_performance = pd.DataFrame(
    store_rows
)


store_performance = store_performance[
    [
        "store_id",
        "samples",
        "actual_mean",
        "predicted_mean",
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
        "WAPE",
        "bias",
        "median_absolute_error",
        "max_absolute_error"
    ]
]


store_performance = (
    store_performance
    .sort_values(
        "MAE",
        ascending=False
    )
)


store_path = os.path.join(
    ANALYSIS_DIR,
    "store_performance.csv"
)

store_performance.to_csv(
    store_path,
    index=False
)


print(
    f"\nSaved: {store_path}"
)

print("\n10 WORST STORES BY MAE:")

print(
    store_performance
    .head(10)
    .to_string(index=False)
)


# ================================================================
# 2. GATE-WISE PERFORMANCE
# ================================================================

print("\n")
print("=" * 75)
print("2. GATE-WISE PERFORMANCE")
print("=" * 75)


gate_rows = []

for gate_id, group in df.groupby(
    "gate_id"
):

    metrics = calculate_metrics(
        group
    )

    metrics["gate_id"] = gate_id

    gate_rows.append(
        metrics
    )


gate_performance = pd.DataFrame(
    gate_rows
)


gate_performance = gate_performance[
    [
        "gate_id",
        "samples",
        "actual_mean",
        "predicted_mean",
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
        "WAPE",
        "bias",
        "median_absolute_error",
        "max_absolute_error"
    ]
]


gate_performance = (
    gate_performance
    .sort_values(
        "MAE",
        ascending=False
    )
)


gate_path = os.path.join(
    ANALYSIS_DIR,
    "gate_performance.csv"
)

gate_performance.to_csv(
    gate_path,
    index=False
)


print(
    f"\nSaved: {gate_path}"
)

print("\nGATE RESULTS:")

print(
    gate_performance
    .to_string(index=False)
)


# ================================================================
# 3. WEEKDAY PERFORMANCE
# ================================================================

print("\n")
print("=" * 75)
print("3. WEEKDAY PERFORMANCE")
print("=" * 75)


df["weekday_number"] = (
    df["date"].dt.weekday
)


weekday_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}


weekday_rows = []


for weekday, group in df.groupby(
    "weekday_number"
):

    metrics = calculate_metrics(
        group
    )

    metrics["weekday"] = (
        weekday_names[weekday]
    )

    metrics["weekday_number"] = (
        weekday
    )

    weekday_rows.append(
        metrics
    )


weekday_performance = pd.DataFrame(
    weekday_rows
)


weekday_performance = (
    weekday_performance
    .sort_values(
        "weekday_number"
    )
)


weekday_path = os.path.join(
    ANALYSIS_DIR,
    "weekday_performance.csv"
)

weekday_performance.to_csv(
    weekday_path,
    index=False
)


print(
    f"\nSaved: {weekday_path}"
)

print(
    weekday_performance
    .to_string(index=False)
)


# ================================================================
# 4. MONTHLY PERFORMANCE
# ================================================================

print("\n")
print("=" * 75)
print("4. MONTHLY PERFORMANCE")
print("=" * 75)


df["month_number"] = (
    df["date"].dt.month
)


month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


monthly_rows = []


for month, group in df.groupby(
    "month_number"
):

    metrics = calculate_metrics(
        group
    )

    metrics["month"] = (
        month_names[month]
    )

    metrics["month_number"] = (
        month
    )

    monthly_rows.append(
        metrics
    )


monthly_performance = pd.DataFrame(
    monthly_rows
)


monthly_performance = (
    monthly_performance
    .sort_values(
        "month_number"
    )
)


monthly_path = os.path.join(
    ANALYSIS_DIR,
    "monthly_performance.csv"
)

monthly_performance.to_csv(
    monthly_path,
    index=False
)


print(
    f"\nSaved: {monthly_path}"
)

print(
    monthly_performance
    .to_string(index=False)
)


# ================================================================
# 5. LOW / NORMAL PERFORMANCE
# ================================================================

print("\n")
print("=" * 75)
print("5. LOW / NORMAL PERFORMANCE")
print("=" * 75)


category_rows = []


for category, group in df.groupby(
    "footfall_category"
):

    metrics = calculate_metrics(
        group
    )

    metrics["footfall_category"] = (
        category
    )

    metrics["mean_absolute_error"] = (
        metrics["MAE"]
    )

    category_rows.append(
        metrics
    )


low_normal_performance = pd.DataFrame(
    category_rows
)


low_normal_path = os.path.join(
    ANALYSIS_DIR,
    "low_normal_performance.csv"
)

low_normal_performance.to_csv(
    low_normal_path,
    index=False
)


print(
    f"\nSaved: {low_normal_path}"
)

print(
    low_normal_performance
    .to_string(index=False)
)


# ================================================================
# 6. WORST 100 PREDICTIONS
# ================================================================

print("\n")
print("=" * 75)
print("6. WORST 100 PREDICTIONS")
print("=" * 75)


worst_columns = [
    "date",
    "store_id",
    "gate_id",
    "total_footfall",
    "predicted_footfall",
    "error",
    "absolute_error",
    "error_percent",
    "prediction_direction",
    "footfall_category"
]


worst_100 = (
    df[
        worst_columns
    ]
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(100)
)


worst_path = os.path.join(
    ANALYSIS_DIR,
    "worst_100_predictions.csv"
)

worst_100.to_csv(
    worst_path,
    index=False
)


print(
    f"\nSaved: {worst_path}"
)

print("\nTOP 20 WORST PREDICTIONS:")

print(
    worst_100
    .head(20)
    .to_string(index=False)
)


# ================================================================
# 7. OVERPREDICTION ANALYSIS
# ================================================================

print("\n")
print("=" * 75)
print("7. OVERPREDICTION ANALYSIS")
print("=" * 75)


over_df = df[
    df["error"] > 0
].copy()


over_metrics = {

    "count":
        len(over_df),

    "percentage_of_predictions":
        len(over_df) / len(df) * 100,

    "mean_overprediction":
        over_df["error"].mean()
        if len(over_df)
        else np.nan,

    "median_overprediction":
        over_df["error"].median()
        if len(over_df)
        else np.nan,

    "maximum_overprediction":
        over_df["error"].max()
        if len(over_df)
        else np.nan,

    "mean_actual":
        over_df["total_footfall"].mean()
        if len(over_df)
        else np.nan,

    "mean_predicted":
        over_df["predicted_footfall"].mean()
        if len(over_df)
        else np.nan
}


overprediction_analysis = pd.DataFrame(
    list(
        over_metrics.items()
    ),
    columns=[
        "metric",
        "value"
    ]
)


over_path = os.path.join(
    ANALYSIS_DIR,
    "overprediction_analysis.csv"
)

overprediction_analysis.to_csv(
    over_path,
    index=False
)


print(
    f"\nSaved: {over_path}"
)

print(
    overprediction_analysis
    .to_string(index=False)
)


# ================================================================
# 8. UNDERPREDICTION ANALYSIS
# ================================================================

print("\n")
print("=" * 75)
print("8. UNDERPREDICTION ANALYSIS")
print("=" * 75)


under_df = df[
    df["error"] < 0
].copy()


under_df["underprediction"] = (
    -under_df["error"]
)


under_metrics = {

    "count":
        len(under_df),

    "percentage_of_predictions":
        len(under_df) / len(df) * 100,

    "mean_underprediction":
        under_df["underprediction"].mean()
        if len(under_df)
        else np.nan,

    "median_underprediction":
        under_df["underprediction"].median()
        if len(under_df)
        else np.nan,

    "maximum_underprediction":
        under_df["underprediction"].max()
        if len(under_df)
        else np.nan,

    "mean_actual":
        under_df["total_footfall"].mean()
        if len(under_df)
        else np.nan,

    "mean_predicted":
        under_df["predicted_footfall"].mean()
        if len(under_df)
        else np.nan
}


underprediction_analysis = pd.DataFrame(
    list(
        under_metrics.items()
    ),
    columns=[
        "metric",
        "value"
    ]
)


under_path = os.path.join(
    ANALYSIS_DIR,
    "underprediction_analysis.csv"
)

underprediction_analysis.to_csv(
    under_path,
    index=False
)


print(
    f"\nSaved: {under_path}"
)

print(
    underprediction_analysis
    .to_string(index=False)
)


# ================================================================
# 9. ERROR DISTRIBUTION
# ================================================================

print("\n")
print("=" * 75)
print("9. ERROR DISTRIBUTION")
print("=" * 75)


bins = [
    0,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    np.inf
]


labels = [
    "0-5",
    "5-10",
    "10-20",
    "20-50",
    "50-100",
    "100-200",
    "200-500",
    "500-1000",
    "1000+"
]


df["error_bucket"] = pd.cut(
    df["absolute_error"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)


error_distribution = (
    df.groupby(
        "error_bucket",
        observed=False
    )
    .agg(
        samples=(
            "absolute_error",
            "size"
        ),

        mean_actual=(
            "total_footfall",
            "mean"
        ),

        mean_predicted=(
            "predicted_footfall",
            "mean"
        ),

        mean_absolute_error=(
            "absolute_error",
            "mean"
        )
    )
    .reset_index()
)


error_distribution[
    "percentage"
] = (
    error_distribution["samples"]
    /
    len(df)
) * 100


error_path = os.path.join(
    ANALYSIS_DIR,
    "error_distribution.csv"
)

error_distribution.to_csv(
    error_path,
    index=False
)


print(
    f"\nSaved: {error_path}"
)

print(
    error_distribution
    .to_string(index=False)
)


print("\nError percentiles:")

for percentile in [
    50,
    75,
    90,
    95,
    99
]:

    value = np.percentile(
        df["error_percent"].dropna(),
        percentile
    )

    print(
        f"P{percentile}: "
        f"{value:.2f}%"
    )


# ================================================================
# 10. FEATURE IMPORTANCE
# ================================================================

print("\n")
print("=" * 75)
print("10. FEATURE IMPORTANCE")
print("=" * 75)


if os.path.exists(
    FEATURE_IMPORTANCE_FILE
):

    feature_importance = pd.read_csv(
        FEATURE_IMPORTANCE_FILE
    )

    print("\nColumns:")
    print(
        feature_importance.columns.tolist()
    )


    # ------------------------------------------------------------
    # CLASSIFIER IMPORTANCE
    # ------------------------------------------------------------

    if "classifier_importance" in feature_importance.columns:

        classifier_importance = (
            feature_importance[
                [
                    "feature",
                    "classifier_importance"
                ]
            ]
            .sort_values(
                "classifier_importance",
                ascending=False
            )
        )

        classifier_path = os.path.join(
            ANALYSIS_DIR,
            "classifier_feature_importance.csv"
        )

        classifier_importance.to_csv(
            classifier_path,
            index=False
        )

        print(
            "\nTOP 15 CLASSIFIER FEATURES:"
        )

        print(
            classifier_importance
            .head(15)
            .to_string(index=False)
        )


    # ------------------------------------------------------------
    # NORMAL MODEL
    # ------------------------------------------------------------

    if "normal_model_importance" in feature_importance.columns:

        normal_importance = (
            feature_importance[
                [
                    "feature",
                    "normal_model_importance"
                ]
            ]
            .sort_values(
                "normal_model_importance",
                ascending=False
            )
        )

        print(
            "\nTOP 15 NORMAL/HIGH FEATURES:"
        )

        print(
            normal_importance
            .head(15)
            .to_string(index=False)
        )


    # ------------------------------------------------------------
    # LOW MODEL
    # ------------------------------------------------------------

    if "low_model_importance" in feature_importance.columns:

        low_importance = (
            feature_importance[
                [
                    "feature",
                    "low_model_importance"
                ]
            ]
            .sort_values(
                "low_model_importance",
                ascending=False
            )
        )

        print(
            "\nTOP 15 LOW-FOOTFALL FEATURES:"
        )

        print(
            low_importance
            .head(15)
            .to_string(index=False)
        )


    feature_importance.to_csv(
        os.path.join(
            ANALYSIS_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

else:

    print(
        "\nFeature importance file not found:"
    )

    print(
        FEATURE_IMPORTANCE_FILE
    )


# ================================================================
# 11. ERROR CONCENTRATION
# ================================================================

print("\n")
print("=" * 75)
print("ERROR CONCENTRATION ANALYSIS")
print("=" * 75)


store_error = (
    df.groupby("store_id")
    ["absolute_error"]
    .sum()
    .sort_values(
        ascending=False
    )
)


total_error = (
    df["absolute_error"].sum()
)


top10_error = (
    store_error
    .head(10)
    .sum()
)


top10_percentage = (
    top10_error
    /
    total_error
) * 100


print(
    f"\nTop 10 stores contribute "
    f"{top10_percentage:.2f}% "
    f"of total absolute error."
)


store_error_df = pd.DataFrame({

    "store_id":
        store_error.index,

    "total_absolute_error":
        store_error.values
})


store_error_df[
    "percentage_of_total_error"
] = (
    store_error_df[
        "total_absolute_error"
    ]
    /
    total_error
) * 100


store_error_df.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "store_error_concentration.csv"
    ),
    index=False
)


# ================================================================
# 12. STAGE 1 ROUTING ANALYSIS
# ================================================================

print("\n")
print("=" * 75)
print("STAGE 1 ROUTING ANALYSIS")
print("=" * 75)


if "model_used" in df.columns:

    routing_rows = []


    for model_used, group in df.groupby(
        "model_used"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics["model_used"] = (
            model_used
        )

        routing_rows.append(
            metrics
        )


    routing_performance = pd.DataFrame(
        routing_rows
    )


    routing_path = os.path.join(
        ANALYSIS_DIR,
        "routing_performance.csv"
    )


    routing_performance.to_csv(
        routing_path,
        index=False
    )


    print(
        f"\nSaved: {routing_path}"
    )

    print(
        routing_performance
        .to_string(index=False)
    )


    routing_table = pd.crosstab(

        df["footfall_category"],

        df["model_used"],

        margins=True
    )


    routing_table.to_csv(
        os.path.join(
            ANALYSIS_DIR,
            "routing_confusion_table.csv"
        )
    )


    print(
        "\nRouting table:"
    )

    print(
        routing_table
    )


# ================================================================
# 13. SYSTEMATIC BIAS
# ================================================================

print("\n")
print("=" * 75)
print("SYSTEMATIC BIAS ANALYSIS")
print("=" * 75)


mean_error = df["error"].mean()

median_error = df["error"].median()

over_count = (
    df["prediction_direction"]
    ==
    "OVERPREDICTED"
).sum()

under_count = (
    df["prediction_direction"]
    ==
    "UNDERPREDICTED"
).sum()

exact_count = (
    df["prediction_direction"]
    ==
    "EXACT"
).sum()


print(
    f"\nMean error: "
    f"{mean_error:.4f}"
)

print(
    f"Median error: "
    f"{median_error:.4f}"
)

print(
    f"Overpredictions: "
    f"{over_count:,} "
    f"({over_count / len(df) * 100:.2f}%)"
)

print(
    f"Underpredictions: "
    f"{under_count:,} "
    f"({under_count / len(df) * 100:.2f}%)"
)

print(
    f"Exact predictions: "
    f"{exact_count:,}"
)


if mean_error < -10:

    bias_conclusion = (
        "MODEL SHOWS UNDERPREDICTION BIAS"
    )

elif mean_error > 10:

    bias_conclusion = (
        "MODEL SHOWS OVERPREDICTION BIAS"
    )

else:

    bias_conclusion = (
        "NO STRONG SYSTEMATIC BIAS"
    )


print(
    f"\nConclusion: "
    f"{bias_conclusion}"
)


# ================================================================
# 14. STORE × GATE ANALYSIS
# ================================================================

print("\n")
print("=" * 75)
print("STORE × GATE ANALYSIS")
print("=" * 75)


store_gate_rows = []


for (
    store_id,
    gate_id
), group in df.groupby(
    [
        "store_id",
        "gate_id"
    ]
):

    metrics = calculate_metrics(
        group
    )

    metrics["store_id"] = store_id

    metrics["gate_id"] = gate_id

    store_gate_rows.append(
        metrics
    )


store_gate_performance = pd.DataFrame(
    store_gate_rows
)


store_gate_performance = (
    store_gate_performance
    .sort_values(
        "MAE",
        ascending=False
    )
)


store_gate_path = os.path.join(
    ANALYSIS_DIR,
    "store_gate_performance.csv"
)


store_gate_performance.to_csv(
    store_gate_path,
    index=False
)


print(
    f"\nSaved: {store_gate_path}"
)

print(
    "\nTOP 20 STORE × GATE COMBINATIONS:"
)

print(
    store_gate_performance
    .head(20)
    .to_string(index=False)
)


# ================================================================
# 15. WEEKEND VS WEEKDAY
# ================================================================

print("\n")
print("=" * 75)
print("WEEKEND VS WEEKDAY ANALYSIS")
print("=" * 75)


df["day_type"] = np.where(

    df["date"].dt.weekday >= 5,

    "Weekend",

    "Weekday"
)


day_type_rows = []


for day_type, group in df.groupby(
    "day_type"
):

    metrics = calculate_metrics(
        group
    )

    metrics["day_type"] = day_type

    day_type_rows.append(
        metrics
    )


day_type_performance = pd.DataFrame(
    day_type_rows
)


day_type_path = os.path.join(
    ANALYSIS_DIR,
    "weekend_vs_weekday.csv"
)


day_type_performance.to_csv(
    day_type_path,
    index=False
)


print(
    day_type_performance
    .to_string(index=False)
)


# ================================================================
# 16. HIGH FOOTFALL ERROR ANALYSIS
# ================================================================

print("\n")
print("=" * 75)
print("HIGH FOOTFALL ERROR ANALYSIS")
print("=" * 75)


high_threshold = df[
    "total_footfall"
].quantile(0.90)


high_df = df[
    df["total_footfall"]
    >= high_threshold
].copy()


high_metrics = calculate_metrics(
    high_df
)


print(
    f"\n90th percentile threshold: "
    f"{high_threshold:.2f}"
)

for key, value in high_metrics.items():

    if key == "samples":

        print(
            f"{key:<25}: "
            f"{value:,}"
        )

    else:

        print(
            f"{key:<25}: "
            f"{value:.4f}"
        )


high_df.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "high_footfall_predictions.csv"
    ),
    index=False
)


# ================================================================
# 17. WORST UNDERPREDICTIONS
# ================================================================

print("\n")
print("=" * 75)
print("TOP 50 UNDERPREDICTIONS")
print("=" * 75)


worst_under = (
    df[
        df["error"] < 0
    ]
    .sort_values(
        "error"
    )
    .head(50)
)


worst_under.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "worst_50_underpredictions.csv"
    ),
    index=False
)


print(
    worst_under[
        [
            "date",
            "store_id",
            "gate_id",
            "total_footfall",
            "predicted_footfall",
            "error",
            "absolute_error"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ================================================================
# 18. WORST OVERPREDICTIONS
# ================================================================

print("\n")
print("=" * 75)
print("TOP 50 OVERPREDICTIONS")
print("=" * 75)


worst_over = (
    df[
        df["error"] > 0
    ]
    .sort_values(
        "error",
        ascending=False
    )
    .head(50)
)


worst_over.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "worst_50_overpredictions.csv"
    ),
    index=False
)


print(
    worst_over[
        [
            "date",
            "store_id",
            "gate_id",
            "total_footfall",
            "predicted_footfall",
            "error",
            "absolute_error"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ================================================================
# 19. FINAL V3.2 RECOMMENDATION
# ================================================================

print("\n")
print("=" * 75)
print("GENERATING V3.2 RECOMMENDATION")
print("=" * 75)


recommendations = []


# ------------------------------------------------
# Overall MAPE
# ------------------------------------------------

if overall["MAPE"] > 20:

    recommendations.append(
        "1. Overall MAPE remains above 20%. "
        "V3.2 should focus on reducing percentage error, "
        "especially for normal/high footfall."
    )

else:

    recommendations.append(
        "1. Overall MAPE is below 20%. "
        "Continue improving percentage error without "
        "sacrificing MAE/RMSE."
    )


# ------------------------------------------------
# WAPE
# ------------------------------------------------

if overall["WAPE"] > 15:

    recommendations.append(
        "2. WAPE is high. Investigate large absolute "
        "errors and high-footfall underprediction."
    )

else:

    recommendations.append(
        "2. WAPE is reasonably controlled, but large "
        "absolute errors should still be investigated."
    )


# ------------------------------------------------
# Low vs Normal
# ------------------------------------------------

low_result = low_normal_performance[
    low_normal_performance[
        "footfall_category"
    ]
    ==
    "LOW"
]


normal_result = low_normal_performance[
    low_normal_performance[
        "footfall_category"
    ]
    ==
    "NORMAL/HIGH"
]


if (
    len(low_result) > 0
    and
    len(normal_result) > 0
):

    low_mae = low_result.iloc[0]["MAE"]

    normal_mae = normal_result.iloc[0]["MAE"]

    if normal_mae > low_mae * 2:

        recommendations.append(
            f"3. NORMAL/HIGH regression is the main "
            f"weakness. MAE = {normal_mae:.2f} versus "
            f"LOW MAE = {low_mae:.2f}. "
            f"Prioritize the normal/high model in V3.2."
        )

    else:

        recommendations.append(
            "3. Both LOW and NORMAL/HIGH models need "
            "balanced improvement in V3.2."
        )


# ------------------------------------------------
# Routing
# ------------------------------------------------

if "model_used" in df.columns:

    correct_low = (
        (
            df["footfall_category"] == "LOW"
        )
        &
        (
            df["model_used"] == "low_model"
        )
    ).sum()

    actual_low = (
        df["footfall_category"]
        ==
        "LOW"
    ).sum()

    if actual_low > 0:

        routing_recall = (
            correct_low
            /
            actual_low
        ) * 100

    else:

        routing_recall = np.nan


    if routing_recall < 90:

        recommendations.append(
            f"4. Stage-1 LOW recall is only "
            f"{routing_recall:.2f}%. "
            f"Consider improving classifier recall."
        )

    else:

        recommendations.append(
            f"4. Stage-1 routing is strong "
            f"({routing_recall:.2f}% LOW recall). "
            f"Do not make major classifier changes yet."
        )


# ------------------------------------------------
# Worst weekday
# ------------------------------------------------

worst_weekday = (
    weekday_performance
    .sort_values(
        "MAE",
        ascending=False
    )
    .iloc[0]
)


best_weekday = (
    weekday_performance
    .sort_values(
        "MAE",
        ascending=True
    )
    .iloc[0]
)


recommendations.append(
    f"5. Worst weekday is "
    f"{worst_weekday['weekday']} "
    f"(MAE {worst_weekday['MAE']:.2f}). "
    f"Best weekday is "
    f"{best_weekday['weekday']} "
    f"(MAE {best_weekday['MAE']:.2f}). "
    f"V3.2 should strengthen weekday/weekend interactions."
)


# ------------------------------------------------
# Worst month
# ------------------------------------------------

worst_month = (
    monthly_performance
    .sort_values(
        "MAE",
        ascending=False
    )
    .iloc[0]
)


recommendations.append(
    f"6. Worst month is "
    f"{worst_month['month']} "
    f"(MAE {worst_month['MAE']:.2f}). "
    f"Investigate seasonal, holiday and month-specific patterns."
)


# ------------------------------------------------
# Underprediction
# ------------------------------------------------

if (
    under_count > over_count
):

    recommendations.append(
        "7. Underpredictions are more frequent than "
        "overpredictions. V3.2 should improve high-footfall "
        "peak detection and recent-growth signals."
    )

else:

    recommendations.append(
        "7. Overpredictions are more frequent. "
        "V3.2 should investigate whether lag/rolling "
        "features are causing excessive prediction levels."
    )


# ------------------------------------------------
# High footfall
# ------------------------------------------------

if high_metrics["MAE"] > overall["MAE"] * 1.5:

    recommendations.append(
        f"8. High-footfall observations have substantially "
        f"larger errors (MAE {high_metrics['MAE']:.2f}). "
        f"V3.2 should add peak/event-aware features."
    )

else:

    recommendations.append(
        "8. High-footfall performance is not disproportionately "
        "worse than overall performance."
    )


# ------------------------------------------------
# Feature recommendations
# ------------------------------------------------

recommendations.append(
    "9. Recommended V3.2 features: "
    "weekday-specific lag/rolling statistics, "
    "7-vs-30 and 14-vs-28 growth ratios, "
    "previous/next holiday distance, "
    "store × gate interaction, "
    "recent peak-footfall statistics, "
    "and recent maximum/median footfall."
)


# ------------------------------------------------
# Model recommendation
# ------------------------------------------------

recommendations.append(
    "10. Keep the strict date-based train/test split "
    "from V3.1. Do not change the V3.1 test period when "
    "comparing V3.2."
)


recommendations.append(
    "11. Keep the same 32 core features initially and "
    "add V3.2 features incrementally so the impact of "
    "each feature group can be measured."
)


recommendations.append(
    "12. Do not blindly increase XGBoost depth or "
    "estimators. The major problem appears to be "
    "feature representation of peaks/seasonality rather "
    "than simply insufficient model complexity."
)


# ================================================================
# SAVE RECOMMENDATION
# ================================================================

recommendation_text = (
    "FOOTFALL PREDICTION V3.2 RECOMMENDATION\n"
    + "=" * 75
    + "\n\n"
)


for recommendation in recommendations:

    recommendation_text += (
        recommendation
        + "\n\n"
    )


recommendation_path = os.path.join(
    ANALYSIS_DIR,
    "v3_2_recommendation.txt"
)


with open(
    recommendation_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        recommendation_text
    )


print(
    f"\nSaved: {recommendation_path}"
)


# ================================================================
# ANALYSIS SUMMARY
# ================================================================

summary_text = f"""
FOOTFALL PREDICTION V3.1 ANALYSIS SUMMARY
{'=' * 75}

DATA
----
Rows              : {len(df):,}
Stores            : {df['store_id'].nunique()}
Gates             : {df['gate_id'].nunique()}
Date range        : {df['date'].min().date()} to {df['date'].max().date()}

OVERALL
-------
MAE               : {overall['MAE']:.4f}
RMSE              : {overall['RMSE']:.4f}
R2                : {overall['R2']:.4f}
R2 %              : {overall['R2'] * 100:.2f}%
MAPE              : {overall['MAPE']:.4f}%
WAPE              : {overall['WAPE']:.4f}%
Bias              : {overall['bias']:.4f}

PREDICTION DIRECTION
--------------------
Overprediction    : {over_count:,} ({over_count / len(df) * 100:.2f}%)
Underprediction   : {under_count:,} ({under_count / len(df) * 100:.2f}%)
Exact             : {exact_count:,}

WORST WEEKDAY
-------------
{worst_weekday['weekday']}
MAE               : {worst_weekday['MAE']:.4f}

BEST WEEKDAY
------------
{best_weekday['weekday']}
MAE               : {best_weekday['MAE']:.4f}

WORST MONTH
-----------
{worst_month['month']}
MAE               : {worst_month['MAE']:.4f}

TOP 10 STORES ERROR SHARE
-------------------------
{top10_percentage:.2f}%

HIGH FOOTFALL
-------------
90th percentile  : {high_threshold:.4f}
MAE              : {high_metrics['MAE']:.4f}
RMSE             : {high_metrics['RMSE']:.4f}

BIAS CONCLUSION
---------------
{bias_conclusion}

V3.2 RECOMMENDATIONS
--------------------
"""


for recommendation in recommendations:

    summary_text += (
        recommendation
        + "\n"
    )


summary_path = os.path.join(
    ANALYSIS_DIR,
    "analysis_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary_text
    )


# ================================================================
# FINAL OUTPUT LIST
# ================================================================

print("\n")
print("=" * 75)
print("FILES CREATED")
print("=" * 75)


files_created = [

    "overall_performance.csv",

    "store_performance.csv",

    "gate_performance.csv",

    "weekday_performance.csv",

    "monthly_performance.csv",

    "low_normal_performance.csv",

    "worst_100_predictions.csv",

    "overprediction_analysis.csv",

    "underprediction_analysis.csv",

    "error_distribution.csv",

    "feature_importance.csv",

    "classifier_feature_importance.csv",

    "routing_performance.csv",

    "routing_confusion_table.csv",

    "store_error_concentration.csv",

    "store_gate_performance.csv",

    "weekend_vs_weekday.csv",

    "high_footfall_predictions.csv",

    "worst_50_underpredictions.csv",

    "worst_50_overpredictions.csv",

    "analysis_summary.txt",

    "v3_2_recommendation.txt"
]


for i, filename in enumerate(
    files_created,
    1
):

    full_path = os.path.join(
        ANALYSIS_DIR,
        filename
    )

    if os.path.exists(full_path):

        print(
            f"✓ {filename}"
        )


# ================================================================
# FINAL RECOMMENDATION DISPLAY
# ================================================================

print("\n")
print("=" * 75)
print("FINAL V3.2 RECOMMENDATION")
print("=" * 75)

print()

print(
    recommendation_text
)


print("=" * 75)
print("V3.1 ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"\nAll analysis files are available in:\n"
    f"{os.path.abspath(ANALYSIS_DIR)}"
)