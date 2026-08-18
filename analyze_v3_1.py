# ================================================================
# FOOTFALL PREDICTION - V3.1 COMPLETE ERROR ANALYSIS
# ================================================================
#
# INPUT:
#   models_v3_1/test_predictions_v3_1.csv
#   models_v3_1/feature_importance_v3_1.csv
#
# OUTPUT:
#   analysis_v3_1/
#
# ANALYSIS:
#   1. Store-wise performance
#   2. Gate-wise performance
#   3. Weekday performance
#   4. Monthly performance
#   5. Low/Normal performance
#   6. Worst 100 predictions
#   7. Overprediction analysis
#   8. Underprediction analysis
#   9. Error distribution
#   10. Feature importance ranking
#   11. V3.2 recommendation
#
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG
# ================================================================

PREDICTION_FILE = os.path.join(
    "models_v3_1",
    "test_predictions_v3_1.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    "models_v3_1",
    "feature_importance_v3_1.csv"
)

OUTPUT_DIR = "analysis_v3_1"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_mape(actual, predicted):
    """
    MAPE calculated only where actual > 0.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual > 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


def safe_wape(actual, predicted):
    """
    WAPE = sum absolute error / sum actual * 100
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(actual - predicted))
        / denominator
    ) * 100


def calculate_metrics(group):
    """
    Calculate standard performance metrics
    for a dataframe group.
    """

    actual = group["actual"].values
    predicted = group["predicted"].values

    error = predicted - actual

    absolute_error = np.abs(error)

    mae = np.mean(
        absolute_error
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    mape = safe_mape(
        actual,
        predicted
    )

    wape = safe_wape(
        actual,
        predicted
    )

    bias = np.mean(
        error
    )

    median_absolute_error = np.median(
        absolute_error
    )

    return pd.Series({

        "samples": len(group),

        "actual_mean":
            np.mean(actual),

        "predicted_mean":
            np.mean(predicted),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "MAPE":
            mape,

        "WAPE":
            wape,

        "bias":
            bias,

        "median_absolute_error":
            median_absolute_error,

        "max_absolute_error":
            np.max(absolute_error)

    })


def print_section(title):

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


# ================================================================
# START
# ================================================================

print("=" * 75)
print("FOOTFALL PREDICTION V3.1 COMPLETE ANALYSIS")
print("=" * 75)


# ================================================================
# CHECK FILES
# ================================================================

print_section(
    "CHECKING INPUT FILES"
)

if not os.path.exists(
    PREDICTION_FILE
):

    raise FileNotFoundError(
        f"""
Prediction file not found:

{PREDICTION_FILE}

Make sure V3.1 training completed successfully.
"""
    )


if not os.path.exists(
    FEATURE_IMPORTANCE_FILE
):

    print(
        "\nWARNING:"
    )

    print(
        f"Feature importance file not found:"
    )

    print(
        FEATURE_IMPORTANCE_FILE
    )

    feature_importance_available = False

else:

    feature_importance_available = True


print(
    "\nPrediction file found:"
)

print(
    PREDICTION_FILE
)


# ================================================================
# LOAD PREDICTIONS
# ================================================================

print_section(
    "LOADING V3.1 PREDICTIONS"
)

df = pd.read_csv(
    PREDICTION_FILE
)

print(
    f"\nRows loaded: {len(df):,}"
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ================================================================
# NORMALIZE COLUMN NAMES
# ================================================================

column_mapping = {}


if "total_footfall" in df.columns:
    column_mapping["total_footfall"] = "actual"

elif "actual" in df.columns:
    column_mapping["actual"] = "actual"

else:
    raise ValueError(
        """
Actual footfall column not found.

Expected:
total_footfall
or
actual
"""
    )


if "predicted_footfall" in df.columns:
    column_mapping[
        "predicted_footfall"
    ] = "predicted"

elif "predicted" in df.columns:
    column_mapping[
        "predicted"
    ] = "predicted"

else:
    raise ValueError(
        """
Prediction column not found.

Expected:
predicted_footfall
or
predicted
"""
    )


df = df.rename(
    columns=column_mapping
)


# ================================================================
# DATE
# ================================================================

if "date" not in df.columns:

    raise ValueError(
        "date column not found."
    )


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

df["actual"] = pd.to_numeric(
    df["actual"],
    errors="coerce"
)

df["predicted"] = pd.to_numeric(
    df["predicted"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "actual",
        "predicted"
    ]
).copy()


# ================================================================
# BASIC ERROR FEATURES
# ================================================================

print_section(
    "CREATING ERROR FEATURES"
)

df["error"] = (
    df["predicted"]
    -
    df["actual"]
)

df["absolute_error"] = np.abs(
    df["error"]
)

df["squared_error"] = (
    df["error"] ** 2
)

df["error_percent"] = np.where(

    df["actual"] != 0,

    (
        df["absolute_error"]
        /
        df["actual"]
    ) * 100,

    np.nan
)


df["signed_error_percent"] = np.where(

    df["actual"] != 0,

    (
        df["error"]
        /
        df["actual"]
    ) * 100,

    np.nan
)


# ================================================================
# OVER / UNDER PREDICTION
# ================================================================

df["prediction_direction"] = np.where(

    df["error"] > 0,

    "OVERPREDICTED",

    np.where(

        df["error"] < 0,

        "UNDERPREDICTED",

        "EXACT"

    )
)


# ================================================================
# DATE FEATURES
# ================================================================

df["year"] = (
    df["date"].dt.year
)

df["month_number"] = (
    df["date"].dt.month
)

df["month"] = (
    df["date"]
    .dt.strftime("%B")
)

df["weekday_number"] = (
    df["date"].dt.weekday
)

df["weekday"] = (
    df["date"]
    .dt.strftime("%A")
)


# ================================================================
# LOW / NORMAL CATEGORY
# ================================================================

if "low_footfall" in df.columns:

    df["footfall_category"] = np.where(

        df["low_footfall"] == 1,

        "LOW",

        "NORMAL/HIGH"
    )

else:

    print(
        "\nWARNING: low_footfall column not found."
    )

    df["footfall_category"] = "UNKNOWN"


# ================================================================
# BASIC DATA SUMMARY
# ================================================================

print_section(
    "DATA SUMMARY"
)

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
# OVERALL METRICS
# ================================================================

overall = calculate_metrics(
    df
)

print_section(
    "OVERALL PERFORMANCE"
)

print(
    f"\nSamples : "
    f"{int(overall['samples']):,}"
)

print(
    f"Actual mean : "
    f"{overall['actual_mean']:.2f}"
)

print(
    f"Predicted mean : "
    f"{overall['predicted_mean']:.2f}"
)

print(
    f"MAE : "
    f"{overall['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{overall['RMSE']:.4f}"
)

print(
    f"MAPE : "
    f"{overall['MAPE']:.4f}%"
)

print(
    f"WAPE : "
    f"{overall['WAPE']:.4f}%"
)

print(
    f"Bias : "
    f"{overall['bias']:.4f}"
)


# ================================================================
# 1. STORE-WISE PERFORMANCE
# ================================================================

print_section(
    "1. STORE-WISE PERFORMANCE"
)

store_performance = (
    df
    .groupby("store_id")
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

store_performance = (
    store_performance
    .sort_values(
        "MAE",
        ascending=False
    )
)

store_path = os.path.join(
    OUTPUT_DIR,
    "store_performance.csv"
)

store_performance.to_csv(
    store_path,
    index=False
)

print(
    f"\nSaved: {store_path}"
)

print(
    "\n10 WORST STORES BY MAE:"
)

print(
    store_performance[
        [
            "store_id",
            "samples",
            "MAE",
            "RMSE",
            "MAPE",
            "WAPE",
            "bias"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\n10 BEST STORES BY MAE:"
)

print(
    store_performance[
        [
            "store_id",
            "samples",
            "MAE",
            "RMSE",
            "MAPE",
            "WAPE",
            "bias"
        ]
    ]
    .sort_values(
        "MAE"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ================================================================
# 2. GATE-WISE PERFORMANCE
# ================================================================

print_section(
    "2. GATE-WISE PERFORMANCE"
)

gate_performance = (
    df
    .groupby("gate_id")
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

gate_performance = (
    gate_performance
    .sort_values(
        "MAE",
        ascending=False
    )
)

gate_path = os.path.join(
    OUTPUT_DIR,
    "gate_performance.csv"
)

gate_performance.to_csv(
    gate_path,
    index=False
)

print(
    f"\nSaved: {gate_path}"
)

print(
    "\n10 WORST GATES BY MAE:"
)

print(
    gate_performance[
        [
            "gate_id",
            "samples",
            "MAE",
            "RMSE",
            "MAPE",
            "WAPE",
            "bias"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ================================================================
# 3. WEEKDAY PERFORMANCE
# ================================================================

print_section(
    "3. WEEKDAY PERFORMANCE"
)

weekday_performance = (
    df
    .groupby(
        [
            "weekday_number",
            "weekday"
        ]
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

weekday_performance = (
    weekday_performance
    .sort_values(
        "weekday_number"
    )
)

weekday_path = os.path.join(
    OUTPUT_DIR,
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
    "\nWEEKDAY RESULTS:"
)

print(
    weekday_performance[
        [
            "weekday",
            "samples",
            "actual_mean",
            "predicted_mean",
            "MAE",
            "RMSE",
            "MAPE",
            "WAPE",
            "bias"
        ]
    ]
    .to_string(
        index=False
    )
)


# ================================================================
# 4. MONTHLY PERFORMANCE
# ================================================================

print_section(
    "4. MONTHLY PERFORMANCE"
)

monthly_performance = (
    df
    .groupby(
        [
            "month_number",
            "month"
        ]
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

monthly_performance = (
    monthly_performance
    .sort_values(
        "month_number"
    )
)

monthly_path = os.path.join(
    OUTPUT_DIR,
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
    "\nMONTHLY RESULTS:"
)

print(
    monthly_performance[
        [
            "month",
            "samples",
            "actual_mean",
            "predicted_mean",
            "MAE",
            "RMSE",
            "MAPE",
            "WAPE",
            "bias"
        ]
    ]
    .to_string(
        index=False
    )
)


# ================================================================
# 5. LOW / NORMAL PERFORMANCE
# ================================================================

print_section(
    "5. LOW / NORMAL PERFORMANCE"
)

if (
    df["footfall_category"]
    != "UNKNOWN"
).any():

    low_normal_performance = (
        df
        .groupby(
            "footfall_category"
        )
        .apply(
            calculate_metrics,
            include_groups=False
        )
        .reset_index()
    )

else:

    low_normal_performance = pd.DataFrame()


low_normal_path = os.path.join(
    OUTPUT_DIR,
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
    "\nLOW / NORMAL RESULTS:"
)

print(
    low_normal_performance.to_string(
        index=False
    )
)


# ================================================================
# 6. WORST 100 PREDICTIONS
# ================================================================

print_section(
    "6. WORST 100 PREDICTIONS"
)

worst_100 = (
    df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(100)
    .copy()
)

worst_path = os.path.join(
    OUTPUT_DIR,
    "worst_100_predictions.csv"
)

worst_100.to_csv(
    worst_path,
    index=False
)

print(
    f"\nSaved: {worst_path}"
)

print(
    "\nTOP 20 WORST PREDICTIONS:"
)

worst_columns = [
    "date",
    "store_id",
    "gate_id",
    "actual",
    "predicted",
    "error",
    "absolute_error",
    "error_percent",
    "prediction_direction",
    "footfall_category"
]

available_worst_columns = [
    c for c in worst_columns
    if c in worst_100.columns
]

print(
    worst_100[
        available_worst_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ================================================================
# 7. OVERPREDICTION ANALYSIS
# ================================================================

print_section(
    "7. OVERPREDICTION ANALYSIS"
)

over_df = df[
    df["error"] > 0
].copy()

if len(over_df) > 0:

    over_summary = pd.DataFrame({

        "metric": [
            "count",
            "percentage_of_predictions",
            "mean_overprediction",
            "median_overprediction",
            "maximum_overprediction",
            "mean_actual",
            "mean_predicted"
        ],

        "value": [

            len(over_df),

            len(over_df)
            / len(df)
            * 100,

            over_df["error"].mean(),

            over_df["error"].median(),

            over_df["error"].max(),

            over_df["actual"].mean(),

            over_df["predicted"].mean()
        ]
    })

else:

    over_summary = pd.DataFrame()


over_path = os.path.join(
    OUTPUT_DIR,
    "overprediction_analysis.csv"
)

over_summary.to_csv(
    over_path,
    index=False
)

print(
    f"\nSaved: {over_path}"
)

print(
    over_summary.to_string(
        index=False
    )
)


# ================================================================
# 8. UNDERPREDICTION ANALYSIS
# ================================================================

print_section(
    "8. UNDERPREDICTION ANALYSIS"
)

under_df = df[
    df["error"] < 0
].copy()

if len(under_df) > 0:

    under_summary = pd.DataFrame({

        "metric": [
            "count",
            "percentage_of_predictions",
            "mean_underprediction",
            "median_underprediction",
            "maximum_underprediction",
            "mean_actual",
            "mean_predicted"
        ],

        "value": [

            len(under_df),

            len(under_df)
            / len(df)
            * 100,

            np.abs(
                under_df["error"]
            ).mean(),

            np.abs(
                under_df["error"]
            ).median(),

            np.abs(
                under_df["error"]
            ).max(),

            under_df["actual"].mean(),

            under_df["predicted"].mean()
        ]
    })

else:

    under_summary = pd.DataFrame()


under_path = os.path.join(
    OUTPUT_DIR,
    "underprediction_analysis.csv"
)

under_summary.to_csv(
    under_path,
    index=False
)

print(
    f"\nSaved: {under_path}"
)

print(
    under_summary.to_string(
        index=False
    )
)


# ================================================================
# 9. ERROR DISTRIBUTION
# ================================================================

print_section(
    "9. ERROR DISTRIBUTION"
)

error_bins = [
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

error_labels = [
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
    bins=error_bins,
    labels=error_labels,
    right=False
)

error_distribution = (
    df
    .groupby(
        "error_bucket",
        observed=False
    )
    .agg(

        samples=(
            "absolute_error",
            "size"
        ),

        mean_actual=(
            "actual",
            "mean"
        ),

        mean_predicted=(
            "predicted",
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
    OUTPUT_DIR,
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
    "\nERROR DISTRIBUTION:"
)

print(
    error_distribution.to_string(
        index=False
    )
)


# ================================================================
# ADD ERROR PERCENTILE INFORMATION
# ================================================================

print(
    "\nError percentiles:"
)

percentiles = [
    50,
    75,
    90,
    95,
    99
]

for p in percentiles:

    value = np.percentile(
        df["absolute_error"],
        p
    )

    print(
        f"P{p}: "
        f"{value:.2f}"
    )


# ================================================================
# 10. FEATURE IMPORTANCE
# ================================================================

print_section(
    "10. FEATURE IMPORTANCE"
)

if feature_importance_available:

    importance_df = pd.read_csv(
        FEATURE_IMPORTANCE_FILE
    )

    print(
        "\nColumns:"
    )

    print(
        importance_df.columns.tolist()
    )

    # ------------------------------------------------------------
    # NORMAL MODEL IMPORTANCE
    # ------------------------------------------------------------

    if (
        "normal_model_importance"
        in importance_df.columns
    ):

        normal_importance = (
            importance_df[
                [
                    "feature",
                    "normal_model_importance"
                ]
            ]
            .sort_values(
                "normal_model_importance",
                ascending=False
            )
            .copy()
        )

        print(
            "\nTOP 15 NORMAL/HIGH FEATURES:"
        )

        print(
            normal_importance
            .head(15)
            .to_string(
                index=False
            )
        )

    else:

        normal_importance = pd.DataFrame()


    # ------------------------------------------------------------
    # LOW MODEL IMPORTANCE
    # ------------------------------------------------------------

    if (
        "low_model_importance"
        in importance_df.columns
    ):

        low_importance = (
            importance_df[
                [
                    "feature",
                    "low_model_importance"
                ]
            ]
            .sort_values(
                "low_model_importance",
                ascending=False
            )
            .copy()
        )

        print(
            "\nTOP 15 LOW-FOOTFALL FEATURES:"
        )

        print(
            low_importance
            .head(15)
            .to_string(
                index=False
            )
        )

    else:

        low_importance = pd.DataFrame()


    # ------------------------------------------------------------
    # COMBINED IMPORTANCE
    # ------------------------------------------------------------

    importance_df["average_importance"] = (
        importance_df[
            [
                c for c in [
                    "low_model_importance",
                    "normal_model_importance"
                ]
                if c in importance_df.columns
            ]
        ]
        .mean(axis=1)
    )

    importance_df = (
        importance_df
        .sort_values(
            "average_importance",
            ascending=False
        )
    )

    importance_output = os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_output,
        index=False
    )

    print(
        f"\nSaved: {importance_output}"
    )

else:

    importance_df = pd.DataFrame()


# ================================================================
# ADDITIONAL STORE ERROR CONCENTRATION
# ================================================================

print_section(
    "ERROR CONCENTRATION ANALYSIS"
)

store_sorted = (
    store_performance
    .sort_values(
        "MAE",
        ascending=False
    )
    .copy()
)

total_absolute_error = (
    df["absolute_error"].sum()
)

if total_absolute_error > 0:

    store_error_detail = (
        df
        .groupby("store_id")
        .agg(
            samples=(
                "absolute_error",
                "size"
            ),
            total_absolute_error=(
                "absolute_error",
                "sum"
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean"
            )
        )
        .reset_index()
    )

    store_error_detail[
        "error_contribution_percent"
    ] = (

        store_error_detail[
            "total_absolute_error"
        ]
        /
        total_absolute_error
    ) * 100

    store_error_detail = (
        store_error_detail
        .sort_values(
            "error_contribution_percent",
            ascending=False
        )
    )

    top_10_error_contribution = (
        store_error_detail
        .head(10)[
            "error_contribution_percent"
        ]
        .sum()
    )

else:

    top_10_error_contribution = 0


print(
    f"\nTop 10 stores contribute "
    f"{top_10_error_contribution:.2f}% "
    f"of total absolute error."
)


# ================================================================
# MODEL ROUTING ANALYSIS
# ================================================================

print_section(
    "STAGE 1 ROUTING ANALYSIS"
)

if "model_used" in df.columns:

    routing = pd.crosstab(

        df["footfall_category"],

        df["model_used"],

        margins=True
    )

    print(
        routing.to_string()
    )

else:

    routing = None

    print(
        "\nmodel_used column not found."
    )


# ================================================================
# ROUTING ERROR ANALYSIS
# ================================================================

if "model_used" in df.columns:

    routing_performance = (
        df
        .groupby("model_used")
        .apply(
            calculate_metrics,
            include_groups=False
        )
        .reset_index()
    )

    routing_path = os.path.join(
        OUTPUT_DIR,
        "routing_performance.csv"
    )

    routing_performance.to_csv(
        routing_path,
        index=False
    )

    print(
        f"\nRouting performance saved:"
    )

    print(
        routing_path
    )

    print(
        "\nRouting performance:"
    )

    print(
        routing_performance.to_string(
            index=False
        )
    )


# ================================================================
# DETECT SYSTEMATIC BIAS
# ================================================================

print_section(
    "SYSTEMATIC BIAS ANALYSIS"
)

mean_error = df["error"].mean()

median_error = df["error"].median()

over_count = (
    df["error"] > 0
).sum()

under_count = (
    df["error"] < 0
).sum()

exact_count = (
    df["error"] == 0
).sum()

total_count = len(df)


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
    f"({over_count / total_count * 100:.2f}%)"
)

print(
    f"Underpredictions: "
    f"{under_count:,} "
    f"({under_count / total_count * 100:.2f}%)"
)

print(
    f"Exact predictions: "
    f"{exact_count:,}"
)


if mean_error > 10:

    bias_direction = (
        "SYSTEMATIC OVERPREDICTION"
    )

elif mean_error < -10:

    bias_direction = (
        "SYSTEMATIC UNDERPREDICTION"
    )

else:

    bias_direction = (
        "NO STRONG SYSTEMATIC BIAS"
    )


print(
    f"\nConclusion: {bias_direction}"
)


# ================================================================
# GENERATE AUTOMATIC RECOMMENDATION
# ================================================================

print_section(
    "GENERATING V3.2 RECOMMENDATION"
)

recommendations = []

# ------------------------------------------------
# Overall MAPE
# ------------------------------------------------

overall_mape = overall["MAPE"]

if overall_mape > 20:

    recommendations.append(
        "Overall MAPE remains above 20%. "
        "V3.2 should focus on reducing percentage "
        "error, especially for normal/high footfall."
    )


# ------------------------------------------------
# WAPE
# ------------------------------------------------

overall_wape = overall["WAPE"]

if overall_wape > 12:

    recommendations.append(
        "WAPE is relatively high. "
        "Investigate systematic underprediction "
        "and large absolute errors."
    )


# ------------------------------------------------
# Normal/high performance
# ------------------------------------------------

if not low_normal_performance.empty:

    normal_rows = low_normal_performance[
        low_normal_performance[
            "footfall_category"
        ] == "NORMAL/HIGH"
    ]

    low_rows = low_normal_performance[
        low_normal_performance[
            "footfall_category"
        ] == "LOW"
    ]

    if len(normal_rows) > 0:

        normal_mae_value = (
            normal_rows.iloc[0]["MAE"]
        )

        recommendations.append(
            f"Normal/high footfall has MAE "
            f"{normal_mae_value:.2f}. "
            "Prioritize improving the normal/high "
            "regressor in V3.2."
        )

    if len(low_rows) > 0:

        low_mae_value = (
            low_rows.iloc[0]["MAE"]
        )

        if low_mae_value < overall["MAE"]:

            recommendations.append(
                f"Low-footfall MAE is only "
                f"{low_mae_value:.2f}. "
                "Do not aggressively modify the "
                "low-footfall model unless detailed "
                "analysis shows a specific weakness."
            )


# ------------------------------------------------
# Stage 1
# ------------------------------------------------

if "low_footfall" in df.columns:

    stage1_accuracy_estimate = None

    if "model_used" in df.columns:

        routing_correct = (

            (
                (df["footfall_category"] == "LOW")
                &
                (df["model_used"] == "low_model")
            )
            |
            (
                (df["footfall_category"] == "NORMAL/HIGH")
                &
                (df["model_used"] == "normal_model")
            )
        ).sum()

        stage1_accuracy_estimate = (
            routing_correct
            /
            len(df)
        ) * 100

        if stage1_accuracy_estimate < 95:

            recommendations.append(
                "Stage 1 routing accuracy is below 95%. "
                "Consider probability threshold tuning "
                "for the low-footfall classifier."
            )

        else:

            recommendations.append(
                "Stage 1 routing is strong. "
                "Do not make major classifier changes "
                "before improving the regression models."
            )


# ------------------------------------------------
# Store concentration
# ------------------------------------------------

if top_10_error_contribution > 40:

    recommendations.append(
        f"The top 10 stores contribute "
        f"{top_10_error_contribution:.2f}% of total "
        "absolute error. Investigate store-specific "
        "patterns and consider store-level features."
    )

elif top_10_error_contribution > 25:

    recommendations.append(
        f"The top 10 stores contribute "
        f"{top_10_error_contribution:.2f}% of total "
        "absolute error. Store-specific behavior "
        "should be investigated."
    )


# ------------------------------------------------
# Weekday
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
        "MAE"
    )
    .iloc[0]
)

recommendations.append(
    f"Worst weekday by MAE is "
    f"{worst_weekday['weekday']} "
    f"({worst_weekday['MAE']:.2f}). "
    f"Best weekday is "
    f"{best_weekday['weekday']} "
    f"({best_weekday['MAE']:.2f}). "
    "Check weekend/weekday and day-specific "
    "seasonality before V3.2."
)


# ------------------------------------------------
# Month
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
    f"Worst month by MAE is "
    f"{worst_month['month']} "
    f"({worst_month['MAE']:.2f}). "
    "Investigate seasonal and holiday effects."
)


# ------------------------------------------------
# Bias
# ------------------------------------------------

if abs(mean_error) > 20:

    if mean_error > 0:

        recommendations.append(
            "The model shows noticeable systematic "
            "OVERPREDICTION. Consider calibration "
            "or improving spike/peak detection."
        )

    else:

        recommendations.append(
            "The model shows noticeable systematic "
            "UNDERPREDICTION. Consider adding stronger "
            "recent-trend and peak-demand features."
        )

else:

    recommendations.append(
        "Overall prediction bias is relatively balanced."
    )


# ------------------------------------------------
# Feature importance
# ------------------------------------------------

if not importance_df.empty:

    top_features = (
        importance_df
        .head(10)["feature"]
        .tolist()
    )

    recommendations.append(
        "Top features currently driving the model: "
        +
        ", ".join(
            top_features
        )
        +
        ". V3.2 should add features complementary "
        "to these rather than blindly adding many "
        "new variables."
    )


# ------------------------------------------------
# Final recommended feature ideas
# ------------------------------------------------

recommendations.append(
    "Recommended V3.2 feature candidates: "
    "lag/rolling features by weekday, recent "
    "7-vs-30 and 14-vs-28 trends, holiday proximity "
    "(previous/next holiday), weekend indicators, "
    "store-gate interaction features, and recent "
    "peak-footfall statistics."
)

recommendations.append(
    "Keep the strict date-based split used by V3.1 "
    "so V3.2 can be compared fairly against V3.1."
)

recommendations.append(
    "Do not change the test period when comparing "
    "V3.2 with V3.1."
)


# ================================================================
# SAVE RECOMMENDATIONS
# ================================================================

recommendation_text = []

recommendation_text.append(
    "FOOTFALL V3.1 → V3.2 RECOMMENDATION"
)

recommendation_text.append(
    "=" * 70
)

recommendation_text.append("")

recommendation_text.append(
    "OVERALL PERFORMANCE"
)

recommendation_text.append(
    f"MAE  : {overall['MAE']:.4f}"
)

recommendation_text.append(
    f"RMSE : {overall['RMSE']:.4f}"
)

recommendation_text.append(
    f"MAPE : {overall['MAPE']:.4f}%"
)

recommendation_text.append(
    f"WAPE : {overall['WAPE']:.4f}%"
)

recommendation_text.append(
    f"Bias : {overall['bias']:.4f}"
)

recommendation_text.append("")

recommendation_text.append(
    "RECOMMENDATIONS"
)

recommendation_text.append(
    "-" * 70
)

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    recommendation_text.append(
        f"{i}. {recommendation}"
    )

recommendation_text.append("")

recommendation_text.append(
    "PRIORITY FOR V3.2"
)

recommendation_text.append(
    "-" * 70
)

recommendation_text.append(
    "1. Keep leakage-safe feature generation."
)

recommendation_text.append(
    "2. Keep strict chronological train/test split."
)

recommendation_text.append(
    "3. Keep Stage 1 classifier as the baseline."
)

recommendation_text.append(
    "4. Focus primarily on the normal/high regressor."
)

recommendation_text.append(
    "5. Investigate store/gate-specific errors."
)

recommendation_text.append(
    "6. Add stronger seasonal and holiday features."
)

recommendation_text.append(
    "7. Add recent trend/peak-demand features."
)

recommendation_text.append(
    "8. Compare V3.2 using exactly the same test period."
)


recommendation_path = os.path.join(
    OUTPUT_DIR,
    "v3_2_recommendation.txt"
)

with open(
    recommendation_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            recommendation_text
        )
    )


print(
    f"\nSaved: {recommendation_path}"
)


# ================================================================
# COMPLETE SUMMARY FILE
# ================================================================

summary_lines = []

summary_lines.append(
    "FOOTFALL V3.1 COMPLETE ANALYSIS"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append("")

summary_lines.append(
    f"Rows analyzed: {len(df):,}"
)

summary_lines.append(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

summary_lines.append(
    f"Stores: {df['store_id'].nunique()}"
)

summary_lines.append(
    f"Gates: {df['gate_id'].nunique()}"
)

summary_lines.append("")

summary_lines.append(
    "OVERALL METRICS"
)

summary_lines.append(
    f"MAE  : {overall['MAE']:.4f}"
)

summary_lines.append(
    f"RMSE : {overall['RMSE']:.4f}"
)

summary_lines.append(
    f"MAPE : {overall['MAPE']:.4f}%"
)

summary_lines.append(
    f"WAPE : {overall['WAPE']:.4f}%"
)

summary_lines.append(
    f"Bias : {overall['bias']:.4f}"
)

summary_lines.append("")

summary_lines.append(
    "ERROR DISTRIBUTION"
)

for _, row in error_distribution.iterrows():

    summary_lines.append(

        f"{row['error_bucket']}: "
        f"{int(row['samples']):,} samples "
        f"({row['percentage']:.2f}%)"
    )

summary_lines.append("")

summary_lines.append(
    "WORST STORES"
)

for _, row in (
    store_performance
    .head(10)
    .iterrows()
):

    summary_lines.append(

        f"Store {row['store_id']}: "
        f"MAE={row['MAE']:.2f}, "
        f"MAPE={row['MAPE']:.2f}%"
    )

summary_lines.append("")

summary_lines.append(
    "WORST GATES"
)

for _, row in (
    gate_performance
    .head(10)
    .iterrows()
):

    summary_lines.append(

        f"Gate {row['gate_id']}: "
        f"MAE={row['MAE']:.2f}, "
        f"MAPE={row['MAPE']:.2f}%"
    )

summary_lines.append("")

summary_lines.append(
    "V3.2 RECOMMENDATIONS"
)

summary_lines.append(
    "-" * 70
)

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    summary_lines.append(
        f"{i}. {recommendation}"
    )


summary_path = os.path.join(
    OUTPUT_DIR,
    "analysis_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# ================================================================
# FINAL TERMINAL SUMMARY
# ================================================================

print_section(
    "FINAL V3.2 RECOMMENDATION"
)

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"\n{i}. {recommendation}"
    )


print_section(
    "FILES CREATED"
)

files_created = [
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
    "routing_performance.csv",
    "analysis_summary.txt",
    "v3_2_recommendation.txt"
]

for filename in files_created:

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(filepath):

        print(
            f"✓ {filepath}"
        )


print("\n" + "=" * 75)
print("V3.1 ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"\nAll analysis files are available in:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print("\n" + "=" * 75)