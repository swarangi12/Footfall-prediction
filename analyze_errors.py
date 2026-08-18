import pandas as pd
import numpy as np

FILE = "prediction_log_new_model.csv"

df = pd.read_csv(FILE)

# Keep only rows where actual > 0
df = df[df["actual"] > 0].copy()

# Calculate errors
df["absolute_error"] = (
    df["predicted"] - df["actual"]
).abs()

df["error_percent"] = (
    df["absolute_error"]
    / df["actual"]
    * 100
)

# ---------------------------------------------------------
# Store + Gate performance
# ---------------------------------------------------------

summary = (
    df.groupby(
        ["store_id", "gate_id"]
    )
    .agg(
        predictions=("actual", "count"),
        actual_total=("actual", "sum"),
        predicted_total=("predicted", "sum"),
        MAE=("absolute_error", "mean"),
        MAPE=("error_percent", "mean")
    )
    .reset_index()
)

summary = summary.sort_values(
    "MAPE",
    ascending=False
)

print("\n" + "=" * 70)
print("WORST STORE + GATE COMBINATIONS")
print("=" * 70)

print(
    summary.head(30).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Best combinations
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BEST STORE + GATE COMBINATIONS")
print("=" * 70)

print(
    summary.sort_values(
        "MAPE"
    ).head(20).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Store-level performance
# ---------------------------------------------------------

store_summary = (
    df.groupby("store_id")
    .agg(
        predictions=("actual", "count"),
        MAE=("absolute_error", "mean"),
        MAPE=("error_percent", "mean")
    )
    .reset_index()
    .sort_values(
        "MAPE",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("WORST STORES")
print("=" * 70)

print(
    store_summary.head(20).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# High-error rows
# ---------------------------------------------------------

high_error = df[
    df["error_percent"] > 50
].sort_values(
    "error_percent",
    ascending=False
)

print("\n" + "=" * 70)
print("PREDICTIONS WITH >50% ERROR")
print("=" * 70)

print(
    high_error[
        [
            "date",
            "store_id",
            "gate_id",
            "predicted",
            "actual",
            "error_percent"
        ]
    ]
    .head(50)
    .to_string(index=False)
)

# ---------------------------------------------------------
# Save analysis
# ---------------------------------------------------------

summary.to_csv(
    "store_gate_mape_analysis.csv",
    index=False
)

store_summary.to_csv(
    "store_mape_analysis.csv",
    index=False
)

high_error.to_csv(
    "high_error_predictions.csv",
    index=False
)

print("\nAnalysis files saved:")
print("  store_gate_mape_analysis.csv")
print("  store_mape_analysis.csv")
print("  high_error_predictions.csv")