import pandas as pd

df = pd.read_csv("prediction_log_new_model.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store_id", "gate_id", "date"]
)

# Low-footfall flags
df["low_5"] = df["actual"] <= 5
df["low_10"] = df["actual"] <= 10

# Previous day's actual
df["previous_actual"] = (
    df.groupby(
        ["store_id", "gate_id"]
    )["actual"]
    .shift(1)
)

# Low -> low transitions
low_consecutive = df[
    (df["actual"] <= 5) &
    (df["previous_actual"] <= 5)
]

print("=" * 70)
print("LOW FOOTFALL ANALYSIS")
print("=" * 70)

print(
    f"Actual <= 5: "
    f"{(df['actual'] <= 5).sum():,} rows"
)

print(
    f"Actual <= 10: "
    f"{(df['actual'] <= 10).sum():,} rows"
)

print(
    f"Consecutive <= 5: "
    f"{len(low_consecutive):,} rows"
)

# Percentage of low values that are consecutive
low_count = (df["actual"] <= 5).sum()

if low_count > 0:
    print(
        f"Consecutive-low percentage: "
        f"{len(low_consecutive) / low_count * 100:.2f}%"
    )

# ---------------------------------------------------------
# Stores/gates with most low-footfall observations
# ---------------------------------------------------------

summary = (
    df.groupby(
        ["store_id", "gate_id"]
    )
    .agg(
        total_rows=("actual", "count"),
        low_5=("actual", lambda x: (x <= 5).sum()),
        low_10=("actual", lambda x: (x <= 10).sum()),
        median_actual=("actual", "median"),
        mean_actual=("actual", "mean")
    )
    .reset_index()
)

summary["low_5_percent"] = (
    summary["low_5"]
    / summary["total_rows"]
    * 100
)

print("\n" + "=" * 70)
print("STORE/GATE WITH MOST LOW FOOTFALL")
print("=" * 70)

print(
    summary
    .sort_values(
        "low_5_percent",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)

summary.to_csv(
    "low_footfall_store_gate_analysis.csv",
    index=False
)