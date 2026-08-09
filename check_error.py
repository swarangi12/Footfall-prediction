import pandas as pd
import os
import numpy as np
import subprocess
import sys

# -----------------------------
# CHECK FILES
# -----------------------------

if not os.path.exists("prediction_log.csv"):
    print("prediction_log.csv not found")
    exit()

if not os.path.exists("actual_footfall.csv"):
    print("actual_footfall.csv not found")
    exit()


# -----------------------------
# READ FILES
# -----------------------------

pred = pd.read_csv("prediction_log.csv")
actual = pd.read_csv("actual_footfall.csv")


# -----------------------------
# REMOVE DUPLICATE PREDICTIONS
# Keep the latest prediction
# -----------------------------

pred = pred.drop_duplicates(
    subset=["date", "store_id", "gate_id"],
    keep="last"
)


# -----------------------------
# MERGE
# -----------------------------

df = pred.merge(
    actual,
    on=["date", "store_id", "gate_id"],
    how="inner"
)


if df.empty:
    print("No matching prediction and actual data found.")
    exit()


# -----------------------------
# ERROR
# -----------------------------

df["error"] = abs(
    df["actual"] - df["predicted"]
)


# -----------------------------
# ERROR %
# -----------------------------

df["error_percent"] = np.where(
    df["actual"] == 0,
    0,
    (df["error"] / df["actual"]) * 100
)


# -----------------------------
# SAVE ERROR LOG
# -----------------------------

df.to_csv(
    "error_log.csv",
    index=False
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\nPrediction Error:")

print(
    df[
        [
            "date",
            "store_id",
            "gate_id",
            "predicted",
            "actual",
            "error_percent"
        ]
    ]
)


# -----------------------------
# LAST 7 ERROR AVERAGE
# -----------------------------

average_error = df[
    "error_percent"
].tail(7).mean()

print(
    f"\nAverage Error = {average_error:.2f}%"
)


# -----------------------------
# RETRAINING
# -----------------------------

THRESHOLD = 25

if average_error > THRESHOLD:

    print(
        "⚠️ Retraining Recommended"
    )

    print(
        "🔄 Starting model retraining..."
    )

    try:

        subprocess.run(
            [sys.executable, "retrain_model.py"],
            check=True
        )

        print(
            "✅ Model retrained successfully."
        )

    except subprocess.CalledProcessError as e:

        print(
            f"❌ Retraining failed: {e}"
        )

else:

    print(
        "✅ Model Performance is Good"
    )