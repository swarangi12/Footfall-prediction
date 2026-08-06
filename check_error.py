import pandas as pd
import os
import numpy as np

# Check if files exist
if not os.path.exists("prediction_log.csv"):
    print("prediction_log.csv not found")
    exit()

if not os.path.exists("actual_footfall.csv"):
    print("actual_footfall.csv not found")
    exit()

# Read files
pred = pd.read_csv("prediction_log.csv")
actual = pd.read_csv("actual_footfall.csv")

# Merge prediction and actual
df = pred.merge(
    actual,
    on=["date", "store_id", "gate_id"]
)

if df.empty:
    print("No matching prediction and actual data found.")
    exit()

# Absolute Error
df["error"] = abs(
    df["actual"] - df["predicted"]
)

# Percentage Error
df["error_percent"] = np.where(
    df["actual"] == 0,
    0,
    (df["error"] / df["actual"]) * 100
)

# Save error log
df.to_csv("error_log.csv", index=False)

# Average error of last 7 entries
average_error = df["error_percent"].tail(7).mean()

print(f"Average Error = {average_error:.2f}%")

import subprocess
import sys

THRESHOLD = 25

if average_error > THRESHOLD:
    print("⚠️ Retraining Recommended")
    print("🔄 Starting model retraining...")

    try:
        subprocess.run(
            [sys.executable, "retrain_model.py"],
            check=True
        )
        print("✅ Model retrained successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Retraining failed: {e}")

else:
    print("✅ Model Performance is Good")