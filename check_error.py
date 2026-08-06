import pandas as pd
import subprocess

pred = pd.read_csv("prediction_log.csv")

actual = pd.read_csv("actual_footfall.csv")

df = pred.merge(
    actual,
    on=["date","store_id","gate_id"]
)
df["error_percent"] = (
    abs(df["actual"] - df["predicted"])
    / df["actual"]
) * 100

df["error"] = abs(
    df["actual"] -
    df["predicted"]
)

df.to_csv("error_log.csv", index=False)

average_error = df["error"].tail(7).mean()

print("Average Error =", average_error)
import matplotlib.pyplot as plt

colors = []

for error in df["error_percent"]:
    if error > 25:
        colors.append("yellow")
    else:
        colors.append("steelblue")

plt.figure(figsize=(10,5))

plt.bar(
    df["date"],
    df["error_percent"],
    color=colors
)

plt.axhline(
    y=25,
    color="red",
    linestyle="--",
    label="25% Threshold"
)

plt.xlabel("Date")
plt.ylabel("Prediction Error (%)")
plt.title("Prediction Error Analysis")
plt.xticks(rotation=45)
plt.legend()

plt.show()
import streamlit as st
if (df["error_percent"] > 25).any():

    st.warning(
        "⚠️ Prediction error exceeded 25% on one or more days. The yellow bars indicate high-error days."
    )
else:

    st.success(
        "✅ All prediction errors are below 25%."
    )

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