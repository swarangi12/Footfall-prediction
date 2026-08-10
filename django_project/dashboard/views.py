from django.shortcuts import render
import pandas as pd
import os


BASE_DIR = r"C:\Users\swara\Downloads\footfall"


def weekly_report(request):

    prediction_file = os.path.join(BASE_DIR, "prediction_log.csv")
    actual_file = os.path.join(BASE_DIR, "actual_footfall.csv")

    # Read CSV files
    pred = pd.read_csv(prediction_file)
    actual = pd.read_csv(actual_file)

    # Make sure dates are same format
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    actual["date"] = pd.to_datetime(actual["date"]).dt.date

    # Merge prediction and actual
    df = pred.merge(
        actual,
        on=["date", "store_id", "gate_id"],
        how="inner"
    )

    # Calculate error
    if not df.empty:
        df["error_percent"] = (
            abs(df["predicted"] - df["actual"])
            / df["actual"].replace(0, 1)
        ) * 100

        df["error_percent"] = df["error_percent"].round(2)

        # Sort newest first
        df = df.sort_values("date", ascending=False)

    # Convert dataframe into dictionary
    rows = df.to_dict("records")

    print("Rows sent to template:")
    print(rows)

    return render(
        request,
        "dashboard/weekly_report.html",
        {
            "rows": rows
        }
    )