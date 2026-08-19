import pandas as pd
import numpy as np

from django.shortcuts import render
from .models import DailyHourlyFootfall
from django.db.models import Sum
import os
from django.conf import settings


def dashboard(request):

    store_gate_data = (
        DailyHourlyFootfall.objects
        .values(
            "store_id",
            "gate_id"
        )
        .annotate(
            footfall=Sum("total_footfall")
        )
        .order_by(
            "store_id",
            "gate_id"
        )
    )
    week_dates = []

    for i in range(7):
        current_date = week_start + pd.Timedelta(days=i)

        week_dates.append({
            "name": current_date.strftime("%A"),
            "date": current_date.strftime("%d-%b-%Y")
        })
        
    context = {
        "store_gate_data": list(store_gate_data),
        "week_dates": week_dates,
    }

    return render(
        request,
        "dashboard.html",
        context
    )


def weekly_report(request):

    # ---------------------------------
    # LOAD PREDICTED DATA
    # ---------------------------------

    prediction_file = os.path.join(
        settings.BASE_DIR.parent, "prediction_log.csv"
    )

    if os.path.exists(prediction_file):
        pred_df = pd.read_csv(prediction_file)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
    else:
        pred_df = pd.DataFrame(
            columns=["date", "store_id", "predicted"]
        )

    # ---------------------------------
    # LOAD ACTUAL DATA FROM SQL BACKUP
    # ---------------------------------

    actual_file = os.path.join(
        settings.BASE_DIR.parent, "actual_footfall.csv"
    )

    if os.path.exists(actual_file):
        actual_df = pd.read_csv(actual_file)
        actual_df["date"] = pd.to_datetime(
            actual_df["date"], errors="coerce"
        )
        actual_df = actual_df.dropna(subset=["date"])
        actual_df = (
            actual_df.groupby(["date", "store_id"])["total_footfall"]
            .sum()
            .reset_index()
            .rename(columns={"total_footfall": "actual"})
        )
    else:
        actual_df = pd.DataFrame(
            columns=["date", "store_id", "actual"]
        )

    # ---------------------------------
    # MERGE PREDICTED + ACTUAL
    # ---------------------------------

    if not pred_df.empty and "store_id" in pred_df.columns:
        if "actual" in pred_df.columns:
            pred_df = pred_df.drop(columns=["actual"])
        df = pd.merge(
            pred_df, actual_df,
            on=["date", "store_id"], how="outer"
        )
    elif not actual_df.empty:
        df = actual_df.copy()
        df["predicted"] = 0
    else:
        df = pd.DataFrame(
            columns=["date", "store_id", "predicted", "actual"]
        )

    df["predicted"] = df["predicted"].fillna(0)
    df["actual"] = df["actual"].fillna(0)

    # ---------------------------------
    # CALCULATE VARIATION
    # ---------------------------------

    df["variation"] = (
        (df["actual"] - df["predicted"]).abs()
        / df["actual"].replace(0, pd.NA)
        * 100
    )

    df["variation"] = df["variation"].fillna(0)

    df["predicted"] = df["predicted"].round(0)
    df["actual"] = df["actual"].round(0)
    df["variation"] = df["variation"].round(0)

    # ---------------------------------
    # STORE NAME
    # ---------------------------------

    if "store_name" not in df.columns:
        df["store_name"] = df["store_id"].astype(str)

    # ---------------------------------
    # VARIATION FILTER
    # ---------------------------------

    variation_filter = request.GET.get(
        "variation",
        "all"
    )

    # ---------------------------------
    # BUILD ROWS (one per store per date)
    # ---------------------------------

    df = df.sort_values(["store_id", "date"])

    rows = []

    for _, r in df.iterrows():
        predicted = int(r["predicted"])
        actual = int(r["actual"])
        variation = int(r["variation"])

        if variation_filter != "all":
            if variation_filter == "0-10":
                minimum, maximum = 0, 10
            elif variation_filter == "11-15":
                minimum, maximum = 11, 15
            elif variation_filter == "16-20":
                minimum, maximum = 16, 20
            elif variation_filter == "21-25":
                minimum, maximum = 21, 25
            elif variation_filter == "26-30":
                minimum, maximum = 26, 30
            elif variation_filter == "30-plus":
                minimum, maximum = 31, float("inf")
            else:
                minimum, maximum = 0, float("inf")

            if not (minimum <= variation <= maximum):
                continue

        rows.append({
            "store_id": r["store_id"],
            "store_name": r["store_name"],
            "date": r["date"].strftime("%d-%b-%Y"),
            "day": r["date"].strftime("%A"),
            "predicted": predicted,
            "actual": actual,
            "variation": variation,
        })

    # ---------------------------------
    # CONTEXT
    # ---------------------------------

    context = {
        "rows": rows,
        "total_rows": len(rows),
        "variation_filter": variation_filter,

        "variation_options": [
            ("all", "All"),
            ("0-10", "0-10%"),
            ("11-15", "11-15%"),
            ("16-20", "16-20%"),
            ("21-25", "21-25%"),
            ("26-30", "26-30%"),
            ("30-plus", "Above 30%"),
        ],
    }

    return render(
        request,
        "dashboard/weekly_report.html",
        context
    )