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
    # WEEK START / END
    # ---------------------------------

    selected_date = request.GET.get("date")

    if selected_date:
        selected_date = pd.to_datetime(selected_date)
    else:
        selected_date = df["date"].max()

    # Monday of selected week
    week_start = selected_date - pd.Timedelta(
        days=selected_date.weekday()
    )

    week_end = week_start + pd.Timedelta(days=6)

    weekly_df = df[
        (df["date"] >= week_start) &
        (df["date"] <= week_end)
    ].copy()

    # ---------------------------------
    # VARIATION FILTER
    # ---------------------------------

    variation_filter = request.GET.get(
        "variation",
        "all"
    )

    # ---------------------------------
    # CREATE ONE ROW PER STORE
    # ---------------------------------

    stores = []

    for (store_id, store_name), group in weekly_df.groupby(
        ["store_id", "store_name"]
    ):

        row = {
            "store_id": store_id,
            "store_name": store_name
        }

        for day_number in range(7):

            current_date = week_start + pd.Timedelta(
                days=day_number
            )

            day_name = current_date.strftime("%a").lower()

            day_data = group[
                group["date"].dt.date
                == current_date.date()
            ]

            if len(day_data) > 0:

                predicted = day_data["predicted"].sum()
                actual = day_data["actual"].sum()

                if actual != 0:
                    variation = (
                        abs(actual - predicted)
                        / actual
                        * 100
                    )
                else:
                    variation = 0

                row[f"{day_name}_predicted"] = round(
                    predicted
                )

                row[f"{day_name}_actual"] = round(
                    actual
                )

                row[f"{day_name}_variation"] = round(
                    variation
                )

            else:

                row[f"{day_name}_predicted"] = 0
                row[f"{day_name}_actual"] = 0
                row[f"{day_name}_variation"] = 0

        stores.append(row)

    # ---------------------------------
    # APPLY VARIATION FILTER
    # ---------------------------------

    if variation_filter != "all":

        filtered_stores = []

        if variation_filter == "0-10":
            minimum = 0
            maximum = 10

        elif variation_filter == "11-15":
            minimum = 11
            maximum = 15

        elif variation_filter == "16-20":
            minimum = 16
            maximum = 20

        elif variation_filter == "21-25":
            minimum = 21
            maximum = 25

        elif variation_filter == "26-30":
            minimum = 26
            maximum = 30

        elif variation_filter == "30-plus":
            minimum = 31
            maximum = float("inf")

        else:
            minimum = 0
            maximum = float("inf")

        for store in stores:

            variations = [
                store["mon_variation"],
                store["tue_variation"],
                store["wed_variation"],
                store["thu_variation"],
                store["fri_variation"],
                store["sat_variation"],
                store["sun_variation"],
            ]

            # Keep store if at least one day's
            # variation falls in selected range
            if any(
                minimum <= value <= maximum
                for value in variations
            ):
                filtered_stores.append(store)

        stores = filtered_stores

    # ---------------------------------
    # CONTEXT
    # ---------------------------------

    context = {
        "stores": stores,
        "week_start": week_start,
        "week_end": week_end,
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