import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from django.shortcuts import render
from django.conf import settings
from django.db.models import Sum
from django.db import connection

from .models import DailyHourlyFootfall


# ============================================================
# DASHBOARD
# ============================================================

def dashboard(request):

    # --------------------------------------------------------
    # GET DATE FROM CALENDAR
    # --------------------------------------------------------

    selected_date = request.GET.get("date")

    if selected_date:

        try:
            week_start = datetime.strptime(
                selected_date,
                "%Y-%m-%d"
            ).date()

        except ValueError:
            week_start = None

    else:
        week_start = None


    # --------------------------------------------------------
    # DEFAULT DATE
    # USE LATEST DATE FROM DATABASE
    # --------------------------------------------------------

    if week_start is None:

        latest_record = (
            DailyHourlyFootfall.objects
            .order_by("-date")
            .first()
        )

        if latest_record:
            week_start = latest_record.date

        else:
            week_start = datetime.today().date()


    # --------------------------------------------------------
    # NEXT 6 DAYS
    # TOTAL 7 DAYS
    # --------------------------------------------------------

    week_end = week_start + timedelta(days=6)


    # --------------------------------------------------------
    # LOAD ONLY 7 DAYS
    # --------------------------------------------------------

    store_gate_data = (

        DailyHourlyFootfall.objects

        .filter(
            date__gte=week_start,
            date__lte=week_end
        )

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


    # --------------------------------------------------------
    # CREATE 7 DATES
    # --------------------------------------------------------

    week_dates = []

    for i in range(7):

        current_date = (
            week_start +
            timedelta(days=i)
        )

        week_dates.append({

            "name": current_date.strftime(
                "%A"
            ),

            "date": current_date.strftime(
                "%d-%b-%Y"
            ),

            "raw_date": current_date.strftime(
                "%Y-%m-%d"
            )

        })


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = {

        "store_gate_data": list(
            store_gate_data
        ),

        "week_dates": week_dates,

        "selected_date": week_start.strftime(
            "%Y-%m-%d"
        ),

        "week_start": week_start.strftime(
            "%d-%b-%Y"
        ),

        "week_end": week_end.strftime(
            "%d-%b-%Y"
        ),

    }


    return render(
        request,
        "dashboard.html",
        context
    )


# ============================================================
# WEEKLY REPORT
# ============================================================

# ============================================================
# WEEKLY REPORT
# ============================================================

def weekly_report(request):

    # ========================================================
    # PREDICTION FILE
    # ========================================================

    prediction_file = os.path.join(
        settings.BASE_DIR.parent,
        "prediction_log.csv"
    )

    # ========================================================
    # GET SELECTED DATE
    # ========================================================

    selected_date = request.GET.get("date")

    if selected_date:
        try:
            selected = pd.to_datetime(
                selected_date
            ).normalize()

            # Always move selected date to Monday
            week_start = (
                selected
                - pd.Timedelta(days=selected.weekday())
            )

        except Exception:
            week_start = None

    else:
        week_start = None

    # ========================================================
    # DEFAULT DATE
    # ========================================================

    if week_start is None:

        if os.path.exists(prediction_file):

            temp_df = pd.read_csv(
                prediction_file,
                usecols=["date"]
            )

            temp_df["date"] = pd.to_datetime(
                temp_df["date"],
                errors="coerce"
            )

            max_date = temp_df["date"].max()

            if pd.isna(max_date):
                max_date = pd.Timestamp.today().normalize()

            week_start = (
                max_date.normalize()
                - pd.Timedelta(days=max_date.weekday())
            )

        else:

            latest_record = (
                DailyHourlyFootfall.objects
                .order_by("-date")
                .first()
            )

            if latest_record:

                latest_date = pd.Timestamp(
                    latest_record.date
                ).normalize()

                week_start = (
                    latest_date
                    - pd.Timedelta(days=latest_date.weekday())
                )

            else:

                week_start = (
                    pd.Timestamp.today().normalize()
                )

    # ========================================================
    # 7 DAY RANGE
    # ========================================================

    week_start = pd.Timestamp(
        week_start
    ).normalize()

    week_end = (
        week_start
        + pd.Timedelta(days=6)
    )

    # ========================================================
    # LOAD PREDICTED DATA
    # ========================================================

    if os.path.exists(prediction_file):

        pred_df = pd.read_csv(
            prediction_file
        )

        print("\n==============================")
        print("PREDICTION FILE")
        print("==============================")
        print(
            "Prediction columns:",
            pred_df.columns.tolist()
        )

        # DATE
        pred_df["date"] = pd.to_datetime(
            pred_df["date"],
            errors="coerce"
        ).dt.normalize()

        # STORE ID
        pred_df["store_id"] = pd.to_numeric(
            pred_df["store_id"],
            errors="coerce"
        )

        # PREDICTED
        pred_df["predicted"] = pd.to_numeric(
            pred_df["predicted"],
            errors="coerce"
        )

        # REMOVE INVALID
        pred_df = pred_df.dropna(
            subset=[
                "date",
                "store_id"
            ]
        )

        # FILTER 7 DAYS
        pred_df = pred_df[
            (pred_df["date"] >= week_start)
            &
            (pred_df["date"] <= week_end)
        ].copy()

        print(
            "Prediction rows after filter:",
            len(pred_df)
        )

    else:

        print(
            "prediction_log.csv NOT FOUND:",
            prediction_file
        )

        pred_df = pd.DataFrame(
            columns=[
                "date",
                "store_id",
                "predicted"
            ]
        )

    # ========================================================
    # GROUP PREDICTIONS
    # ========================================================

    if not pred_df.empty:

        pred_grouped = (
            pred_df
            .groupby(
                [
                    "date",
                    "store_id"
                ],
                as_index=False
            )["predicted"]
            .sum()
        )

    else:

        pred_grouped = pd.DataFrame(
            columns=[
                "date",
                "store_id",
                "predicted"
            ]
        )

    # ========================================================
    # LOAD ACTUAL DATA
    # ========================================================

    start_date_str = week_start.strftime(
        "%Y-%m-%d"
    )

    end_date_str = week_end.strftime(
        "%Y-%m-%d"
    )

    print("\n==============================")
    print("ACTUAL DATABASE QUERY")
    print("==============================")

    print(
        "Start date:",
        start_date_str
    )

    print(
        "End date:",
        end_date_str
    )

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                DATE(date) AS date,
                store_id,
                SUM(total_footfall) AS actual
            FROM app_hourlyfootfall
            WHERE DATE(date) >= %s
              AND DATE(date) <= %s
            GROUP BY
                DATE(date),
                store_id
            ORDER BY
                DATE(date),
                store_id
            """,
            [
                start_date_str,
                end_date_str
            ]
        )

        actual_rows = cursor.fetchall()

    print(
        "Actual database rows:",
        len(actual_rows)
    )

    # ========================================================
    # ACTUAL DATAFRAME
    # ========================================================

    actual_df = pd.DataFrame(
        actual_rows,
        columns=[
            "date",
            "store_id",
            "actual"
        ]
    )

    if not actual_df.empty:

        actual_df["date"] = pd.to_datetime(
            actual_df["date"],
            errors="coerce"
        ).dt.normalize()

        actual_df["store_id"] = pd.to_numeric(
            actual_df["store_id"],
            errors="coerce"
        )

        actual_df["actual"] = pd.to_numeric(
            actual_df["actual"],
            errors="coerce"
        ).fillna(0)

        actual_df = actual_df.dropna(
            subset=[
                "date",
                "store_id"
            ]
        )

    else:

        # IMPORTANT:
        # Keep actual_df defined even when database has no data.
        actual_df = pd.DataFrame(
            columns=[
                "date",
                "store_id",
                "actual"
            ]
        )

    # ========================================================
    # MERGE PREDICTED + ACTUAL
    # ========================================================

    if (
        not pred_grouped.empty
        and not actual_df.empty
    ):

        df = pd.merge(
            pred_grouped,
            actual_df,
            on=[
                "date",
                "store_id"
            ],
            how="outer"
        )

    elif not pred_grouped.empty:

        df = pred_grouped.copy()
        df["actual"] = 0

    elif not actual_df.empty:

        df = actual_df.copy()
        df["predicted"] = 0

    else:

        df = pd.DataFrame(
            columns=[
                "date",
                "store_id",
                "predicted",
                "actual"
            ]
        )

    # ========================================================
    # FILL MISSING VALUES
    # ========================================================

    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="coerce"
    ).fillna(0)

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce"
    ).fillna(0)

    # ========================================================
    # STORE NAME
    # ========================================================

    # If you have a real store-name field in your database,
    # replace this with that field.
    df["store_name"] = (
        df["store_id"]
        .astype(str)
    )

    # ========================================================
    # CALCULATE VARIATION
    # ========================================================

    if not df.empty:

        df["variation"] = np.where(
            df["actual"] == 0,
            0,
            (
                abs(
                    df["predicted"]
                    -
                    df["actual"]
                )
                /
                df["actual"]
                *
                100
            )
        )

        df["variation"] = (
            df["variation"]
            .round()
            .astype(int)
        )

    else:

        df["variation"] = pd.Series(
            dtype=int
        )

    # ========================================================
    # SORT
    # ========================================================

    if not df.empty:

        df = df.sort_values(
            by=[
                "store_id",
                "date"
            ]
        )

    # ========================================================
    # VARIATION FILTER
    # ========================================================

    variation_filter = request.GET.get(
        "variation",
        "all"
    )

    # ========================================================
    # CREATE 7 DATE HEADERS
    # ========================================================

    week_dates = []

    for i in range(7):

        current_date = (
            week_start
            +
            pd.Timedelta(days=i)
        )

        week_dates.append({
            "name": current_date.strftime(
                "%A"
            ),

            "date": current_date.strftime(
                "%d-%b-%Y"
            ),

            "raw_date": current_date.strftime(
                "%Y-%m-%d"
            )
        })

    # ========================================================
    # HELPER FOR VARIATION FILTER
    # ========================================================

    def variation_matches(value):

        value = int(value)

        if variation_filter == "all":
            return True

        if variation_filter == "0-10":
            return 0 <= value <= 10

        if variation_filter == "11-15":
            return 11 <= value <= 15

        if variation_filter == "16-20":
            return 16 <= value <= 20

        if variation_filter == "21-25":
            return 21 <= value <= 25

        if variation_filter == "26-30":
            return 26 <= value <= 30

        if variation_filter == "30-plus":
            return value >= 31

        return True

    # ========================================================
    # CREATE LOOKUP
    # ========================================================

    data_lookup = {}

    for _, r in df.iterrows():

        date_key = (
            pd.Timestamp(r["date"])
            .strftime("%Y-%m-%d")
        )

        store_key = str(
            int(r["store_id"])
        )

        data_lookup[
            (store_key, date_key)
        ] = {
            "predicted": round(
                float(r["predicted"]),
                2
            ),

            "actual": round(
                float(r["actual"]),
                2
            ),

            "variation": int(
                r["variation"]
            )
        }

    # ========================================================
    # GET ALL STORES
    # ========================================================

    store_ids = sorted(
        set(
            str(int(x))
            for x in df["store_id"]
            if pd.notna(x)
        )
    )

    # ========================================================
    # CREATE WEEKLY TABLE
    # ========================================================

    weekly_rows = []

    for store_id in store_ids:

        store_row = {
            "store_id": store_id,
            "store_name": store_id,
            "days": []
        }

        store_has_matching_variation = (
            variation_filter == "all"
        )

        for day in week_dates:

            lookup_key = (
                store_id,
                day["raw_date"]
            )

            record = data_lookup.get(
                lookup_key
            )

            if record:

                matches_filter = (
                    variation_matches(
                        record["variation"]
                    )
                )

                if matches_filter:
                    store_has_matching_variation = True

                    day_data = {
                        "predicted": record["predicted"],
                        "actual": record["actual"],
                        "variation": record["variation"],
                        "show": True
                    }

                else:

                    day_data = {
                        "predicted": "-",
                        "actual": "-",
                        "variation": "-",
                        "show": False
                    }

            else:

                day_data = {
                    "predicted": "-",
                    "actual": "-",
                    "variation": "-",
                    "show": False
                }

            store_row["days"].append(
                day_data
            )

        # Only show store if at least one
        # day matches the selected variation.
        if store_has_matching_variation:
            weekly_rows.append(
                store_row
            )

    # ========================================================
    # CONTEXT
    # ========================================================

    context = {

        "weekly_rows": weekly_rows,

        "total_rows": len(
            weekly_rows
        ),

        "variation_filter":
            variation_filter,

        "variation_options": [

            (
                "all",
                "All"
            ),

            (
                "0-10",
                "0-10%"
            ),

            (
                "11-15",
                "11-15%"
            ),

            (
                "16-20",
                "16-20%"
            ),

            (
                "21-25",
                "21-25%"
            ),

            (
                "26-30",
                "26-30%"
            ),

            (
                "30-plus",
                "Above 30%"
            )
        ],

        "selected_date":
            week_start.strftime(
                "%Y-%m-%d"
            ),

        "week_start":
            week_start.strftime(
                "%d-%b-%Y"
            ),

        "week_end":
            week_end.strftime(
                "%d-%b-%Y"
            ),

        "week_dates":
            week_dates
    }

    # ========================================================
    # RENDER
    # ========================================================

    return render(
        request,
        "dashboard/weekly_report.html",
        context
    )