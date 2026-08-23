import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from django.shortcuts import render
from django.conf import settings
from django.db import connection

from .models import DailyHourlyFootfall


# ============================================================
# HELPER: GET PREDICTION FILE
# ============================================================

def get_prediction_file():
    """
    prediction_log.csv is located one level above django_project.
    """

    possible_paths = [
        os.path.join(
            settings.BASE_DIR,
            "prediction_log.csv"
        ),

        os.path.join(
            settings.BASE_DIR.parent,
            "prediction_log.csv"
        ),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return possible_paths[1]


# ============================================================
# HELPER: GET WEEK START
# ============================================================

def get_week_start(request, prediction_file):

    selected_date = request.GET.get("date")

    # --------------------------------------------------------
    # USER SELECTED DATE
    # --------------------------------------------------------

    if selected_date:

        try:

            selected = pd.to_datetime(
                selected_date,
                errors="coerce"
            )

            if not pd.isna(selected):

                selected = selected.normalize()

                # Monday of selected week
                week_start = (
                    selected
                    - pd.Timedelta(
                        days=selected.weekday()
                    )
                )

                return week_start

        except Exception:
            pass

    # --------------------------------------------------------
    # DEFAULT:
    # LATEST DATE FROM PREDICTION FILE
    # --------------------------------------------------------

    if os.path.exists(prediction_file):

        try:

            temp_df = pd.read_csv(
                prediction_file,
                usecols=["date"]
            )

            temp_df["date"] = pd.to_datetime(
                temp_df["date"],
                errors="coerce"
            )

            temp_df = temp_df.dropna(
                subset=["date"]
            )

            if not temp_df.empty:

                latest_date = (
                    temp_df["date"]
                    .max()
                    .normalize()
                )

                return (
                    latest_date
                    - pd.Timedelta(
                        days=latest_date.weekday()
                    )
                )

        except Exception as e:

            print(
                "Could not determine date "
                "from prediction file:",
                e
            )

    # --------------------------------------------------------
    # FALLBACK TO DATABASE
    # --------------------------------------------------------

    try:

        latest_record = (
            DailyHourlyFootfall.objects
            .order_by("-date")
            .first()
        )

        if latest_record:

            latest_date = pd.Timestamp(
                latest_record.date
            ).normalize()

            return (
                latest_date
                - pd.Timedelta(
                    days=latest_date.weekday()
                )
            )

    except Exception as e:

        print(
            "Could not determine latest "
            "database date:",
            e
        )

    # --------------------------------------------------------
    # FINAL FALLBACK
    # --------------------------------------------------------

    today = pd.Timestamp.today().normalize()

    return (
        today
        - pd.Timedelta(
            days=today.weekday()
        )
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard(request):

    prediction_file = get_prediction_file()

    week_start = get_week_start(
        request,
        prediction_file
    )

    week_end = (
        week_start
        + pd.Timedelta(days=6)
    )

    print("\n")
    print("=" * 70)
    print("DASHBOARD")
    print("=" * 70)

    print(
        "Week:",
        week_start.strftime("%Y-%m-%d"),
        "to",
        week_end.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    pred_df = pd.DataFrame()

    if os.path.exists(prediction_file):

        try:

            pred_df = pd.read_csv(
                prediction_file
            )

            required_columns = [
                "date",
                "store_id",
                "gate_id",
                "predicted"
            ]

            missing = [
                col
                for col in required_columns
                if col not in pred_df.columns
            ]

            if not missing:

                pred_df["date"] = pd.to_datetime(
                    pred_df["date"],
                    errors="coerce"
                ).dt.normalize()

                pred_df["store_id"] = pd.to_numeric(
                    pred_df["store_id"],
                    errors="coerce"
                )

                pred_df["gate_id"] = pd.to_numeric(
                    pred_df["gate_id"],
                    errors="coerce"
                )

                pred_df["predicted"] = pd.to_numeric(
                    pred_df["predicted"],
                    errors="coerce"
                )

                pred_df = pred_df.dropna(
                    subset=[
                        "date",
                        "store_id",
                        "predicted"
                    ]
                )

                pred_df = pred_df[
                    (pred_df["date"] >= week_start)
                    &
                    (pred_df["date"] <= week_end)
                ].copy()

            else:

                print(
                    "Missing prediction columns:",
                    missing
                )

        except Exception as e:

            print(
                "Prediction file error:",
                e
            )

    # --------------------------------------------------------
    # PREDICTION GROUP BY STORE + DATE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ACTUAL DATA
    # --------------------------------------------------------

    actual_df = get_actual_data(
        week_start,
        week_end
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    if not pred_grouped.empty:

        df = pd.merge(
            pred_grouped,
            actual_df,
            on=[
                "date",
                "store_id"
            ],
            how="left"
        )

        # If actual does not exist,
        # show 0 rather than creating errors.
        df["actual"] = (
            pd.to_numeric(
                df["actual"],
                errors="coerce"
            )
            .fillna(0)
        )

    else:

        df = pd.DataFrame(
            columns=[
                "date",
                "store_id",
                "predicted",
                "actual"
            ]
        )

    # --------------------------------------------------------
    # VARIATION
    # --------------------------------------------------------

    if not df.empty:

        df["variation"] = np.where(
            df["actual"] == 0,
            "NA",
            (
                (
                    abs(
                        df["predicted"]
                        -
                        df["actual"]
                    )
                    /
                    df["actual"]
                )
                * 100
            )
        )

        df["variation"] = df[
            "variation"
        ].apply(
            lambda x:
                "NA"
                if x == "NA"
                else round(float(x), 2)
        )

    # --------------------------------------------------------
    # WEEK DATES
    # --------------------------------------------------------

    week_dates = []

    for i in range(7):

        current_date = (
            week_start
            + pd.Timedelta(days=i)
        )

        week_dates.append(
            {
                "name": current_date.strftime(
                    "%A"
                ),

                "date": current_date.strftime(
                    "%d-%b-%Y"
                ),

                "raw_date": current_date.strftime(
                    "%Y-%m-%d"
                )
            }
        )

    # --------------------------------------------------------
    # BUILD LOOKUP
    # --------------------------------------------------------

    data_lookup = {}

    for _, row in df.iterrows():

        store_id = str(
            int(row["store_id"])
        )

        date_key = pd.Timestamp(
            row["date"]
        ).strftime(
            "%Y-%m-%d"
        )

        data_lookup[
            (store_id, date_key)
        ] = {
            "predicted": round(
                float(row["predicted"]),
                2
            ),

            "actual": round(
                float(row["actual"]),
                2
            ),

            "variation": row[
                "variation"
            ]
        }

    # --------------------------------------------------------
    # STORE LIST
    # IMPORTANT:
    # STORES COME ONLY FROM PREDICTIONS
    # --------------------------------------------------------

    if not pred_grouped.empty:

        store_ids = sorted(
            set(
                str(int(x))
                for x in pred_grouped[
                    "store_id"
                ]
                if pd.notna(x)
            )
        )

    else:

        store_ids = []

    # --------------------------------------------------------
    # VARIATION FILTER
    # --------------------------------------------------------

    variation_filter = request.GET.get(
        "variation",
        "all"
    )

    # --------------------------------------------------------
    # CHECK VARIATION FILTER
    # --------------------------------------------------------

    def variation_matches(value):

        if value == "NA":
            return variation_filter == "all"

        try:
            value = float(value)
        except (TypeError, ValueError):
            return variation_filter == "all"

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
            return value > 30

        return True

    # --------------------------------------------------------
    # BUILD WEEKLY ROWS
    # --------------------------------------------------------

    weekly_rows = []

    for store_id in store_ids:

        store_row = {
            "store_id": store_id,
            "store_name": store_id,
            "days": []
        }

        has_matching_day = (
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

                matches = variation_matches(
                    record["variation"]
                )

                if matches:

                    has_matching_day = True

                    day_data = {
                        "predicted":
                            record["predicted"],

                        "actual":
                            record["actual"],

                        "variation":
                            record["variation"],

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

        if has_matching_day:

            weekly_rows.append(
                store_row
            )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print(
        "Prediction rows:",
        len(pred_grouped)
    )

    print(
        "Actual rows:",
        len(actual_df)
    )

    print(
        "Merged rows:",
        len(df)
    )

    if not df.empty:

        print(
            "Actual values found:",
            (df["actual"] != 0).sum()
        )

        print("\nFINAL SAMPLE:")

        print(
            df[
                [
                    "date",
                    "store_id",
                    "predicted",
                    "actual",
                    "variation"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = {

        "weekly_rows":
            weekly_rows,

        "total_rows":
            len(weekly_rows),

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

    return render(
        request,
        "dashboard/weekly_report.html",
        context
    )


# ============================================================
# GET ACTUAL DATA
# ============================================================

def get_actual_data(
    week_start,
    week_end
):

    start_date = (
        pd.Timestamp(
            week_start
        ).date()
    )

    end_date = (
        pd.Timestamp(
            week_end
        ).date()
    )

    print("\n")
    print("=" * 70)
    print("ACTUAL DATABASE QUERY")
    print("=" * 70)

    print(
        "Start date:",
        start_date
    )

    print(
        "End date:",
        end_date
    )

    actual_rows = []

    # --------------------------------------------------------
    # QUERY CURRENT actual_footfall TABLE
    #
    # SCHEMA:
    #
    # id
    # date
    # store_id
    # gate_id
    # actual
    # created_at
    #
    # --------------------------------------------------------

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    DATE(date) AS date,
                    store_id,
                    SUM(actual) AS actual
                FROM actual_footfall
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
                    start_date,
                    end_date
                ]
            )

            actual_rows = cursor.fetchall()

    except Exception as e:

        print(
            "ACTUAL DATABASE ERROR:",
            e
        )

        actual_rows = []

    print(
        "Actual database rows:",
        len(actual_rows)
    )

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    actual_df = pd.DataFrame(
        actual_rows,
        columns=[
            "date",
            "store_id",
            "actual"
        ]
    )

    if actual_df.empty:

        print(
            "WARNING: No actual data found."
        )

        return pd.DataFrame(
            columns=[
                "date",
                "store_id",
                "actual"
            ]
        )

    # --------------------------------------------------------
    # CLEAN TYPES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DEBUG SAMPLE
    # --------------------------------------------------------

    print("\nACTUAL SAMPLE:")

    print(
        actual_df.head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # GROUP AGAIN TO GUARANTEE ONE ROW
    # PER DATE + STORE
    # --------------------------------------------------------

    actual_df = (
        actual_df
        .groupby(
            [
                "date",
                "store_id"
            ],
            as_index=False
        )["actual"]
        .sum()
    )

    return actual_df


# ============================================================
# OPTIONAL OLD DASHBOARD URL
# ============================================================
#
# If your urls.py currently points "/" to dashboard,
# this function redirects the dashboard to the weekly report.
#
# You can keep this if your main page is the weekly report.
# ============================================================

def home(request):

    return weekly_report(request)