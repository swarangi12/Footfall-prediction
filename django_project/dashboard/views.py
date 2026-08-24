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

    # If a specific `date` query parameter is provided, align to Monday of that week
    date_param = request.GET.get("date")
    if date_param:
        try:
            param_date = pd.to_datetime(date_param, errors="coerce").normalize()
            if not pd.isna(param_date):
                week_start = param_date - pd.Timedelta(days=param_date.weekday())
        except Exception:
            pass

    week_end = week_start + pd.Timedelta(days=6)

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
    # LOAD PREDICTIONS & CSV ACTUALS
    # --------------------------------------------------------

    pred_df = pd.DataFrame()

    if os.path.exists(prediction_file):
        try:
            pred_df = pd.read_csv(prediction_file)
            required_columns = ["date", "store_id", "gate_id", "predicted"]
            missing = [col for col in required_columns if col not in pred_df.columns]

            if not missing:
                pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce").dt.normalize()
                pred_df["store_id"] = pd.to_numeric(pred_df["store_id"], errors="coerce")
                pred_df["gate_id"] = pd.to_numeric(pred_df["gate_id"], errors="coerce")
                pred_df["predicted"] = pd.to_numeric(pred_df["predicted"], errors="coerce").fillna(0)
                
                if "actual" in pred_df.columns:
                    pred_df["actual"] = pd.to_numeric(pred_df["actual"], errors="coerce").fillna(0)
                else:
                    pred_df["actual"] = 0

                pred_df = pred_df.dropna(subset=["date", "store_id"])
                pred_df = pred_df[
                    (pred_df["date"] >= week_start) & (pred_df["date"] <= week_end)
                ].copy()
            else:
                print("Missing prediction columns:", missing)
        except Exception as e:
            print("Prediction file error:", e)

    # --------------------------------------------------------
    # PREDICTION & CSV ACTUALS GROUP BY STORE + GATE + DATE
    # --------------------------------------------------------

    if not pred_df.empty:
        pred_grouped = (
            pred_df.groupby(["date", "store_id", "gate_id"], as_index=False)[["predicted", "actual"]].sum()
        )
        pred_grouped["store_id"] = pred_grouped["store_id"].astype(int).astype(str)
        pred_grouped["gate_id"] = pred_grouped["gate_id"].astype(int).astype(str)
        pred_grouped["date"] = pd.to_datetime(pred_grouped["date"]).dt.strftime("%Y-%m-%d")
    else:
        pred_grouped = pd.DataFrame(columns=["date", "store_id", "gate_id", "predicted", "actual"])

    # --------------------------------------------------------
    # ACTUAL DATA FROM DATABASE
    # --------------------------------------------------------

    actual_df = get_actual_data(week_start, week_end)
    if not actual_df.empty:
        actual_df["store_id"] = actual_df["store_id"].astype(int).astype(str)
        actual_df["gate_id"] = actual_df["gate_id"].astype(int).astype(str)
        actual_df["date"] = pd.to_datetime(actual_df["date"]).dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # MERGE (OUTER JOIN ON date, store_id, gate_id)
    # --------------------------------------------------------

    if not pred_grouped.empty and not actual_df.empty:
        df = pd.merge(
            pred_grouped,
            actual_df,
            on=["date", "store_id", "gate_id"],
            how="outer",
            suffixes=("_csv", "_db")
        )
        if "actual_csv" in df.columns and "actual_db" in df.columns:
            df["actual"] = np.where(df["actual_csv"] > 0, df["actual_csv"], df["actual_db"])
        elif "actual_csv" in df.columns:
            df["actual"] = df["actual_csv"]
        elif "actual_db" in df.columns:
            df["actual"] = df["actual_db"]
    elif not pred_grouped.empty:
        df = pred_grouped.copy()
    elif not actual_df.empty:
        df = actual_df.copy()
        df["predicted"] = 0
    else:
        df = pd.DataFrame(columns=["date", "store_id", "gate_id", "predicted", "actual"])

    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce").fillna(0)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0)


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

    # Build lookup and store_gate_pairs
    data_lookup = {}
    store_gate_pairs = set()

    for _, row in df.iterrows():
        store_id = str(int(row["store_id"]))
        # Ensure gate_id is a string (empty if missing)
        gate_id = (
            str(int(row["gate_id"]))
            if not pd.isna(row.get("gate_id"))
            else ""
        )
        date_key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        data_lookup[(store_id, gate_id, date_key)] = {
            "predicted": int(float(row["predicted"])),
            "actual": int(float(row["actual"])),
            "variation": row["variation"],
        }
        store_gate_pairs.add((store_id, gate_id))
    # Debug: print sample record for store 272 on 2026-06-17
    sample_key = ("272", "2026-06-17")
    if sample_key in data_lookup:
        print("Sample record for store 272 on 2026-06-17:", data_lookup[sample_key])


    # --------------------------------------------------------
    # STORE LIST & VARIATION FILTER
    # --------------------------------------------------------
    variation_filter = request.GET.get(
        "variation",
        "all"
    )

    def variation_matches(value):
        if value == "NA":
            return variation_filter == "all"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return variation_filter == "all"

        if variation_filter == "all":
            return True
        # Expanded ranges
        if variation_filter == "0-10":
            return 0 <= value <= 10
        if variation_filter == "11-20":
            return 11 <= value <= 20
        if variation_filter == "21-30":
            return 21 <= value <= 30
        if variation_filter == "31-40":
            return 31 <= value <= 40
        if variation_filter == "41-50":
            return 41 <= value <= 50
        if variation_filter == "51-60":
            return 51 <= value <= 60
        if variation_filter == "61-70":
            return 61 <= value <= 70
        if variation_filter == "71-80":
            return 71 <= value <= 80
        if variation_filter == "81-90":
            return 81 <= value <= 90
        if variation_filter == "91-100":
            return 91 <= value <= 100
        return False

    # Build a sorted list of (store_id, gate_id) tuples from the lookup
    sorted_store_gate_pairs = sorted(
        list(store_gate_pairs),
        key=lambda x: (int(x[0]), int(x[1]) if x[1] != "" else -1)
    )

    # --------------------------------------------------------
    # BUILD WEEKLY ROWS
    # --------------------------------------------------------
    weekly_rows = []

    for store_id, gate_id in sorted_store_gate_pairs:
        store_row = {
            "store_id": store_id,
            "gate_id": gate_id,
            "days": []
        }

        for day in week_dates:
            lookup_key = (store_id, gate_id, day["raw_date"])
            record = data_lookup.get(lookup_key)

            if record:
                matches = variation_matches(record["variation"])
                if matches:
                    day_data = {
                        "predicted": int(record["predicted"]),
                        "actual": int(record["actual"]),
                        # Variation may be numeric or the string "NA"
                        "variation": (
                            int(record["variation"]) if isinstance(record["variation"], (int, float))
                            else (int(float(record["variation"])) if isinstance(record["variation"], str) and record["variation"].upper() != "NA" else "-")
                        ),
                        "variation_display": (
                            f"{int(record['variation'])}%" if isinstance(record["variation"], (int, float))
                            else (f"{int(float(record['variation']))}%" if isinstance(record["variation"], str) and record["variation"].upper() != "NA" else "NA%")
                        ),
                        "show": True,
                    }
                else:
                    day_data = {
                        "predicted": "-",
                        "actual": "-",
                        "variation": "-",
                        "variation_display": "-",
                        "show": False,
                    }
            else:
                day_data = {
                    "predicted": "-",
                    "actual": "-",
                    "variation": "-",
                    "variation_display": "-",
                    "show": False,
                }

            store_row["days"].append(day_data)

        # Always include the store row (even if no matching day)
        weekly_rows.append(store_row)

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
        "headers": week_dates,
        "stores": weekly_rows,
        "total_records": len(weekly_rows),
        "total_rows": len(weekly_rows),
        "variation_filter": variation_filter,
        "variation_options": [
            ("all", "All"),
            ("0-10", "0-10%"),
            ("11-20", "11-20%"),
            ("21-30", "21-30%"),
            ("31-40", "31-40%"),
            ("41-50", "41-50%"),
            ("51-60", "51-60%"),
            ("61-70", "61-70%"),
            ("71-80", "71-80%"),
            ("81-90", "81-90%"),
            ("91-100", "91-100%"),
        ],
        "selected_date": week_start.strftime("%Y-%m-%d"),
        "week_start": week_start,
        "week_end": week_end,
        "week_dates": week_dates,
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
                    gate_id,
                    SUM(actual) AS actual
                FROM actual_footfall
                WHERE DATE(date) >= %s
                  AND DATE(date) <= %s
                GROUP BY
                    DATE(date),
                    store_id,
                    gate_id
                ORDER BY
                    DATE(date),
                    store_id,
                    gate_id
                """,
                [
                    start_date,
                    end_date
                ]
            )

            actual_rows = cursor.fetchall()

    except Exception as e:
        print("ACTUAL DATABASE ERROR:", e)
        actual_rows = []

    print("Actual database rows:", len(actual_rows))

    actual_df = pd.DataFrame(
        actual_rows,
        columns=["date", "store_id", "gate_id", "actual"]
    )

    if actual_df.empty:
        print("WARNING: No actual data found.")
        return pd.DataFrame(columns=["date", "store_id", "gate_id", "actual"])

    actual_df["date"] = pd.to_datetime(actual_df["date"], errors="coerce").dt.normalize()
    actual_df["store_id"] = pd.to_numeric(actual_df["store_id"], errors="coerce")
    actual_df["gate_id"] = pd.to_numeric(actual_df["gate_id"], errors="coerce")
    actual_df["actual"] = pd.to_numeric(actual_df["actual"], errors="coerce").fillna(0)

    actual_df = actual_df.dropna(subset=["date", "store_id"])

    # Ensure grouping by date, store_id, gate_id
    actual_df = actual_df.groupby(["date", "store_id", "gate_id"], as_index=False)["actual"].sum()

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