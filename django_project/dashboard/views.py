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

    context = {
        "store_gate_data": list(store_gate_data),
    }

    return render(
        request,
        "dashboard.html",
        context
    )


def weekly_report(request):

    # =========================================================
    # 1. PREDICTION CSV
    # =========================================================

    csv_path = os.path.join(settings.BASE_DIR, "prediction_log.csv")
    

    try:
       df = pd.read_csv(csv_path)
       pred = df.copy()

    except FileNotFoundError:

        return render(
            request,
            "dashboard/weekly_report.html",
            {
                "error": (
                    "prediction_log.csv not found. "
                    "Please check the file path."
                )
            }
        )

    # Convert prediction columns

    pred["date"] = pd.to_datetime(
        pred["date"],
        errors="coerce"
    )

    pred["store_id"] = pd.to_numeric(
        pred["store_id"],
        errors="coerce"
    )

    pred["gate_id"] = pd.to_numeric(
        pred["gate_id"],
        errors="coerce"
    )

    pred["predicted"] = pd.to_numeric(
        pred["predicted"],
        errors="coerce"
    )

    pred = pred.dropna(
        subset=[
            "date",
            "store_id",
            "gate_id",
            "predicted"
        ]
    )

    # Prediction cannot be negative

    pred["predicted"] = pred["predicted"].clip(
        lower=0
    )


    # =========================================================
    # 2. HOURLY FIELD NAMES
    # =========================================================

    hourly_fields = [

        "t7_00_8_00",
        "t8_00_9_00",
        "t9_00_10_00",
        "t10_00_11_00",
        "t11_00_12_00",
        "t12_00_13_00",
        "t13_00_14_00",
        "t14_00_15_00",
        "t15_00_16_00",
        "t16_00_17_00",
        "t17_00_18_00",
        "t18_00_19_00",
        "t19_00_20_00",
        "t20_00_21_00",

    ]


    # =========================================================
    # 3. LOAD ACTUAL + HOURLY DATA FROM DATABASE
    # =========================================================

    actual_data = list(
        DailyHourlyFootfall.objects.values(
            "date",
            "store_id",
            "gate_id",
            "total_footfall",
            *hourly_fields
        )
    )

    actual = pd.DataFrame(actual_data)

    if actual.empty:

        return render(
            request,
            "dashboard/weekly_report.html",
            {
                "error": "No actual footfall data found."
            }
        )


    # =========================================================
    # 4. CLEAN ACTUAL DATA
    # =========================================================

    actual["date"] = pd.to_datetime(
        actual["date"],
        errors="coerce"
    )

    actual["store_id"] = pd.to_numeric(
        actual["store_id"],
        errors="coerce"
    )

    actual["gate_id"] = pd.to_numeric(
        actual["gate_id"],
        errors="coerce"
    )

    actual["actual"] = pd.to_numeric(
        actual["total_footfall"],
        errors="coerce"
    )

    actual = actual.dropna(
        subset=[
            "date",
            "store_id",
            "gate_id",
            "actual"
        ]
    )


    # =========================================================
    # 5. CLEAN HOURLY VALUES
    # =========================================================

    for field in hourly_fields:

        actual[field] = pd.to_numeric(
            actual[field],
            errors="coerce"
        ).fillna(0)


    # =========================================================
    # 6. MERGE PREDICTION + ACTUAL
    # =========================================================

    merge_columns = [
        "date",
        "store_id",
        "gate_id"
    ]

    actual_columns = (
        merge_columns
        + ["actual"]
        + hourly_fields
    )

    df = pd.merge(

        pred[
            merge_columns
            + ["predicted"]
        ],

        actual[
            actual_columns
        ],

        on=merge_columns,

        how="inner"
    )


    if df.empty:

        return render(
            request,
            "dashboard/weekly_report.html",
            {
                "error": (
                    "No matching prediction and actual "
                    "records were found."
                )
            }
        )


    # =========================================================
    # 7. CALCULATE ERRORS
    # =========================================================

    df["absolute_error"] = (
        df["predicted"]
        -
        df["actual"]
    ).abs()


    # Error % is NOT calculated when actual = 0
    #
    # This prevents misleading values such as:
    #
    # Actual = 0
    # Predicted = 120
    #
    # from becoming 12000%.

    df["error_percent"] = np.where(

        df["actual"] > 0,

        (
            df["absolute_error"]
            /
            df["actual"]
            *
            100
        ),

        np.nan
    )


    # =========================================================
    # 8. MAE
    # =========================================================

    mae = df["absolute_error"].mean()


    # =========================================================
    # 9. MAPE
    # =========================================================

    positive_actual = (
        df["actual"] > 0
    )

    if positive_actual.any():

        mape = (

            df.loc[
                positive_actual,
                "absolute_error"
            ]

            /

            df.loc[
                positive_actual,
                "actual"
            ]

        ).mean() * 100

    else:

        mape = np.nan


    # =========================================================
    # 10. RMSE
    # =========================================================

    rmse = np.sqrt(

        (
            df["predicted"]
            -
            df["actual"]
        ) ** 2

    ).mean()

    # Correct RMSE calculation

    rmse = np.sqrt(

        (
            df["predicted"]
            -
            df["actual"]
        ) ** 2
    ).mean()

    # Actually calculate root mean squared error

    rmse = np.sqrt(
        (
            (
                df["predicted"]
                -
                df["actual"]
            ) ** 2
        ).mean()
    )


    # =========================================================
    # 11. R2
    # =========================================================

    actual_mean = df["actual"].mean()

    ss_total = (
        (
            df["actual"]
            -
            actual_mean
        ) ** 2
    ).sum()

    ss_residual = (
        (
            df["actual"]
            -
            df["predicted"]
        ) ** 2
    ).sum()

    if ss_total != 0:

        r2 = 1 - (
            ss_residual
            /
            ss_total
        )

    else:

        r2 = np.nan


    # =========================================================
    # 12. ROUND VALUES
    # =========================================================

    df["predicted"] = (
        df["predicted"]
        .round(2)
    )

    df["actual"] = (
        df["actual"]
        .round(2)
    )

    df["absolute_error"] = (
        df["absolute_error"]
        .round(2)
    )

    df["error_percent"] = (
        df["error_percent"]
        .round(2)
    )


    # =========================================================
    # 13. ROUND HOURLY VALUES
    # =========================================================

    for field in hourly_fields:

        df[field] = (
            df[field]
            .fillna(0)
            .round(0)
            .astype(int)
        )


    # =========================================================
    # 14. SORT
    # =========================================================

    df = df.sort_values(
        "date",
        ascending=False
    )


    # =========================================================
    # 15. CREATE TABLE DATA
    # =========================================================

    table_data = []


    for _, row in df.iterrows():

        # -----------------------------------------------------
        # Error percentage
        # -----------------------------------------------------

        if pd.isna(
            row["error_percent"]
        ):

            error_percent_display = "N/A"

        else:

            error_percent_display = (
                f'{row["error_percent"]:.2f}%'
            )


        # -----------------------------------------------------
        # Basic row
        # -----------------------------------------------------

        row_data = {

            "date": row["date"].strftime(
                "%Y-%m-%d"
            ),

            "store_id": int(
                row["store_id"]
            ),

            "gate_id": int(
                row["gate_id"]
            ),

            "predicted": float(
                row["predicted"]
            ),

            "actual": float(
                row["actual"]
            ),

            "absolute_error": float(
                row["absolute_error"]
            ),

            "error_percent":
                error_percent_display
        }


        # -----------------------------------------------------
        # Add all hourly fields
        # -----------------------------------------------------

        for field in hourly_fields:

            row_data[field] = int(
                row[field]
            )


        table_data.append(
            row_data
        )


    # =========================================================
    # 16. SUMMARY
    # =========================================================

    context = {

        "table_data":
            table_data,

        "mae":
            round(mae, 2),

        "rmse":
            round(rmse, 2),

        "r2": (
            round(r2, 4)
            if not np.isnan(r2)
            else "N/A"
        ),

        "mape": (
            round(mape, 2)
            if not np.isnan(mape)
            else "N/A"
        ),

        "total_records":
            len(df),

        "zero_actual_records":
            int(
                (
                    df["actual"] == 0
                ).sum()
            ),

        "positive_actual_records":
            int(
                (
                    df["actual"] > 0
                ).sum()
            ),

        "prediction_start":
            df["date"]
            .min()
            .strftime("%Y-%m-%d"),

        "prediction_end":
            df["date"]
            .max()
            .strftime("%Y-%m-%d")
    }


    # =========================================================
    # 17. RENDER
    # =========================================================

    return render(
        request,
        "dashboard/weekly_report.html",
        context
    )