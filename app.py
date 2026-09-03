import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

try:
    st.set_page_config(
        page_title="Footfall Prediction Dashboard",
        page_icon="📊",
        layout="wide"
    )
except Exception:
    pass


# =========================================================
# IMPORTS
# =========================================================

import os
import pickle
import shutil
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    import holidays
except Exception:
    holidays = None

try:
    from supabase import create_client
except Exception:
    create_client = None


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# SUPABASE
# =========================================================

SUPABASE_URL = None
SUPABASE_KEY = None
supabase = None

try:
    sec = getattr(st, "secrets", None)

    if sec is not None:
        try:
            SUPABASE_URL = sec.get("SUPABASE_URL", None)
            SUPABASE_KEY = sec.get("SUPABASE_KEY", None)
        except Exception:
            pass

except Exception:
    pass


if not SUPABASE_URL:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")

if not SUPABASE_KEY:
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


if SUPABASE_URL and SUPABASE_KEY and create_client:

    try:
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

    except Exception as e:
        print("Could not connect to Supabase:", e)


# =========================================================
# STYLE
# =========================================================

try:

    style_path = BASE_DIR / "style.css"

    if style_path.exists():

        with open(
            style_path,
            encoding="utf-8"
        ) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

except Exception:
    pass


# =========================================================
# FILE PATH FINDER
# =========================================================

def find_file(filename):

    paths = [
        BASE_DIR / filename,
        BASE_DIR / "models_v3_1" / filename,
        BASE_DIR.parent / filename,
    ]

    for path in paths:

        if path.exists():
            return path

    return BASE_DIR / filename


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = find_file(
    "hourlyfootfall_till_current_date1.csv"
)

STAGE1_MODEL_PATH = find_file(
    "stage1_low_classifier.pkl"
)

LOW_MODEL_PATH = find_file(
    "low_footfall_model.pkl"
)

NORMAL_MODEL_PATH = find_file(
    "normal_footfall_model.pkl"
)

PREDICTION_CONFIG_PATH = find_file(
    "prediction_config.pkl"
)

MODEL_INFO_PATH = find_file(
    "model_info.pkl"
)

PREDICTION_LOG = find_file(
    "prediction_log.csv"
)

ACTUAL_FILE = find_file(
    "actual_footfall.csv"
)

ERROR_LOG = find_file(
    "error_log.csv"
)


# =========================================================
# REQUIRED MODEL FEATURES
# =========================================================

FEATURES = [

    "store_id",
    "gate_id",

    "year",
    "month",
    "day",
    "weekday",
    "week",
    "quarter",

    "is_weekend",
    "holiday",

    "weekday_sin",
    "weekday_cos",

    "month_sin",
    "month_cos",

    "lag1",
    "lag7",
    "lag14",
    "lag21",
    "lag28",
    "lag30",

    "rolling7",
    "rolling14",
    "rolling30",

    "trend",

    "store_mean",
    "gate_mean",

    "store_weekday_mean",
    "gate_weekday_mean",

    "is_month_start",
    "is_month_end",

    "is_quarter_start",
    "is_quarter_end"
]


# =========================================================
# RECURSIVE METADATA SEARCH
# =========================================================

def recursively_find_value(
    obj,
    keys_to_find
):

    keys_to_find = {
        str(k).lower()
        for k in keys_to_find
    }

    if isinstance(obj, dict):

        for key, value in obj.items():

            key_lower = str(key).lower()

            if key_lower in keys_to_find:
                return value

            result = recursively_find_value(
                value,
                keys_to_find
            )

            if result is not None:
                return result

    elif isinstance(obj, (list, tuple)):

        for item in obj:

            result = recursively_find_value(
                item,
                keys_to_find
            )

            if result is not None:
                return result

    return None


# =========================================================
# LOAD METADATA
# =========================================================

@st.cache_resource
def load_metadata():

    prediction_config = None
    model_info = None

    if PREDICTION_CONFIG_PATH.exists():

        try:

            with open(
                PREDICTION_CONFIG_PATH,
                "rb"
            ) as f:

                prediction_config = pickle.load(f)

        except Exception:

            prediction_config = None


    if MODEL_INFO_PATH.exists():

        try:

            with open(
                MODEL_INFO_PATH,
                "rb"
            ) as f:

                model_info = pickle.load(f)

        except Exception:

            model_info = None


    return (
        prediction_config,
        model_info
    )


prediction_config, model_info = load_metadata()


# =========================================================
# DETECT TARGET TRANSFORMATION
# =========================================================

def detect_target_transform(
    prediction_config,
    model_info
):

    objects = [
        prediction_config,
        model_info
    ]

    search_keys = [

        "target_transform",
        "target_transformation",
        "target_transform_type",

        "transform",
        "transformation",

        "target_type",
        "target_scale",
        "prediction_scale",

        "use_log_target",
        "log_target",
        "log_transform"
    ]


    for obj in objects:

        if obj is None:
            continue

        value = recursively_find_value(
            obj,
            search_keys
        )

        if value is None:
            continue


        if isinstance(value, bool):

            if value:
                return "log1p"

            return "raw"


        value_string = str(
            value
        ).lower().strip()


        if (
            "log1p" in value_string
            or
            "log(1+x)" in value_string
            or
            "log_plus_one" in value_string
        ):

            return "log1p"


        if (
            value_string == "log"
            or
            "natural_log" in value_string
            or
            value_string == "ln"
        ):

            return "log"


        if (
            value_string == "raw"
            or
            value_string == "none"
            or
            value_string == "identity"
            or
            value_string == "original"
        ):

            return "raw"


    for obj in objects:

        if obj is None:
            continue

        try:

            text = str(obj).lower()

            if (
                "log1p" in text
                or
                "log(1+x)" in text
                or
                "log_plus_one" in text
            ):

                return "log1p"

        except Exception:
            pass


    # V3.1 fallback
    return "log1p"


TARGET_TRANSFORM = detect_target_transform(
    prediction_config,
    model_info
)


# =========================================================
# CONVERT MODEL OUTPUT
# =========================================================

def convert_prediction_to_footfall(
    raw_prediction
):

    try:

        raw_prediction = float(
            raw_prediction
        )

    except Exception as e:

        raise ValueError(
            f"Invalid model prediction: {raw_prediction}"
        ) from e


    if not np.isfinite(
        raw_prediction
    ):

        raise ValueError(
            f"Model returned invalid prediction: {raw_prediction}"
        )


    if TARGET_TRANSFORM == "log1p":

        raw_prediction = np.clip(
            raw_prediction,
            -20,
            20
        )

        prediction = np.expm1(
            raw_prediction
        )


    elif TARGET_TRANSFORM == "log":

        raw_prediction = np.clip(
            raw_prediction,
            -20,
            20
        )

        prediction = np.exp(
            raw_prediction
        )


    else:

        prediction = raw_prediction


    prediction = max(
        0.0,
        float(prediction)
    )

    return prediction


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    with open(
        STAGE1_MODEL_PATH,
        "rb"
    ) as f:

        stage1 = pickle.load(f)


    with open(
        LOW_MODEL_PATH,
        "rb"
    ) as f:

        low_model = pickle.load(f)


    with open(
        NORMAL_MODEL_PATH,
        "rb"
    ) as f:

        normal_model = pickle.load(f)


    return (
        stage1,
        low_model,
        normal_model
    )


# =========================================================
# LOAD MODELS SAFELY
# =========================================================

try:

    (
        stage1_model,
        low_model,
        normal_model
    ) = load_models()

except Exception as e:

    st.error(
        f"""
        ❌ Unable to load models.

        Error:
        {e}

        Make sure these files exist:

        • stage1_low_classifier.pkl
        • low_footfall_model.pkl
        • normal_footfall_model.pkl
        """
    )

    st.stop()


# =========================================================
# VERIFY MODEL FEATURES
# =========================================================

def get_model_features(model):

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    return None


stage1_features = get_model_features(
    stage1_model
)

low_features = get_model_features(
    low_model
)

normal_features = get_model_features(
    normal_model
)


feature_errors = []


if stage1_features is not None:

    if stage1_features != FEATURES:

        feature_errors.append(
            "Stage 1 classifier"
        )


if low_features is not None:

    if low_features != FEATURES:

        feature_errors.append(
            "Low-footfall model"
        )


if normal_features is not None:

    if normal_features != FEATURES:

        feature_errors.append(
            "Normal-footfall model"
        )


if feature_errors:

    st.error(
        "❌ Model feature mismatch: "
        +
        ", ".join(
            feature_errors
        )
    )

    st.stop()


# =========================================================
# LOAD DATA - MEMORY OPTIMIZED
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )


    # IMPORTANT:
    # Only load columns actually needed.
    data = pd.read_csv(
        DATA_PATH,
        usecols=[
            "date",
            "store_id",
            "gate_id",
            "total_footfall"
        ]
    )


    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )


    data["store_id"] = pd.to_numeric(
        data["store_id"],
        errors="coerce"
    )


    data["gate_id"] = pd.to_numeric(
        data["gate_id"],
        errors="coerce"
    )


    data["total_footfall"] = pd.to_numeric(
        data["total_footfall"],
        errors="coerce"
    )


    data = data.dropna(
        subset=[
            "date",
            "store_id",
            "gate_id",
            "total_footfall"
        ]
    )


    # Smaller numeric types
    data["store_id"] = data[
        "store_id"
    ].astype("int32")


    data["gate_id"] = data[
        "gate_id"
    ].astype("int16")


    data["total_footfall"] = data[
        "total_footfall"
    ].astype("float32")


    data = data.drop_duplicates(
        subset=[
            "date",
            "store_id",
            "gate_id"
        ],
        keep="last"
    )


    data = data.sort_values(
        [
            "store_id",
            "gate_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )


    return data


# =========================================================
# LOAD DATA SAFELY
# =========================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"❌ Unable to load dataset: {e}"
    )

    st.stop()


# =========================================================
# HOURLY DISTRIBUTION HELPERS
# =========================================================

HOURLY_COLUMNS = [
    "t7_00_8_00", "t8_00_9_00", "t9_00_10_00", "t10_00_11_00", "t11_00_12_00",
    "t12_00_13_00", "t13_00_14_00", "t14_00_15_00", "t15_00_16_00", "t16_00_17_00",
    "t17_00_18_00", "t18_00_19_00", "t19_00_20_00", "t20_00_21_00", "t21_00_22_00",
    "t22_00_23_00", "t23_00_23_59"
]

HOURLY_LABELS = {
    "t7_00_8_00": "07:00 - 08:00",
    "t8_00_9_00": "08:00 - 09:00",
    "t9_00_10_00": "09:00 - 10:00",
    "t10_00_11_00": "10:00 - 11:00",
    "t11_00_12_00": "11:00 - 12:00",
    "t12_00_13_00": "12:00 - 13:00",
    "t13_00_14_00": "13:00 - 14:00",
    "t14_00_15_00": "14:00 - 15:00",
    "t15_00_16_00": "15:00 - 16:00",
    "t16_00_17_00": "16:00 - 17:00",
    "t17_00_18_00": "17:00 - 18:00",
    "t18_00_19_00": "18:00 - 19:00",
    "t19_00_20_00": "19:00 - 20:00",
    "t20_00_21_00": "20:00 - 21:00",
    "t21_00_22_00": "21:00 - 22:00",
    "t22_00_23_00": "22:00 - 23:00",
    "t23_00_23_59": "23:00 - 23:59"
}


@st.cache_data
def load_hourly_distribution():
    if not DATA_PATH.exists():
        return None

    usecols = ["date", "store_id", "gate_id"] + HOURLY_COLUMNS

    raw = pd.read_csv(
        DATA_PATH,
        usecols=usecols
    )

    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="coerce"
    )

    raw = raw.dropna(
        subset=[
            "date",
            "store_id",
            "gate_id"
        ]
    )

    raw["store_id"] = raw[
        "store_id"
    ].astype("int32")

    raw["gate_id"] = raw[
        "gate_id"
    ].astype("int16")

    raw["day_of_week"] = raw[
        "date"
    ].dt.dayofweek

    for col in HOURLY_COLUMNS:
        raw[col] = pd.to_numeric(
            raw[col],
            errors="coerce"
        ).fillna(0)

    by_dayofweek = raw.groupby(
        [
            "store_id",
            "gate_id",
            "day_of_week"
        ]
    )[HOURLY_COLUMNS].mean()

    by_store_gate = raw.groupby(
        [
            "store_id",
            "gate_id"
        ]
    )[HOURLY_COLUMNS].mean()

    global_avg = raw[
        HOURLY_COLUMNS
    ].mean()

    return {
        "by_dayofweek": by_dayofweek,
        "by_store_gate": by_store_gate,
        "global_avg": global_avg,
        "hourly_cols": HOURLY_COLUMNS
    }


try:

    hourly_dist_data = load_hourly_distribution()

except Exception:

    hourly_dist_data = None


def predict_hourwise(
    daily_total,
    store,
    gate,
    target_date,
    dist_data
):
    if dist_data is None:
        return pd.DataFrame()

    hourly_cols = dist_data["hourly_cols"]
    day_of_week = pd.Timestamp(target_date).dayofweek

    ratios = None

    try:
        if (store, gate, day_of_week) in dist_data["by_dayofweek"].index:
            ratios = dist_data["by_dayofweek"].loc[(store, gate, day_of_week)].values.copy()
    except Exception:
        ratios = None

    if ratios is None or ratios.sum() == 0:
        try:
            if (store, gate) in dist_data["by_store_gate"].index:
                ratios = dist_data["by_store_gate"].loc[(store, gate)].values.copy()
        except Exception:
            ratios = None

    if ratios is None or ratios.sum() == 0:
        ratios = dist_data["global_avg"].values.copy()

    total_ratio = ratios.sum()

    if total_ratio <= 0:
        normalized_ratios = np.ones(len(hourly_cols)) / len(hourly_cols)
    else:
        normalized_ratios = ratios / total_ratio

    predicted_hourly = daily_total * normalized_ratios
    labels = [HOURLY_LABELS.get(col, col) for col in hourly_cols]

    return pd.DataFrame({
        "Time Slot": labels,
        "Predicted Footfall": np.round(predicted_hourly, 1)
    })


# =========================================================
# CONTINUOUS LEARNING & ADAPTIVE CALIBRATION HELPERS
# =========================================================

def get_recent_calibration_factor(store, gate):
    if not ERROR_LOG.exists():
        return 1.0, 0, 0.0

    try:
        df_err = pd.read_csv(ERROR_LOG)
        if df_err.empty:
            return 1.0, 0, 0.0

        mask = (df_err["store_id"] == store) & (df_err["gate_id"] == gate) & (df_err["actual"] > 0)
        recent = df_err[mask].tail(7)

        if len(recent) == 0:
            recent = df_err[df_err["actual"] > 0].tail(7)

        if len(recent) == 0:
            return 1.0, 0, 0.0

        mean_act = recent["actual"].mean()
        mean_pred = recent["predicted"].mean()

        if mean_pred <= 0 or mean_act <= 0:
            return 1.0, len(recent), float(recent["error_percent"].mean()) if "error_percent" in recent.columns else 0.0

        raw_ratio = mean_act / mean_pred
        bounded_ratio = max(0.7, min(1.3, raw_ratio))
        calib_factor = 0.7 * bounded_ratio + 0.3 * 1.0
        avg_err = float(recent["error_percent"].mean()) if "error_percent" in recent.columns else 0.0

        return round(calib_factor, 4), len(recent), round(avg_err, 2)
    except Exception:
        return 1.0, 0, 0.0


def parse_sql_dump_actuals():
    sql_path = find_file("shoppersstop_backup.sql")
    if not sql_path.exists():
        return pd.DataFrame()
    insert_prefix = "INSERT INTO `app_agehourlyfootfall`"
    parsed_records = []
    try:
        import csv
        import re
        from io import StringIO
        with sql_path.open("r", encoding="utf-8", errors="ignore") as f:
            buffer = []
            for line in f:
                if not buffer and insert_prefix not in line:
                    continue
                buffer.append(line.rstrip())
                if line.rstrip().endswith(";"):
                    stmt = " ".join(buffer)
                    buffer.clear()
                    match = re.search(r"VALUES\s*(.*);$", stmt, flags=re.IGNORECASE | re.DOTALL)
                    if not match:
                        continue
                    values_blob = match.group(1).strip().replace("\n", " ")
                    if values_blob.startswith("(") and values_blob.endswith(")"):
                        values_blob = values_blob[1:-1]
                    raw_rows = re.split(r"\),\s*\(", values_blob)
                    for raw in raw_rows:
                        raw = f"({raw})"
                        raw_clean = raw.replace("\\'", "__SINGLE_QUOTE__")
                        inner = raw_clean[1:-1]
                        try:
                            csv_reader = csv.reader(StringIO(inner), delimiter=",", quotechar="'", escapechar="\\")
                            parts = next(csv_reader)
                            parts = [p.replace("__SINGLE_QUOTE__", "'") for p in parts]
                            store_id = int(parts[3])
                            date_str = parts[4].strip("'")
                            gate_id = int(parts[5])
                            hourly_vals = [int(v) if v.isdigit() else 0 for v in parts[7:24]]
                            if len(hourly_vals) < 17:
                                hourly_vals.extend([0] * (17 - len(hourly_vals)))
                            record = [date_str, store_id, gate_id] + hourly_vals
                            parsed_records.append(record)
                        except Exception:
                            continue
        if not parsed_records:
            return pd.DataFrame()
        cols = ["date", "store_id", "gate_id"] + HOURLY_COLUMNS
        df_parsed = pd.DataFrame(parsed_records, columns=cols)
        df_agg = df_parsed.groupby(["date", "store_id", "gate_id"])[HOURLY_COLUMNS].sum().reset_index()
        df_agg["total_footfall"] = df_agg[HOURLY_COLUMNS].sum(axis=1)
        return df_agg
    except Exception:
        return pd.DataFrame()


def sync_actuals_to_dataset():
    if not DATA_PATH.exists():
        return False, "Required dataset file not found."

    try:
        df_main = pd.read_csv(DATA_PATH)
        df_main["date"] = pd.to_datetime(df_main["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        updated_count = 0
        added_count = 0

        # 1. Sync actuals from shoppersstop_backup.sql
        df_sql = parse_sql_dump_actuals()
        if not df_sql.empty:
            df_sql["date"] = pd.to_datetime(df_sql["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            for _, row in df_sql.iterrows():
                d = str(row["date"])
                s = int(row["store_id"])
                g = int(row["gate_id"])
                act = float(row["total_footfall"])

                mask = (df_main["date"] == d) & (df_main["store_id"] == s) & (df_main["gate_id"] == g)
                if mask.any():
                    df_main.loc[mask, "total_footfall"] = act
                    for hcol in HOURLY_COLUMNS:
                        if hcol in df_main.columns and hcol in row:
                            df_main.loc[mask, hcol] = row[hcol]
                    updated_count += 1
                else:
                    new_row = {"date": d, "store_id": s, "gate_id": g, "total_footfall": act}
                    for hcol in HOURLY_COLUMNS:
                        if hcol in row:
                            new_row[hcol] = row[hcol]
                    df_main = pd.concat([df_main, pd.DataFrame([new_row])], ignore_index=True)
                    added_count += 1

        # 2. Sync actuals from actual_footfall.csv
        if ACTUAL_FILE.exists():
            actuals = pd.read_csv(ACTUAL_FILE)
            if not actuals.empty:
                actuals["date"] = pd.to_datetime(actuals["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                for _, row in actuals.iterrows():
                    d = str(row["date"])
                    s = int(row["store_id"])
                    g = int(row["gate_id"])
                    act = float(row["actual"])

                    mask = (df_main["date"] == d) & (df_main["store_id"] == s) & (df_main["gate_id"] == g)
                    if mask.any():
                        df_main.loc[mask, "total_footfall"] = act
                        updated_count += 1
                    else:
                        new_row = {"date": d, "store_id": s, "gate_id": g, "total_footfall": act}
                        df_main = pd.concat([df_main, pd.DataFrame([new_row])], ignore_index=True)
                        added_count += 1

        df_main.to_csv(DATA_PATH, index=False)
        return True, f"Synchronized {updated_count} updated and {added_count} new entries to training dataset."
    except Exception as e:
        return False, f"Sync error: {e}"


def trigger_background_retrain():
    def _worker():
        try:
            import subprocess
            import sys
            subprocess.run([sys.executable, "retrain_model.py"], capture_output=True, text=True)
        except Exception:
            pass

    import threading
    threading.Thread(target=_worker, daemon=True).start()


# =========================================================
# HOLIDAYS
# =========================================================

try:

    if holidays and hasattr(
        holidays,
        "India"
    ):

        india_holidays = holidays.India()

    else:

        india_holidays = {}

except Exception:

    india_holidays = {}


# =========================================================
# CREATE FEATURES
#
# IMPORTANT:
# This function is NOT called on the complete dataset.
# It is only called for the selected store/gate history.
# =========================================================

def create_features(data):

    data = data.copy()


    data["date"] = pd.to_datetime(
        data["date"]
    )


    data = data.sort_values(
        [
            "store_id",
            "gate_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )


    # =====================================================
    # CALENDAR
    # =====================================================

    data["year"] = (
        data["date"].dt.year
    )

    data["month"] = (
        data["date"].dt.month
    )

    data["day"] = (
        data["date"].dt.day
    )

    data["weekday"] = (
        data["date"].dt.weekday
    )

    data["week"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["quarter"] = (
        data["date"].dt.quarter
    )

    data["is_weekend"] = (
        data["weekday"] >= 5
    ).astype(int)

    data["holiday"] = (
        data["date"]
        .dt.date
        .isin(india_holidays)
        .astype(int)
    )


    # =====================================================
    # CYCLICAL FEATURES
    # =====================================================

    data["weekday_sin"] = np.sin(
        2 * np.pi *
        data["weekday"] / 7
    )

    data["weekday_cos"] = np.cos(
        2 * np.pi *
        data["weekday"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi *
        (data["month"] - 1) / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi *
        (data["month"] - 1) / 12
    )


    # =====================================================
    # GROUP
    # =====================================================

    group = data.groupby(
        [
            "store_id",
            "gate_id"
        ],
        sort=False
    )["total_footfall"]


    # =====================================================
    # LAGS
    # =====================================================

    data["lag1"] = group.shift(1)

    data["lag7"] = group.shift(7)

    data["lag14"] = group.shift(14)

    data["lag21"] = group.shift(21)

    data["lag28"] = group.shift(28)

    data["lag30"] = group.shift(30)


    # =====================================================
    # ROLLING
    # =====================================================

    data["rolling7"] = group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )


    data["rolling14"] = group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )


    data["rolling30"] = group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=1
        )
        .mean()
    )


    # =====================================================
    # TREND
    # =====================================================

    data["trend"] = (
        data["rolling7"]
        /
        data["rolling30"].replace(
            0,
            np.nan
        )
    )


    # =====================================================
    # PAST VALUES
    # =====================================================

    data["_past"] = group.shift(1)


    # =====================================================
    # STORE MEAN
    # =====================================================

    data["store_mean"] = (
        data.groupby(
            "store_id"
        )["_past"]
        .transform(
            lambda x:
            x.expanding()
            .mean()
        )
    )


    # =====================================================
    # GATE MEAN
    # =====================================================

    data["gate_mean"] = (
        data.groupby(
            "gate_id"
        )["_past"]
        .transform(
            lambda x:
            x.expanding()
            .mean()
        )
    )


    # =====================================================
    # STORE + WEEKDAY MEAN
    # =====================================================

    data["store_weekday_mean"] = (
        data.groupby(
            [
                "store_id",
                "weekday"
            ]
        )["_past"]
        .transform(
            lambda x:
            x.expanding()
            .mean()
        )
    )


    # =====================================================
    # GATE + WEEKDAY MEAN
    # =====================================================

    data["gate_weekday_mean"] = (
        data.groupby(
            [
                "gate_id",
                "weekday"
            ]
        )["_past"]
        .transform(
            lambda x:
            x.expanding()
            .mean()
        )
    )


    # =====================================================
    # DATE FLAGS
    # =====================================================

    data["is_month_start"] = (
        data["date"]
        .dt.is_month_start
        .astype(int)
    )

    data["is_month_end"] = (
        data["date"]
        .dt.is_month_end
        .astype(int)
    )

    data["is_quarter_start"] = (
        data["date"]
        .dt.is_quarter_start
        .astype(int)
    )

    data["is_quarter_end"] = (
        data["date"]
        .dt.is_quarter_end
        .astype(int)
    )


    # =====================================================
    # CLEAN
    # =====================================================

    data = data.drop(
        columns=[
            "_past"
        ]
    )


    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    return data


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <h1 style="color:#0F4C81;text-align:center;">
    📊 Footfall Prediction Dashboard
    </h1>

    <p style="text-align:center;font-size:20px;">
    Two-Stage XGBoost Footfall Prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Prediction Inputs"
)


store_list = sorted(
    df["store_id"].unique()
)


store_id = st.sidebar.selectbox(
    "🏪 Select Store",
    store_list
)


gate_list = sorted(
    df[
        df["store_id"] == store_id
    ]["gate_id"].unique()
)


gate_id = st.sidebar.selectbox(
    "🚪 Select Gate",
    gate_list
)


selected_date = st.sidebar.date_input(
    "📅 Prediction Date",
    value=date.today()
)


predict_clicked = st.sidebar.button(
    "🚀 Predict Footfall",
    use_container_width=True
)


# =========================================================
# HISTORY
# =========================================================

history = df[
    (df["store_id"] == store_id)
    &
    (df["gate_id"] == gate_id)
].sort_values(
    "date"
).copy()


if history.empty:

    st.error(
        "No historical data available."
    )

    st.stop()


# =========================================================
# BUILD FUTURE FEATURES
# =========================================================

def build_future_features(
    history_df,
    store,
    gate,
    target_date
):

    target_date = pd.Timestamp(
        target_date
    )


    # =====================================================
    # ONLY SELECTED STORE + GATE
    # =====================================================

    history = history_df[
        (
            history_df["store_id"]
            == store
        )
        &
        (
            history_df["gate_id"]
            == gate
        )
        &
        (
            history_df["date"]
            < target_date
        )
    ].copy()


    history = history.sort_values(
        "date"
    )


    if history.empty:

        raise ValueError(
            "No historical data available "
            "before selected date."
        )


    # =====================================================
    # TEMP DATA
    # =====================================================

    future_row = pd.DataFrame({

        "date": [
            target_date
        ],

        "store_id": [
            store
        ],

        "gate_id": [
            gate
        ],

        "total_footfall": [
            np.nan
        ]
    })


    temp = pd.concat(
        [
            history,
            future_row
        ],
        ignore_index=True
    )


    # =====================================================
    # CREATE FEATURES
    # =====================================================

    temp_features = create_features(
        temp
    )


    row = temp_features[
        temp_features["date"]
        == target_date
    ].tail(1)


    if row.empty:

        raise ValueError(
            "Could not create prediction features."
        )


    # =====================================================
    # CHECK FEATURES
    # =====================================================

    missing_features = [

        feature

        for feature in FEATURES

        if feature not in row.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing model features: "
            +
            ", ".join(
                missing_features
            )
        )


    X = row[
        FEATURES
    ].copy()


    # =====================================================
    # FALLBACK VALUES
    # =====================================================

    fallback_columns = [

        "lag1",
        "lag7",
        "lag14",
        "lag21",
        "lag28",
        "lag30",

        "rolling7",
        "rolling14",
        "rolling30",

        "store_mean",
        "gate_mean",

        "store_weekday_mean",
        "gate_weekday_mean"
    ]


    last_value = float(
        history[
            "total_footfall"
        ].iloc[-1]
    )


    mean_value = float(
        history[
            "total_footfall"
        ].mean()
    )


    for col in fallback_columns:

        if pd.isna(
            X.iloc[0][col]
        ):

            if col.startswith(
                "lag"
            ):

                X.loc[
                    X.index[0],
                    col
                ] = last_value

            else:

                X.loc[
                    X.index[0],
                    col
                ] = mean_value


    # =====================================================
    # TREND
    # =====================================================

    if pd.isna(
        X.iloc[0]["trend"]
    ):

        X.loc[
            X.index[0],
            "trend"
        ] = 1.0


    # =====================================================
    # FINAL CLEANUP
    # =====================================================

    X = X.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    X = X.fillna(0)


    return X


# =========================================================
# PREDICT ONE DATE
# =========================================================

def predict_one_date(
    working_df,
    store,
    gate,
    target_date
):

    X = build_future_features(
        working_df,
        store,
        gate,
        target_date
    )


    # =====================================================
    # STAGE 1
    # =====================================================

    stage_prediction = stage1_model.predict(
        X
    )[0]


    try:

        stage_prediction = int(
            stage_prediction
        )

    except Exception:

        stage_prediction = 0


    # =====================================================
    # MODEL SELECTION
    # =====================================================

    if stage_prediction == 1:

        selected_model = low_model

        model_name = (
            "Low-footfall model"
        )

    else:

        selected_model = normal_model

        model_name = (
            "Normal-footfall model"
        )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    raw_prediction = selected_model.predict(
        X
    )[0]


    # =====================================================
    # CONVERT LOG → REAL FOOTFALL & ADAPTIVE CALIBRATION
    # =====================================================

    raw_val = convert_prediction_to_footfall(
        raw_prediction
    )

    calib_factor, _, _ = get_recent_calibration_factor(store, gate)

    prediction = raw_val * calib_factor


    return (
        prediction,
        model_name,
        X,
        float(raw_prediction)
    )


# =========================================================
# 7-DAY FORECAST
# =========================================================

def make_7_day_forecast(
    base_data,
    store,
    gate,
    start_date
):

    # Only copy the selected store/gate.
    working = base_data[
        (base_data["store_id"] == store)
        &
        (base_data["gate_id"] == gate)
    ].copy()


    results = []


    start_date = pd.Timestamp(
        start_date
    )


    for offset in range(7):

        target = (
            start_date
            +
            pd.Timedelta(
                days=offset
            )
        )


        (
            prediction,
            model_name,
            _,
            raw_prediction
        ) = predict_one_date(

            working,
            store,
            gate,
            target
        )


        results.append({

            "date":
                target,

            "Predicted_Footfall":
                prediction,

            "Model":
                model_name
        })


        # =================================================
        # RECURSIVE PREDICTION
        # =================================================

        new_row = pd.DataFrame({

            "date": [
                target
            ],

            "store_id": [
                store
            ],

            "gate_id": [
                gate
            ],

            "total_footfall": [
                prediction
            ]
        })


        working = pd.concat(
            [
                working,
                new_row
            ],
            ignore_index=True
        )


    return pd.DataFrame(
        results
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_clicked:

    try:

        (
            prediction,
            model_name,
            X,
            raw_prediction
        ) = predict_one_date(

            df,
            store_id,
            gate_id,
            selected_date
        )


        # =================================================
        # SESSION STATE
        # =================================================

        st.session_state.store = (
            store_id
        )

        st.session_state.gate = (
            gate_id
        )

        st.session_state.prediction_date = (
            pd.Timestamp(
                selected_date
            )
        )

        st.session_state.prediction = (
            prediction
        )


        # =================================================
        # RESULT
        # =================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Predicted Footfall",
                f"{prediction:.2f}"
            )


        with col2:

            st.metric(
                "Selected Store",
                str(store_id)
            )


        # =================================================
        # SAVE PREDICTION
        # =================================================

        prediction_date_str = (
            pd.Timestamp(
                selected_date
            ).strftime(
                "%Y-%m-%d"
            )
        )


        new_prediction = pd.DataFrame({

            "date": [
                prediction_date_str
            ],

            "store_id": [
                store_id
            ],

            "gate_id": [
                gate_id
            ],

            "predicted": [
                round(
                    prediction,
                    2
                )
            ]
        })


        if PREDICTION_LOG.exists():

            old = pd.read_csv(
                PREDICTION_LOG,
                usecols=[
                    "date",
                    "store_id",
                    "gate_id",
                    "predicted"
                ]
            )


            old["date"] = (
                old["date"].astype(str)
            )


            old["store_id"] = pd.to_numeric(
                old["store_id"],
                errors="coerce"
            )


            old["gate_id"] = pd.to_numeric(
                old["gate_id"],
                errors="coerce"
            )


            mask = (

                (
                    old["date"]
                    == prediction_date_str
                )

                &

                (
                    old["store_id"]
                    == store_id
                )

                &

                (
                    old["gate_id"]
                    == gate_id
                )
            )


            if mask.any():

                old.loc[
                    mask,
                    "predicted"
                ] = round(
                    prediction,
                    2
                )

            else:

                old = pd.concat(
                    [
                        old,
                        new_prediction
                    ],
                    ignore_index=True
                )


            old.to_csv(
                PREDICTION_LOG,
                index=False
            )


        else:

            new_prediction.to_csv(
                PREDICTION_LOG,
                index=False
            )

        try:
            django_pred_path = BASE_DIR / "django_project" / "prediction_log.csv"
            if django_pred_path.parent.exists():
                shutil.copy(PREDICTION_LOG, django_pred_path)
        except Exception:
            pass


        # =================================================
        # HOLIDAY
        # =================================================

        selected_timestamp = pd.Timestamp(
            selected_date
        )


        if (
            selected_timestamp.date()
            in india_holidays
        ):

            holiday_name = (
                india_holidays.get(
                    selected_timestamp.date()
                )
            )


            st.warning(
                f"🎉 {holiday_name}"
            )


        # =================================================
        # 7 DAY FORECAST
        # =================================================

        future = make_7_day_forecast(

            df,
            store_id,
            gate_id,
            selected_date
        )


        future["holiday"] = (
            future["date"].apply(
                lambda x:
                int(
                    x.date()
                    in india_holidays
                )
            )
        )


        st.markdown("---")


        st.subheader(
            "📅 Next 7 Days Prediction"
        )


        display_table = future[
            [
                "date",
                "Predicted_Footfall"
            ]
        ].copy()


        display_table["date"] = (
            display_table["date"]
            .dt.strftime(
                "%d-%m-%Y"
            )
        )


        display_table[
            "Predicted_Footfall"
        ] = (
            display_table[
                "Predicted_Footfall"
            ].round(2)
        )


        st.dataframe(
            display_table,
            use_container_width=True
        )


        # =================================================
        # CHART
        # =================================================

        st.subheader(
            "📊 Footfall Forecast"
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.plot(
            future["date"],
            future[
                "Predicted_Footfall"
            ],
            marker="o",
            linewidth=3
        )


        ax.fill_between(
            future["date"],
            future[
                "Predicted_Footfall"
            ],
            alpha=0.20
        )


        ax.set_xlabel(
            "Date"
        )


        ax.set_ylabel(
            "Predicted Footfall"
        )


        ax.set_title(
            "Next 7 Days Forecast"
        )


        ax.grid(
            True
        )


        plt.xticks(
            rotation=45
        )


        fig.tight_layout()


        st.pyplot(
            fig
        )


        plt.close(fig)


        # =================================================
        # HIGH / LOW
        # =================================================

        highest = future.loc[
            future[
                "Predicted_Footfall"
            ].idxmax()
        ]


        lowest = future.loc[
            future[
                "Predicted_Footfall"
            ].idxmin()
        ]


        c1, c2 = st.columns(2)


        with c1:

            st.success(
                f"""
                Highest Footfall:

                **{highest['Predicted_Footfall']:.2f}**

                {highest['date'].strftime('%d-%m-%Y')}
                """
            )


        with c2:

            st.info(
                f"""
                Lowest Footfall:

                **{lowest['Predicted_Footfall']:.2f}**

                {lowest['date'].strftime('%d-%m-%Y')}
                """
            )


        # =================================================
        # DOWNLOAD
        # =================================================

        csv = future[
            [
                "date",
                "Predicted_Footfall",
                "Model"
            ]
        ].copy()


        csv["date"] = (
            csv["date"].astype(str)
        )


        st.download_button(

            "📥 Download Prediction CSV",

            data=csv.to_csv(
                index=False
            ),

            file_name=(
                "footfall_prediction.csv"
            ),

            mime="text/csv"
        )


        # =================================================
        # HOURWISE PREDICTION
        # =================================================

        st.markdown("---")

        st.subheader("⏰ Hourwise Footfall Prediction")

        forecast_dates_str = future["date"].dt.strftime("%d-%m-%Y").tolist()

        selected_date_fmt = pd.Timestamp(selected_date).strftime("%d-%m-%Y")

        default_idx = forecast_dates_str.index(selected_date_fmt) if selected_date_fmt in forecast_dates_str else 0

        col_h1, col_h2 = st.columns([2, 1])

        with col_h1:

            chosen_date_fmt = st.selectbox(
                "Select Date for Hourwise Breakdown:",
                options=forecast_dates_str,
                index=default_idx,
                key="hourly_date_select"
            )

        chosen_row = future[future["date"].dt.strftime("%d-%m-%Y") == chosen_date_fmt]

        if not chosen_row.empty:

            chosen_daily_pred = chosen_row.iloc[0]["Predicted_Footfall"]

            chosen_target_date = chosen_row.iloc[0]["date"]

        else:

            chosen_daily_pred = prediction

            chosen_target_date = selected_date

        hourly_df = predict_hourwise(
            chosen_daily_pred,
            store_id,
            gate_id,
            chosen_target_date,
            hourly_dist_data
        )

        if not hourly_df.empty and len(hourly_df) > 0:

            peak_idx = hourly_df["Predicted Footfall"].idxmax()

            peak_row = hourly_df.loc[peak_idx]

            lowest_idx = hourly_df["Predicted Footfall"].idxmin()

            lowest_row = hourly_df.loc[lowest_idx]

            morning_val = hourly_df.iloc[0:5]["Predicted Footfall"].sum()

            afternoon_val = hourly_df.iloc[5:10]["Predicted Footfall"].sum()

            evening_val = hourly_df.iloc[10:15]["Predicted Footfall"].sum()

            night_val = hourly_df.iloc[15:]["Predicted Footfall"].sum()

            m1, m2, m3, m4 = st.columns(4)

            with m1:

                st.metric(
                    "🔥 Peak Hour",
                    f"{peak_row['Time Slot']}",
                    f"{peak_row['Predicted Footfall']:.0f} visitors"
                )

            with m2:

                st.metric(
                    "🌅 Morning (07-12)",
                    f"{morning_val:.0f}",
                    f"{(morning_val / chosen_daily_pred * 100):.1f}%"
                )

            with m3:

                st.metric(
                    "☀️ Afternoon (12-17)",
                    f"{afternoon_val:.0f}",
                    f"{(afternoon_val / chosen_daily_pred * 100):.1f}%"
                )

            with m4:

                st.metric(
                    "🌆 Evening (17-22)",
                    f"{evening_val:.0f}",
                    f"{(evening_val / chosen_daily_pred * 100):.1f}%"
                )

            # Chart
            fig_h, ax_h = plt.subplots(figsize=(10, 4.5))

            bars = ax_h.bar(
                hourly_df["Time Slot"],
                hourly_df["Predicted Footfall"],
                color="#2b5c8f",
                edgecolor="none",
                alpha=0.85
            )

            bars[peak_idx].set_color("#e05d06")

            ax_h.set_title(
                f"Predicted Hourly Distribution ({chosen_date_fmt}) - Total: {chosen_daily_pred:.0f} visitors",
                fontsize=11,
                fontweight="bold"
            )

            ax_h.set_ylabel("Predicted Visitors")

            ax_h.set_xlabel("Time Slot")

            plt.xticks(rotation=45, ha="right")

            ax_h.grid(axis="y", linestyle="--", alpha=0.5)

            fig_h.tight_layout()

            st.pyplot(fig_h)

            plt.close(fig_h)

            with st.expander("📋 Detailed Hourly Breakdown Data Table"):

                st.dataframe(
                    hourly_df,
                    use_container_width=True
                )

            hourly_export = hourly_df.copy()

            hourly_export["Date"] = pd.Timestamp(chosen_target_date).strftime("%Y-%m-%d")

            hourly_export["Store_ID"] = store_id

            hourly_export["Gate_ID"] = gate_id

            st.download_button(
                "📥 Download Hourwise Prediction CSV",
                data=hourly_export.to_csv(index=False),
                file_name=f"hourly_footfall_store_{store_id}_gate_{gate_id}_{pd.Timestamp(chosen_target_date).strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


        # =================================================
        # ACTUAL FOOTFALL
        # =================================================

        st.markdown("---")


        st.subheader(
            "📝 Enter Actual Footfall"
        )


        actual_value = st.number_input(

            "Actual Footfall",

            min_value=0,

            step=1
        )


        if st.button(
            "💾 Save Actual Footfall"
        ):

            actual_data = {

                "date":
                    prediction_date_str,

                "store_id":
                    int(store_id),

                "gate_id":
                    int(gate_id),

                "actual":
                    int(actual_value)
            }


            try:

                # =================================================
                # SUPABASE
                # =================================================

                if supabase:

                    try:

                        supabase.table(
                            "actual_footfall"
                        ).upsert(

                            actual_data,

                            on_conflict=(
                                "date,store_id,gate_id"
                            )

                        ).execute()


                        st.success(
                            "✅ Actual Footfall Saved to Supabase!"
                        )


                    except Exception as e:

                        st.warning(
                            f"Could not save to Supabase: {e}"
                        )


                # =================================================
                # LOCAL BACKUP
                # =================================================

                new_actual = pd.DataFrame({

                    "date": [
                        prediction_date_str
                    ],

                    "store_id": [
                        store_id
                    ],

                    "gate_id": [
                        gate_id
                    ],

                    "actual": [
                        actual_value
                    ]
                })


                if ACTUAL_FILE.exists():

                    old_actual = pd.read_csv(
                        ACTUAL_FILE,
                        usecols=[
                            "date",
                            "store_id",
                            "gate_id",
                            "actual"
                        ]
                    )


                    old_actual["date"] = (
                        old_actual[
                            "date"
                        ].astype(str)
                    )


                    old_actual["store_id"] = (
                        pd.to_numeric(
                            old_actual[
                                "store_id"
                            ],
                            errors="coerce"
                        )
                    )


                    old_actual["gate_id"] = (
                        pd.to_numeric(
                            old_actual[
                                "gate_id"
                            ],
                            errors="coerce"
                        )
                    )


                    mask = (

                        (
                            old_actual["date"]
                            == prediction_date_str
                        )

                        &

                        (
                            old_actual["store_id"]
                            == store_id
                        )

                        &

                        (
                            old_actual["gate_id"]
                            == gate_id
                        )
                    )


                    if mask.any():

                        old_actual.loc[
                            mask,
                            "actual"
                        ] = actual_value

                    else:

                        old_actual = pd.concat(
                            [
                                old_actual,
                                new_actual
                            ],
                            ignore_index=True
                        )


                    old_actual.to_csv(
                        ACTUAL_FILE,
                        index=False
                    )


                else:

                    new_actual.to_csv(
                        ACTUAL_FILE,
                        index=False
                    )


                # =================================================
                # ERROR LOG
                # =================================================

                if (
                    PREDICTION_LOG.exists()
                    and
                    ACTUAL_FILE.exists()
                ):

                    pred = pd.read_csv(
                        PREDICTION_LOG
                    )


                    actual = pd.read_csv(
                        ACTUAL_FILE
                    )


                    error_df = pred.merge(

                        actual,

                        on=[
                            "date",
                            "store_id",
                            "gate_id"
                        ]
                    )


                    if not error_df.empty:

                        error_df[
                            "absolute_error"
                        ] = (

                            error_df[
                                "actual"
                            ]

                            -

                            error_df[
                                "predicted"
                            ]

                        ).abs()


                        error_df[
                            "error_percent"
                        ] = np.nan


                        positive = (
                            error_df[
                                "actual"
                            ] > 0
                        )


                        error_df.loc[
                            positive,
                            "error_percent"
                        ] = (

                            error_df.loc[
                                positive,
                                "absolute_error"
                            ]

                            /

                            error_df.loc[
                                positive,
                                "actual"
                            ]

                            *

                            100
                        )


                        error_df[
                            "error_percent"
                        ] = (

                            error_df[
                                "error_percent"
                            ].round(2)
                        )


                        error_df.to_csv(
                            ERROR_LOG,
                            index=False
                        )


                sync_actuals_to_dataset()
                trigger_background_retrain()

                st.success(
                    "✅ Actual Footfall Saved! Model background learning triggered."
                )


                st.rerun()


            except Exception as e:

                st.error(
                    f"❌ Failed to save actual footfall: {e}"
                )


        # =================================================
        # HISTORICAL DATA
        # =================================================

        with st.expander(
            "📂 View Historical Data"
        ):

            st.dataframe(
                history.tail(30),
                use_container_width=True
            )


            # =================================================
            # ERROR ANALYSIS
            # =================================================

            if ERROR_LOG.exists():

                try:

                    error_df = pd.read_csv(
                        ERROR_LOG
                    )


                    if (
                        "error_percent"
                        in error_df.columns
                        and
                        not error_df.empty
                    ):

                        st.subheader(
                            "📊 Prediction Error Analysis"
                        )


                        fig2, ax2 = plt.subplots(
                            figsize=(10, 5)
                        )


                        ax2.bar(

                            error_df[
                                "date"
                            ].astype(str),

                            error_df[
                                "error_percent"
                            ]
                        )


                        ax2.axhline(

                            y=25,

                            linestyle="--",

                            label="25% Threshold"
                        )


                        ax2.set_xlabel(
                            "Date"
                        )


                        ax2.set_ylabel(
                            "Error (%)"
                        )


                        ax2.set_title(
                            "Prediction Error"
                        )


                        plt.xticks(
                            rotation=45
                        )


                        ax2.legend()


                        fig2.tight_layout()


                        st.pyplot(
                            fig2
                        )


                        plt.close(fig2)


                except Exception as e:

                    st.warning(
                        f"Could not load error analysis: {e}"
                    )


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")


st.markdown(
    """
    <center>

    <p>📊 Footfall Prediction System</p>

    <p>
    Developed using
    <b>Python | XGBoost | Pandas | Streamlit</b>
    </p>

    </center>
    """,
    unsafe_allow_html=True
)