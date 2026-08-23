from supabase import create_client
import streamlit as st
from datetime import date
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import holidays
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Footfall Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)


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
        except BaseException:
            pass
except BaseException:
    pass

if not SUPABASE_URL:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
if not SUPABASE_KEY:
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except BaseException as e:
        print("Could not connect to Supabase:", e)


# =========================================================
# STYLE
# =========================================================

try:
    with open(BASE_DIR / "style.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except Exception:
    pass


# =========================================================
# FILE PATHS
# =========================================================

def find_file(filename):
    paths = [
        BASE_DIR / filename,
        BASE_DIR / "models_v3_1" / filename,
        BASE_DIR.parent / filename,
    ]
    for p in paths:
        if p.exists():
            return p
    return BASE_DIR / filename

DATA_PATH = find_file("hourlyfootfall_till_current_date1.csv")

STAGE1_MODEL_PATH = find_file("stage1_low_classifier.pkl")
LOW_MODEL_PATH = find_file("low_footfall_model.pkl")
NORMAL_MODEL_PATH = find_file("normal_footfall_model.pkl")

# V3.1 metadata
PREDICTION_CONFIG_PATH = find_file("prediction_config.pkl")
MODEL_INFO_PATH = find_file("model_info.pkl")

PREDICTION_LOG = find_file("prediction_log.csv")
ACTUAL_FILE = find_file("actual_footfall.csv")
ERROR_LOG = find_file("error_log.csv")


# =========================================================
# EXACT FEATURES USED BY V3.1 MODELS
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
# LOAD V3.1 METADATA
# =========================================================

def recursively_find_value(obj, keys_to_find):
    """
    Safely search nested dictionaries/objects for useful
    configuration information.

    This is only used to detect whether the target was
    log-transformed during V3.1 training.
    """

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


    return prediction_config, model_info


prediction_config, model_info = load_metadata()


# =========================================================
# DETECT TARGET TRANSFORMATION
# =========================================================

def detect_target_transform(
    prediction_config,
    model_info
):
    """
    Detect how V3.1 target was transformed.

    Expected possibilities:

        log1p
        log
        raw

    V3.1 log-target models normally require:

        expm1(prediction)

    before displaying footfall.
    """

    objects = [
        prediction_config,
        model_info
    ]

    # -----------------------------------------------------
    # Search for explicit target-transform information
    # -----------------------------------------------------

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

        # Boolean information
        if isinstance(value, bool):

            if value:
                return "log1p"

            return "raw"

        value_string = str(
            value
        ).lower().strip()

        # log1p
        if (
            "log1p"
            in value_string
            or
            "log(1+x)"
            in value_string
            or
            "log_plus_one"
            in value_string
        ):

            return "log1p"

        # natural log
        if (
            value_string == "log"
            or
            "natural_log"
            in value_string
            or
            value_string == "ln"
        ):

            return "log"

        # raw
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


    # -----------------------------------------------------
    # Search text representation of metadata
    # -----------------------------------------------------

    for obj in objects:

        if obj is None:
            continue

        try:

            text = str(obj).lower()

            if (
                "log1p"
                in text
                or
                "log(1+x)"
                in text
                or
                "log_plus_one"
                in text
            ):

                return "log1p"

        except Exception:
            pass


    # -----------------------------------------------------
    # V3.1 fallback
    #
    # Your V3.1 models are log-target models.
    # Therefore default to log1p if metadata does not
    # explicitly say otherwise.
    # -----------------------------------------------------

    return "log1p"


TARGET_TRANSFORM = detect_target_transform(
    prediction_config,
    model_info
)


# =========================================================
# CONVERT MODEL OUTPUT TO REAL FOOTFALL
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
            f"Model returned invalid prediction: "
            f"{raw_prediction}"
        )


    # -----------------------------------------------------
    # LOG1P TARGET
    # -----------------------------------------------------

    if TARGET_TRANSFORM == "log1p":

        # Prevent overflow in extremely unusual cases.
        raw_prediction = np.clip(
            raw_prediction,
            -20,
            20
        )

        prediction = np.expm1(
            raw_prediction
        )


    # -----------------------------------------------------
    # NATURAL LOG TARGET
    # -----------------------------------------------------

    elif TARGET_TRANSFORM == "log":

        raw_prediction = np.clip(
            raw_prediction,
            -20,
            20
        )

        prediction = np.exp(
            raw_prediction
        )


    # -----------------------------------------------------
    # RAW TARGET
    # -----------------------------------------------------

    else:

        prediction = raw_prediction


    # Footfall cannot be negative.

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

        {e}

        Make sure these files are present beside app.py:

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


# =========================================================
# CHECK FEATURES
# =========================================================

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

    st.write(
        "Dashboard features:",
        FEATURES
    )

    if stage1_features:

        st.write(
            "Stage 1 features:",
            stage1_features
        )

    if low_features:

        st.write(
            "Low model features:",
            low_features
        )

    if normal_features:

        st.write(
            "Normal model features:",
            normal_features
        )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH
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


try:

    df = load_data()

except Exception as e:

    st.error(
        f"❌ Unable to load dataset: {e}"
    )

    st.stop()


india_holidays = holidays.India()


# =========================================================
# CREATE MODEL FEATURES
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
    # CALENDAR FEATURES
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
        .isin(
            india_holidays
        )
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
# BUILD FEATURES
# =========================================================

@st.cache_data
def get_feature_data(data):

    return create_features(
        data
    )


feature_df = get_feature_data(
    df
)


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


    # -----------------------------------------------------
    # ONLY DATA BEFORE TARGET DATE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CREATE TEMP DATASET
    # -----------------------------------------------------

    temp = history.copy()


    future_row = pd.DataFrame({

        "date":
            [target_date],

        "store_id":
            [store],

        "gate_id":
            [gate],

        "total_footfall":
            [np.nan]

    })


    temp = pd.concat(
        [
            temp,
            future_row
        ],
        ignore_index=True
    )


    # -----------------------------------------------------
    # CREATE FEATURES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # SELECT EXACT MODEL FEATURES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # FALLBACK VALUES
    # -----------------------------------------------------

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
                ] = history[
                    "total_footfall"
                ].iloc[-1]

            else:

                X.loc[
                    X.index[0],
                    col
                ] = history[
                    "total_footfall"
                ].mean()


    # -----------------------------------------------------
    # TREND FALLBACK
    # -----------------------------------------------------

    if pd.isna(
        X.iloc[0]["trend"]
    ):

        X.loc[
            X.index[0],
            "trend"
        ] = 1.0


    # -----------------------------------------------------
    # FINAL CLEANUP
    # -----------------------------------------------------

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


    # =====================================================
    # HANDLE CLASS LABEL
    # =====================================================

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
    # RAW MODEL PREDICTION
    # =====================================================

    raw_prediction = selected_model.predict(
        X
    )[0]


    # =====================================================
    # IMPORTANT:
    #
    # V3.1 MODEL USES LOG TARGET
    #
    # raw prediction could be:
    #
    # 7.7
    #
    # But actual footfall is:
    #
    # expm1(7.7)
    #
    # =====================================================

    prediction = convert_prediction_to_footfall(
        raw_prediction
    )


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

    working = base_data.copy()

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


        # -------------------------------------------------
        # Add REAL-SCALE prediction to working data.
        #
        # This is extremely important.
        #
        # Future recursive lags must use actual footfall
        # scale, NOT the log prediction.
        # -------------------------------------------------

        new_row = pd.DataFrame({

            "date":
                [target],

            "store_id":
                [store],

            "gate_id":
                [gate],

            "total_footfall":
                [prediction]

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
        # SHOW FEATURES FOR DEBUGGING
        # =================================================

        


        


        # =================================================
        # SESSION
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

            "date":
                [prediction_date_str],

            "store_id":
                [store_id],

            "gate_id":
                [gate_id],

            "predicted":
                [round(
                    prediction,
                    2
                )]

        })


        if PREDICTION_LOG.exists():

            old = pd.read_csv(
                PREDICTION_LOG
            )


            old["date"] = (
                old["date"]
                .astype(str)
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
            future["date"]
            .apply(
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
                "Predicted_Footfall",
                "Model"
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


        st.pyplot(
            fig
        )


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
            csv["date"]
            .astype(str)
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
                        st.warning(f"Could not save to Supabase: {e}")


                # -----------------------------------------
                # LOCAL BACKUP
                # -----------------------------------------

                new_actual = pd.DataFrame({

                    "date":
                        [prediction_date_str],

                    "store_id":
                        [store_id],

                    "gate_id":
                        [gate_id],

                    "actual":
                        [actual_value]

                })


                if ACTUAL_FILE.exists():

                    old_actual = pd.read_csv(
                        ACTUAL_FILE
                    )


                    old_actual["date"] = (
                        old_actual["date"]
                        .astype(str)
                    )


                    old_actual["store_id"] = pd.to_numeric(
                        old_actual["store_id"],
                        errors="coerce"
                    )


                    old_actual["gate_id"] = pd.to_numeric(
                        old_actual["gate_id"],
                        errors="coerce"
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


                # -----------------------------------------
                # ERROR LOG
                # -----------------------------------------

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


            if ERROR_LOG.exists():

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


                    fig, ax = plt.subplots(
                        figsize=(10, 5)
                    )


                    ax.bar(

                        error_df[
                            "date"
                        ].astype(str),

                        error_df[
                            "error_percent"
                        ]

                    )


                    ax.axhline(

                        y=25,

                        linestyle="--",

                        label="25% Threshold"

                    )


                    ax.set_xlabel(
                        "Date"
                    )


                    ax.set_ylabel(
                        "Error (%)"
                    )


                    ax.set_title(
                        "Prediction Error"
                    )


                    plt.xticks(
                        rotation=45
                    )


                    ax.legend()


                    st.pyplot(
                        fig
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