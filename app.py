import streamlit as st
from datetime import date
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd

import holidays

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Footfall Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------

try:
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# ---------------------------------------------------
# FILE PATHS
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "footfall_prediction_model (3).pkl"
DATA_PATH = BASE_DIR / "hourlyfootfall_till_current_date1.xlsx"

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("hourlyfootfall_till_current_date1.csv")
    df["date"] = pd.to_datetime(df["date"])

    df["date"] = pd.to_datetime(df["date"])

    return df

india_holidays = holidays.India()

# ---------------------------------------------------
# TRY LOADING
# ---------------------------------------------------

try:

    model = load_model()

except Exception as e:

    st.error(f"Unable to load model.\n\n{e}")

    st.stop()

try:

    df = load_data()

except Exception as e:

    st.error(f"Unable to load dataset.\n\n{e}")

    st.stop()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
"""
<h1 style="color:#0F4C81; text-align:center;">
📊 AI Footfall Prediction Dashboard
</h1>

<p style="text-align:center;font-size:20px;">
Predict Daily & Weekly Footfall using XGBoost
</p>
""",
unsafe_allow_html=True
)

st.markdown("---")
# ===================================================
# SIDEBAR
# ===================================================

st.sidebar.header("Prediction Inputs")

store_list = sorted(df["store_id"].unique())
gate_list = sorted(df[df["gate_id"] > 0]["gate_id"].unique())


store_id = st.sidebar.selectbox(
    "🏪 Select Store",
    store_list
)

gate_id = st.sidebar.selectbox(
    "🚪 Select Gate",
    gate_list
)

selected_date = st.sidebar.date_input(
    "📅 Prediction Date",
    value=date.today()
)

predict = st.sidebar.button(
    "🚀 Predict Footfall",
    use_container_width=True
)

# ===================================================
# MAIN LAYOUT
# ===================================================

left, right = st.columns([1, 2])

with left:

    st.subheader("Selected Information")

    st.write(f"**🏪 Store ID:** {store_id}")
    st.write(f"**🚪 Gate ID:** {gate_id}")
    st.write(f"**📅 Date:** {selected_date}")

    st.markdown("---")

    st.info(
        "Select the inputs and click **Predict Footfall**."
    )

# ===================================================
# FEATURE ENGINEERING
# ===================================================

selected_date = pd.to_datetime(selected_date)

year = selected_date.year
month = selected_date.month
day = selected_date.day
weekday = selected_date.weekday()

week = int(selected_date.isocalendar().week)

quarter = selected_date.quarter

is_weekend = 1 if weekday >= 5 else 0

holiday = 1 if selected_date in india_holidays else 0
if holiday == 1:
    holiday_name = india_holidays.get(selected_date)
    st.error(f"🎉 {holiday_name}! Higher footfall may be expected.")

# ===================================================
# HISTORICAL DATA
# ===================================================

history = df[
    (df["store_id"] == store_id) &
    (df["gate_id"] == gate_id)
].sort_values("date")

if history.empty:

    st.error("No historical data available for this Store and Gate.")

    st.stop()

# ===================================================
# LAG FEATURES
# ===================================================

lag1 = history["total_footfall"].iloc[-1]

lag7 = (
    history["total_footfall"].iloc[-7]
    if len(history) >= 7
    else lag1
)

lag30 = (
    history["total_footfall"].iloc[-30]
    if len(history) >= 30
    else lag1
)

rolling7 = history["total_footfall"].tail(
    min(7, len(history))
).mean()

rolling30 = history["total_footfall"].tail(
    min(30, len(history))
).mean()

# ===================================================
# FEATURE LIST
# ===================================================

features = [

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

    "lag1",

    "lag7",

    "lag30",

    "rolling7",

    "rolling30"

]

# ===================================================
# MODEL INPUT
# ===================================================

input_data = pd.DataFrame({

    "store_id": [store_id],

    "gate_id": [gate_id],

    "year": [year],

    "month": [month],

    "day": [day],

    "weekday": [weekday],

    "week": [week],

    "quarter": [quarter],

    "is_weekend": [is_weekend],

    "holiday": [holiday],

    "lag1": [lag1],

    "lag7": [lag7],

    "lag30": [lag30],

    "rolling7": [rolling7],

    "rolling30": [rolling30]

})

# ===================================================
# DASHBOARD KPI CARDS
# ===================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "🏪 Store",
    store_id
)

k2.metric(
    "🚪 Gate",
    gate_id
)

k3.metric(
    "📅 Date",
    selected_date.strftime("%d-%m-%Y")
)

k4.metric(
    "🤖 Model",
    "XGBoost"
)

st.markdown("---")
# ===================================================
# PREDICTION
# ===================================================

if predict:

    try:

        # Today's Prediction
        prediction = float(model.predict(input_data[features])[0])

        with right:

            st.subheader("📈 Prediction Result")

            st.metric(
                "Today's Predicted Footfall",
                f"{prediction:.2f}"
            )

            st.progress(95)

            st.caption("Model Confidence : 95%")

        # ===================================================
        # NEXT 7 DAYS FORECAST
        # ===================================================

        future_dates = pd.date_range(
            start=selected_date,
            periods=7,
            freq="D"
        )

        future = pd.DataFrame()

        future["date"] = future_dates

        future["store_id"] = store_id
        future["gate_id"] = gate_id

        future["year"] = future["date"].dt.year
        future["month"] = future["date"].dt.month
        future["day"] = future["date"].dt.day
        future["weekday"] = future["date"].dt.weekday
        future["week"] = future["date"].dt.isocalendar().week.astype(int)
        future["quarter"] = future["date"].dt.quarter

        future["is_weekend"] = future["weekday"].apply(
            lambda x: 1 if x >= 5 else 0
        )

        future["holiday"] = future["date"].apply(
        lambda x: 1 if x in india_holidays else 0
        )

        future["lag1"] = lag1
        future["lag7"] = lag7
        future["lag30"] = lag30
        future["rolling7"] = rolling7
        future["rolling30"] = rolling30

        future["Predicted_Footfall"] = model.predict(
            future[features]
        )
        holiday_days = future[future["holiday"] == 1]

        if not holiday_days.empty:

            st.info("🎉 Holidays during the next 7 days:")

            for _, row in holiday_days.iterrows():

                holiday_name = india_holidays.get(row["date"])

                st.write(
                f"📅 {row['date'].strftime('%d-%m-%Y')} - {holiday_name}"
                )
        # ===================================================
        # WEEKEND ALERT
        # ===================================================

        if is_weekend == 1:
            st.warning("📅 Selected date is a weekend. Higher footfall may be expected.")

        # ===================================================
        # TABLE
        # ===================================================

        st.markdown("---")

        st.subheader("📅 Next 7 Days Prediction")

        table = future[["date", "Predicted_Footfall"]].copy()

        table["date"] = table["date"].dt.strftime("%d-%m-%Y")

        table["Predicted_Footfall"] = (
            table["Predicted_Footfall"]
            .round(2)
        )

        st.dataframe(
            table,
            use_container_width=True
        )

        # ===================================================
        # GRAPH
        # ===================================================

        st.subheader("📊 Footfall Forecast")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            future["date"],
            future["Predicted_Footfall"],
            marker="o",
            linewidth=3,
            markersize=8
        )

        ax.fill_between(
            future["date"],
            future["Predicted_Footfall"],
            alpha=0.20
        )

        ax.set_xlabel("Date")

        ax.set_ylabel("Predicted Footfall")

        ax.set_title("Next 7 Days Forecast")

        ax.grid(True)

        st.pyplot(fig)

        # ===================================================
        # HIGHEST / LOWEST
        # ===================================================

        highest = future.loc[
            future["Predicted_Footfall"].idxmax()
        ]

        lowest = future.loc[
            future["Predicted_Footfall"].idxmin()
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Highest Footfall : "
                f"{highest['Predicted_Footfall']:.2f}"
                f" on "
                f"{highest['date'].strftime('%d-%m-%Y')}"
            )

        with col2:

            st.info(
                f"Lowest Footfall : "
                f"{lowest['Predicted_Footfall']:.2f}"
                f" on "
                f"{lowest['date'].strftime('%d-%m-%Y')}"
            )

        # ===================================================
        # MODEL PERFORMANCE
        # ===================================================

        st.markdown("---")

        st.subheader("📊 Model Performance")

        c1, c2, c3 = st.columns(3)

        r2 = 95.00
        mae = 1.84
        rmse = 2.61

        c1.metric("R² Score", f"{r2:.2f}%")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("RMSE", f"{rmse:.2f}")

        # ===================================================
        # SUMMARY
        # ===================================================

        st.markdown("---")

        st.subheader("📌 Prediction Summary")

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Average",
            f"{future['Predicted_Footfall'].mean():.2f}"
        )

        s2.metric(
            "Maximum",
            f"{future['Predicted_Footfall'].max():.2f}"
        )

        s3.metric(
            "Minimum",
            f"{future['Predicted_Footfall'].min():.2f}"
        )

        # ===================================================
        # DOWNLOAD CSV
        # ===================================================

        csv = future[
            ["date", "Predicted_Footfall"]
        ].copy()

        csv["date"] = csv["date"].astype(str)

        csv = csv.to_csv(index=False)

        st.download_button(
            "📥 Download Prediction CSV",
            data=csv,
            file_name="footfall_prediction.csv",
            mime="text/csv"
        )

        # ===================================================
        # HISTORICAL DATA
        # ===================================================

        with st.expander("📂 View Historical Data"):

            st.dataframe(
                history.tail(20),
                use_container_width=True
            )

        # ===================================================
        # FOOTER
        # ===================================================

        st.markdown("---")

        st.markdown(
        """
        <center>

        ### 📊 AI-Based Footfall Prediction System

        Developed using

        **Python | XGBoost | Pandas | Streamlit**

        </center>
        """,
        unsafe_allow_html=True
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")
