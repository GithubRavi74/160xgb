import streamlit as st
import pandas as pd
import numpy as np
import pickle

# using FEB 5 App.py
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="iPoultry AI – Growth Forecast",
    layout="wide"
)

st.title("🐔 iPoultry AI – Batch Growth Forecast")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ml_ready_daily.csv")
    df["farm_id"] = df["farm_id"].astype(str).str.strip()
    df["batch_id"] = df["batch_id"].astype(str).str.strip()
    return df

df = load_data()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

weight_model = load_model("weight_xgb_model.pkl")
mort_model   = load_model("mortality_xgb_model.pkl")
fcr_model    = load_model("fcr_xgb_model.pkl")

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
st.subheader("📥 Select Farm and Batch")

farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))

batch_id = st.selectbox(
    "Batch ID",
    sorted(df[df["farm_id"] == farm_id]["batch_id"].unique())
)

# -------------------------------------------------
# READINESS CHECK
# -------------------------------------------------
def compute_readiness(batch_df):

    total_days = batch_df["day_number"].max()
    available_days = batch_df["day_number"].nunique()

    coverage_score = min(available_days / total_days, 1.0)

    critical_cols = [
        "feed_today_kg", "birds_alive",
        "mortality_today", "temp", "rh"
    ]
    completeness = 1 - batch_df[critical_cols].isna().mean().mean()

    last_days = batch_df.sort_values("day_number").tail(3)
    recency_score = 1.0 if last_days.isna().sum().sum() == 0 else 0.5

    readiness = (
        0.4 * coverage_score +
        0.3 * completeness +
        0.2 * 1.0 +
        0.1 * recency_score
    ) * 100

    return readiness, available_days, total_days, completeness

# -------------------------------------------------
# ACTION BUTTON
# -------------------------------------------------
if st.button("📈 Forecast Next 7 Days"):

    batch_df = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].copy()

    if batch_df.empty:
        st.error("❌ No data found for this Farm & Batch")
        st.stop()

    readiness, available_days, total_days, completeness = compute_readiness(batch_df)

    # -------------------------------------------------
    # READINESS GATE
    # -------------------------------------------------
    st.subheader("📊 Data Readiness")

    st.metric("Readiness Score", f"{readiness:.1f} / 100")
    st.write(f"📅 Days available: {available_days} / {total_days}")
    st.write(f"🧾 Data completeness: {completeness*100:.1f}%")

    if readiness < 60:
        st.error("❌ Data quality too low for reliable prediction")
        st.stop()
    elif readiness < 80:
        st.warning("⚠️ Prediction possible, but confidence is medium")
    else:
        st.success("✅ Data quality excellent for prediction")

    # -------------------------------------------------
    # FEATURE RECONSTRUCTION
    # -------------------------------------------------
    batch_df = batch_df.sort_values("day_number")

    batch_df["mortality_rate"] = (
        batch_df["mortality_today"] / batch_df["birds_alive"]
    )

    batch_df["feed_per_bird"] = (
        batch_df["feed_today_kg"] / batch_df["birds_alive"]
    )

    batch_df["rolling_7d_feed"] = (
        batch_df["feed_today_kg"]
        .rolling(7, min_periods=3)
        .mean()
    )

    batch_df["rolling_7d_gain"] = (
        batch_df["sample_weight_kg"]
        .diff()
        .rolling(7, min_periods=3)
        .mean()
    )

    batch_df = batch_df.dropna()

    if batch_df.empty:
        st.error("❌ Not enough continuous data to compute rolling features")
        st.stop()

    # -------------------------------------------------
    # FORECAST LOOP
    # -------------------------------------------------
    horizon = 7
    last_row = batch_df.iloc[-1].copy()

    forecast_rows = []

    birds_alive = last_row["birds_alive"]

    for i in range(1, horizon + 1):

        features = {
            "day_number": last_row["day_number"] + i,
            "birds_alive": birds_alive,
            "feed_today_kg": last_row["feed_today_kg"],
            "feed_per_bird": last_row["feed_per_bird"],
            "mortality_today": last_row["mortality_today"],
            "mortality_rate": last_row["mortality_rate"],
            "rolling_7d_feed": last_row["rolling_7d_feed"],
            "rolling_7d_gain": last_row["rolling_7d_gain"],
            "temp": last_row["temp"],
            "rh": last_row["rh"],
            "co": last_row["co"],
            "nh": last_row["nh"],
        }

        X_pred = pd.DataFrame([features])

        pred_weight = weight_model.predict(X_pred)[0]
        pred_mort   = max(0, mort_model.predict(X_pred)[0])
        pred_fcr    = fcr_model.predict(X_pred)[0]

        birds_alive = max(0, birds_alive - int(round(pred_mort)))

        forecast_rows.append({
            "Day": features["day_number"],
            "Predicted Weight (kg)": round(pred_weight, 3),
            "Predicted Mortality": int(round(pred_mort)),
            "Birds Alive": birds_alive,
            "Predicted Feed / Bird": round(pred_fcr, 3)
        })

    forecast_df = pd.DataFrame(forecast_rows)

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.subheader("🔮 7-Day Growth Forecast")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.info(
        f"📌 Forecast confidence: "
        f"{'High' if readiness >= 80 else 'Medium'}  \n"
        f"Based on {available_days} days of historical batch data."
    )
