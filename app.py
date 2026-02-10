import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

gain_model = load_model("weight_gain_xgb_model.pkl")
mort_model = load_model("mortality_xgb_model.pkl")

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🐔 iPoultry AI – Daily Health Prediction")

st.subheader("📥 Enter Today’s Farm Data")

col1, col2 = st.columns(2)

with col1:
    day_number = st.number_input("📅 Day of Cycle", min_value=1, max_value=60)
    birds_alive = st.number_input("🐔 Birds Alive Today", min_value=1)
    mortality_today = st.number_input("☠ Mortality Today", min_value=0)
    feed_today = st.number_input("🌾 Feed Given Today (kg)", min_value=0.0)

with col2:
    sample_weight = st.number_input(
        "⚖ Sample Avg Weight (kg) (optional)",
        min_value=0.0,
        value=0.0
    )
    temp = st.number_input("🌡 Temperature (°C)", value=30.0)
    rh = st.number_input("💧 Humidity (%)", value=65.0)
    nh = st.number_input("🧪 Ammonia (ppm)", value=15.0)
    co = st.number_input("🧪 CO₂ (ppm)", value=2500.0)

# -------------------------------------------------
# PREDICT
# -------------------------------------------------
if st.button("🔮 Predict Today"):

    feed_per_bird = feed_today / max(birds_alive, 1)
    mortality_rate = mortality_today / max(birds_alive, 1)

    # Minimal rolling placeholders (V1)
    rolling_feed = feed_today
    rolling_gain = 0.05  # fallback if no weight history

    X = pd.DataFrame([{
        "day_number": day_number,
        "birds_alive": birds_alive,
        "feed_today_kg": feed_today,
        "feed_per_bird": feed_per_bird,
        "mortality_rate": mortality_rate,
        "rolling_7d_feed": rolling_feed,
        "rolling_7d_gain": rolling_gain,
        "temp": temp,
        "rh": rh,
        "co": co,
        "nh": nh
    }])

    X = X[gain_model.get_booster().feature_names]

    # --- Predictions ---
    gain_pred = max(0, gain_model.predict(X)[0])
    mort_pred = max(0, int(mort_model.predict(X)[0]))

    # Derived FCR
    if gain_pred > 0:
        fcr = feed_today / (gain_pred * birds_alive)
    else:
        fcr = np.nan

    # -------------------------------------------------
    # HEALTH LOGIC
    # -------------------------------------------------
    health_score = 100

    if mortality_rate > 0.005:
        health_score -= 30
    if temp > 34 or temp < 26:
        health_score -= 20
    if nh > 25:
        health_score -= 20
    if fcr > 2.2:
        health_score -= 15

    if health_score >= 75:
        status = "🟢 Normal"
    elif health_score >= 50:
        status = "🟡 Watch"
    else:
        status = "🔴 Risk"

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.subheader("📊 Today’s AI Assessment")

    st.metric("Health Status", status)
    st.metric("Expected Daily Gain (kg)", round(gain_pred, 3))
    st.metric("Expected Mortality Tomorrow", mort_pred)

    if not np.isnan(fcr):
        st.metric("Feed Efficiency (FCR)", round(fcr, 2))

    # Alerts
    st.subheader("⚠ Alerts & Suggestions")

    if temp > 34:
        st.warning("High temperature — check ventilation & cooling")
    if nh > 25:
        st.warning("High ammonia — increase air exchange")
    if mortality_rate > 0.005:
        st.warning("Mortality above normal — inspect flock closely")

    if health_score >= 75:
        st.success("Flock condition appears stable 👍")
