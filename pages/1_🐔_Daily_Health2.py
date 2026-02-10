import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Daily Health", layout="wide")
st.title("🐔 iPoultry AI – Daily Health Prediction")

# -------------------------------------------------
# LOAD DATA & MODELS
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

df = load_data()
gain_model = load_model("weight_gain_xgb_model.pkl")
mort_model = load_model("mortality_xgb_model.pkl")

# -------------------------------------------------
# CLEAN IDS
# -------------------------------------------------
df["farm_id"] = df["farm_id"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()

# -------------------------------------------------
# SELECT FARM & BATCH
# -------------------------------------------------
st.subheader("🏭 Select Farm & Batch")

farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))
batch_id = st.selectbox(
    "Batch ID",
    sorted(df[df["farm_id"] == farm_id]["batch_id"].unique())
)

batch_hist = df[
    (df["farm_id"] == farm_id) &
    (df["batch_id"] == batch_id)
].sort_values("day_number")

if batch_hist.empty:
    st.error("No historical data found for this batch")
    st.stop()

last = batch_hist.iloc[-1]

# Yesterday reference
y_mort_rate = last["mortality_today"] / max(last["birds_alive"], 1)
y_gain = last["daily_weight_gain_kg"]


# -------------------------------------------------
# AUTO CONTEXT (SYSTEM ONLY)
# -------------------------------------------------
current_day = int(last["day_number"]) + 1
prev_weight = float(last["sample_weight_kg"])
birds_alive_yesterday = int(last["birds_alive"])

rolling_feed = batch_hist.tail(7)["feed_today_kg"].mean()
rolling_gain = batch_hist.tail(7)["daily_weight_gain_kg"].mean()

# fallback if early days
rolling_feed = rolling_feed if not np.isnan(rolling_feed) else last["feed_today_kg"]
rolling_gain = rolling_gain if not np.isnan(rolling_gain) else 0.05

# -------------------------------------------------
# FARMER INPUTS (TODAY)
# -------------------------------------------------
st.subheader("📥 Enter Today’s Data")

col1, col2 = st.columns(2)

with col1:
    birds_alive = st.number_input(
        "🐔 Birds Alive Today",
        min_value=1,
        value=birds_alive_yesterday
    )
    mortality_today = st.number_input(
        "☠ Mortality Today",
        min_value=0
    )
    feed_today = st.number_input(
        "🌾 Feed Given Today (kg)",
        min_value=0.0
    )

with col2:
    sample_weight = st.number_input(
        "⚖ Sample Avg Weight (kg) (optional)",
        min_value=0.0,
        value=0.0
    )
    temp = st.number_input("🌡 Temperature (°C)", value=float(last["temp"]))
    rh = st.number_input("💧 Humidity (%)", value=float(last["rh"]))
    nh = st.number_input("🧪 Ammonia (ppm)", value=float(last["nh"]))
    co = st.number_input("🧪 CO₂ (ppm)", value=float(last["co"]))

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔮 Predict Today"):

    feed_per_bird = feed_today / max(birds_alive, 1)
    mortality_rate = mortality_today / max(birds_alive, 1)

    X = pd.DataFrame([{
        "day_number": current_day,
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

    gain_pred = max(0, gain_model.predict(X)[0])
    mort_pred = max(0, int(mort_model.predict(X)[0]))

    # --- Trend arrows ---
    mort_trend = "⬆️" if mortality_rate > y_mort_rate else "⬇️"
    gain_trend = "⬆️" if gain_pred > y_gain else "⬇️"


    # Derived FCR
    if gain_pred > 0:
        fcr = feed_today / (gain_pred * birds_alive)
    else:
        fcr = np.nan

    # -------------------------------------------------
    # HEALTH SCORING (EXPLAINABLE)
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
    st.subheader("📊 AI Assessment for Today")

    st.metric("Health Status", status)
    st.metric("Expected Daily Gain (kg)", round(gain_pred, 3))
    st.metric("Expected Mortality Tomorrow", mort_pred)

    if not np.isnan(fcr):
        st.metric("Derived FCR", round(fcr, 2))

    st.subheader("⚠ Alerts & Actions")

    if temp > 34:
        st.warning("High temperature — check ventilation / cooling")
    if nh > 25:
        st.warning("High ammonia — increase air exchange")
    if mortality_rate > 0.005:
        st.warning("Mortality above normal — inspect flock health")

    if health_score >= 75:
        st.success("Flock condition appears stable 👍")
