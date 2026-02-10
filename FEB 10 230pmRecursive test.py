import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="iPoultry AI – Recursive Forecast",
    layout="wide"
)
st.title("📈 iPoultry AI – Recursive Growth Forecast & Validation")

# -------------------------------------------------
# LOAD DATA + MODELS
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

df = load_data()

# ✅ ONLY LOAD MODELS THAT SHOULD EXIST
gain_model = load_model("weight_gain_xgb_model.pkl")
mort_model = load_model("mortality_xgb_model.pkl")

# -------------------------------------------------
# CLEAN IDS
# -------------------------------------------------
df["farm_id"] = df["farm_id"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
st.subheader("🏭 Select Farm & Batch")

farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))
batch_id = st.selectbox(
    "Batch ID",
    sorted(df[df["farm_id"] == farm_id]["batch_id"].unique())
)

# -------------------------------------------------
# RUN FORECAST
# -------------------------------------------------
if st.button("📈 Run Recursive Forecast"):

    batch_full = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].sort_values("day_number")

    if batch_full.empty:
        st.error("No data found for selected Farm & Batch")
        st.stop()

    st.subheader("📋 Full Batch Data")
    st.dataframe(batch_full, use_container_width=True)

    max_day = int(batch_full["day_number"].max())

    cutoff_day = st.slider(
        "Select Cutoff Day",
        min_value=5,
        max_value=max_day - 1,
        value=max_day - 7
    )

    history = batch_full[batch_full["day_number"] <= cutoff_day]
    future_actual = batch_full[
        (batch_full["day_number"] > cutoff_day) &
        (batch_full["day_number"] <= cutoff_day + 7)
    ]

    last = history.iloc[-1]

    # -------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------
    current_day = int(last["day_number"])
    current_weight = float(last["sample_weight_kg"])
    current_birds = int(last["birds_alive"])
    feed_today = float(last["feed_today_kg"])

    mortality_today = 0  # predicted recursively

    temp, rh, co, nh = last["temp"], last["rh"], last["co"], last["nh"]

    rolling_feed = history.tail(7)["feed_today_kg"].mean()
    rolling_gain = history.tail(7)["daily_weight_gain_kg"].mean()

    # -------------------------------------------------
    # RECURSIVE SIMULATION
    # -------------------------------------------------
    horizon = 7
    predictions = []

    for step in range(1, horizon + 1):

        day = current_day + step

        X = pd.DataFrame([{
            "day_number": day,
            "birds_alive": current_birds,
            "feed_today_kg": feed_today,
            "feed_per_bird": feed_today / max(current_birds, 1),
            "mortality_rate": mortality_today / max(current_birds, 1),
            "rolling_7d_feed": rolling_feed,
            "rolling_7d_gain": rolling_gain,
            "temp": temp,
            "rh": rh,
            "co": co,
            "nh": nh
        }])

        X = X[gain_model.get_booster().feature_names]

        # --- MODEL PREDICTIONS ---
        gain_pred = max(0, gain_model.predict(X)[0])
        mort_pred = max(0, int(mort_model.predict(X)[0]))

        # --- BIOLOGICAL UPDATE ---
        current_weight += gain_pred
        current_birds = max(1, current_birds - mort_pred)
        mortality_today = mort_pred

        rolling_gain = (rolling_gain * 6 + gain_pred) / 7

        # ✅ DERIVED FCR (NO MODEL)
        if gain_pred > 0 and current_birds > 0:
            fcr = feed_today / (gain_pred * current_birds)
        else:
            fcr = np.nan

        predictions.append({
            "Day": day,
            "Predicted Weight (kg)": round(current_weight, 3),
            "Predicted Daily Gain (kg)": round(gain_pred, 4),
            "Predicted Mortality": mort_pred,
            "Birds Alive": current_birds,
            "Derived FCR": round(fcr, 3)
        })

    pred_df = pd.DataFrame(predictions)

    st.subheader("🔮 Recursive Predictions")
    st.dataframe(pred_df, use_container_width=True)
