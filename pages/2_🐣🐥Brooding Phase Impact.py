import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Brooding Phase Impact Forecasting", layout="wide")

st.markdown(
    """
    <h1 style='color: black;'>
        📈 iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='color: green;'> Brooding Phase Impact Forecasting</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

df = load_data()
df.columns = df.columns.str.strip()

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
    st.error("No historical data found.")
    st.stop()

# -------------------------------------------------
# BUTTON TRIGGER
# -------------------------------------------------
if st.button("🚀 Analyze Brooding Life Impact On Harvest Weight"):

    # -------------------------------------------------
    # BROODING PERIOD (Day 1–10)
    # -------------------------------------------------
    brooding_data = batch_hist[batch_hist["day_number"] <= 10]

    if len(brooding_data) < 5:
        st.error("Insufficient brooding data (Need at least 5 days).")
        st.stop()

    avg_temp = brooding_data["temp"].mean()
    avg_rh = brooding_data["rh"].mean()

    # -------------------------------------------------
    # EARLY HEAT STRESS CALCULATION
    # -------------------------------------------------
    optimal_brooding_temp = 32  # ideal average brooding temp

    temp_deviation = abs(avg_temp - optimal_brooding_temp)
    heat_stress_index = max(0, temp_deviation / 5)

    humidity_penalty = 0
    if avg_rh > 75:
        humidity_penalty = (avg_rh - 75) / 20

    early_stress_score = min(heat_stress_index + humidity_penalty, 2)

    # Convert to growth suppression
    early_suppression_factor = 1 - min(early_stress_score * 0.05, 0.10)
    impact_pct = (1 - early_suppression_factor) * 100

    # -------------------------------------------------
    # IDEAL GENETIC GROWTH CURVE (Example)
    # -------------------------------------------------
    ideal_days = np.array([1, 7, 14, 21, 28, 35])
    ideal_weights = np.array([0.042, 0.18, 0.45, 0.9, 1.5, 2.2])

    growth_curve = interp1d(
        ideal_days,
        ideal_weights,
        kind="cubic",
        fill_value="extrapolate"
    )

    ideal_final_weight = float(growth_curve(35))
    predicted_final_weight = ideal_final_weight * early_suppression_factor

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------
    st.subheader("🔥 Brooding Period Analysis (Day 1–10)")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Brooding Temp (°C)", f"{avg_temp:.1f}")
    col2.metric("Avg Brooding Humidity (%)", f"{avg_rh:.1f}")
    col3.metric("Early Stress Index", f"{early_stress_score:.2f}")

    # Status
    if early_stress_score < 0.5:
        st.success("🟢 Brooding Management Optimal")
    elif early_stress_score < 1:
        st.warning("🟡 Minor Brooding Deviation Detected")
    else:
        st.error("🔴 Significant Early Heat Stress Detected")

    # -------------------------------------------------
    # HARVEST IMPACT FORECAST
    # -------------------------------------------------
    st.subheader("🚀 Day 35 Harvest Impact Forecast")

    c1, c2, c3 = st.columns(3)

    c1.metric("Ideal Target Weight(kg)", f"{ideal_final_weight:.2f}")
    c2.metric("Predicted Harvest Weight (kg)", f"{predicted_final_weight:.2f}")
    c3.metric("Early Life Impact", f"-{impact_pct:.1f}%")

    # -------------------------------------------------
    # GROWTH CURVE VISUALIZATION
    # -------------------------------------------------
    st.subheader("📈 Growth Curve Projection")

    days = np.arange(1, 36)
    ideal_curve = growth_curve(days)
    adjusted_curve = ideal_curve * early_suppression_factor

    plt.figure()
    plt.plot(days, ideal_curve)
    plt.plot(days, adjusted_curve)
    plt.xlabel("Day")
    plt.ylabel("Weight (kg)")
    st.pyplot(plt)

    # -------------------------------------------------
    # EXECUTIVE INTERPRETATION
    # -------------------------------------------------
    st.subheader("🧠 Executive Insight")

    if impact_pct < 2:
        st.success("Early brooding conditions are unlikely to affect final harvest weight.")
    elif impact_pct < 5:
        st.warning("Minor early-life impact detected. Monitor flock performance.")
    else:
        st.error("Brooding stress may significantly reduce final harvest performance.")
