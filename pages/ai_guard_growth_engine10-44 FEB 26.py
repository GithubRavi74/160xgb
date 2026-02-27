import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")
st.title("📈 iPoultry AI Guard – Smart Harvest Weight Prediction")

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

farm_batches = (
    df[df["farm_id"] == farm_id][["batch_id"]]
    .drop_duplicates()
    .sort_values("batch_id")
)

batch_id = st.selectbox("Batch ID", farm_batches["batch_id"])

batch_hist = df[
    (df["farm_id"] == farm_id) &
    (df["batch_id"] == batch_id)
].sort_values("day_number")

if batch_hist.empty:
    st.error("No historical data found.")
    st.stop()

last = batch_hist.iloc[-1]

# -------------------------------------------------
# MANUAL INPUT SECTION
# -------------------------------------------------
st.subheader("📝 Enter Current Bird Information")

col1, col2 = st.columns(2)

current_day = col1.number_input(
    "Bird Age (Day)",
    min_value=1,
    max_value=60,
    value=int(last["day_number"])
)

today_weight = col2.number_input(
    "Today's Average Weight (kg)",
    min_value=0.01,
    max_value=5.0,
    value=float(last.get("sample_weight_kg", 1.0)),
    step=0.01
)

birds_alive = int(last["birds_alive"])

# -------------------------------------------------
# ENVIRONMENTAL CONTEXT (Last 7 Days)
# -------------------------------------------------
rolling_7 = batch_hist.tail(7)

temp_7d = rolling_7["temp"].mean()
nh_7d = rolling_7["nh"].mean()
co_7d = rolling_7["co"].mean()

# -------------------------------------------------
# 7-DAY ENVIRONMENT SNAPSHOT (Non-Technical View)
# -------------------------------------------------
st.subheader("🌤 7-Day Environmental Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Temperature (°C)", f"{temp_7d:.1f}")
col2.metric("Avg Humidity (%)", f"{rh_7d:.1f}")
col3.metric("Avg Ammonia (ppm)", f"{nh_7d:.1f}")
col4.metric("Avg CO₂ (ppm)", f"{co_7d:.0f}")

# -------------------------------------------------
# Simple Environmental Status Logic
# -------------------------------------------------
env_score = 0

# Temperature scoring
if 26 <= temp_7d <= 32:
    env_score += 1

# Ammonia scoring
if nh_7d <= 25:
    env_score += 1

# CO2 scoring
if co_7d <= 3000:
    env_score += 1

# Final Status
if env_score == 3:
    st.success("🟢 Environment Stable – Conditions support optimal growth.")
elif env_score == 2:
    st.warning("🟡 Minor Environmental Deviation – Monitor ventilation and litter.")
else:
    st.error("🔴 Environmental Stress Detected – Immediate correction recommended.")


# -------------------------------------------------
# IDEAL GENETIC CURVE (Arbor Acres Example)
# -------------------------------------------------
ideal_days = np.array([1, 7, 14, 21, 28, 35])
ideal_weights = np.array([0.042, 0.18, 0.45, 0.9, 1.5, 2.2])

growth_curve = interp1d(
    ideal_days,
    ideal_weights,
    kind="cubic",
    fill_value="extrapolate"
)

ideal_current_weight = float(growth_curve(current_day))
ideal_final_weight = float(growth_curve(35))

# -------------------------------------------------
# PERFORMANCE VS IDEAL
# -------------------------------------------------
performance_ratio = today_weight / ideal_current_weight
adjusted_final_weight = ideal_final_weight * performance_ratio

# Status classification
if performance_ratio > 1.05:
    status = "Ahead of Target"
    status_color = "success"
elif performance_ratio >= 0.95:
    status = "On Track"
    status_color = "warning"
else:
    status = "Behind Target"
    status_color = "error"

# -------------------------------------------------
# ENVIRONMENTAL STRESS CALCULATION
# -------------------------------------------------
optimal_temp = 30 if current_day < 21 else 28

heat_index = max(0, (temp_7d - optimal_temp) / 5)

gas_index = 0
if nh_7d > 25:
    gas_index += (nh_7d - 25) / 20
if co_7d > 3000:
    gas_index += (co_7d - 3000) / 2000

ventilation_index = 0.5 * heat_index + 0.5 * gas_index

total_stress = 0.4*heat_index + 0.4*gas_index + 0.2*ventilation_index
total_stress = min(total_stress, 1.5)

suppression_factor = 1 - min(total_stress * 0.08, 0.12)

# -------------------------------------------------
# FINAL PREDICTION
# -------------------------------------------------
predicted_final_weight = adjusted_final_weight * suppression_factor
growth_impact_pct = (1 - suppression_factor) * 100

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
st.subheader("📊 Current Status")

colA, colB, colC = st.columns(3)

colA.metric("Ideal Weight Today (kg)", f"{ideal_current_weight:.2f}")
colB.metric("Actual Weight Today (kg)", f"{today_weight:.2f}")
colC.metric("Birds Alive", birds_alive)

if status_color == "success":
    st.success(f"🟢 {status}")
elif status_color == "warning":
    st.warning(f"🟡 {status}")
else:
    st.error(f"🔴 {status}")

# -------------------------------------------------
# HARVEST FORECAST
# -------------------------------------------------
st.subheader("🚀 Harvest Forecast (Day 35 Projection)")

c1, c2, c3 = st.columns(3)

c1.metric("Genetic Target (kg)", f"{ideal_final_weight:.2f}")
c2.metric("Predicted Harvest Weight (kg)", f"{predicted_final_weight:.2f}")
c3.metric("Environmental Impact", f"-{growth_impact_pct:.1f}%")

# -------------------------------------------------
# GROWTH CURVE VISUALIZATION
# -------------------------------------------------
st.subheader("📈 Growth Curve Projection")

days = np.arange(1, 36)
ideal_curve = growth_curve(days)
performance_curve = ideal_curve * performance_ratio
adjusted_curve = performance_curve * suppression_factor

plt.figure()
plt.plot(days, ideal_curve)
plt.plot(days, performance_curve)
plt.plot(days, adjusted_curve)
plt.scatter(current_day, today_weight)
plt.xlabel("Day")
plt.ylabel("Weight (kg)")
st.pyplot(plt)

# -------------------------------------------------
# CONFIDENCE SCORE
# -------------------------------------------------
st.subheader("🎯 Forecast Confidence")

confidence = 100

if abs(temp_7d - batch_hist["temp"].mean()) > 5:
    confidence -= 10
if abs(nh_7d - batch_hist["nh"].mean()) > 10:
    confidence -= 10
if birds_alive < batch_hist["birds_alive"].max() * 0.9:
    confidence -= 10

confidence = max(60, confidence)

if confidence >= 85:
    st.success(f"High Confidence ({confidence}%)")
elif confidence >= 70:
    st.warning(f"Moderate Confidence ({confidence}%)")
else:
    st.error(f"Lower Confidence ({confidence}%)")


