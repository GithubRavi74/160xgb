import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")
st.title("📈 iPoultry AI Guard – Harvest Growth Forecast")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

df = load_data()
st.write(df.columns)

# Clean IDs
df["farm_id"] = df["farm_id"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()
#df["batchName"] = df["batchName"].astype(str).str.strip()
 

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
    st.error("No historical data found for this batch.")
    st.stop()

last = batch_hist.iloc[-1]
if batch_hist.empty:
    st.error("No historical data found for this batch.")
    st.stop()

last = batch_hist.iloc[-1]

# -------------------------------------------------
# AUTO CONTEXT
# -------------------------------------------------
current_day = int(last["day_number"])
birds_alive = int(last["birds_alive"])

rolling_7 = batch_hist.tail(7)

temp_7d = rolling_7["temp"].mean()
rh_7d = rolling_7["rh"].mean()
nh_7d = rolling_7["nh"].mean()
co_7d = rolling_7["co"].mean()

# -------------------------------------------------
# IDEAL GROWTH BASELINE (Arbor Acres Example)
# -------------------------------------------------
ideal_days = np.array([1, 7, 14, 21, 28, 35])
ideal_weights = np.array([0.042, 0.18, 0.45, 0.9, 1.5, 2.2])  # in kg

growth_curve = interp1d(
    ideal_days,
    ideal_weights,
    kind="cubic",
    fill_value="extrapolate"
)

ideal_current_weight = float(growth_curve(current_day))
ideal_final_weight = float(growth_curve(35))

# -------------------------------------------------
# ENVIRONMENTAL STRESS CALCULATION
# -------------------------------------------------
# Age-based optimal temp
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

predicted_final_weight = ideal_final_weight * suppression_factor

growth_impact_pct = (1 - suppression_factor) * 100

# -------------------------------------------------
# DISPLAY CONTEXT
# -------------------------------------------------
st.subheader("📊 Current Batch Context")

colA, colB, colC = st.columns(3)

colA.metric("Current Age (Day)", current_day)
colB.metric("Birds Alive", birds_alive)
colC.metric("Ideal Weight Today (kg)", f"{ideal_current_weight:.2f}")

# -------------------------------------------------
# FORECAST RESULTS
# -------------------------------------------------
st.subheader("🚀 Harvest Forecast (Day 35 Projection)")

col1, col2, col3 = st.columns(3)

col1.metric("Ideal Harvest Weight (kg)", f"{ideal_final_weight:.2f}")
col2.metric("Predicted Harvest Weight (kg)", f"{predicted_final_weight:.2f}")
col3.metric("Growth Suppression Impact", f"-{growth_impact_pct:.1f}%")

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
st.subheader("🧠 AI Interpretation")

if suppression_factor > 0.97:
    st.success("Environment stable. Growth trajectory aligned with genetic potential.")
elif suppression_factor > 0.92:
    st.warning("Mild environmental suppression detected. Monitor ventilation and gas levels.")
else:
    st.error("Significant growth suppression risk detected. Immediate ventilation and gas correction recommended.")

# -------------------------------------------------
# STRESS BREAKDOWN
# -------------------------------------------------
st.subheader("🔍 Environmental Stress Breakdown")

st.write(f"Heat Stress Index: {heat_index:.2f}")
st.progress(min(heat_index / 2, 1.0))

st.write(f"Gas Stress Index: {gas_index:.2f}")
st.progress(min(gas_index / 2, 1.0))

st.write(f"Ventilation Risk Index: {ventilation_index:.2f}")
st.progress(min(ventilation_index / 2, 1.0))

# -------------------------------------------------
# GROWTH CURVE VISUALIZATION
# -------------------------------------------------
st.subheader("📈 Growth Curve Projection")

days = np.arange(1, 36)
ideal_curve = growth_curve(days)
adjusted_curve = ideal_curve * suppression_factor

import matplotlib.pyplot as plt

plt.figure()
plt.plot(days, ideal_curve)
plt.plot(days, adjusted_curve)
plt.scatter(current_day, ideal_current_weight)
plt.xlabel("Day")
plt.ylabel("Weight (kg)")
st.pyplot(plt)

# -------------------------------------------------
# CONFIDENCE INDICATOR
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
    st.error(f"Lower Confidence ({confidence}%) – Environmental variability detected")
