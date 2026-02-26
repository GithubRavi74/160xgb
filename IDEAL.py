import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

st.set_page_config(page_title="AI Guard - Growth Forecast", layout="wide")

st.title("🐔 AI Guard - Broiler Growth Forecast Engine")

# ---------------------------------------------------
# IDEAL GROWTH DATA (Arbor Acres Example)
# ---------------------------------------------------
ideal_days = np.array([1, 7, 14, 21, 28, 35])
ideal_weights = np.array([42, 180, 450, 900, 1500, 2200])

# Smooth interpolation
growth_curve = interp1d(ideal_days, ideal_weights, kind='cubic', fill_value="extrapolate")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
st.subheader("📥 Current Batch Inputs")

current_day = st.slider("Current Bird Age (Day)", 1, 35, 12)
current_weight = st.number_input("Current Average Weight (g)", 100, 2000, 420)

avg_temp = st.number_input("Avg Temperature (Last 3 Days °C)", 20.0, 40.0, 32.0)
avg_humidity = st.number_input("Avg Humidity (%)", 40.0, 95.0, 75.0)
avg_nh3 = st.number_input("Avg Ammonia (ppm)", 0.0, 60.0, 28.0)
avg_co2 = st.number_input("Avg CO2 (ppm)", 500.0, 5000.0, 2800.0)

# ---------------------------------------------------
# STRESS CALCULATIONS
# ---------------------------------------------------

# Heat stress index
optimal_temp = 30 if current_day < 21 else 28
heat_index = max(0, (avg_temp - optimal_temp) / 5)

# Gas stress index
gas_index = 0
if avg_nh3 > 25:
    gas_index += (avg_nh3 - 25) / 20
if avg_co2 > 3000:
    gas_index += (avg_co2 - 3000) / 2000

# Ventilation proxy
ventilation_index = heat_index * 0.5 + gas_index * 0.5

# Total stress
total_stress = 0.4*heat_index + 0.4*gas_index + 0.2*ventilation_index
total_stress = min(total_stress, 1.5)

suppression_factor = 1 - min(total_stress * 0.08, 0.12)

# ---------------------------------------------------
# IDEAL VS ADJUSTED FORECAST
# ---------------------------------------------------
ideal_final_weight = growth_curve(35)
predicted_final_weight = ideal_final_weight * suppression_factor

# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------
st.subheader("📊 Growth Forecast Result")

col1, col2, col3 = st.columns(3)

col1.metric("Ideal Harvest Weight (Day 35)", f"{ideal_final_weight:.0f} g")
col2.metric("Predicted Harvest Weight", f"{predicted_final_weight:.0f} g")
col3.metric("Growth Impact", f"-{(1-suppression_factor)*100:.1f}%")

# ---------------------------------------------------
# PLOT GROWTH CURVE
# ---------------------------------------------------
days = np.arange(1, 36)
ideal_curve = growth_curve(days)
adjusted_curve = ideal_curve * suppression_factor

plt.figure()
plt.plot(days, ideal_curve)
plt.plot(days, adjusted_curve)
plt.scatter(current_day, current_weight)
plt.xlabel("Day")
plt.ylabel("Weight (g)")
st.pyplot(plt)

# ---------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------
st.subheader("🧠 AI Interpretation")

if suppression_factor > 0.97:
    st.success("Environment stable. Growth trajectory healthy.")
elif suppression_factor > 0.92:
    st.warning("Mild growth suppression detected due to environmental stress.")
else:
    st.error("Significant growth suppression detected. Improve ventilation and reduce gas exposure.")
