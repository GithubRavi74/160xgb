import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – FCR Forecast", layout="wide")

st.markdown(
    """
    <h1 style='color: black;'>
        📈 iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='color: green;'>FCR Prediction</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
st.subheader("📝 Enter the following information of the Batch")

#initial_flock = st.number_input(
#   "Initial Flock Size",
#   min_value=1,
#   value=10000
#)

#st.markdown("### CURRENT DAY INPUT")

col1, col2 = st.columns(2)

current_day = col1.number_input(
    "Bird Age (Day)",
    min_value=1,
    max_value=60,
    value=10
)

birds_alive = col2.number_input(
    "Birds Alive Today",
    min_value=1,
    value=9800
)

col3, col4 = st.columns(2)

avg_weight = col3.number_input(
    "Average Bird Weight (kg) At This Age",
    min_value=0.01,
    value=0.35,
    step=0.01
)

feed_today = col4.number_input(
    "Total Feed Consumed Until Today (kg)",
    min_value=0.0,
    value=4500.0
)

# -------------------------------------------------
# AUTO CALCULATIONS
# -------------------------------------------------
chick_weight = 0.04

current_total_weight = birds_alive * avg_weight
# IF INITIAL FLOCK SIZE IS CONSIDERED INTO FCR CALCULATON
#initial_total_weight = initial_flock * chick_weight
#weight_gain = current_total_weight - initial_total_weight
#current_fcr = feed_today / weight_gain if weight_gain > 0 else 0

current_fcr = feed_today / current_total_weight if current_total_weight > 0 else 0

st.markdown("### 📊 Auto Calculated Values")

c1, c2 = st.columns(2)

c1.metric("Total Flock Weight At This Age (kg)", f"{current_total_weight:,.0f}")
c2.metric("Current FCR", f"{current_fcr:.2f}")

# -------------------------------------------------
# BUTTON
# -------------------------------------------------
if st.button("🚀 Provide FCR Forecasting"):

    st.text("\n")

    # -------------------------------------------------
    # IDEAL FCR CURVE
    # -------------------------------------------------
    ideal_days = np.array([7, 14, 21, 28, 35])

    ideal_fcr_values = np.array([
        0.95, 1.15, 1.35, 1.50, 1.60
    ])

    fcr_curve = interp1d(
        ideal_days,
        ideal_fcr_values,
        kind="linear",
        fill_value="extrapolate"
    )

    ideal_current_fcr = float(fcr_curve(current_day))
    ideal_final_fcr = float(fcr_curve(35))

    # -------------------------------------------------
    # PERFORMANCE VS IDEAL
    # -------------------------------------------------
    performance_ratio = current_fcr / ideal_current_fcr
    predicted_final_fcr = ideal_final_fcr * performance_ratio

    if performance_ratio < 0.95:
        status = "TODAY'S FCR STATUS : Better than Target"
        status_color = "success"
    elif performance_ratio <= 1.05:
        status = "TODAY'S FCR STATUS : On Track"
        status_color = "warning"
    else:
        status = "TODAY'S FCR STATUS : Worse than Target"
        status_color = "error"

    # -------------------------------------------------
    # DISPLAY TODAY COMPARISON
    # -------------------------------------------------
    st.subheader("📊 Today's FCR Comparison with Ideal FCR")

    colA, colB = st.columns(2)

    colA.metric("Ideal FCR Today", f"{ideal_current_fcr:.2f}")
    colB.metric("Actual FCR Today", f"{current_fcr:.2f}")

    if status_color == "success":
        st.success(f"🟢 {status}")
    elif status_color == "warning":
        st.warning(f"🟡 {status}")
    else:
        st.error(f"🔴 {status}")

    st.text("\n")

    # -------------------------------------------------
    # FINAL FCR FORECAST
    # -------------------------------------------------
    st.markdown(
        "<h2 style='text-align: center;'>🚀 Harvest FCR Forecast</h2>",
        unsafe_allow_html=True
    )

    st.subheader("Day 35 FCR Projection")

    c1, c2 = st.columns(2)

    c1.metric("Ideal Target FCR (Day 35)", f"{ideal_final_fcr:.2f}")
    c2.metric("Predicted Final FCR", f"{predicted_final_fcr:.2f}")

    # -------------------------------------------------
    # FCR CURVE VISUALIZATION
    # -------------------------------------------------
    st.subheader("📈 FCR Projection Curve")

    days = np.arange(7, 36)
    ideal_curve = fcr_curve(days)
    predicted_curve = ideal_curve * performance_ratio

    plt.figure()

    plt.plot(days, ideal_curve, marker='o', linewidth=1)
    plt.plot(days, predicted_curve, marker='o', linewidth=1)

    plt.scatter(current_day, current_fcr, s=150)

    plt.xlabel("Day")
    plt.ylabel("FCR")
    plt.grid(True)

    legend = plt.legend(["Ideal FCR", "Predicted FCR", "Today's FCR"])

    for text in legend.get_texts():
        text.set_color("green")

    st.pyplot(plt)
