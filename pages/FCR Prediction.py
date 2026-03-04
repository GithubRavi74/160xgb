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

st.markdown("<h2 style='color: green;'>Bird FCR Prediction</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# MANUAL INPUT SECTION
# -------------------------------------------------
st.subheader("📝 Enter the Bird Information")

col1, col2 = st.columns(2)

current_day = col1.number_input(
    "Bird Age (Day)",
    min_value=1,
    max_value=60,
    value=10
)

today_fcr = col2.number_input(
    "Current FCR",
    min_value=0.5,
    max_value=3.0,
    value=1.20,
    step=0.01
)

# -------------------------------------------------
# BUTTON
# -------------------------------------------------
if st.button("🚀 Provide FCR Forecasting"):

    st.text("\n")

    # -------------------------------------------------
    # IDEAL FCR CURVE (Example Standard Broiler FCR)
    # -------------------------------------------------
    ideal_days = np.array([
        7, 14, 21, 28, 35
    ])

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
    # (Lower FCR is better)
    # -------------------------------------------------
    performance_ratio = today_fcr / ideal_current_fcr
    adjusted_final_fcr = ideal_final_fcr * performance_ratio

    # Status classification
    if performance_ratio < 0.95:
        status = "TODAY'S FCR STATUS : Better than Target"
        status_color = "success"
    elif performance_ratio <= 1.05:
        status = "TODAY'S FCR STATUS : On Track"
        status_color = "warning"
    else:
        status = "TODAY'S FCR STATUS : Worse than Target"
        status_color = "error"

    predicted_final_fcr = adjusted_final_fcr

    # -------------------------------------------------
    # DISPLAY TODAY COMPARISON
    # -------------------------------------------------
    st.subheader("📊 Today's FCR Comparison with Ideal FCR")

    colA, colB = st.columns(2)

    colA.metric("Ideal FCR Today", f"{ideal_current_fcr:.2f}")
    colB.metric("Actual FCR Today", f"{today_fcr:.2f}")

    if status_color == "success":
        st.success(f"🟢 {status}")
    elif status_color == "warning":
        st.warning(f"🟡 {status}")
    else:
        st.error(f"🔴 {status}")

    st.text("\n")
    st.text("\n")

    # -------------------------------------------------
    # FINAL FCR FORECAST
    # -------------------------------------------------
    st.markdown(
        "<h2 style='text-align: center;'>🚀 Harvest FCR Forecast</h2>",
        unsafe_allow_html=True
    )

    st.text("\n")
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
    performance_curve = ideal_curve * performance_ratio
    
    plt.figure()
    
    # Ideal curve with highlighted points
    plt.plot(
        days,
        ideal_curve,
        marker='o',
        markersize=6,
        linewidth=2
    )
    
    # Predicted curve with highlighted points
    plt.plot(
        days,
        performance_curve,
        marker='o',
        markersize=6,
        linewidth=2
    )
    
    # Highlight today's datapoint clearly
    plt.scatter(
        current_day,
        today_fcr,
        s=120
    )
    
    #plt.xlabel("Day")
    #plt.ylabel("FCR")
    #plt.grid(True)
    
    plt.plot(days, ideal_curve, marker='o', markersize=5, linewidth=2)
    plt.plot(days, performance_curve, marker='o', markersize=5, linewidth=2)

    plt.scatter(current_day, today_fcr, s=150)
    st.pyplot(plt)

    
    
