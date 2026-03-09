import streamlit as st
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
st.subheader("📝 Enter the Following Information Of The Batch")

col1, col2 = st.columns(2)

initial_flock = col1.number_input(
    "Initial Flock Size",
    min_value=1,
    value=10000
)

bird_age = col2.number_input(
    "Bird Age (Day)",
    min_value=1,
    max_value=60,
    value=10
)

#st.markdown("### CURRENT DAY INPUT")

c1, c2, c3 = st.columns(3)

birds_alive = c1.number_input(
    "Total Birds Alive At This Age",
    min_value=1,
    value=9800
)

avg_weight = c2.number_input(
    "Average Weight (kg) At This Age",
    min_value=0.01,
    value=0.35,
    step=0.01
)

total_feed = c3.number_input(
    "Total Feed Consumed (kg) Till Now",
    min_value=0.01,
    value=4200.0
)

# -------------------------------------------------
# AUTO CALCULATIONS
# -------------------------------------------------
total_weight = birds_alive * avg_weight

current_fcr = total_feed / total_weight

# -------------------------------------------------
# DISPLAY AUTO CALCULATED VALUES
# -------------------------------------------------
st.markdown("### 📊 Auto Calculated Values")

st.markdown(
    """
    <style>
    .calc-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1E90FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="calc-box">', unsafe_allow_html=True)

    c4, c5 = st.columns(2)

    c4.metric("Accumulated Live Weight (kg)", f"{total_weight:.2f}")
    c5.metric("Current FCR", f"{current_fcr:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

st.text("")

# -------------------------------------------------
# BUTTON
# -------------------------------------------------
if st.button("🚀 Provide FCR Forecasting"):

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

    ideal_current_fcr = float(fcr_curve(bird_age))
    ideal_final_fcr = float(fcr_curve(35))

    # -------------------------------------------------
    # PERFORMANCE VS IDEAL
    # -------------------------------------------------
    performance_ratio = current_fcr / ideal_current_fcr
    predicted_final_fcr = ideal_final_fcr * performance_ratio

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

    # -------------------------------------------------
    # TODAY COMPARISON
    # -------------------------------------------------
    st.subheader("📊 Today's FCR Comparison with Ideal")

    colA, colB = st.columns(2)

    colA.metric("Ideal FCR Today", f"{ideal_current_fcr:.2f}")
    colB.metric("Actual FCR Today", f"{current_fcr:.2f}")

    if status_color == "success":
        st.success(f"🟢 {status}")
    elif status_color == "warning":
        st.warning(f"🟡 {status}")
    else:
        st.error(f"🔴 {status}")

    st.text("")
    st.text("")

    # -------------------------------------------------
    # FINAL FCR FORECAST
    # -------------------------------------------------
    st.markdown(
        "<h2 style='text-align: center;'>🚀 Harvest FCR Forecast</h2>",
        unsafe_allow_html=True
    )

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

    plt.plot(days, ideal_curve, marker='o')
    plt.plot(days, predicted_curve, marker='o')

    plt.scatter(bird_age, current_fcr, s=150)

    plt.xlabel("Day")
    plt.ylabel("FCR")
    plt.grid(True)

    legend = plt.legend(
        ["Ideal FCR", "Predicted FCR", "Today's FCR"]
    )

    for text in legend.get_texts():
        text.set_color("green")

    st.pyplot(plt)
