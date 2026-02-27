import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")

# -------------------------------------------------
# CUSTOM STYLING (Enterprise Agriculture Theme)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f8f2;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #1b5e20;
}

.sub-title {
    font-size: 20px;
    color: #2e7d32;
}

.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.input-card {
    background-color: #eef7ea;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #d8e8d2;
    margin-bottom: 25px;
}

.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1b5e20;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("<div class='main-title'>📈 iPoultry AI Guard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Bird Harvest Weight Prediction</div>", unsafe_allow_html=True)
st.write("")

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
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🏭 Select Farm & Batch")

    farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))

    farm_batches = (
        df[df["farm_id"] == farm_id][["batch_id"]]
        .drop_duplicates()
        .sort_values("batch_id")
    )

    batch_id = st.selectbox("Batch ID", farm_batches["batch_id"])
    st.markdown("</div>", unsafe_allow_html=True)

batch_hist = df[
    (df["farm_id"] == farm_id) &
    (df["batch_id"] == batch_id)
].sort_values("day_number")

if batch_hist.empty:
    st.error("No historical data found.")
    st.stop()

last = batch_hist.iloc[-1]
birds_alive = int(last["birds_alive"])

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Bird Information")

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

predict_button = st.button("🚀 Provide Forecasting")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# RUN FORECAST ONLY AFTER BUTTON CLICK
# -------------------------------------------------
if predict_button:

    # -------------------------------------------------
    # ENVIRONMENT SNAPSHOT
    # -------------------------------------------------
    rolling_7 = batch_hist.tail(7)

    temp_7d = rolling_7["temp"].mean()
    nh_7d = rolling_7["nh"].mean()
    rh_7d = rolling_7["rh"].mean()
    co_7d = rolling_7["co"].mean()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🌤 Last 7-Day Environmental Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Temperature (°C)", f"{temp_7d:.1f}")
    c2.metric("Avg Humidity (%)", f"{rh_7d:.1f}")
    c3.metric("Avg Ammonia (ppm)", f"{nh_7d:.1f}")
    c4.metric("Avg CO₂ (ppm)", f"{co_7d:.0f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # IDEAL GENETIC CURVE
    # -------------------------------------------------
    ideal_days = np.array([1, 7, 14, 21, 28, 35])
    ideal_weights = np.array([0.042, 0.18, 0.45, 0.9, 1.5, 2.2])

    growth_curve = interp1d(ideal_days, ideal_weights, kind="cubic", fill_value="extrapolate")

    ideal_current_weight = float(growth_curve(current_day))
    ideal_final_weight = float(growth_curve(35))

    performance_ratio = today_weight / ideal_current_weight
    adjusted_final_weight = ideal_final_weight * performance_ratio

    # -------------------------------------------------
    # ENVIRONMENTAL STRESS
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

    predicted_final_weight = adjusted_final_weight * suppression_factor
    growth_impact_pct = (1 - suppression_factor) * 100

    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🚀 Harvest Forecast (Day 35 Projection)")

    r1, r2, r3 = st.columns(3)
    r1.metric("Genetic Target (kg)", f"{ideal_final_weight:.2f}")
    r2.metric("Predicted Harvest Weight (kg)", f"{predicted_final_weight:.2f}")
    r3.metric("Environmental Impact", f"-{growth_impact_pct:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # GROWTH CURVE
    # -------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📈 Growth Curve Projection")

    days = np.arange(1, 36)
    ideal_curve = growth_curve(days)
    performance_curve = ideal_curve * performance_ratio
    adjusted_curve = performance_curve * suppression_factor

    plt.figure()
    plt.plot(days, ideal_curve, label="Genetic Target")
    plt.plot(days, performance_curve, label="Performance Adjusted")
    plt.plot(days, adjusted_curve, label="Final Forecast")
    plt.scatter(current_day, today_weight)
    plt.xlabel("Day")
    plt.ylabel("Weight (kg)")
    plt.legend()
    st.pyplot(plt)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
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

    st.markdown("</div>", unsafe_allow_html=True)
