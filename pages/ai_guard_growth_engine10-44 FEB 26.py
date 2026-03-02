import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")
#st.title("📈 iPoultry AI Guard")
#st.subheader("Bird Harvest Weight Prediction")

st.markdown(
    """
    <h1 style='color: black;'>
        📈 iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h2 style='color: green;'>Bird Harvest Weight Prediction</h2>", unsafe_allow_html=True)

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
st.subheader("🏭 Select Your Farm & Batch")

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
st.subheader("📝 Enter the Bird Information")

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

if st.button("🚀 Provide Forecasting"):
    # -------------------------------------------------
    # ENVIRONMENTAL CONTEXT (Last 7 Days)
    # -------------------------------------------------
    rolling_7 = batch_hist.tail(7)
    
    temp_7d = rolling_7["temp"].mean()
    nh_7d = rolling_7["nh"].mean()
    rh_7d = rolling_7["rh"].mean()  
    co_7d = rolling_7["co"].mean()
    
    # -------------------------------------------------
    # 7-DAY ENVIRONMENT SNAPSHOT (Non-Technical View)
    # -------------------------------------------------
   
    st.write("")
    st.write("")
    st.subheader("🌤 Last 7-Day Environmental Snapshot")
    
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
    # IDEAL WEIGHT CURVE (Ross 308 Example)
    # -------------------------------------------------
    #ideal_days = np.array([0, 7, 14, 21, 28, 35])
    #ideal_weights = np.array([0.043, 0.208, 0.519, 0.985, 1.573, 2.235])

    ideal_days = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35
    ])

    ideal_weights = np.array([
    0.043, 0.061, 0.079, 0.099, 0.122, 0.148, 0.176, 0.208, 0.242, 0.28,
    0.321, 0.366, 0.414, 0.465, 0.519, 0.576, 0.637, 0.701, 0.768, 0.837,
    0.91, 0.985, 1.062, 1.142, 1.225, 1.309, 1.395, 1.483, 1.573, 1.664,
    1.757, 1.851, 1.946, 2.041, 2.138, 2.235
    ])
    
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
    st.subheader("📊 Current Status Of Birds")
    
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


