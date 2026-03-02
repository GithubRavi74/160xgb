import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI", layout="wide")
#st.title("📈 iPoultry AI Guard")
#st.subheader("Environment Score For Barn")

st.markdown(
    """
    <h1 style='color: black;'>
        📈 iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h2 style='color: green;'>Environment Score For Barn</h2>", unsafe_allow_html=True)

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

 
birds_alive = int(last["birds_alive"])

# -------------------------------------------------
# MANUAL INPUT SECTION
# -------------------------------------------------
st.subheader("📝 Enter the Bird Age")

col1 = st.columns(1)

current_day = col1.number_input(
    "Bird Age (Day)",
    min_value=1,
    max_value=60,
    value=int(last["day_number"])
)

if st.button("🚀 Show Barn Score"):
    # -------------------------------------------------
    # ENVIRONMENTAL CONTEXT (Last 7 Days)
    # -------------------------------------------------
    rolling_7 = batch_hist.tail(7)
    
    temp_7d = rolling_7["temp"].mean()
    nh_7d = rolling_7["nh"].mean()
    rh_7d = rolling_7["rh"].mean()  
    co_7d = rolling_7["co"].mean()
    
      
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
    growth_impact_pct = (1 - suppression_factor) * 100
    
    # -------------------------------------------------
    # 7-DAY ENVIRONMENT SNAPSHOT (Non-Technical View)
    # -------------------------------------------------
   
    st.write("\n")
    st.write("")
    st.subheader("🌤 Last 7-Day Environmental Snapshot")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Avg Temperature (°C)", f"{temp_7d:.1f}")
    col2.metric("Avg Humidity (%)", f"{rh_7d:.1f}")
    col3.metric("Avg Ammonia (ppm)", f"{nh_7d:.1f}")
    col4.metric("Avg CO₂ (ppm)", f"{co_7d:.0f}")
    #col5.metric("Growth Impact (%)", f"{growth_impact_pct:.1f}")
    #col5.metric("Growth Impact (%)", f"{growth_impact_pct:.1f}")
    col5.metric("Estimated Growth Suppression (%)", f"{growth_impact_pct:.1f}")
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
        st.success("🟢 Environment is stable – Conditions support optimal growth.")
    elif env_score == 2:
        st.warning("🟡 Minor Environmental Deviation exists – Monitor ventilation and litter")
    else:
        st.error("🔴 Environmental Stress is detected – Immediate correction is recommended.")

    
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
