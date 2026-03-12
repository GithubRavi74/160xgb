import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from datetime import datetime

# --- 1. IDEAL WEIGHT REFERENCE ---
IDEAL_WEIGHT = {
    1:0.061, 2:0.079, 3:0.099, 4:0.122, 5:0.148, 6:0.176, 7:0.208,
    8:0.242, 9:0.28, 10:0.321, 11:0.366, 12:0.414, 13:0.465,
    14:0.519, 15:0.576, 16:0.637, 17:0.701, 18:0.768, 19:0.837,
    20:0.91, 21:0.985, 22:1.062, 23:1.142, 24:1.225, 25:1.309,
    26:1.395, 27:1.483, 28:1.573, 29:1.664, 30:1.757,
    31:1.851, 32:1.946, 33:2.041, 34:2.138, 35:2.235,
    36:2.332, 37:2.43, 38:2.527, 39:2.625, 40:2.723
}

# --- 2. CONFIG ---
st.set_page_config(page_title="iPoultry AI Guard", layout="wide")

st.markdown("""
    <h1 style='color: black;'>📈 iPoultry <span style='color: #FFD700;'>AI Guard</span></h1>
    <h2 style='color: green;'>Harvest Weight & FCR Prediction</h2>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
MODEL_PATH = "kishorebatches_weight_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error(f"❌ Model file '{MODEL_PATH}' not found.")
else:
    # --- 4. INPUT SECTION (ALWAYS VISIBLE) ---
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📅 Batch & Mortality")
            initial_flock = st.number_input("Initial Flock Size", 100, 100000, 5000)
            total_mortality = st.number_input("Total Mortality to Date", 0, initial_flock, 150)
            day_number = st.number_input("Day Number (Age)", 1, 40, 21)
            birds_alive = initial_flock - total_mortality
            st.info(f"**Current Birds Alive:** {birds_alive}")

        with col2:
            st.subheader("🌡️ Environment & Feed")
            feed_today = st.number_input("Total Feed Today (kg)", 0.0, 5000.0, 450.0)
            temp = st.slider("Mean Temperature (°C)", 15.0, 40.0, 28.5)
            rh = st.slider("Relative Humidity (%)", 20.0, 100.0, 65.0)
            nh = st.number_input("Ammonia (NH3) Level", 0.0, 50.0, 10.0)

    # Sidebar inputs for FCR and Harvest
    st.sidebar.header("⚙️ Target Settings")
    harvest_day = st.sidebar.number_input("Target Harvest Day", 30, 45, 40)
    # MOVED OUT OF BUTTON LOGIC:
    total_feed_to_date = st.sidebar.number_input("Total Feed Consumed to Date (kg)", value=feed_today * day_number)
    
    # Advanced logic
    heat_index = temp + (0.33 * rh) - 0.7
    feed_per_bird = feed_today / birds_alive if birds_alive > 0 else 0
    
    with st.expander("Advanced: Historical Trends"):
        roll_feed = st.number_input("7-Day Avg Feed (kg)", value=feed_today)
        roll_gain = st.number_input("7-Day Avg Gain (kg)", value=0.05)
        co_level = st.number_input("CO Level", 0.0, 50.0, 5.0)

    st.markdown("---")

    # --- 5. PREDICTION LOGIC ---
    if st.button("🚀 Run AI Analysis", use_container_width=True):
        input_data = pd.DataFrame([{
            'day_number': day_number, 'birds_alive': birds_alive, 'feed_today_kg': feed_today,
            'temp': temp, 'rh': rh, 'co': co_level, 'nh': nh, 'heat_index': heat_index,
            'feed_per_bird': feed_per_bird, 'rolling_7d_feed': roll_feed,
            'rolling_7d_gain': roll_gain, 'initial_flock': initial_flock,
            'cumulative_mortality': total_mortality
        }])

        # 1. Weights
        current_pred = model.predict(input_data)[0]
        ideal_today = IDEAL_WEIGHT.get(day_number, 1.0)
        performance_ratio = current_pred / ideal_today
        ideal_harvest = IDEAL_WEIGHT.get(harvest_day, 2.723)
        projected_weight = ideal_harvest * performance_ratio
   
        # 2. FCR Calculations
        current_total_biomass = current_pred * birds_alive
        current_fcr = total_feed_to_date / current_total_biomass if current_total_biomass > 0 else 0
        
        projected_total_feed = total_feed_to_date + (roll_feed * (harvest_day - day_number))
        projected_biomass = projected_weight * birds_alive
        harvest_fcr = projected_total_feed / projected_biomass if projected_biomass > 0 else 0

        # --- 6. DISPLAY RESULTS ---
        st.subheader("⚖️ Weight Predictions")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Today's Estimated Weight", f"{current_pred:.3f} kg")
        with res_col2:
            st.metric("Projected Harvest Weight", f"{projected_weight:.3f} kg", 
                      delta=f"{(performance_ratio-1)*100:.1f}% vs Ideal")

        st.markdown("### 📈 Efficiency Metrics (FCR)")
        fcr_col1, fcr_col2 = st.columns(2)
        with fcr_col1:
            st.metric("Estimated Current FCR", f"{current_fcr:.2f}")
        with fcr_col2:
            fcr_delta = harvest_fcr - 1.60 
            st.metric("Projected Harvest FCR", f"{harvest_fcr:.2f}", delta=f"{fcr_delta:.2f}", delta_color="inverse")

        # Alerts
        if heat_index > 32:
            st.warning(f"⚠️ **Heat Stress:** Index {heat_index:.1f}. Growth may be stunted.")
        else:
            st.success(f"✅ **Optimal Environment:** (Heat Index: {heat_index:.1f})")

        # --- 7. DOWNLOAD REPORT ---
        report_text = f"""
        iPoultry AI Guard - Prediction Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        -------------------------------------------
        WEIGHT: Today: {current_pred:.3f} kg | Harvest: {projected_weight:.3f} kg
        EFFICIENCY: Current FCR: {current_fcr:.2f} | Harvest FCR: {harvest_fcr:.2f}
        STATUS: {performance_ratio*100:.1f}% of Ideal Growth
        -------------------------------------------
        """
        st.download_button("📄 Download Detailed Report", report_text, 
                           file_name=f"Report_Day{day_number}.txt", use_container_width=True)

st.divider()
st.caption("iPoultry AI Guard © 2026")
