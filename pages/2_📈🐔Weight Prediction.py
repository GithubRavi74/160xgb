import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Weight Prediction Based On Farm Trained Model", layout="wide")
#st.title("📈 iPoultry AI Guard")
#st.subheader("Bird Harvest Weight Prediction by Farm Trained AI Model")

st.markdown(
    """
    <h1 style='color: black;'>
        📈 iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h2 style='color: green;'>Bird Harvest Weight Prediction by Farm Trained AI Model</h2>", unsafe_allow_html=True)

# 1. PAGE TITLE (The sidebar menu will use the filename)
#st.title("🐥 iPoultry AI Guard: Weight Prediction")
#st.markdown("Use this tool to estimate bird weight based on current environmental and feed data.")

# 2. LOAD THE TRAINED MODEL
# We look for the pickle file in the root directory
MODEL_PATH = "kishorebatches_weight_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error(f"❌ Model file '{MODEL_PATH}' not found in the root folder.")
    st.info("Please ensure the .pkl file is uploaded to your GitHub repository.")
else:
    # 3. INPUT LAYOUT
    # We use columns in the main pane to keep the sidebar clean for your navigation menu
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📅 Batch Stats")
        day_number = st.number_input("Day Number (Age)", 1, 45, 21)
        birds_alive = st.number_input("Birds Currently Alive", 1, 50000, 5000)
        feed_today = st.number_input("Total Feed Today (kg)", 0.0, 5000.0, 450.0)

    with col_b:
        st.subheader("🌡️ Environment")
        temp = st.slider("Mean Temperature (°C)", 15.0, 40.0, 28.5)
        rh = st.slider("Relative Humidity (%)", 20.0, 100.0, 65.0)
        co = st.number_input("CO Level", 0.0, 50.0, 5.0)
        nh = st.number_input("Ammonia (NH3)", 0.0, 50.0, 10.0)

    st.markdown("---")
    
    # 4. CALCULATED FEATURES
    # Match the logic used during training
    heat_index = temp + (0.33 * rh) - 0.7
    feed_per_bird = feed_today / birds_alive
    
    # Optional inputs for rolling averages
    expander = st.expander("Advanced: Historical Trends (Rolling Averages)")
    with expander:
        roll_feed = st.number_input("7-Day Avg Feed (kg)", value=feed_today)
        roll_gain = st.number_input("7-Day Avg Gain (kg)", value=0.05)

    # 5. PREDICT BUTTON
    if st.button("🚀 Predict Current Weight", use_container_width=True):
        # Build the input dataframe exactly as the XGBoost model expects
        input_data = pd.DataFrame([{
            'day_number': day_number,
            'birds_alive': birds_alive,
            'feed_today_kg': feed_today,
            'temp': temp,
            'rh': rh,
            'co': co,
            'nh': nh,
            'heat_index': heat_index,
            'feed_per_bird': feed_per_bird,
            'rolling_7d_feed': roll_feed,
            'rolling_7d_gain': roll_gain
        }])

        prediction = model.predict(input_data)[0]

        # 6. RESULTS DISPLAY
        st.success(f"### Predicted Average Weight: **{prediction:.3f} kg**")
        
        # Environmental Health logic
        if heat_index > 32:
            st.warning(f"⚠️ **Caution:** Heat Index is {heat_index:.1f}. Birds may experience heat stress.")
        else:
            st.info(f"✅ **Comfort Zone:** Heat Index is {heat_index:.1f}.")

st.caption("iPoultry AI Guard Predictor Module")
