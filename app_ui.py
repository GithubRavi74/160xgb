import streamlit as st
import pandas as pd

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="iPoultry AI – Batch Growth Forecast",
    layout="wide"
)

st.title("🐔 iPoultry AI – Batch Growth Forecast")

st.caption(
    "Select a farm and batch. The system will use recorded data "
    "to forecast growth for the coming days."
)

# ----------------------------------------
# LOAD DATA (READ-ONLY)
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ml_ready_daily.csv")
    return df

df = load_data()

# ----------------------------------------
# FARM & BATCH SELECTION
# ----------------------------------------
st.subheader("📍 Select Batch")

col1, col2 = st.columns(2)

with col1:
    farm_ids = sorted(df["farm_id"].unique())
    farm_id = st.selectbox("Farm ID", farm_ids)

with col2:
    batch_ids = (
        df[df["farm_id"] == farm_id]["batch_id"]
        .unique()
        .tolist()
    )
    batch_id = st.selectbox("Batch ID", batch_ids)

batch_df = df[df["batch_id"] == batch_id].sort_values("day_number")

# ----------------------------------------
# BATCH SUMMARY PANEL
# ----------------------------------------
st.subheader("📊 Batch Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Batch Age (days)", int(batch_df["day_number"].max()))

with col2:
    st.metric("Days of Data", batch_df["day_number"].nunique())

with col3:
    last_date = batch_df["date"].max()
    st.metric("Last Data Date", last_date.strftime("%Y-%m-%d"))

# ----------------------------------------
# DATA SUFFICIENCY CHECK
# ----------------------------------------
st.subheader("🧪 Data Readiness Check")

days_of_data = batch_df["day_number"].nunique()

if days_of_data < 7:
    st.error("❌ Not enough data to predict (minimum 7 days required).")
    ready = False
elif days_of_data < 14:
    st.warning("⚠️ Limited data. Predictions will have low confidence.")
    ready = True
else:
    st.success("✅ Sufficient data available for prediction.")
    ready = True

# ----------------------------------------
# FORECAST SETTINGS
# ----------------------------------------
st.subheader("🔮 Forecast Settings")

forecast_days = st.radio(
    "Forecast Horizon",
    options=[7, 14],
    horizontal=True
)

# ----------------------------------------
# PREDICT BUTTON (PLACEHOLDER)
# ----------------------------------------
if st.button("📈 Predict Growth", disabled=not ready):
    st.info(
        "Prediction engine will run here.\n\n"
        "✔ Historical data loaded\n"
        "✔ Features will be reconstructed automatically\n"
        "✔ Model inference coming next"
    )
