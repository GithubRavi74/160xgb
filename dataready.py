import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="iPoultry AI – Data Readiness", layout="wide")
st.title("📊 Batch Data Readiness Check")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

df = load_data()

# CLEAN THE VALUES
df["farm_id"] = df["farm_id"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()

#st.dataframe(df)

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
#farm_id = st.text_input("Farm ID")
#batch_id = st.text_input("Batch ID")

st.write("SELECT FARM AND BATCH TO CHECK IF AI CAN BE USED)
farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))
batch_id = st.selectbox(
    "Batch ID",
    sorted(df[df["farm_id"] == farm_id]["batch_id"].unique())
)

# CLEAN THE USER INPUT
#farm_id = farm_id.strip()
#batch_id = batch_id.strip()

st.write("FARM"+ farm_id)
st.write("BATCH"+ batch_id)
if st.button("🔍 Check Data Readiness"):

    #batch_df = df[(df["farm_id"] == farm_id) & (df["batch_id"] == batch_id)]
    
    batch_df = df[
    (df["farm_id"] == farm_id) &
    (df["batch_id"] == batch_id)
    ]
    st.dataframe(batch_df)

    st.write("Matching rows count:", len(batch_df))
    
    if batch_df.empty:
        st.error("❌ No data found for this Farm & Batch")
        st.stop()

    total_days = batch_df["day_number"].max()
    available_days = batch_df["day_number"].nunique()

    # -----------------------------
    # COVERAGE
    # -----------------------------
    coverage_score = min(available_days / total_days, 1.0)

    # -----------------------------
    # COMPLETENESS
    # -----------------------------
    critical_cols = [
        "feed_today_kg", "birds_alive",
        "mortality_today", "temp", "rh"
    ]

    completeness = 1 - batch_df[critical_cols].isna().mean().mean()

    # -----------------------------
    # RECENCY
    # -----------------------------
    last_days = batch_df.sort_values("day_number").tail(3)
    recency_score = 1.0 if last_days.isna().sum().sum() == 0 else 0.5

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    readiness_score = (
        0.4 * coverage_score +
        0.3 * completeness +
        0.2 * 1.0 +          # placeholder for consistency (next step)
        0.1 * recency_score
    ) * 100

    # -----------------------------
    # DISPLAY
    # -----------------------------
    st.subheader("📈 Readiness Summary")

    st.metric("Readiness Score", f"{readiness_score:.1f} / 100")
    st.write(f"📅 Days available: {available_days} / {total_days}")
    st.write(f"🧾 Data completeness: {completeness*100:.1f}%")

    if readiness_score >= 80:
        st.success("✅ Data is excellent for prediction")
    elif readiness_score >= 60:
        st.warning("⚠️ Data is usable but not ideal")
    else:
        st.error("❌ Data quality too low for prediction")
