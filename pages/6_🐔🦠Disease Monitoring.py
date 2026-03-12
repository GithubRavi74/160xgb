import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import load_farm_data

st.set_page_config(page_title="Disease Dashboard", layout="wide")
st.title("📊 Disease Monitoring Dashboard")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = load_farm_data()



#######################################################
# COMMENTED BELOW SO AS TO NOT HARDCODE THE CSV
#def load_data():
    #return pd.read_csv("farm_data.csv")
#df = load_data()
#######################################################
 
if df.empty:
    st.error("No farm data available.")
    st.stop()

# -------------------------------------------------
# Basic KPIs
# -------------------------------------------------
st.subheader("📌 Current Batch Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Mortality", f"{df['mortality_today'].mean():.2f}")
col2.metric("Avg Temperature", f"{df['temp'].mean():.1f} °C")
col3.metric("Avg Ammonia", f"{df['nh'].mean():.1f} ppm")

# -------------------------------------------------
# Mortality Trend
# -------------------------------------------------
st.subheader("📈 Mortality Trend")

plt.figure()
plt.plot(df["day_number"], df["mortality_today"])
plt.xlabel("Day")
plt.ylabel("Mortality")
st.pyplot(plt)

# -------------------------------------------------
# Environmental Trends
# -------------------------------------------------
st.subheader("🌡 Environmental Trends")

plt.figure()
plt.plot(df["day_number"], df["temp"])
plt.xlabel("Day")
plt.ylabel("Temperature")
st.pyplot(plt)

plt.figure()
plt.plot(df["day_number"], df["nh"])
plt.xlabel("Day")
plt.ylabel("Ammonia")
st.pyplot(plt)


# -------------------------------------------------
# Risk Indicator
# -------------------------------------------------
st.subheader("⚠ Risk Indicator")

latest = df.iloc[-1]

risk_score = 0

if latest["temp"] > 32:
    risk_score += 1
if latest["nh"] > 25:
    risk_score += 1
if latest["mortality_today"] > df["mortality_today"].mean() * 1.5:
    risk_score += 1


if risk_score >= 2:
    st.error("High Disease Risk Detected")
elif risk_score == 1:
    st.warning("Moderate Risk")
else:
    st.success("Low Risk")
