import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Disease Dashboard", layout="wide")

st.title("📊 Disease Monitoring Dashboard")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("farm_data.csv")

df = load_data()

if df.empty:
    st.error("No farm data available.")
    st.stop()

# -------------------------------------------------
# Basic KPIs
# -------------------------------------------------
st.subheader("📌 Current Batch Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Mortality", f"{df['mortality'].mean():.2f}")
col2.metric("Avg Temperature", f"{df['temperature'].mean():.1f} °C")
col3.metric("Avg Ammonia", f"{df['ammonia'].mean():.1f} ppm")

# -------------------------------------------------
# Mortality Trend
# -------------------------------------------------
st.subheader("📈 Mortality Trend")

plt.figure()
plt.plot(df["day"], df["mortality"])
plt.xlabel("Day")
plt.ylabel("Mortality")
st.pyplot(plt)

# -------------------------------------------------
# Environmental Trends
# -------------------------------------------------
st.subheader("🌡 Environmental Trends")

plt.figure()
plt.plot(df["day"], df["temperature"])
plt.xlabel("Day")
plt.ylabel("Temperature")
st.pyplot(plt)

plt.figure()
plt.plot(df["day"], df["ammonia"])
plt.xlabel("Day")
plt.ylabel("Ammonia")
st.pyplot(plt)

# -------------------------------------------------
# Risk Indicator
# -------------------------------------------------
st.subheader("⚠ Risk Indicator")

latest = df.iloc[-1]

risk_score = 0

if latest["temperature"] > 32:
    risk_score += 1
if latest["ammonia"] > 25:
    risk_score += 1
if latest["mortality"] > df["mortality"].mean() * 1.5:
    risk_score += 1

if risk_score >= 2:
    st.error("High Disease Risk Detected")
elif risk_score == 1:
    st.warning("Moderate Risk")
else:
    st.success("Low Risk")
