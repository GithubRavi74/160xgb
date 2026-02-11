import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Daily Health", layout="wide")
st.title("🐔 iPoultry AI – Daily Health Prediction")

# -------------------------------------------------
# LOAD DATA & MODELS
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Load
df = load_data()
gain_model = load_model("weight_gain_xgb_model.pkl")
mort_model = load_model("mortality_xgb_model.pkl")

# -------------------------------------------------
# CLEAN IDS
# -------------------------------------------------
df["farm_id"] = df["farm_id"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()

# -------------------------------------------------
# SELECT FARM & BATCH
# -------------------------------------------------
st.subheader("🏭 Select Farm & Batch")

farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))
batch_id = st.selectbox(
    "Batch ID",
    sorted(df[df["farm_id"] == farm_id]["batch_id"].unique())
)

batch_hist = df[(df["farm_id"] == farm_id) & (df["batch_id"] == batch_id)].sort_values("day_number")

if batch_hist.empty:
    st.error("No historical data found for this batch")
    st.stop()

last = batch_hist.iloc[-1]

# -------------------------------------------------
# YESTERDAY + 3-DAY REFERENCES (SANITY SAFE)
# -------------------------------------------------
y_gain = float(last.get("daily_weight_gain_kg", 0))
y_mort_rate = last.get("mortality_today", 0) / max(last.get("birds_alive", 1), 1)

last3 = batch_hist.tail(3)

avg3_gain = last3["daily_weight_gain_kg"].mean() if len(last3) >= 2 else y_gain
avg3_mort = (last3["mortality_today"].sum() / max(last3["birds_alive"].mean(), 1)) if len(last3) >= 2 else y_mort_rate

# -------------------------------------------------
# AUTO CONTEXT
# -------------------------------------------------
current_day = int(last["day_number"]) + 1
birds_alive_yesterday = int(last["birds_alive"])

rolling_feed = batch_hist.tail(7)["feed_today_kg"].mean()
rolling_gain = batch_hist.tail(7)["daily_weight_gain_kg"].mean()

rolling_feed = rolling_feed if not np.isnan(rolling_feed) else last["feed_today_kg"]
rolling_gain = rolling_gain if not np.isnan(rolling_gain) else 0.05

# -------------------------------------------------
# FARMER INPUTS
# -------------------------------------------------
st.subheader("📥 Enter Today’s Data")

col1, col2 = st.columns(2)

with col1:
    birds_alive = st.number_input("🐔 Birds Alive Today", min_value=1, value=birds_alive_yesterday)
    mortality_today = st.number_input("☠ Mortality Today", min_value=0)
    feed_today = st.number_input("🌾 Feed Given Today (kg)", min_value=0.0)

with col2:
    sample_weight = st.number_input("⚖ Sample Avg Weight (kg) (optional)", min_value=0.0, value=0.0)
    temp = st.number_input("🌡 Temperature (°C)", value=float(last["temp"]))
    rh = st.number_input("💧 Humidity (%)", value=float(last["rh"]))
    nh = st.number_input("🧪 Ammonia (ppm)", value=float(last["nh"]))
    co = st.number_input("🧪 CO₂ (ppm)", value=float(last["co"]))

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔮 Predict Today"):

    feed_per_bird = feed_today / max(birds_alive, 1)
    mortality_rate = mortality_today / max(birds_alive, 1)

    X = pd.DataFrame([{
        "day_number": current_day,
        "birds_alive": birds_alive,
        "feed_today_kg": feed_today,
        "feed_per_bird": feed_per_bird,
        "mortality_rate": mortality_rate,
        "rolling_7d_feed": rolling_feed,
        "rolling_7d_gain": rolling_gain,
        "temp": temp,
        "rh": rh,
        "co": co,
        "nh": nh
    }])

    X = X[gain_model.get_booster().feature_names]

    gain_pred = max(0, float(gain_model.predict(X)[0]))
    mort_pred = max(0, int(mort_model.predict(X)[0]))

    # -------------------------------------------------
    # SAFE TREND ARROWS (YESTERDAY + 3 DAYS)
    # -------------------------------------------------
    def arrow(curr, ref):
        if ref <= 0:
            return "➖"
        return "⬆️" if curr > ref else "⬇️"

    gain_arrow = arrow(gain_pred, y_gain)
    mort_arrow = arrow(mortality_rate, y_mort_rate)

    gain_3d = arrow(gain_pred, avg3_gain)
    mort_3d = arrow(mortality_rate, avg3_mort)

    # -------------------------------------------------
    # DERIVED FCR
    # -------------------------------------------------
    fcr = feed_today / (gain_pred * birds_alive) if gain_pred > 0 else np.nan

    # -------------------------------------------------
    # HEALTH SCORE
    # -------------------------------------------------
    health_score = 100
    if mortality_rate > 0.005: health_score -= 30
    if temp > 34 or temp < 26: health_score -= 20
    if nh > 25: health_score -= 20
    if not np.isnan(fcr) and fcr > 2.2: health_score -= 15

    if health_score >= 75:
        status = "🟢 Normal - Flock condition appears stable 👍"
    elif health_score >= 50:
        status = "🟡 Watch - Monitor closely"
    else:
        status = "🔴 Risk - Immediate attention required"

    
    # -------------------------------------------------
    # CONFIDENCE SCORE (Model Stability Check)
    # -------------------------------------------------
    #st.subheader("🎯 Prediction Confidence")

    confidence = 100

    # Penalize if inputs deviate strongly from historical averages
    if abs(temp - batch_hist["temp"].mean()) > 5:
        confidence -= 15
    if abs(nh - batch_hist["nh"].mean()) > 10:
        confidence -= 15
    if abs(feed_today - batch_hist["feed_today_kg"].mean()) > batch_hist["feed_today_kg"].std():
        confidence -= 10
    if birds_alive < birds_alive_yesterday * 0.95:
        confidence -= 10

    confidence = max(50, confidence)

    if confidence >= 85:
        #st.success(f"High Confidence ({confidence}%)")
        confidence_label = "🟢🟢Very High Confidence"
    if confidence >= 75:
        #st.success(f"High Confidence ({confidence}%)")
        confidence_label = "🟢Good Condifence"    
    elif confidence >= 50:
        confidence_label = "🟡 Medium Confidence"
        #st.warning(f"Moderate Confidence ({confidence}%)")
    else:
        confidence_label = "🔴Low Confidence"
        #st.error(f"Low Confidence ({confidence}%) – Inputs far from historical patterns")

    
    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.subheader("📊 AI Assessment for Today")

    st.metric("Prediction Confidence", confidence_label)

    colA, colB, colC = st.columns(3)

    colA.metric("Health Status", status)
    colB.metric("Expected Daily Gain (kg)", round(gain_pred, 3), f"{gain_arrow} vs yesterday | {gain_3d} 3‑day")
    colC.metric("Mortality Risk Tomorrow", mort_pred, f"{mort_arrow} vs yesterday | {mort_3d} 3‑day")

    if not np.isnan(fcr):
        st.metric("Derived FCR", round(fcr, 2))

    st.subheader("📘 How to read this")
    st.info(
        "⬆️ means higher than recent days, ⬇️ means lower, ➖ means no reliable comparison. "
        "We compare today’s prediction against yesterday and the last 3 days to avoid false alarms."
    )


    # -------------------------------------------------
    # DISEASE-TYPE PROBABILITY (Explainable Rule Engine)
    # -------------------------------------------------
    disease_scores = {
        "Respiratory Stress": 0,
        "Gut / Enteric Stress": 0,
        "Heat Stress": 0
     }

    # Respiratory pattern
    if nh > 25:
        disease_scores["Respiratory Stress"] += 30
    if co > 3000:
        disease_scores["Respiratory Stress"] += 20
    if mortality_rate > 0.004:
        disease_scores["Respiratory Stress"] += 20
    if gain_pred < rolling_gain * 0.9:
        disease_scores["Respiratory Stress"] += 10

    # Gut / Enteric pattern
    if not np.isnan(fcr) and fcr > 2.3:
        disease_scores["Gut / Enteric Stress"] += 35
    if gain_pred < rolling_gain * 0.9:
        disease_scores["Gut / Enteric Stress"] += 25
    if mortality_rate > 0.003:
        disease_scores["Gut / Enteric Stress"] += 15

    # Heat stress pattern
    if temp > 32:
        disease_scores["Heat Stress"] += 40
    if rh > 75:
        disease_scores["Heat Stress"] += 20
    if gain_pred < y_gain * 0.9:
        disease_scores["Heat Stress"] += 15
    if mortality_rate > 0.004:
        disease_scores["Heat Stress"] += 10

    # Confidence adjustment
    confidence_factor = confidence / 100
    for k in disease_scores:
        disease_scores[k] = min(int(disease_scores[k] * confidence_factor), 95)

    # Sort highest first
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)

    
    # -------------------------------------------------
    # DETAILED DISEASE DASHBOARD SECTION
    # -------------------------------------------------
    st.subheader("🧬 Stress Probability Breakdown")

    for disease, score in sorted_diseases:
        st.write(f"**{disease} — {score}% likelihood**")
        st.progress(score / 100)

    top_disease, top_score = sorted_diseases[0]

    if top_score >= 50:
        st.error(f"Primary Concern: {top_disease}")
    elif top_score >= 30:
        st.warning(f"Emerging Risk: {top_disease}")
    else:
        st.success("No dominant stress pattern detected.")

    # -------------------------------------------------
    # RECOMMENDED ACTIONS PER STRESS TYPE
    # -------------------------------------------------
    st.subheader("🛠 Recommended Actions")

    if top_disease == "Respiratory Stress":
        st.write("• Increase ventilation rate immediately")
        st.write("• Remove wet litter patches")
        st.write("• Inspect birds for coughing or nasal discharge")
        st.write("• Validate ammonia and CO₂ sensor readings")

    elif top_disease == "Gut / Enteric Stress":
        st.write("• Check feed quality and storage")
        st.write("• Inspect droppings consistency")
        st.write("• Review recent feed changes")
        st.write("• Consider gut health supplements")

    elif top_disease == "Heat Stress":
        st.write("• Activate cooling / foggers")
        st.write("• Increase airflow speed")
        st.write("• Ensure cool drinking water availability")
        st.write("• Reduce stocking heat load if possible")

    # -------------------------------------------------
    # HISTORICAL COMPARISON (Batch Benchmarking)
    # -------------------------------------------------
    st.subheader("📊 Batch Benchmark Comparison")

    farm_avg_gain = df[df["farm_id"] == farm_id]["daily_weight_gain_kg"].mean()
    farm_avg_mort = df[df["farm_id"] == farm_id]["mortality_today"].mean()

    st.write(f"Farm Avg Daily Gain: {round(farm_avg_gain,3)} kg")
    st.write(f"Farm Avg Mortality: {round(farm_avg_mort,2)} birds/day")

    if gain_pred > farm_avg_gain:
        st.success("Gain above farm average")
    else:
        st.warning("Gain below farm average")

    if mort_pred < farm_avg_mort:
        st.success("Mortality below farm average")
    else:
        st.error("Mortality above farm average")
