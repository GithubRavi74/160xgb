import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")
st.title("📈 iPoultry AI – Batch Growth Forecast")

# -------------------------------------------------
# IDEAL CURVE (REFERENCE ONLY)
# -------------------------------------------------
IDEAL_WEIGHT = {
    1:0.061,2:0.079,3:0.099,4:0.122,5:0.148,6:0.176,7:0.208,
    8:0.242,9:0.28,10:0.321,11:0.366,12:0.414,13:0.465,
    14:0.519,15:0.576,16:0.637,17:0.701,18:0.768,19:0.837,
    20:0.91,21:0.985,22:1.062,23:1.142,24:1.225,25:1.309,
    26:1.395,27:1.483,28:1.573,29:1.664,30:1.757,
    31:1.851,32:1.946,33:2.041,34:2.138,35:2.235,
    36:2.332,37:2.43,38:2.527,39:2.625,40:2.723
}

# -------------------------------------------------
# LOAD DATA & MODELS
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ml_ready_daily.csv")

@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

df = load_data()
weight_model = load_model("weight_xgb_model.pkl")
mort_model   = load_model("mortality_xgb_model.pkl")
fcr_model    = load_model("fcr_xgb_model.pkl")

df["farm_id"] = df["farm_id"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
st.subheader("🏭 Select Farm & Batch")

farm_id = st.selectbox("Farm ID", sorted(df["farm_id"].unique()))
batch_id = st.selectbox(
    "Batch ID",
    sorted(df[df["farm_id"] == farm_id]["batch_id"].unique())
)

#ideal_mode = st.checkbox("🧪 Ideal Condition Test (validation only)", value=False)

# -------------------------------------------------
# FORECAST
# -------------------------------------------------
if st.button("📈 Forecast Next 7 Days"):

    batch_df = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].sort_values("day_number")

    if batch_df.empty:
        st.error("❌ No data available for this batch")
        st.stop()

    last_row = batch_df.iloc[-1]

    age_today = int(last_row["day_number"])
    birds_alive = int(last_row["birds_alive"])
    feed_today = float(last_row["feed_today_kg"])
    mortality_today = float(last_row["mortality_today"])
    temp, rh, co, nh = last_row[["temp","rh","co","nh"]]

    # -------------------------------------------------
    # DATA QUALITY CHECK
    # -------------------------------------------------
    recent = batch_df.tail(7)
    continuous_days = recent["day_number"].diff().dropna().eq(1).sum()
    rolling_real = False

    if len(recent) >= 3 and continuous_days >= 2:
        rolling_feed = recent["feed_today_kg"].mean()
        rolling_gain = recent["daily_weight_gain_kg"].mean()
        rolling_real = True
    else:
        rolling_feed = feed_today
        rolling_gain = 0.045

    # -------------------------------------------------
    # GOLD BATCH LOGIC
    # -------------------------------------------------
    coverage = batch_df["day_number"].nunique() / batch_df["day_number"].max()
    avg_mortality = batch_df["mortality_rate"].mean()

    is_gold_batch = (
        coverage >= 0.8 and
        rolling_real and
        avg_mortality < 0.04
    )

    # -------------------------------------------------
    # CONFIDENCE BASE SCORE
    # -------------------------------------------------
    continuity_score = min(continuous_days / 6, 1.0)
    rolling_score = 1.0 if rolling_real else 0.6
    completeness_score = 1 - recent[["feed_today_kg","mortality_today","temp","rh"]].isna().mean().mean()

    base_confidence = (
        0.4 * continuity_score +
        0.3 * rolling_score +
        0.2 * completeness_score +
        0.1 * 1.0
    ) * 100

    if is_gold_batch:
        base_confidence += 7

    base_confidence = round(min(base_confidence, 98), 1)

    # -------------------------------------------------
    # FUTURE FEATURES
    # -------------------------------------------------
    horizon = 7
    days = np.arange(age_today + 1, age_today + horizon + 1)

    rows, confidences, conf_colors = [], [], []

    for i, d in enumerate(days):

        #if ideal_mode:
        #    feed = feed_today * 1.35
        #   mortality = 0
        #   temp_i, rh_i, co_i, nh_i = 28, 60, 3, 5
        #else:
        #    feed = feed_today
        #    mortality = mortality_today
        #    temp_i, rh_i, co_i, nh_i = temp, rh, co, nh

        feed = feed_today
        mortality = mortality_today
        temp_i, rh_i, co_i, nh_i = temp, rh, co, nh
        
        confidence = max(base_confidence - i * 0.8, 60)

        if confidence >= 85:
            conf_colors.append("🟢")
        elif confidence >= 70:
            conf_colors.append("🟡")
        else:
            conf_colors.append("🔴")

        rows.append({
            "day_number": d,
            "birds_alive": birds_alive,
            "feed_today_kg": feed,
            "feed_per_bird": feed / birds_alive,
            "mortality_today": mortality,
            "mortality_rate": mortality / birds_alive,
            "rolling_7d_feed": rolling_feed,
            "rolling_7d_gain": rolling_gain,
            "temp": temp_i,
            "rh": rh_i,
            "co": co_i,
            "nh": nh_i
        })

        confidences.append(round(confidence, 1))

    X = pd.DataFrame(rows)

    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------
    pred_weight = weight_model.predict(X)
    pred_mort   = mort_model.predict(X)
    pred_fcr    = fcr_model.predict(X)

    # -------------------------------------------------
    # OUTPUT TABLE
    # -------------------------------------------------
    out = pd.DataFrame({
        "Day": days,
        "Predicted Weight (kg)": np.round(pred_weight, 3),
        "Ideal Weight (kg)": [IDEAL_WEIGHT.get(int(d), None) for d in days],
        "Confidence": [f"{c}% {e}" for c, e in zip(confidences, conf_colors)]
    })

    st.subheader("📊 Forecast Results")
    st.dataframe(out, use_container_width=True, hide_index=True)

    # -------------------------------------------------
    # CONFIDENCE BAR CHART
    # -------------------------------------------------
    st.subheader("📶 Prediction Confidence")

    fig_conf = go.Figure()
    fig_conf.add_bar(
        x=days,
        y=confidences,
        marker_color=["green" if c>=85 else "orange" if c>=70 else "red" for c in confidences]
    )
    fig_conf.update_layout(
        yaxis_title="Confidence %",
        xaxis_title="Day",
        height=300
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # -------------------------------------------------
    # COMPARATIVE CURVE
    # -------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days,
        y=pred_weight,
        name="Predicted Weight (XGB)",
        mode="lines+markers"
    ))

    fig.add_trace(go.Scatter(
        x=days,
        y=[IDEAL_WEIGHT.get(int(d), None) for d in days],
        name="Ideal Weight (Ross)",
        mode="lines",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title="Predicted vs Ideal Growth Curve",
        xaxis_title="Bird Age (days)",
        yaxis_title="Weight (kg)",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # FARMER GUIDANCE
    # -------------------------------------------------
    st.subheader("🧑‍🌾 Farmer Guidance")

    if is_gold_batch:
        st.success("🏆 Gold Batch detected — excellent discipline and data quality.")
    elif rolling_real:
        st.info("👍 Good data. Maintain daily records for best results.")
    else:
        st.warning("⚠️ Data gaps detected. Daily feed and mortality records will improve accuracy.")
