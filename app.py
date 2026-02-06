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
# LOAD DATA + MODELS
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

ideal_mode = st.checkbox("🧪 Ideal Condition Test (validation only)", value=False)

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
    # ROLLING FEATURE SAFETY
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
        rolling_gain = 0.045  # learned baseline

    # -------------------------------------------------
    # CONFIDENCE SCORE (BASE)
    # -------------------------------------------------
    continuity_score = min(continuous_days / 6, 1.0)
    rolling_score = 1.0 if rolling_real else 0.6
    completeness_score = 1 - recent[["feed_today_kg","mortality_today","temp","rh"]].isna().mean().mean()
    recency_score = 1.0

    base_confidence = (
        0.4 * continuity_score +
        0.3 * rolling_score +
        0.2 * completeness_score +
        0.1 * recency_score
    ) * 100

    base_confidence = round(min(base_confidence, 95), 1)

    # -------------------------------------------------
    # BUILD FUTURE FEATURES
    # -------------------------------------------------
    horizon = 7
    days = np.arange(age_today + 1, age_today + horizon + 1)

    rows, confidences = [], []

    for i, d in enumerate(days):

        if ideal_mode:
            feed = feed_today * 1.35
            mortality = 0
            temp_i, rh_i, co_i, nh_i = 28, 60, 3, 5
        else:
            feed = feed_today
            mortality = mortality_today
            temp_i, rh_i, co_i, nh_i = temp, rh, co, nh

        confidence = max(base_confidence - i * 0.8, 60)

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
    # OUTPUT
    # -------------------------------------------------
    out = pd.DataFrame({
        "Day": days,
        "Predicted Weight (kg)": np.round(pred_weight, 3),
        "Predicted Mortality": np.maximum(0, np.round(pred_mort).astype(int)),
        "Predicted Feed / Bird": np.round(pred_fcr, 3),
        "Confidence (%)": confidences
    })

    st.subheader("📊 Forecast Results")
    st.dataframe(out, use_container_width=True, hide_index=True)

    # -------------------------------------------------
    # FARMER GUIDANCE PANEL
    # -------------------------------------------------
    st.subheader("🧑‍🌾 Farmer Guidance")

    if rolling_real:
        st.success("✅ Excellent data discipline. Rolling trends are calculated from your actual records.")
    else:
        st.warning(
            "⚠️ Data gaps detected. Record feed, mortality, and environment daily "
            "for higher prediction accuracy."
        )

    if base_confidence >= 85:
        st.success("📈 High confidence forecast")
    elif base_confidence >= 70:
        st.info("ℹ️ Medium confidence forecast — accuracy will improve with continuous data")
    else:
        st.warning("❗ Low confidence — predictions are approximate")

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=out["Day"],
        y=out["Predicted Weight (kg)"],
        name="Predicted Weight",
        mode="lines+markers"
    ))

    fig.update_layout(
        title="7-Day Growth Forecast",
        xaxis_title="Bird Age (days)",
        yaxis_title="Weight (kg)",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)
