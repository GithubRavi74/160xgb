import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Recursive Forecast", layout="wide")
st.title("📈 iPoultry AI – Recursive Growth Forecast & Validation")

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

gain_model = load_model("weight_gain_xgb_model.pkl")
mort_model = load_model("mortality_xgb_model.pkl")
fcr_model  = load_model("fcr_xgb_model.pkl")

# CLEAN IDS
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

if st.button("📈 Run Recursive Forecast"):

    # -------------------------------------------------
    # FULL BATCH DATA
    # -------------------------------------------------
    batch_full = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].sort_values("day_number")

    if batch_full.empty:
        st.error("No data found")
        st.stop()

    st.subheader("📋 Full Batch Data")
    st.dataframe(batch_full, use_container_width=True)

    max_day = int(batch_full["day_number"].max())

    cutoff_day = st.slider(
        "Select Cutoff Day (simulate unknown future)",
        min_value=5,
        max_value=max_day - 1,
        value=max_day - 7
    )

    # -------------------------------------------------
    # CUTOFF HISTORY
    # -------------------------------------------------
    history = batch_full[batch_full["day_number"] <= cutoff_day]
    future_actual = batch_full[
        (batch_full["day_number"] > cutoff_day) &
        (batch_full["day_number"] <= cutoff_day + 7)
    ]

    st.subheader("✂️ Data Visible to Model (Cutoff)")
    st.dataframe(history, use_container_width=True)

    last = history.iloc[-1]

    # -------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------
    current_day = int(last["day_number"])
    current_weight = float(last["sample_weight_kg"])
    current_birds = int(last["birds_alive"])

    feed_today = float(last["feed_today_kg"])
    mortality_today = float(last["mortality_today"])

    temp, rh, co, nh = last["temp"], last["rh"], last["co"], last["nh"]

    rolling_feed = history.tail(7)["feed_today_kg"].mean()
    rolling_gain = history.tail(7)["daily_weight_gain_kg"].mean()

    # -------------------------------------------------
    # RECURSIVE SIMULATION
    # -------------------------------------------------
    horizon = 7
    predictions = []

    for step in range(1, horizon + 1):

        day = current_day + step

        X = pd.DataFrame([{
            "day_number": day,
            "birds_alive": current_birds,
            "feed_today_kg": feed_today,
            "feed_per_bird": feed_today / current_birds,
            "mortality_today": mortality_today,
            "mortality_rate": mortality_today / current_birds,
            "rolling_7d_feed": rolling_feed,
            "rolling_7d_gain": rolling_gain,
            "temp": temp,
            "rh": rh,
            "co": co,
            "nh": nh
        }])

        gain_pred = gain_model.predict(X)[0]
        mort_pred = max(0, int(mort_model.predict(X)[0]))
        fcr_pred  = fcr_model.predict(X)[0]

        # UPDATE BIOLOGICAL STATE
        current_weight += gain_pred
        current_birds = max(1, current_birds - mort_pred)

        # Update rolling gain
        rolling_gain = (rolling_gain * 6 + gain_pred) / 7

        predictions.append({
            "Day": day,
            "Predicted Weight (kg)": round(current_weight, 3),
            "Predicted Daily Gain (kg)": round(gain_pred, 4),
            "Predicted Mortality": mort_pred,
            "Birds Alive": current_birds,
            "Predicted Feed / Bird": round(fcr_pred, 3)
        })

    pred_df = pd.DataFrame(predictions)

    st.subheader("🔮 Recursive Predictions")
    st.dataframe(pred_df, use_container_width=True)

    # -------------------------------------------------
    # COMPARISON
    # -------------------------------------------------
    if not future_actual.empty:
        actual = future_actual[[
            "day_number",
            "sample_weight_kg"
        ]].copy()

        actual.columns = ["Day", "Actual Weight (kg)"]

        compare = pred_df.merge(actual, on="Day", how="left")

        st.subheader("📊 Prediction vs Actual")
        st.dataframe(compare, use_container_width=True)

        mae = np.mean(
            np.abs(compare["Predicted Weight (kg)"] - compare["Actual Weight (kg)"])
        )

        st.metric("📉 Weight MAE (7-day)", round(mae, 4))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=compare["Day"],
            y=compare["Predicted Weight (kg)"],
            name="Predicted",
            mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=compare["Day"],
            y=compare["Actual Weight (kg)"],
            name="Actual",
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Recursive Forecast vs Actual",
            xaxis_title="Day",
            yaxis_title="Weight (kg)",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
