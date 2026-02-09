import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")
st.title("📈 iPoultry AI – Batch Forecast & Backtest")

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

if st.button("📈 Run Forecast & Backtest"):

    # -------------------------------------------------
    # FULL BATCH DATA
    # -------------------------------------------------
    batch_df_full = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].sort_values("day_number")

    if batch_df_full.empty:
        st.error("❌ No data for selected batch")
        st.stop()

    st.subheader("📋 Full Batch Data (All Rows)")
    st.dataframe(batch_df_full, use_container_width=True)

    max_day = int(batch_df_full["day_number"].max())

    cutoff_day = st.slider(
        "Select Cutoff Day (simulate deleted future rows)",
        min_value=5,
        max_value=max_day - 1,
        value=max_day - 7
    )

    # -------------------------------------------------
    # CUTOFF DATA (MODEL INPUT HISTORY)
    # -------------------------------------------------
    batch_df_cut = batch_df_full[
        batch_df_full["day_number"] <= cutoff_day
    ]

    st.subheader("✂️ Data Used For Prediction (Cutoff History)")
    st.dataframe(batch_df_cut, use_container_width=True)

    # -------------------------------------------------
    # FUTURE ACTUAL (Hidden Data)
    # -------------------------------------------------
    future_actual = batch_df_full[
        (batch_df_full["day_number"] > cutoff_day) &
        (batch_df_full["day_number"] <= cutoff_day + 7)
    ]

    # -------------------------------------------------
    # BUILD FORECAST
    # -------------------------------------------------
    last_row = batch_df_cut.iloc[-1]

    age_today = int(last_row["day_number"])
    birds_alive = int(last_row["birds_alive"])
    feed_today = float(last_row["feed_today_kg"])
    mortality_today = float(last_row["mortality_today"])

    temp = float(last_row["temp"])
    rh   = float(last_row["rh"])
    co   = float(last_row["co"])
    nh   = float(last_row["nh"])

    recent = batch_df_cut.tail(7)

    if len(recent) >= 3:
        rolling_feed = recent["feed_today_kg"].mean()
        rolling_gain = recent["daily_weight_gain_kg"].mean()
    else:
        rolling_feed = feed_today
        rolling_gain = 0.045

    horizon = 7
    days = np.arange(age_today + 1, age_today + horizon + 1)

    rows = []

    for d in days:
        rows.append({
            "day_number": d,
            "birds_alive": birds_alive,
            "feed_today_kg": feed_today,
            "feed_per_bird": feed_today / birds_alive,
            "mortality_today": mortality_today,
            "mortality_rate": mortality_today / birds_alive,
            "rolling_7d_feed": rolling_feed,
            "rolling_7d_gain": rolling_gain,
            "temp": temp,
            "rh": rh,
            "co": co,
            "nh": nh
        })

    X_future = pd.DataFrame(rows)

    # -------------------------------------------------
    # PREDICT
    # -------------------------------------------------
    pred_weight = weight_model.predict(X_future)
    pred_mort   = mort_model.predict(X_future)
    pred_fcr    = fcr_model.predict(X_future)

    prediction_table = pd.DataFrame({
        "Day": days,
        "Predicted Weight (kg)": np.round(pred_weight, 3),
        "Predicted Mortality": np.maximum(0, np.round(pred_mort).astype(int)),
        "Predicted Feed / Bird": np.round(pred_fcr, 3)
    })

    st.subheader("🔮 Predictions From Cutoff")
    st.dataframe(prediction_table, use_container_width=True)

    # -------------------------------------------------
    # COMPARE WITH ACTUAL
    # -------------------------------------------------
    if not future_actual.empty:

        actual_table = future_actual[[
            "day_number",
            "sample_weight_kg",
            "mortality_today",
            "feed_per_bird"
        ]].copy()

        actual_table.columns = [
            "Day",
            "Actual Weight (kg)",
            "Actual Mortality",
            "Actual Feed / Bird"
        ]

        comparison = prediction_table.merge(
            actual_table,
            on="Day",
            how="left"
        )

        st.subheader("📊 Prediction vs Actual Comparison")
        st.dataframe(comparison, use_container_width=True)

        # PLOT
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=comparison["Day"],
            y=comparison["Predicted Weight (kg)"],
            name="Predicted",
            mode="lines+markers"
        ))

        fig.add_trace(go.Scatter(
            x=comparison["Day"],
            y=comparison["Actual Weight (kg)"],
            name="Actual",
            mode="lines+markers"
        ))

        fig.update_layout(
            title="Prediction vs Actual Weight",
            xaxis_title="Day",
            yaxis_title="Weight (kg)",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        if comparison["Actual Weight (kg)"].notna().any():
            mae = np.mean(
                np.abs(
                    comparison["Predicted Weight (kg)"] -
                    comparison["Actual Weight (kg)"]
                )
            )
            st.metric("📉 Weight MAE (7-day)", round(mae, 4))

    else:
        st.info("No actual future data available to compare.")
