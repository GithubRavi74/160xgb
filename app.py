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

# -------------------------------------------------
# FORECAST SECTION
# -------------------------------------------------
if st.button("📈 Forecast Next 7 Days"):

    batch_df_full = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].sort_values("day_number")

    if batch_df_full.empty:
        st.error("❌ No data for selected batch")
        st.stop()

    max_day = int(batch_df_full["day_number"].max())

    if max_day < 10:
        st.error("❌ Not enough historical data for backtesting")
        st.stop()

    # -----------------------------------
    # CUTOFF DAY SLIDER (Simulated deletion)
    # -----------------------------------
    cutoff_day = st.slider(
        "Select Cutoff Day (simulate prediction from this day)",
        min_value=5,
        max_value=max_day - 1,
        value=max_day - 7
    )

    # Data available to model
    batch_df = batch_df_full[
        batch_df_full["day_number"] <= cutoff_day
    ]

    # Hidden future (actual values for comparison)
    future_actual = batch_df_full[
        (batch_df_full["day_number"] > cutoff_day) &
        (batch_df_full["day_number"] <= cutoff_day + 7)
    ]

    if batch_df.empty:
        st.error("❌ No data before cutoff")
        st.stop()

    # -------------------------------------------------
    # LATEST STATE FROM CUTOFF
    # -------------------------------------------------
    last_row = batch_df.iloc[-1]

    age_today = int(last_row["day_number"])
    birds_alive = int(last_row["birds_alive"])
    feed_today = float(last_row["feed_today_kg"])
    mortality_today = float(last_row["mortality_today"])

    temp = float(last_row["temp"])
    rh   = float(last_row["rh"])
    co   = float(last_row["co"])
    nh   = float(last_row["nh"])

    # -------------------------------------------------
    # ROLLING FEATURES
    # -------------------------------------------------
    recent = batch_df.tail(7)

    if len(recent) >= 3:
        rolling_feed = recent["feed_today_kg"].mean()
        rolling_gain = recent["daily_weight_gain_kg"].mean()
        rolling_note = "✅ Rolling features computed from batch data"
    else:
        rolling_feed = feed_today
        rolling_gain = 0.045
        rolling_note = "⚠️ Insufficient continuity — using safe defaults"

    st.info(rolling_note)

    # -------------------------------------------------
    # BUILD FUTURE FEATURE MATRIX
    # -------------------------------------------------
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

    X = pd.DataFrame(rows)

    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------
    pred_weight = weight_model.predict(X)
    pred_mort   = mort_model.predict(X)
    pred_fcr    = fcr_model.predict(X)

    # -------------------------------------------------
    # PREDICTION OUTPUT
    # -------------------------------------------------
    out = pd.DataFrame({
        "Day": days,
        "Predicted Weight (kg)": np.round(pred_weight, 3),
        "Predicted Mortality": np.maximum(0, np.round(pred_mort).astype(int)),
        "Predicted Feed / Bird": np.round(pred_fcr, 3)
    })

    # -------------------------------------------------
    # COMPARE WITH ACTUAL (IF AVAILABLE)
    # -------------------------------------------------
    if not future_actual.empty:

        compare = future_actual[[
            "day_number",
            "sample_weight_kg",
            "mortality_today",
            "feed_per_bird"
        ]].copy()

        compare.columns = [
            "Day",
            "Actual Weight (kg)",
            "Actual Mortality",
            "Actual Feed / Bird"
        ]

        final_compare = out.merge(compare, on="Day", how="left")

        st.subheader("📊 Prediction vs Actual")
        st.dataframe(final_compare, use_container_width=True)

        # -----------------------------
        # METRICS
        # -----------------------------
        if final_compare["Actual Weight (kg)"].notna().any():
            mae = np.mean(
                np.abs(
                    final_compare["Predicted Weight (kg)"] -
                    final_compare["Actual Weight (kg)"]
                )
            )

            st.metric("📉 Weight MAE (7-day)", round(mae, 4))

        # -----------------------------
        # PLOT
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=final_compare["Day"],
            y=final_compare["Predicted Weight (kg)"],
            name="Predicted",
            mode="lines+markers"
        ))

        fig.add_trace(go.Scatter(
            x=final_compare["Day"],
            y=final_compare["Actual Weight (kg)"],
            name="Actual",
            mode="lines+markers"
        ))

        fig.update_layout(
            title="Prediction vs Actual Weight",
            xaxis_title="Bird Age (days)",
            yaxis_title="Weight (kg)",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("📊 Forecast Results")
        st.dataframe(out, use_container_width=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=out["Day"],
            y=out["Predicted Weight (kg)"],
            name="Predicted",
            mode="lines+markers"
        ))

        fig.update_layout(
            title="7-Day Growth Forecast",
            xaxis_title="Bird Age (days)",
            yaxis_title="Weight (kg)",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # CONFIDENCE NOTE
    # -------------------------------------------------
    if "⚠️" in rolling_note:
        st.warning(
            "Prediction used estimated rolling features due to limited history."
        )
    else:
        st.success("Prediction based on strong continuous batch data.")
