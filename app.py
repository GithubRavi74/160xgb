import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="iPoultry AI – Growth Forecast",
    layout="wide"
)

st.title("🐔 iPoultry AI Module – Chicken Growth Forecast")

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

weight_model = load_model("weight_xgb_model.pkl")
mort_model   = load_model("mortality_xgb_model.pkl")
fcr_model    = load_model("fcr_xgb_model.pkl")

# ----------------------------------------------------------
# IDEAL CURVE (REFERENCE ONLY – NOT TRAINED ON)
# ----------------------------------------------------------
IDEAL_WEIGHT = {
    1:0.061,2:0.079,3:0.099,4:0.122,5:0.148,6:0.176,7:0.208,
    8:0.242,9:0.28,10:0.321,11:0.366,12:0.414,13:0.465,
    14:0.519,15:0.576,16:0.637,17:0.701,18:0.768,19:0.837,
    20:0.91,21:0.985,22:1.062,23:1.142,24:1.225,25:1.309,
    26:1.395,27:1.483,28:1.573,29:1.664,30:1.757,
    31:1.851,32:1.946,33:2.041,34:2.138,35:2.235,
    36:2.332,37:2.43,38:2.527,39:2.625,40:2.723
}

# ----------------------------------------------------------
# TEMPORARY IDEAL CHECKBOX FOR TESTING
# ----------------------------------------------------------
ideal_mode = st.checkbox("🧪 Run Ideal Condition Test", value=False)



# ----------------------------------------------------------
# INPUTS
# ----------------------------------------------------------
st.subheader("📥 Enter Current Farm Conditions")

c1, c2, c3 = st.columns(3)

with c1:
    age_today = st.number_input("Bird Age (days)", 1, 45, 14)
    birds_alive = st.number_input("Birds Alive", 1, 200000, 940)

with c2:
    feed_today = st.number_input("Feed Today (kg)", 0.0, 5000.0, 120.0)
    mortality_today = st.number_input("Mortality Today (birds)", 0, 2000, 2)

with c3:
    temp = st.number_input("Avg Temperature (°C)", 15.0, 40.0, 28.0)
    rh   = st.number_input("Avg Humidity (%)", 20.0, 95.0, 70.0)
    co   = st.number_input("CO (ppm)", 0.0, 50.0, 3.0)
    nh   = st.number_input("NH₃ (ppm)", 0.0, 50.0, 2.0)

# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------
if st.button("📈 Predict Next 7 Days"):

    horizon = 7
    days = np.arange(age_today, age_today + horizon + 1)

    rows = []

    # -----------------------------
    # FEATURE BUILDING BLOCK
    # -----------------------------
    # We intentionally override inputs when ideal_mode = True
    # to validate whether the model responds to perfect conditions
    rows = []

    rows = []

    for d in days:

        if ideal_mode:
            feed = feed_today * 1.4
            mortality = 0
            temp_i, rh_i, co_i, nh_i = 28, 60, 3, 2
            rolling_gain = 0.065   # ideal growth momentum
        else:
            feed = feed_today
            mortality = mortality_today
            temp_i, rh_i, co_i, nh_i = temp, rh, co, nh
            rolling_gain = 0.05

        feed_per_bird = feed / birds_alive
        mortality_rate = mortality / birds_alive

        rows.append({
        "day_number": d,
        "birds_alive": birds_alive,
        "feed_today_kg": feed,
        "feed_per_bird": feed_per_bird,
        "mortality_today": mortality,
        "mortality_rate": mortality_rate,
        "rolling_7d_feed": feed * 7,
        "rolling_7d_gain": rolling_gain,
        "temp": temp_i,
        "rh": rh_i,
        "co": co_i,
        "nh": nh_i
        })

        X = pd.DataFrame(rows)


    
    # -----------------------------
    # MODEL PREDICTIONS
    # -----------------------------
    pred_weight = weight_model.predict(X)
    pred_mort   = mort_model.predict(X)
    pred_fcr    = fcr_model.predict(X)

    # -----------------------------
    # OUTPUT TABLE
    # -----------------------------
    df_out = pd.DataFrame({
        "Age (days)": days,
        "Predicted Weight (kg)": np.round(pred_weight, 3),
        "Ideal Weight (kg)": [IDEAL_WEIGHT.get(int(d), None) for d in days],
        "Predicted Mortality": np.maximum(0, np.round(pred_mort).astype(int)),
        "Predicted Feed / Bird": np.round(pred_fcr, 3)
    })

    st.subheader("📊 Forecast Table")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # -----------------------------
    # PLOT
    # -----------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_out["Age (days)"],
        y=df_out["Predicted Weight (kg)"],
        name="Predicted Weight (XGBR)",
        mode="lines+markers",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_out["Age (days)"],
        y=df_out["Ideal Weight (kg)"],
        name="Ideal Weight (Ross)",
        mode="lines",
        line=dict(width=3, dash="dash")
    ))

    fig.update_layout(
        title="Predicted vs Ideal Growth Curve",
        xaxis_title="Bird Age (days)",
        yaxis_title="Weight (kg)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
