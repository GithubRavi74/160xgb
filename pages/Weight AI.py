import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Growth Forecast", layout="wide")
#st.title("📈 iPoultry AI Guard")
#st.subheader("Bird Harvest Weight Prediction")

st.markdown(
    """
    <h1 style='color: black;'>
        📈 iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h2 style='color: green;'>Bird Harvest Weight Prediction</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# MANUAL INPUT SECTION
# -------------------------------------------------
st.subheader("📝 Enter the Bird Information")

col1, col2 = st.columns(2)

current_day = col1.number_input(
    "Bird Age (Day)",
    min_value=1,
    max_value=60,
    value=int(last["day_number"])
)

today_weight = col2.number_input(
    "Today's Average Weight (kg)",
    min_value=0.01,
    max_value=5.0,
    value=float(last.get("sample_weight_kg", 1.0)),
    step=0.01
)

#birds_alive = int(last["birds_alive"])

if st.button("🚀 Provide Forecasting"):
    
    st.text("\n")
    # -------------------------------------------------
    # IDEAL WEIGHT CURVE (Ross 308 Example)
    # -------------------------------------------------
    #ideal_days = np.array([0, 7, 14, 21, 28, 35])
    #ideal_weights = np.array([0.043, 0.208, 0.519, 0.985, 1.573, 2.235])

    ideal_days = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35
    ])

    ideal_weights = np.array([
    0.043, 0.061, 0.079, 0.099, 0.122, 0.148, 0.176, 0.208, 0.242, 0.28,
    0.321, 0.366, 0.414, 0.465, 0.519, 0.576, 0.637, 0.701, 0.768, 0.837,
    0.91, 0.985, 1.062, 1.142, 1.225, 1.309, 1.395, 1.483, 1.573, 1.664,
    1.757, 1.851, 1.946, 2.041, 2.138, 2.235
    ])
    
    growth_curve = interp1d(
        ideal_days,
        ideal_weights,
        kind="cubic",
        fill_value="extrapolate"
    )
    
    ideal_current_weight = float(growth_curve(current_day))
    ideal_final_weight = float(growth_curve(35))
    
    # -------------------------------------------------
    # PERFORMANCE VS IDEAL
    # -------------------------------------------------
    performance_ratio = today_weight / ideal_current_weight
    adjusted_final_weight = ideal_final_weight * performance_ratio
    
    # Status classification
    if performance_ratio > 1.05:
        status = "TODAY'S BIRD WEIGHT STATUS : Ahead of Target"
        status_color = "success"
    elif performance_ratio >= 0.95:
        status = "TODAY'S BIRD WEIGHT STATUS : On Track"
        status_color = "warning"
    else:
        status = "TODAY'S BIRD WEIGHT STATUS : Behind Target"
        status_color = "error"

  
    
    # -------------------------------------------------
    # FINAL PREDICTION
    # -------------------------------------------------
    predicted_final_weight = adjusted_final_weight  
    
    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------
    st.subheader("📊 Today's weight comparison with Today's Ideal Weight")
    
    colA, colB = st.columns(2)
    
    colA.metric("Ideal Weight Today (kg)", f"{ideal_current_weight:.2f}")
    colB.metric("Actual Weight Today (kg)", f"{today_weight:.2f}")
   
    
    if status_color == "success":
        st.success(f"🟢 {status}")
    elif status_color == "warning":
        st.warning(f"🟡 {status}")
    else:
        st.error(f"🔴 {status}")
        
     
    st.text("\n")
    st.text("\n")
    # -------------------------------------------------
    # HARVEST FORECAST
    # -------------------------------------------------
    #st.subheader("🚀 Harvest Weight Forecast\n\n")
    st.markdown(
    "<h2 style='text-align: center;'>🚀 Harvest Weight Forecast</h2>",
    unsafe_allow_html=True
    )
    st.text("\n")
    st.subheader("Day 35 Weight Projection")
    
    c1, c2, c3 = st.columns(3)
    
    c1.metric("Ideal Target Weight (kg)", f"{ideal_final_weight:.2f}")
    c2.metric("Predicted Harvest Weight (kg)", f"{predicted_final_weight:.2f}")
    
    
    # -------------------------------------------------
    # GROWTH CURVE VISUALIZATION
    # -------------------------------------------------
    st.subheader("📈 Growth Curve Projection")
    
    days = np.arange(1, 36)
    ideal_curve = growth_curve(days)
    performance_curve = ideal_curve * performance_ratio
        
    plt.figure()
    plt.plot(days, ideal_curve)
    plt.plot(days, performance_curve)
     
    plt.scatter(current_day, today_weight)
    plt.xlabel("Day")
    plt.ylabel("Weight (kg)")
    st.pyplot(plt)

 


