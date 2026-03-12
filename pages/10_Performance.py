import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- CONFIG ---
st.set_page_config(page_title="iPoultry AI - Accuracy Tracker", layout="wide")

st.markdown("""
    <h1 style='color: black;'>🎯 Model <span style='color: #FFD700;'>Accuracy Tracker</span></h1>
    <h2 style='color: blue;'>Actual vs. Predicted Performance</h2>
    """, unsafe_allow_html=True)

# --- 1. DATA STORAGE SIMULATION ---
# In a real app, you would save this to a database or CSV. 
# For now, let's create a placeholder for historical accuracy.
performance_data = [
    {"Batch": "Batch_001", "Predicted": 2.65, "Actual": 2.62, "Date": "2026-01-15"},
    {"Batch": "Batch_002", "Predicted": 2.72, "Actual": 2.75, "Date": "2026-02-10"},
    {"Batch": "Batch_003", "Predicted": 2.55, "Actual": 2.48, "Date": "2026-02-28"},
]
df_perf = pd.DataFrame(performance_data)

# --- 2. INPUT NEW HARVEST DATA ---
st.subheader("📝 Record Final Harvest Results")
with st.expander("Add New Batch Results"):
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_name = st.text_input("Batch Name/ID", "Batch_004")
    with c2:
        pred_val = st.number_input("AI Predicted Weight (kg)", value=2.700)
    with c3:
        act_val = st.number_input("Actual Harvest Weight (kg)", value=2.680)
    
    if st.button("💾 Save to History"):
        st.success(f"Batch {batch_name} recorded! (Data saving logic would go here)")

st.divider()

# --- 3. ACCURACY ANALYSIS ---
df_perf['Error_KG'] = abs(df_perf['Predicted'] - df_perf['Actual'])
df_perf['Accuracy_Pct'] = (1 - (df_perf['Error_KG'] / df_perf['Actual'])) * 100

# Metrics
avg_accuracy = df_perf['Accuracy_Pct'].mean()
avg_error = df_perf['Error_KG'].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Overall AI Accuracy", f"{avg_accuracy:.2f}%")
m2.metric("Avg. Error Margin", f"{avg_error:.3f} kg")
m3.metric("Total Batches Tracked", len(df_perf))

# --- 4. VISUALIZATION ---
st.subheader("📈 Performance Trend")
fig = px.bar(df_perf, x="Batch", y=["Predicted", "Actual"], 
             barmode="group", 
             title="Predicted vs. Actual Harvest Weights",
             color_discrete_map={"Predicted": "#FFD700", "Actual": "#2E8B57"})

st.plotly_chart(fig, use_container_width=True)

# --- 5. BOSS EXPLANATION ---
st.info("""
**💡 Why this matters:** By tracking the difference between AI predictions and actual weights, we can 'fine-tune' the model. 
High accuracy (95%+) means we can confidently pre-sell our birds to buyers weeks before the harvest.
""")

st.divider()
st.caption("iPoultry AI Guard © 2026")
