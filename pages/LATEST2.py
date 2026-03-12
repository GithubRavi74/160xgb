import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import plotly.express as px
from datetime import datetime

# --- 1. IDEAL WEIGHT REFERENCE ---
IDEAL_WEIGHT = {
    1:0.061, 2:0.079, 3:0.099, 4:0.122, 5:0.148, 6:0.176, 7:0.208,
    8:0.242, 9:0.28, 10:0.321, 11:0.366, 12:0.414, 13:0.465,
    14:0.519, 15:0.576, 16:0.637, 17:0.701, 18:0.768, 19:0.837,
    20:0.91, 21:0.985, 22:1.062, 23:1.142, 24:1.225, 25:1.309,
    26:1.395, 27:1.483, 28:1.573, 29:1.664, 30:1.757,
    31:1.851, 32:1.946, 33:2.041, 34:2.138, 35:2.235,
    36:2.332, 37:2.43, 38:2.527, 39:2.625, 40:2.723,
    41:2.820, 42:2.915, 43:3.010, 44:3.100, 45:3.190
}

# --- 2. CONFIG ---
st.set_page_config(page_title="iPoultry AI Guard", layout="wide")

st.markdown("""
    <h1 style='color: black;'>📈 iPoultry <span style='color: #FFD700;'>AI Guard</span></h1>
    <h2 style='color: green;'> Farm Data Trained AI Model's Weight, FCR & Profit Analytics</h2>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
MODEL_PATH = "kishorebatches_weight_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error(f"❌ Model file '{MODEL_PATH}' not found.")
else:
    # --- 4. MAIN INPUT AREA ---
    st.subheader("📝 Live Farm & Market Inputs")
    
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    with t_col1:
        st.markdown("**📊 Batch Stats**")
        initial_flock = st.number_input("Initial Flock Size", 100, 100000, 5000)
        total_mortality = st.number_input("Total Mortality", 0, initial_flock, 150)
        day_number = st.number_input("Day Number (Age)", 1, 40, 21)
        birds_alive = initial_flock - total_mortality

    with t_col2:
        st.markdown("**🌡️ Environment**")
        temp = st.slider("Mean Temp (°C)", 15.0, 40.0, 28.5)
        rh = st.slider("Humidity (%)", 20.0, 100.0, 65.0)
        feed_today = st.number_input("Feed Today (kg)", 0.0, 5000.0, 450.0)

    with t_col3:
        st.markdown("**🎯 Forecast Targets**")
        harvest_day = st.number_input("Target Harvest Day", 30, 45, 40)
        total_feed_to_date = st.number_input("Total Feed Used (kg)", value=feed_today * day_number)
        st.info(f"Birds Currently Alive: {birds_alive}")

    with t_col4:
        st.markdown("**💰 Market Prices**")
        price_per_kg = st.number_input("Chicken Price ($/kg)", value=2.50)
        feed_cost_per_kg = st.number_input("Feed Cost ($/kg)", value=0.65)
        chick_cost = st.number_input("Cost per Chick ($)", value=0.45)

    # Internal Calculations
    heat_index = temp + (0.33 * rh) - 0.7
    feed_per_bird = feed_today / birds_alive if birds_alive > 0 else 0
    
    with st.expander("🛠️ Advanced Settings"):
        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            roll_feed = st.number_input("7-Day Avg Feed (kg)", value=feed_today)
        with adv_c2:
            roll_gain = st.number_input("7-Day Avg Gain (kg)", value=0.05)
        co_level, nh_level = st.number_input("CO Level", value=5.0), st.number_input("NH3 Level", value=10.0)

    st.markdown("---")

    # --- 5. PREDICTION & ANALYTICS LOGIC ---
    if st.button("🚀 Run AI Business Analysis", use_container_width=True):
        # AI Weight Prediction for Today
        input_data = pd.DataFrame([{
            'day_number': day_number, 'birds_alive': birds_alive, 'feed_today_kg': feed_today,
            'temp': temp, 'rh': rh, 'co': co_level, 'nh': nh_level, 'heat_index': heat_index,
            'feed_per_bird': feed_per_bird, 'rolling_7d_feed': roll_feed,
            'rolling_7d_gain': roll_gain, 'initial_flock': initial_flock,
            'cumulative_mortality': total_mortality
        }])
        current_pred = model.predict(input_data)[0]
        
        # Growth & Harvest Projections
        perf_ratio = current_pred / IDEAL_WEIGHT.get(day_number, 1.0)
        projected_weight = IDEAL_WEIGHT.get(harvest_day, 2.7) * perf_ratio
   
        # FCR Calculations
        current_total_weight = current_pred * birds_alive
        current_fcr = total_feed_to_date / current_total_weight if current_total_weight > 0 else 0
        proj_total_feed = total_feed_to_date + (roll_feed * (harvest_day - day_number))
        harvest_fcr = proj_total_feed / (projected_weight * birds_alive) if birds_alive > 0 else 0

        # Financials
        total_chick_cost = initial_flock * chick_cost
        total_cost = total_chick_cost + (proj_total_feed * feed_cost_per_kg)
        revenue = (projected_weight * birds_alive) * price_per_kg
        profit = revenue - total_cost

        # --- 6. CHART LOGIC (Profit Trend) ---
        days = list(range(30, 46))
        profit_data = []
        for d in days:
            w = IDEAL_WEIGHT.get(d, 2.5) * perf_ratio
            f = total_feed_to_date + (roll_feed * (d - day_number))
            cost = total_chick_cost + (f * feed_cost_per_kg)
            rev = (w * birds_alive) * price_per_kg
            profit_data.append(rev - cost)
        
        df_chart = pd.DataFrame({"Day": days, "Profit": profit_data})
        fig = px.line(df_chart, x="Day", y="Profit", title="Profitability Trend by Harvest Day", markers=True)
        fig.add_vline(x=harvest_day, line_dash="dash", line_color="green", annotation_text="Selected Target")

        # --- 7. DISPLAY RESULTS ---
        st.subheader("📊 Business Intelligence Dashboard")
        
        # Performance Row
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            st.metric("Estimated Weight Today", f"{current_pred:.3f} kg")
            st.metric("Current FCR", f"{current_fcr:.2f}")
        with r1_c2:
            st.metric("Proj. Harvest Weight", f"{projected_weight:.3f} kg", delta=f"{(perf_ratio-1)*100:.1f}%")
            st.metric("Proj. Harvest FCR", f"{harvest_fcr:.2f}", delta_color="inverse")
        with r1_c3:
            st.metric("Projected Net Profit", f"${profit:,.2f}")
            st.write(f"**Growth Performance:** {int(perf_ratio*100)}%")
            st.progress(min(perf_ratio, 1.0))

        # Financial and Visual Row
        r2_c1, r2_c2 = st.columns([1, 2])
        with r2_c1:
            st.markdown("### 💰 Financial Summary")
            st.write(f"**Total Revenue:** ${revenue:,.2f}")
            st.write(f"**Total Expenses:** ${total_cost:,.2f}")
            st.write(f"**ROI:** {(profit/total_cost)*100:.1f}%")
            if heat_index > 32: st.warning(f"Heat Stress Warning! ({heat_index:.1f})")
            else: st.success("Environment is Stable")

        with r2_c2:
            st.plotly_chart(fig, use_container_width=True)


        from fpdf import FPDF
        import io

        # --- 8. PDF REPORT GENERATOR FUNCTION ---
        def create_pdf(data):
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", "B", 20)
            pdf.set_text_color(46, 139, 87) # Dark Green
            pdf.cell(190, 15, "iPoultry AI Guard - Executive Report", ln=True, align="C")
            
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(100)
            pdf.cell(190, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(10)
        
            # Section 1: Batch Overview
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(0)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(190, 10, " 1. Batch & Environmental Status", ln=True, fill=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(95, 10, f"Age: {data['day_number']} Days")
            pdf.cell(95, 10, f"Birds Alive: {data['birds_alive']:,}", ln=True)
            pdf.cell(95, 10, f"Temperature: {data['temp']} C")
            pdf.cell(95, 10, f"Heat Index: {data['heat_index']:.2f}", ln=True)
            pdf.ln(5)
        
            # Section 2: AI Predictions
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, " 2. AI Growth Predictions", ln=True, fill=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(95, 10, f"Current Est. Weight: {data['current_pred']:.3f} kg")
            pdf.cell(95, 10, f"Performance: {data['perf_ratio']:.1%}", ln=True)
            pdf.cell(95, 10, f"Projected Harvest Weight: {data['proj_weight']:.3f} kg")
            pdf.cell(95, 10, f"Harvest Day: {data['harvest_day']}", ln=True)
            pdf.ln(5)
        
            # Section 3: Financials
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, " 3. Financial Forecast", ln=True, fill=True)
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(0, 100, 0) # Profit Green
            pdf.cell(190, 10, f"Projected Net Profit: ${data['profit']:,.2f}", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.set_text_color(0)
            pdf.cell(95, 10, f"Total Revenue: ${data['revenue']:,.2f}")
            pdf.cell(95, 10, f"Total Costs: ${data['total_cost']:,.2f}", ln=True)
            pdf.cell(95, 10, f"Projected FCR: {data['harvest_fcr']:.2f}")
            pdf.cell(95, 10, f"Estimated ROI: {data['roi']:.1f}%", ln=True)
        
            # Footer
            pdf.set_y(-30)
            pdf.set_font("Arial", "I", 8)
            pdf.set_text_color(128)
            pdf.cell(0, 10, "iPoultry AI Guard © 2026 - Confidential Precision Farming Data", align="C")
            
            #return pdf.output()
            return bytes(pdf.output())
        
        # --- INSIDE THE 'if st.button' SECTION ---
        # Create a dictionary of all results for the PDF
        report_data = {
            'day_number': day_number, 'birds_alive': birds_alive, 'temp': temp, 
            'heat_index': heat_index, 'current_pred': current_pred, 
            'perf_ratio': perf_ratio, 'proj_weight': projected_weight, 
            'harvest_day': harvest_day, 'profit': profit, 'revenue': revenue, 
            'total_cost': total_cost, 'harvest_fcr': harvest_fcr, 
            'roi': (profit/total_cost)*100
        }
        
        pdf_bytes = create_pdf(report_data)
        
        
        st.download_button(
            label="📩 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"iPoultry_Official_Report_Day{day_number}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
         

st.divider()
st.caption("iPoultry AI Guard © 2026")
