import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
import io

# --- 1. BREED STANDARDS (The Single Source of Truth) ---
BREED_STANDARDS = {
    "Cobb 500": {
        1:0.061, 2:0.079, 3:0.099, 4:0.122, 5:0.148, 6:0.176, 7:0.208,
        8:0.242, 9:0.28, 10:0.321, 11:0.366, 12:0.414, 13:0.465,
        14:0.519, 15:0.576, 16:0.637, 17:0.701, 18:0.768, 19:0.837,
        20:0.91, 21:0.985, 22:1.062, 23:1.142, 24:1.225, 25:1.309,
        26:1.395, 27:1.483, 28:1.573, 29:1.664, 30:1.757,
        31:1.851, 32:1.946, 33:2.041, 34:2.138, 35:2.235,
        36:2.332, 37:2.43, 38:2.527, 39:2.625, 40:2.723,
        41:2.820, 42:2.915, 43:3.010, 44:3.100, 45:3.190
    },
    "Ross 308": {
        1:0.056, 2:0.071, 3:0.089, 4:0.111, 5:0.136, 6:0.166, 7:0.201,
        8:0.240, 9:0.285, 10:0.334, 11:0.390, 12:0.451, 13:0.518,
        14:0.591, 15:0.670, 16:0.755, 17:0.846, 18:0.943, 19:1.045,
        20:1.154, 21:1.268, 22:1.388, 23:1.514, 24:1.644, 25:1.780,
        26:1.921, 27:2.066, 28:2.215, 29:2.369, 30:2.525,
        31:2.686, 32:2.848, 33:3.014, 34:3.181, 35:3.350,
        36:3.520, 37:3.691, 38:3.861, 39:4.032, 40:4.202,
        41:4.371, 42:4.538, 43:4.705, 44:4.869, 45:5.031
    }
}

# --- 2. PDF GENERATOR FUNCTION ---
def create_pdf(data, lang="English", breed="Cobb 500"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False) 
    pdf.add_page()
    
    t = {
        "English": {
            "title": "iPoultry AI Guard - Executive Report",
            "subtitle": "Precision Analytics for Batch Management",
            "sec1": " 1. Batch & Environmental Status",
            "age": "Bird Age", "breed": "Breed", "alive": "Birds Alive", "temp": "Mean Temp",
            "sec2": " 2. AI Growth Analytics",
            "today_w": "Today's Est. Weight", "perf": "Growth Performance", "proj_w": "Proj. Harvest Weight", "target_d": "Target Harvest Day",
            "sec3": " 3. Financial Intelligence",
            "profit": "Projected Net Profit", "rev": "Estimated Revenue", "cost": "Total Est. Costs", "roi": "Projected ROI"
        },
        "Bahasa Melayu": {
            "title": "iPoultry AI Guard - Laporan Eksekutif",
            "subtitle": "Analitik Kepersisan untuk Pengurusan Kelompok",
            "sec1": " 1. Status Kelompok & Persekitaran",
            "age": "Umur Ayam", "breed": "Baka", "alive": "Ayam Hidup", "temp": "Suhu Purata",
            "sec2": " 2. Analitik Pertumbuhan AI",
            "today_w": "Anggaran Berat Hari Ini", "perf": "Prestasi Pertumbuhan", "proj_w": "Proj. Berat Tuai", "target_d": "Hari Tuaian Sasaran",
            "sec3": " 3. Kecerdasan Kewangan",
            "profit": "Unjuran Untung Bersih", "rev": "Anggaran Hasil", "cost": "Jumlah Anggaran Kos", "roi": "Unjuran ROI"
        }
    }[lang]

    logo_path = "IDEA LOGIC Logo.jpg"
    if os.path.exists(logo_path):
        pdf.image(logo_path, 10, 8, 33) 
    
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(46, 139, 87) 
    pdf.set_xy(50, 10)
    pdf.cell(140, 10, t["title"], ln=True)
    
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100)
    pdf.set_x(50)
    pdf.cell(140, 5, t["subtitle"], ln=True)
    pdf.set_x(50)
    pdf.cell(140, 5, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(10)

    # Section 1
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(185, 8, t["sec1"], ln=True, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.ln(2)
    pdf.cell(60, 7, f"{t['age']}: {data['day_number']} Days")
    pdf.cell(60, 7, f"{t['breed']}: {breed}")
    pdf.cell(65, 7, f"{t['alive']}: {data['birds_alive']:,}", ln=True)
    pdf.cell(60, 7, f"{t['temp']}: {data['temp']} C")
    pdf.cell(60, 7, f"Heat Index: {data['heat_index']:.2f}", ln=True)
    pdf.ln(4)

    # Section 2
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(185, 8, t["sec2"], ln=True, fill=True)
    pdf.set_font("helvetica", "", 10)
    pdf.ln(2)
    pdf.cell(90, 7, f"{t['today_w']}: {data['current_pred']:.3f} kg")
    pdf.cell(90, 7, f"{t['perf']}: {data['perf_ratio']:.1%}", ln=True)
    pdf.cell(90, 7, f"{t['proj_w']}: {data['proj_weight']:.3f} kg")
    pdf.cell(90, 7, f"{t['target_d']}: Day {data['harvest_day']}", ln=True)
    pdf.ln(4)

    # Section 3
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(185, 8, t["sec3"], ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(0, 100, 0) 
    pdf.cell(185, 7, f"{t['profit']}: RM {data['profit']:,.2f}", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(0)
    pdf.cell(90, 7, f"{t['rev']}: RM {data['revenue']:,.2f}")
    pdf.cell(90, 7, f"{t['cost']}: RM {data['total_cost']:,.2f}", ln=True)
    pdf.cell(90, 7, f"Target FCR: {data['harvest_fcr']:.2f}")
    pdf.cell(90, 7, f"{t['roi']}: {data['roi']:.1f}%", ln=True)

    pdf.set_y(265) 
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(150)
    pdf.cell(185, 10, "Developed by Idealogic - Precision Agriculture AI Division (Malaysia)", 0, 0, "C")
    
    return bytes(pdf.output())

# --- 3. CONFIG & STYLING ---
st.set_page_config(page_title="iPoultry AI Guard", layout="wide")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: black; margin-bottom: 0;'>📈 iPoultry <span style='color: #FFD700;'>AI Guard</span></h1>
        <h2 style='color: green; margin-top: 0; margin-bottom: 0;'>Weight, FCR & Profit Analytics</h2>
        <h3 style='color: black; margin-top: 0;'>By AI Model Trained On Farm Data</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #007BFF; color: white; border-radius: 5px; }
    div.stButton > button:first-child:hover { background-color: #0056b3; color: white; }
    div.stDownloadButton > button { background-color: #28a745; color: white; border-radius: 5px; }
    div.stDownloadButton > button:hover { background-color: #218838; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Custom Branded Thick Line
st.markdown('<hr style="height:3px;border:none;color:#2E8B57;background-color:#2E8B57; margin-bottom: 20px;" />', unsafe_allow_html=True)

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
    # --- 4. INPUT SECTION ---
    st.subheader("📝 Enter Farm & Market Inputs")
    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown("**📊 Batch Statistics**")
        initial_flock = st.number_input("Initial Flock Size", 100, 100000, 5000)
        total_mortality = st.number_input("Total Mortality (Cumulative)", 0, initial_flock, 150)
        day_number = st.number_input("Day Number (Age)", 1, 45, 21)
        birds_alive = initial_flock - total_mortality
        selected_breed = st.selectbox("Select Bird Breed", ["Cobb 500", "Ross 308"])
        
    current_standards = BREED_STANDARDS[selected_breed]

    with t2:
        st.markdown("**🌡️ Environment**")
        temp = st.slider("24 hrs Mean Temp (°C)", 15.0, 40.0, 28.5)
        rh = st.slider("Humidity (%)", 20.0, 100.0, 65.0)
        feed_today = st.number_input("Feed Today (kg)", 0.0, 5000.0, 450.0)

    with t3:
        st.markdown("**🎯 Cumulative Data**")
        hist_feed = st.number_input("Total Feed Used UNTIL Yesterday (kg)", value=float(feed_today * (day_number - 1)))
        total_feed_to_date = hist_feed + feed_today
        st.info(f"Total Feed (inc. today): {total_feed_to_date:,.1f} kg")
        harvest_day = st.number_input("Target Harvest Day", 30, 45, 35)

    with t4:
        st.markdown("**💰 Market Prices (RM)**")
        price_per_kg = st.number_input("Chicken Price (RM/kg)", value=9.40)
        feed_cost_per_kg = st.number_input("Feed Cost (RM/kg)", value=2.80)
        chick_cost = st.number_input("Cost per Chick (RM)", value=2.20)

    heat_index = temp + (0.33 * rh) - 0.7
    feed_per_bird = feed_today / birds_alive if birds_alive > 0 else 0
    
    # --- 4.5 ADVANCED SETTINGS (Now Fully Visible) ---
    st.markdown("---")
    st.subheader("🛠️ Enter the Advanced Growth & Air Quality Metrics")
    a1, a2, a3, a4 = st.columns(4)
    
    with a1:
        roll_feed = st.number_input("7-Day Avg Feed (kg)", value=float(feed_today))
    with a2:
        roll_gain = st.number_input("7-Day Avg Gain (kg)", value=0.050, format="%.3f")
    with a3:
        co_level = st.number_input("CO Level (ppm)", value=5.0)
    with a4:
        nh_level = st.number_input("NH3 Level (ppm)", value=10.0)

    report_lang = st.radio("Laporan Bahasa / Report Language:", ["English", "Bahasa Melayu"], horizontal=True)
    st.markdown("---")

    # --- 5. EXECUTION ---
    if st.button("🚀 Run AI Analysis", use_container_width=True):
        input_df = pd.DataFrame([{
            'day_number': day_number, 'birds_alive': birds_alive, 'feed_today_kg': feed_today,
            'temp': temp, 'rh': rh, 'co': co_level, 'nh': nh_level, 'heat_index': heat_index,
            'feed_per_bird': feed_per_bird, 'rolling_7d_feed': roll_feed,
            'rolling_7d_gain': roll_gain, 'initial_flock': initial_flock,
            'cumulative_mortality': total_mortality
        }])
        current_pred = model.predict(input_df)[0]
        
        perf_ratio = current_pred / current_standards.get(day_number, 1.0)
        projected_weight = current_standards.get(harvest_day, 2.7) * perf_ratio
        
        current_fcr = total_feed_to_date / (current_pred * birds_alive) if (current_pred * birds_alive) > 0 else 0
        proj_total_feed = total_feed_to_date + (roll_feed * (harvest_day - day_number))
        harvest_fcr = proj_total_feed / (projected_weight * birds_alive) if (projected_weight * birds_alive) > 0 else 0

        total_chick_cost = initial_flock * chick_cost
        total_cost = total_chick_cost + (proj_total_feed * feed_cost_per_kg)
        revenue = (projected_weight * birds_alive) * price_per_kg
        profit = revenue - total_cost

        # Dashboard UI
        st.subheader("📊 Business Intelligence Dashboard")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            st.metric("Est. Weight Today", f"{current_pred:.3f} kg")
            st.metric("Current FCR", f"{current_fcr:.2f}")
        with r1_c2:
            st.metric("Proj. Harvest Weight", f"{projected_weight:.3f} kg", delta=f"{(perf_ratio-1)*100:.1f}%")
            st.metric("Proj. Harvest FCR", f"{harvest_fcr:.2f}", delta_color="inverse")
        with r1_c3:
            st.metric("Projected Net Profit", f"RM {profit:,.2f}")
            st.write(f"**Growth Performance:** {int(perf_ratio*100)}%")
            st.progress(min(max(perf_ratio, 0.0), 1.0))

        # Financial Summary and Chart Row
        r2_c1, r2_c2 = st.columns([1, 2])
        with r2_c1:
            st.markdown("### 💰 Financial Summary")
            st.write(f"**Total Revenue:** RM {revenue:,.2f}")
            st.write(f"**Total Expenses:** RM {total_cost:,.2f}")
            roi = (profit/total_cost)*100 if total_cost > 0 else 0
            st.write(f"**ROI:** {roi:.1f}%")
            if heat_index > 32: st.warning(f"Heat Stress Warning! ({heat_index:.1f})")
            else: st.success("Environment is Stable")

        with r2_c2:
            days = list(range(30, 46))
            profits = [(((current_standards.get(d, 2.5) * perf_ratio) * birds_alive) * price_per_kg) - 
                       (total_chick_cost + ((total_feed_to_date + (roll_feed * (d - day_number))) * feed_cost_per_kg)) for d in days]
            fig = px.line(x=days, y=profits, title="Profit Trend by Day (RM)", labels={'x':'Day', 'y':'Profit (RM)'}, markers=True)
            st.plotly_chart(fig, use_container_width=True)

        # Store results for PDF
        st.session_state['pdf_data'] = {
            'day_number': day_number, 'birds_alive': birds_alive, 'temp': temp, 
            'heat_index': heat_index, 'current_pred': current_pred, 
            'perf_ratio': perf_ratio, 'proj_weight': projected_weight, 
            'harvest_day': harvest_day, 'profit': profit, 'revenue': revenue, 
            'total_cost': total_cost, 'harvest_fcr': harvest_fcr, 'roi': roi
        }
        
        st.session_state['pdf_bytes'] = create_pdf(st.session_state['pdf_data'], lang=report_lang, breed=selected_breed)

    if 'pdf_bytes' in st.session_state:
        st.markdown("---")
        st.download_button(
            label=f"📩 Download Official {report_lang} PDF Report",
            data=st.session_state['pdf_bytes'],
            file_name=f"iPoultry_Report_Day{day_number}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.divider()
st.caption("iPoultry AI Guard © 2026 | Idealogic Precision Agriculture AI Division")
