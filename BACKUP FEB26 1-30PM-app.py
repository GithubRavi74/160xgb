import streamlit as st

st.set_page_config(
    page_title="iPoultry",
    page_icon="🐔",
    layout="wide"
)

# Hide default "app" label
st.markdown("""
    <style>
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }
        section[data-testid="stSidebarNav"] > ul {
            margin-top: 0px;
        }
        section[data-testid="stSidebarNav"]::before {
            content: "Menu";
            font-size: 22px;
            font-weight: 600;
            display: block;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)


#st.sidebar.title("🐔 iPoultry AI Suite")
#st.sidebar.markdown("Smart Broiler Intelligence Platform")


st.title("🐔 iPoultry AI Shield")
st.markdown("""
Welcome to **iPoultry AI**.

Use the menu on the left to:-
- 🧑‍🌾 Farmer's section for **Daily Health Predictions**
- 🧪 Vendor's section for **Validating Forecast & Models**
""")
