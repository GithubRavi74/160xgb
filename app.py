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

 


#st.title("🐔 iPoultry AI Shield")
st.markdown(
    """
    <h1 style='color: black;'>
        📈🐔iPoultry <span style='color: #FFD700;'>AI Guard</span>
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("""
**Welcome**.

Use the menu on the left to:-
- 🧑‍🌾 **Insights For Farmer**
""")
