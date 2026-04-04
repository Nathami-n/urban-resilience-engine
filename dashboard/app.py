import streamlit as st


st.set_page_config(page_title="Urban Resilience Engine", layout="wide")

st.title("Urban Resilience Engine")
st.caption("Starter dashboard scaffold")

section1, section2, section3 = st.tabs(["Risk map", "Forecast chart", "SHAP plot"])

with section1:
    st.info("Risk map placeholder. Add folium choropleth once predictions are available.")

with section2:
    st.info("Forecast chart placeholder. Add the 2013–2040 trend line here.")

with section3:
    st.info("SHAP plot placeholder. Render models/shap_summary.png when it exists.")
