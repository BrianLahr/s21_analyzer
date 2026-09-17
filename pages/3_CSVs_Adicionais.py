import streamlit as st

from utils.legacy_csv_viewer import create_legacy_csv_viewer


st.set_page_config(
    page_title="CSVs VNA adicionais",
    page_icon="📂",
    layout="wide",
)

create_legacy_csv_viewer()
