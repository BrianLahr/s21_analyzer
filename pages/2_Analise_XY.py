import streamlit as st

from utils.experimental_viewer import load_experimental_data
from utils.spatial_xy_analyzer import create_spatial_xy_analysis


st.set_page_config(
    page_title="Análise XY do VNA",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Análise espacial XY do VNA")
st.caption(
    "Reconstrução automática do grid 6×6 a partir dos nomes dos arquivos e "
    "análise espacial das respostas S11/S21."
)

uploaded_files = st.file_uploader(
    "Selecione os arquivos CSV da varredura",
    type=["csv"],
    accept_multiple_files=True,
    key="xy_page_vna_files",
    help=(
        "Os nomes devem conter a trajetória da varredura, por exemplo "
        "ReT_16a66_65a45_16092026.csv. Os traços Trc1/Trc2 são ignorados; "
        "R1…R9 e T1…T9 são associados às posições na ordem de aquisição."
    ),
)

if not uploaded_files:
    st.info("Selecione os quatro CSVs da varredura para reconstruir o grid completo.")
    st.stop()

parsed_files = []
for uploaded in uploaded_files:
    try:
        df, trace_map = load_experimental_data(uploaded.getvalue(), uploaded.name)
        parsed_files.append(
            {
                "name": uploaded.name,
                "df": df,
                "trace_map": trace_map,
            }
        )
    except Exception as exc:
        st.error(f"{uploaded.name}: {exc}")

if not parsed_files:
    st.stop()

create_spatial_xy_analysis(parsed_files)
