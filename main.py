import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import zipfile

from utils.file_handlers import handle_file_upload, display_file_preview
from utils.data_processors import identify_columns, process_data

# ------------- UI para Circuito Equivalente (Plotly + correlações interativas) ---------------
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

def create_equiv_circuit_ui(equiv_module):
    st.header("⚙️ Circuito Equivalente — Extração RLC")
    st.write(
        "Faça upload de um CSV com colunas geométricas + `Freq [GHz]` + "
        "`dB(S(1,1)) []` + `dB(S(2,1)) []`."
    )

    uploaded = st.file_uploader(
        "Carregue CSV para extração de circuito equivalente",
        type=['csv'],
        key="equiv_uploader"
    )
    if not uploaded:
        st.info("Faça upload de um CSV para começar.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return

    st.write("Preview dos dados:")
    st.dataframe(df.head())

    if st.button("🔎 Analisar e extrair circuito equivalente", type="primary"):
        with st.spinner("Analisando..."):
            try:
                results_df, series = equiv_module.analyze_equiv_circuits_from_df(df)
                st.success("✅ Extração concluída!")
            except Exception as e:
                st.error(f"Erro durante a análise: {e}")
                return

        # ===== Tabela de resultados =====
        st.subheader("📋 Parâmetros extraídos (R, L, C, f₀, f_c)")
        st.dataframe(results_df)

        # Download em Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='equiv_results')
        buf.seek(0)
        st.download_button(
            "📥 Baixar resultados (.xlsx)",
            buf,
            file_name="equiv_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # ===== Plot comparativo =====
        st.subheader("📈 Comparativo Medido vs Modelo (exemplo do primeiro caso)")
        first_key = list(series.keys())[0]
        sr = series[first_key]
        data = equiv_module.get_comparison_data(sr)
        freq = data["freq"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freq, y=data["s21_db"],
            mode="markers",
            marker=dict(size=4, color="gray", opacity=0.6),
            name="S21 medido (dB)"
        ))
        fig.add_trace(go.Scatter(
            x=freq, y=data["s21_smooth"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="S21 suavizado"
        ))
        if data["s21_model_db"] is not None:
            fig.add_trace(go.Scatter(
                x=freq, y=data["s21_model_db"],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="S21 modelo (R||L||C)"
            ))
        fig.update_layout(
            xaxis_title="Frequência [GHz]",
            yaxis_title="S21 [dB]",
            title=f"Comparação Medido vs Modelo — {first_key}",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== Correlações geométricas =====
        st.subheader("📊 Correlação geométrica → L/C (Pearson)")
        corr = equiv_module.geom_to_LC_correlations(results_df)
        st.json(corr)

        # ===== Visualização interativa das correlações =====
        st.subheader("🔍 Visualização das correlações geométricas")
        geom_cols = [c for c in results_df.columns if "[mm]" in c]

        if geom_cols:
            param_sel = st.selectbox("Selecione o parâmetro geométrico", geom_cols)
            fig2 = go.Figure()

            # Scatter para L_nH
            fig2.add_trace(go.Scatter(
                x=results_df[param_sel],
                y=results_df["L_nH"],
                mode="markers",
                name="L_nH",
                marker=dict(color="blue", size=8, symbol="circle"),
            ))
            # Scatter para C_pF
            fig2.add_trace(go.Scatter(
                x=results_df[param_sel],
                y=results_df["C_pF"],
                mode="markers",
                name="C_pF",
                marker=dict(color="red", size=8, symbol="diamond"),
            ))

            fig2.update_layout(
                title=f"Correlação de {param_sel} com L e C",
                xaxis_title=f"{param_sel}",
                yaxis_title="Valor extraído",
                legend=dict(bgcolor="rgba(255,255,255,0.6)"),
                template="plotly_white",
                height=450
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Nenhuma coluna geométrica detectada para correlação.")

        st.info(
            "Observação: L e C foram extraídos analiticamente; "
            "R foi ajustado mantendo L/C fixos. "
            "Para maior precisão ou modelo Pi-type, podemos estender o módulo."
        )


def setup_ui():
    """Configuração inicial de layout"""
    st.set_page_config(
        page_title="Analisador S-Parameters",
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown('<h1 style="text-align: center; margin-bottom: 2rem;">📊 Analisador de Parâmetros S (S11/S21)</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("ℹ️ Sobre")
    st.sidebar.info(
        "Esta aplicação analisa ressonâncias em dados S11 ou S21 de arquivos CSV. "
        "Altere as frequências para calcular automaticamente os parâmetros."
    )

    if 's_param_type' not in st.session_state:
        st.session_state.s_param_type = "S21"

        # CINCO ABAS (adicionada: Circuito Equivalente)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Análise de Dados S11/S21", 
        "📊 Visualização de Resultados",
        "🔄 Comparação de Curvas",
        "🗂️ Organizador de Resultados",
        "⚙️ Circuito Equivalente"
    ])
    
    with tab1:
        run_analysis_tab()
    
    with tab2:
        try:
            from utils.results_visualizer import create_results_visualizer
            create_results_visualizer()
        except ImportError as e:
            st.error(f"❌ Erro ao carregar o visualizador de resultados: {e}")
    
    with tab3:
        try:
            from utils.results_visualizer import create_curves_comparison
            create_curves_comparison()
        except ImportError as e:
            st.error(f"❌ Erro ao carregar o comparador de curvas: {e}")
    
    with tab4:
        try:
            from utils.results_organizer import create_results_organizer
            create_results_organizer()
        except ImportError as e:
            st.error(f"❌ Erro ao carregar o organizador de resultados: {e}")

    with tab5:
        try:
            from utils import equiv_circuit
            create_equiv_circuit_ui(equiv_circuit)
        except Exception as e:
            st.error(f"❌ Erro ao carregar o módulo de circuito equivalente: {e}")

# ======================
# Reset de aplicação
# ======================
def reset_app():
    """Reset completo do estado da aplicação"""
    st.session_state.clear()
    # Restaurar o padrão S21 após reset
    st.session_state.s_param_type = "S21"
    st.rerun()

# ======================
# Exportação de resultados
# ======================
def export_analysis(temp_path, all_results, filename):
    """Gera arquivos Excel e ZIP para download"""
    if not all_results:
        st.warning("⚠️ Nenhuma ressonância calculada para exportar.")
        return

    results_df = pd.DataFrame(all_results)

    # CORREÇÃO: Ordenar por "sample_height [mm]" se a coluna existir
    if 'sample_height [mm]' in results_df.columns:
        try:
            # Converter para numérico para ordenação correta
            results_df['sample_height [mm]'] = pd.to_numeric(results_df['sample_height [mm]'], errors='coerce')
            # Ordenar pelo sample_height
            results_df = results_df.sort_values('sample_height [mm]')
            st.info("📊 Planilha ordenada por 'sample_height [mm]'")
        except Exception as e:
            st.warning(f"⚠️ Não foi possível ordenar por 'sample_height [mm]': {e}")
    
    # Limpar diretório temporário
    for file in temp_path.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)

    # ATUALIZADO: Nome do arquivo adaptado
    excel_path = temp_path / f"resultados_completos_{Path(filename).stem}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Resultados', index=False)

    # ATUALIZADO: Nome do ZIP adaptado
    s_type = st.session_state.get('s_param_type', 'S21')
    zip_path = temp_path / f"resultados_analise_{s_type.lower()}_{Path(filename).stem}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(excel_path, excel_path.name)
        for txt_file in temp_path.glob("*.txt"):
            zip_file.write(txt_file, txt_file.name)

    # Botão de download
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    st.download_button(
        label=f"📥 Baixar Todos os Resultados {s_type} (ZIP)",  # ATUALIZADO
        data=zip_data,
        file_name=zip_path.name,
        mime="application/zip",
        use_container_width=True
    )
    st.success(f"✅ Exportação concluída! Total de {len(all_results)} ressonâncias.")

# ======================
# Função principal
# ======================
def main():
    setup_ui()

def run_analysis_tab():
    """Executa a aba de análise de dados S11/S21"""  # ATUALIZADO
    # Inicializar session_state
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'export_ready' not in st.session_state:
        st.session_state.export_ready = False
    if 's_param_type' not in st.session_state:
        st.session_state.s_param_type = "S21"

    # NOVO: Seleção do tipo de parâmetro S
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Configurações")
    s_param_type = st.sidebar.radio(
        "Tipo de Parâmetro S:",
        ["S11", "S21"],
        index=0 if st.session_state.s_param_type == "S11" else 1,
        key="s_param_selector"
    )
    
    # Atualizar session_state se mudou
    if s_param_type != st.session_state.s_param_type:
        st.session_state.s_param_type = s_param_type
        st.session_state.uploaded_file_data = None
        st.session_state.analysis_results = None
        st.session_state.export_ready = False
        st.session_state.uploader_key += 1
        st.rerun()

    # Upload do arquivo
    uploaded_file = st.file_uploader(
        f"**Selecione o arquivo CSV com dados {s_param_type}**",  # ATUALIZADO
        type=['csv'],
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file:
        try:
            # Processa upload
            df, file_changed = handle_file_upload(uploaded_file, st.session_state.uploaded_file_data)
            if file_changed:
                st.session_state.analysis_results = None
                st.session_state.uploaded_file_data = {
                    'name': uploaded_file.name,
                    'df': df,
                    'freq_col': None,
                    's_col': None,  # ATUALIZADO: nome mais genérico
                    'param_cols': None,
                    'perm_col': None,
                    's_param_type': s_param_type  # NOVO: armazenar tipo S
                }
                st.session_state.export_ready = False

            # Preview do arquivo
            display_file_preview(df, uploaded_file)

            # Identificação de colunas - ATUALIZADO para aceitar S11 ou S21
            data_info = st.session_state.uploaded_file_data
            if data_info['s_col'] is None:  # ATUALIZADO
                freq_col, s_col, param_cols, perm_col = identify_columns(df, s_param_type)  # ATUALIZADO
                data_info.update({
                    'freq_col': freq_col,
                    's_col': s_col,  # ATUALIZADO
                    'param_cols': param_cols,
                    'perm_col': perm_col,
                    's_param_type': s_param_type  # NOVO
                })
            else:
                freq_col = data_info['freq_col']
                s_col = data_info['s_col']  # ATUALIZADO
                param_cols = data_info['param_cols']
                perm_col = data_info['perm_col']

            if not freq_col or not s_col:  # ATUALIZADO
                st.error(f"❌ Não foi possível identificar colunas de frequência e {s_param_type} automaticamente.")
                return
            st.success(f"✅ Colunas identificadas: Frequência='{freq_col}', {s_param_type}='{s_col}'")

            # Processar dados - ATUALIZADO
            analysis_container = st.container()
            
            with analysis_container:
                temp_path, all_results_combined = process_data(
                    df, uploaded_file.name, freq_col, s_col, param_cols, perm_col, s_param_type  # ATUALIZADO
                )
                
                st.session_state.analysis_results = (temp_path, all_results_combined)
                
                # Exportação
                st.markdown("---")
                st.markdown("### 📦 Exportação de Resultados")
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🚀 Exportar Resultados em .ZIP", type="primary", use_container_width=True):
                        st.session_state.export_ready = True
                
                if st.session_state.export_ready:
                    all_results_for_export = []
                    for key in st.session_state:
                        if key.startswith("results_"):
                            all_results_for_export.extend(st.session_state[key])
                    
                    if all_results_for_export:
                        export_analysis(temp_path, all_results_for_export, uploaded_file.name)
                    else:
                        st.warning("⚠️ Nenhum resultado encontrado para exportar. Calcule algumas ressonâncias primeiro.")

        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {e}")
    else:
        st.info(f"👆 Faça upload de um arquivo CSV com dados {s_param_type} para iniciar a análise")  # ATUALIZADO
        st.session_state.analysis_results = None
        st.session_state.uploaded_file_data = None
        st.session_state.export_ready = False
        if st.button("🔄 Resetar Aplicação"):
            reset_app()

            

if __name__ == "__main__":
    main()