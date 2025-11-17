import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import zipfile
import io

from utils.file_handlers import handle_file_upload, display_file_preview
from utils.data_processors import identify_columns, process_data
from utils.correlation_analyzer import create_correlation_tab
from utils.experimental_viewer import create_experimental_tab

# ======================
# Setup da UI
# ======================
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

    # Inicializar session_state
    if 's_param_type' not in st.session_state:
        st.session_state.s_param_type = "S21"
    if 'circuit_analysis_results' not in st.session_state:
        st.session_state.circuit_analysis_results = None

    # Configurações na sidebar
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
        # Limpar resultados específicos do tipo S
        if 'uploaded_file_data' in st.session_state:
            st.session_state.uploaded_file_data = None
        if 'analysis_results' in st.session_state:
            st.session_state.analysis_results = None
        if 'export_ready' in st.session_state:
            st.session_state.export_ready = False
        if 'uploader_key' in st.session_state:
            st.session_state.uploader_key += 1

    #Abas principais - ADICIONE A NOVA ABA
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([  # Mude para 7 abas
        "📈 Análise de Dados S11/S21", 
        "📊 Visualização de Resultados",
        "🔄 Comparação de Curvas", 
        "🗂️ Organizador de Resultados",
        "⚙️ Circuito Equivalente",
        "📊 Análise de Correlação",
        "🔬 Dados Experimentais"  # NOVA ABA
    ])
    
    with tab1:
        run_analysis_tab()
    
    with tab2:
        try:
            from utils.results_visualizer import create_results_visualizer
            create_results_visualizer()
        except ImportError as e:
            st.error(f"❌ Erro ao carregar o visualizador de resultados: {e}")
        except Exception as e:
            st.error(f"❌ Erro no visualizador de resultados: {e}")
    
    with tab3:
        try:
            from utils.results_visualizer import create_curves_comparison
            create_curves_comparison()
        except ImportError as e:
            st.error(f"❌ Erro ao carregar o comparador de curvas: {e}")
        except Exception as e:
            st.error(f"❌ Erro no comparador de curvas: {e}")
    
    with tab4:
        try:
            from utils.results_organizer import create_results_organizer
            create_results_organizer()
        except ImportError as e:
            st.error(f"❌ Erro ao carregar o organizador de resultados: {e}")
        except Exception as e:
            st.error(f"❌ Erro no organizador de resultados: {e}")

    with tab5:
        create_equiv_circuit_tab()

    with tab6:  # NOVA ABA
        create_correlation_tab()

    with tab7:  # NOVA ABA
        create_experimental_tab()

# ======================
# Circuito Equivalente - Versão Melhorada
# ======================
def create_equiv_circuit_tab():
    """Cria a aba de análise de circuito equivalente"""
    st.header("⚙️ Análise de Circuito Equivalente")
    
    try:
        from utils.equiv_circuit import create_circuit_analysis_interface
        
        # Executar análise automaticamente e manter resultados
        results_df, series_store, correlations = create_circuit_analysis_interface()
        
        # Armazenar resultados no session_state para persistência
        if results_df is not None:
            st.session_state.circuit_analysis_results = {
                'results_df': results_df,
                'series_store': series_store,
                'correlations': correlations
            }
        
        # Exibir resultados se disponíveis
        if (st.session_state.circuit_analysis_results and 
            not st.session_state.circuit_analysis_results['results_df'].empty):
            
            display_circuit_analysis_results(st.session_state.circuit_analysis_results)
            
    except ImportError as e:
        st.error(f"❌ Módulo de circuito equivalente não encontrado: {e}")
    except Exception as e:
        st.error(f"❌ Erro na análise de circuito equivalente: {e}")

def display_circuit_analysis_results(analysis_data):
    """Exibe os resultados da análise de circuito equivalente"""
    results_df = analysis_data['results_df']
    series_store = analysis_data['series_store']
    correlations = analysis_data['correlations']
    
    # ===== Tabela de resultados =====
    st.subheader("📋 Parâmetros Extraídos (R, L, C, f₀, f_c)")
    st.dataframe(results_df, use_container_width=True)

    # Download em Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False, sheet_name='equiv_results')
    buf.seek(0)
    
    st.download_button(
        "📥 Baixar Resultados Completos (.xlsx)",
        buf,
        file_name="resultados_circuito_equivalente.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # ===== Plot comparativo =====
    st.subheader("📈 Comparativo Medido vs Modelo")
    
    # Seletor de caso para visualização
    if series_store:
        case_options = list(series_store.keys())
        selected_case = st.selectbox(
            "Selecione o caso para visualização:",
            options=case_options,
            index=0
        )
        
        if selected_case:
            sr = series_store[selected_case]
            plot_circuit_comparison(sr, selected_case)

    # ===== Correlações geométricas =====
    st.subheader("📊 Correlações Geométricas com L/C")
    
    if correlations:
        # Exibir tabela de correlações
        corr_data = []
        for geom_col, corr_vals in correlations.items():
            if corr_vals['r_L'] is not None or corr_vals['r_C'] is not None:
                corr_data.append({
                    'Parâmetro Geométrico': geom_col,
                    'Correlação com L (r)': f"{corr_vals['r_L']:.3f}" if corr_vals['r_L'] is not None else "N/A",
                    'Correlação com C (r)': f"{corr_vals['r_C']:.3f}" if corr_vals['r_C'] is not None else "N/A"
                })
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df, use_container_width=True)
            
            # Visualização gráfica das correlações
            st.subheader("🔍 Visualização Gráfica das Correlações")
            geom_cols = [c for c in results_df.columns if any(x in c for x in ['[mm]', 'radius', 'width', 'height', 'g'])]
            
            if geom_cols:
                param_sel = st.selectbox("Selecione o parâmetro geométrico:", geom_cols)
                plot_geometry_correlations(results_df, param_sel)
        else:
            st.info("ℹ️ Não foram encontradas correlações significativas para exibir.")
    else:
        st.info("ℹ️ Nenhuma correlação geométrica disponível.")

def plot_circuit_comparison(series_data, case_name):
    """Plota comparação entre dados medidos e modelo"""
    import plotly.graph_objects as go
    
    freq = series_data["freq"]
    
    fig = go.Figure()
    
    # Dados medidos
    fig.add_trace(go.Scatter(
        x=freq, y=series_data["s21_db"],
        mode="markers",
        marker=dict(size=4, color="gray", opacity=0.6),
        name="S21 medido (dB)"
    ))
    
    # Dados suavizados
    if "s21_smooth" in series_data:
        fig.add_trace(go.Scatter(
            x=freq, y=series_data["s21_smooth"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="S21 suavizado"
        ))
    
    # Modelo RLC
    if series_data.get("s21_model_db") is not None:
        fig.add_trace(go.Scatter(
            x=freq, y=series_data["s21_model_db"],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="S21 modelo (R||L||C)"
        ))
    
    # Parâmetros do modelo no título
    params = series_data.get('params', {})
    title = f"Comparação Medido vs Modelo — {case_name}"
    if params.get('L_nH') and params.get('C_pF'):
        title += f"<br>L={params['L_nH']:.2f}nH, C={params['C_pF']:.2f}pF"
        if params.get('R_ohm'):
            title += f", R={params['R_ohm']:.0f}Ω"
    
    fig.update_layout(
        xaxis_title="Frequência [GHz]",
        yaxis_title="S21 [dB]",
        title=title,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_geometry_correlations(results_df, param_sel):
    """Plota correlações entre parâmetros geométricos e L/C"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Scatter para L_nH
    valid_L = results_df[[param_sel, 'L_nH']].dropna()
    if not valid_L.empty:
        fig.add_trace(go.Scatter(
            x=valid_L[param_sel],
            y=valid_L["L_nH"],
            mode="markers+lines",
            name="L_nH",
            marker=dict(color="blue", size=8, symbol="circle"),
            line=dict(color="blue", width=1, dash='dot')
        ))
    
    # Scatter para C_pF
    valid_C = results_df[[param_sel, 'C_pF']].dropna()
    if not valid_C.empty:
        fig.add_trace(go.Scatter(
            x=valid_C[param_sel],
            y=valid_C["C_pF"],
            mode="markers+lines", 
            name="C_pF",
            marker=dict(color="red", size=8, symbol="diamond"),
            line=dict(color="red", width=1, dash='dot')
        ))
    
    fig.update_layout(
        title=f"Correlação de {param_sel} com L e C",
        xaxis_title=f"{param_sel}",
        yaxis_title="Valor Extraído",
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
        template="plotly_white",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ======================
# Reset de aplicação
# ======================
def reset_app():
    """Reset completo do estado da aplicação"""
    keys_to_keep = ['s_param_type']  # Manter preferência do usuário
    new_state = {k: st.session_state[k] for k in keys_to_keep if k in st.session_state}
    st.session_state.clear()
    st.session_state.update(new_state)
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

    # Ordenar por altura/deslocamento se existir
    height_cols = ['sample_height [mm]', 'displacement [mm]']
    for col in height_cols:
        if col in results_df.columns:
            try:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                results_df = results_df.sort_values(col)
                st.info(f"📊 Planilha ordenada por '{col}'")
                break
            except Exception as e:
                st.warning(f"⚠️ Não foi possível ordenar por '{col}': {e}")
    
    # Limpar diretório temporário
    for file in temp_path.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)

    # Nome do arquivo adaptado
    excel_path = temp_path / f"resultados_completos_{Path(filename).stem}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Resultados', index=False)

    # Nome do ZIP adaptado
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
        label=f"📥 Baixar Todos os Resultados {s_type} (ZIP)",
        data=zip_data,
        file_name=zip_path.name,
        mime="application/zip",
        use_container_width=True
    )
    st.success(f"✅ Exportação concluída! Total de {len(all_results)} ressonâncias.")

# ======================
# Função principal e aba de análise
# ======================
def main():
    setup_ui()

def run_analysis_tab():
    """Executa a aba de análise de dados S11/S21"""
    # Inicializar session_state
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'export_ready' not in st.session_state:
        st.session_state.export_ready = False

    # Upload do arquivo
    s_param_type = st.session_state.s_param_type
    uploaded_file = st.file_uploader(
        f"**Selecione o arquivo CSV com dados {s_param_type}**",
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
                    's_col': None,
                    'param_cols': None,
                    'perm_col': None,
                    's_param_type': s_param_type
                }
                st.session_state.export_ready = False

            # Preview do arquivo
            display_file_preview(df, uploaded_file)

            # Identificação de colunas
            data_info = st.session_state.uploaded_file_data
            if data_info['s_col'] is None:
                freq_col, s_col, param_cols, perm_col = identify_columns(df, s_param_type)
                data_info.update({
                    'freq_col': freq_col,
                    's_col': s_col,
                    'param_cols': param_cols,
                    'perm_col': perm_col
                })
            else:
                freq_col = data_info['freq_col']
                s_col = data_info['s_col']
                param_cols = data_info['param_cols']
                perm_col = data_info['perm_col']

            if not freq_col or not s_col:
                st.error(f"❌ Não foi possível identificar colunas de frequência e {s_param_type} automaticamente.")
                return
            st.success(f"✅ Colunas identificadas: Frequência='{freq_col}', {s_param_type}='{s_col}'")

            # Processar dados
            analysis_container = st.container()
            
            with analysis_container:
                temp_path, all_results_combined = process_data(
                    df, uploaded_file.name, freq_col, s_col, param_cols, perm_col, s_param_type
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
        st.info(f"👆 Faça upload de um arquivo CSV com dados {s_param_type} para iniciar a análise")
        st.session_state.analysis_results = None
        st.session_state.uploaded_file_data = None
        st.session_state.export_ready = False
        if st.button("🔄 Resetar Aplicação", use_container_width=True):
            reset_app()

if __name__ == "__main__":
    main()