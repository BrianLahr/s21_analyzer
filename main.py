import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import zipfile

from utils.file_handlers import handle_file_upload, display_file_preview
from utils.data_processors import identify_columns, process_data

def setup_ui():
    """Configuração inicial de layout"""
    st.set_page_config(
        page_title="Analisador S21",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CORREÇÃO: REMOVER CSS FIXO para adaptar ao modo escuro automaticamente
    st.markdown('<h1 style="text-align: center; margin-bottom: 2rem;">📊 Analisador de Parâmetros S21</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("ℹ️ Sobre")
    st.sidebar.info(
        "Esta aplicação analisa ressonâncias em dados S21 de arquivos CSV. "
        "Altere as frequências para calcular automaticamente os parâmetros."
    )

        # NOVO: Sistema de abas
    tab1, tab2 = st.tabs(["📈 Análise de Dados S21", "📊 Visualização de Resultados"])
    
    with tab1:
        run_analysis_tab()
    
    with tab2:
        from utils.results_visualizer import create_results_visualizer
        create_results_visualizer()

# ======================
# Reset de aplicação
# ======================
def reset_app():
    """Reset completo do estado da aplicação"""
    st.session_state.clear()
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

    # Salvar Excel
    excel_path = temp_path / f"resultados_completos_{Path(filename).stem}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Resultados', index=False)

    # Criar ZIP
    zip_path = temp_path / f"resultados_analise_s21_{Path(filename).stem}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(excel_path, excel_path.name)
        for txt_file in temp_path.glob("*.txt"):
            zip_file.write(txt_file, txt_file.name)

    # Botão de download
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    st.download_button(
        label="📥 Baixar Todos os Resultados (ZIP)",
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
    """Executa a aba de análise de dados S21"""
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
    uploaded_file = st.file_uploader(
        "**Selecione o arquivo CSV**",
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
                    's21_col': None,
                    'param_cols': None,
                    'perm_col': None
                }
                st.session_state.export_ready = False

            # Preview do arquivo
            display_file_preview(df, uploaded_file)

            # Identificação de colunas
            data_info = st.session_state.uploaded_file_data
            if data_info['freq_col'] is None or data_info['s21_col'] is None:
                freq_col, s21_col, param_cols, perm_col = identify_columns(df)
                data_info.update({
                    'freq_col': freq_col,
                    's21_col': s21_col,
                    'param_cols': param_cols,
                    'perm_col': perm_col
                })
            else:
                freq_col = data_info['freq_col']
                s21_col = data_info['s21_col']
                param_cols = data_info['param_cols']
                perm_col = data_info['perm_col']

            if not freq_col or not s21_col:
                st.error("❌ Não foi possível identificar colunas de frequência e S21 automaticamente.")
                return
            st.success(f"✅ Colunas identificadas: Frequência='{freq_col}', S21='{s21_col}'")

            # Processar dados
            analysis_container = st.container()
            
            with analysis_container:
                temp_path, all_results_combined = process_data(
                    df, uploaded_file.name, freq_col, s21_col, param_cols, perm_col
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
        st.info("👆 Faça upload de um arquivo CSV para iniciar a análise")
        st.session_state.analysis_results = None
        st.session_state.uploaded_file_data = None
        st.session_state.export_ready = False
        if st.button("🔄 Resetar Aplicação"):
            reset_app()

if __name__ == "__main__":
    main()
