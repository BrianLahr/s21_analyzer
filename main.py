import streamlit as st
import pandas as pd
from pathlib import Path

# Importar módulos
from utils.file_handlers import handle_file_upload, display_file_preview
from utils.data_processors import identify_columns, analyze_data
from utils.plot_generators import plot_interactive_curve
from utils.calculation_engines import manual_ressonance_identification

def main():
    st.set_page_config(
        page_title="Analisador S21", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS customizado
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .ressonance-input {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 Analisador de Parâmetros S21</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar com informações
    st.sidebar.title("ℹ️ Sobre")
    st.sidebar.info(
        "Esta aplicação analisa ressonâncias em dados S21 de arquivos CSV. "
        "Plote os gráficos e identifique manualmente as ressonâncias para cálculo dos parâmetros."
    )
    
    # Inicializar session_state de forma mais robusta
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "**Selecione o arquivo CSV**", 
        type=['csv'],
        help="Arquivo CSV contendo dados de frequência e S21",
        key=f"file_uploader_{st.session_state.uploader_key}"
    )
    
    if uploaded_file is not None:
        try:
            # Usar file_handler para processar upload
            df, file_changed = handle_file_upload(uploaded_file, st.session_state.uploaded_file_data)
            
            if file_changed:
                # Resetar análise quando o arquivo muda
                st.session_state.analysis_started = False
                st.session_state.analysis_results = None
            
            # Mostrar informações do arquivo
            display_file_preview(df, uploaded_file)
            
            # CORREÇÃO: Verificar se uploaded_file_data existe antes de acessar suas chaves
            if (st.session_state.uploaded_file_data is None or 
                st.session_state.uploaded_file_data.get('freq_col') is None):
                
                freq_col, s21_col, param_cols, perm_col = identify_columns(df)
                
                # Garantir que uploaded_file_data existe
                if st.session_state.uploaded_file_data is None:
                    st.session_state.uploaded_file_data = {}
                
                st.session_state.uploaded_file_data.update({
                    'freq_col': freq_col,
                    's21_col': s21_col,
                    'param_cols': param_cols,
                    'perm_col': perm_col,
                    'df': df,  # Armazenar o dataframe também
                    'name': uploaded_file.name
                })
            else:
                # Usar dados existentes da session_state
                freq_col = st.session_state.uploaded_file_data['freq_col']
                s21_col = st.session_state.uploaded_file_data['s21_col']
                param_cols = st.session_state.uploaded_file_data['param_cols']
                perm_col = st.session_state.uploaded_file_data['perm_col']
                # Usar o dataframe da session_state para consistência
                df = st.session_state.uploaded_file_data['df']
            
            if freq_col and s21_col:
                st.success(f"✅ Colunas identificadas: Frequência='{freq_col}', S21='{s21_col}'")
                
                # Botão para iniciar análise
                if not st.session_state.analysis_started:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(
                            "🚀 Plotar Gráficos e Identificar Ressonâncias", 
                            type="primary",
                            use_container_width=True,
                            key="start_analysis"
                        ):
                            st.session_state.analysis_started = True
                            st.rerun()
                
                # Se a análise já foi iniciada, mostrar os resultados
                if st.session_state.analysis_started:
                    # Usar um container para evitar recarregamento completo
                    analysis_container = st.container()
                    
                    with analysis_container:
                        with st.spinner("🔬 Processando dados... Isso pode levar alguns segundos"):
                            if st.session_state.analysis_results is None:
                                st.session_state.analysis_results = analyze_data(
                                    df, uploaded_file.name, freq_col, s21_col, param_cols, perm_col
                                )
                            else:
                                # Reutilizar resultados existentes
                                from utils.file_handlers import display_existing_analysis
                                display_existing_analysis(st.session_state.analysis_results, uploaded_file.name)
                
            else:
                st.error("❌ Não foi possível identificar colunas de frequência e S21 automaticamente.")
                if not freq_col:
                    st.error("Coluna de frequência não encontrada. Procure por 'freq', 'frequency' no nome das colunas.")
                if not s21_col:
                    st.error("Coluna S21 não encontrada. Procure por 's21', 'S(2,1)', 'dB(S(2,1))' no nome das colunas.")
                
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
            st.info("💡 Dica: Verifique se o arquivo é um CSV válido e se está corretamente formatado.")
    else:
        # Tela inicial quando não há arquivo
        st.info("👆 Faça upload de um arquivo CSV para iniciar a análise")
        
        # Resetar session_state quando não há arquivo
        st.session_state.analysis_started = False
        st.session_state.uploaded_file_data = None
        st.session_state.analysis_results = None
        
        # Botão para resetar completamente (similar ao seu exemplo)
        if st.button("🔄 Resetar Aplicação"):
            st.session_state.uploader_key += 1
            st.rerun()
        
        # Exemplo de formato esperado
        with st.expander("📋 Exemplo de formato do arquivo CSV"):
            st.code("""
Freq [GHz], dB(S(2,1)), permittivity, other_param
1.0, -2.5, 4.0, 1.0
1.1, -1.8, 4.0, 1.0
1.2, -0.5, 4.0, 1.0
1.3, -4.2, 4.0, 1.0
...            """, language="csv")

if __name__ == "__main__":
    main()