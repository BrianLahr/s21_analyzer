import streamlit as st
import pandas as pd
import zipfile
import tempfile
from pathlib import Path

def handle_file_upload(uploaded_file, uploaded_file_data):
    """Processa o upload do arquivo e gerencia o session_state"""
    if uploaded_file_data is None or uploaded_file_data['name'] != uploaded_file.name:
        # Ler o arquivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Armazenar dados na session_state
        st.session_state.uploaded_file_data = {
            'name': uploaded_file.name,
            'df': df,
            'freq_col': None,
            's21_col': None,
            'param_cols': None,
            'perm_col': None
        }
        return df, True
    else:
        # Usar dados da session_state
        return uploaded_file_data['df'], False

def display_file_preview(df, uploaded_file):
    """Exibe prévia e informações do arquivo"""
    # Mostrar informações do arquivo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Linhas", df.shape[0])
    with col2:
        st.metric("Colunas", df.shape[1])
    with col3:
        st.metric("Tamanho", f"{uploaded_file.size / 1024:.1f} KB")
    
    # Mostrar prévia dos dados
    with st.expander("🔍 Prévia dos dados", expanded=False):
        tab1, tab2 = st.tabs(["Primeiras linhas", "Estatísticas"])
        with tab1:
            st.dataframe(df.head(), use_container_width=True)
        with tab2:
            st.dataframe(df.describe(), use_container_width=True)
        
        st.write(f"**Colunas:** {list(df.columns)}")

def display_existing_analysis(analysis_data, filename):
    """Exibe análise existente da session_state"""
    temp_path, all_results = analysis_data
    
    # Criar dataframe com todos os resultados
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Salvar Excel
        excel_path = temp_path / f"resultados_completos_{Path(filename).stem}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Resultados', index=False)
        
        # Criar arquivo ZIP
        zip_path = temp_path / f"resultados_analise_s21_{Path(filename).stem}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Adicionar Excel
            zip_file.write(excel_path, excel_path.name)
            
            # Adicionar todos os arquivos TXT
            for txt_file in temp_path.glob("*.txt"):
                zip_file.write(txt_file, txt_file.name)
        
        # Mostrar resumo final
        st.success(f"✅ Análise concluída! Total de {len(all_results)} ressonâncias analisadas.")
        
        # Botão para download
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        
        st.download_button(
            label="📥 Baixar Todos os Resultados (ZIP)",
            data=zip_data,
            file_name=zip_path.name,
            mime="application/zip",
            use_container_width=True
        )
        
        # Mostrar resumo dos resultados
        with st.expander("📈 Resumo dos Resultados", expanded=True):
            st.dataframe(results_df, use_container_width=True)
            
            # Estatísticas básicas
            if len(results_df) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_q = results_df['Q_3db'].mean()
                    st.metric("Fator Q Médio (-3dB)", f"{avg_q:.1f}")
                with col2:
                    avg_freq = results_df['frequencia_ressonancia_ghz'].mean()
                    st.metric("Freq. Ressonância Média", f"{avg_freq:.3f} GHz")
                with col3:
                    total_ressonances = len(results_df)
                    st.metric("Total de Ressonâncias", total_ressonances)
    else:
        st.warning("⚠️ Nenhuma ressonância foi analisada.")