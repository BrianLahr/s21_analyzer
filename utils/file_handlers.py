import streamlit as st
import pandas as pd
import zipfile
import tempfile
from pathlib import Path

def handle_file_upload(uploaded_file, uploaded_file_data):
    """Processa o upload do arquivo e gerencia o session_state"""
    
    # CORREÇÃO: Verificação mais robusta
    if (uploaded_file_data is None or 
        'name' not in uploaded_file_data or 
        uploaded_file_data['name'] != uploaded_file.name):
        
        # Ler o arquivo CSV
        try:
            df = pd.read_csv(uploaded_file)
            
            # CORREÇÃO: Retornar os dados sem modificar session_state diretamente
            # O main.py será responsável por atualizar o session_state
            return df, True
            
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivo CSV: {e}")
            # Retornar DataFrame vazio em caso de erro
            return pd.DataFrame(), True
            
    else:
        # Usar dados da session_state existentes
        return uploaded_file_data['df'], False

def display_file_preview(df, uploaded_file):
    """Exibe prévia e informações do arquivo"""
    
    # CORREÇÃO: Verificar se o DataFrame não está vazio
    if df.empty:
        st.warning("⚠️ Nenhum dado válido para exibir.")
        return
        
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
            # CORREÇÃO: Verificar se há colunas numéricas para estatísticas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("ℹ️ Nenhuma coluna numérica para exibir estatísticas.")
        
        st.write(f"**Colunas:** {list(df.columns)}")

def display_existing_analysis(analysis_data, filename):
    """Exibe análise existente da session_state"""
    
    # CORREÇÃO: Verificar se analysis_data é válido
    if analysis_data is None:
        st.warning("⚠️ Nenhum dado de análise disponível.")
        return
        
    temp_path, all_results = analysis_data
    
    # CORREÇÃO: Verificar se all_results existe e não está vazio
    if not all_results:
        st.warning("⚠️ Nenhuma ressonância foi analisada.")
        return
    
    # Criar dataframe com todos os resultados
    try:
        results_df = pd.DataFrame(all_results)
        
        # CORREÇÃO: Usar bloco try-except para operações de arquivo
        try:
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
                use_container_width=True,
                key=f"download_{filename}"  # CORREÇÃO: Chave única para evitar conflitos
            )
            
        except Exception as e:
            st.error(f"❌ Erro ao criar arquivos de exportação: {e}")
            # Continuar mesmo se houver erro na exportação
            
        # Mostrar resumo dos resultados
        with st.expander("📈 Resumo dos Resultados", expanded=True):
            st.dataframe(results_df, use_container_width=True)
            
            # Estatísticas básicas
            if len(results_df) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    # CORREÇÃO: Verificar se a coluna existe
                    if 'Q_3db' in results_df.columns:
                        avg_q = results_df['Q_3db'].mean()
                        st.metric("Fator Q Médio (-3dB)", f"{avg_q:.1f}")
                    else:
                        st.metric("Fator Q Médio", "N/A")
                        
                with col2:
                    if 'frequencia_ressonancia_ghz' in results_df.columns:
                        avg_freq = results_df['frequencia_ressonancia_ghz'].mean()
                        st.metric("Freq. Ressonância Média", f"{avg_freq:.3f} GHz")
                    else:
                        st.metric("Freq. Ressonância Média", "N/A")
                        
                with col3:
                    total_ressonances = len(results_df)
                    st.metric("Total de Ressonâncias", total_ressonances)
                    
    except Exception as e:
        st.error(f"❌ Erro ao processar resultados: {e}")

# CORREÇÃO: Adicionar importação necessária
import numpy as np