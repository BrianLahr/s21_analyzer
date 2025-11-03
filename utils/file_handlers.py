import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import tempfile
from pathlib import Path
import shutil


def handle_file_upload(uploaded_file, uploaded_file_data):
    """Processa o upload do arquivo e gerencia o session_state.
    Suporta CSV, Excel e TXT. Se uploaded_file for None, retorna dados já armazenados."""

    # Caso nenhum arquivo enviado nesta execução, usar dados em session_state (se existirem)
    if uploaded_file is None:
        if uploaded_file_data and 'df' in uploaded_file_data:
            return uploaded_file_data['df'], False
        return pd.DataFrame(), False

    # Se o nome mudou / não existia, ler o novo arquivo
    if (uploaded_file_data is None or
        'name' not in uploaded_file_data or
        uploaded_file_data['name'] != uploaded_file.name):

        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(uploaded_file)
            elif suffix in [".xls", ".xlsx"]:
                df = pd.read_excel(uploaded_file)
            elif suffix == ".txt":
                df = pd.read_csv(uploaded_file, sep="\t")
            else:
                st.error(f"❌ Formato de arquivo não suportado: {suffix}")
                return pd.DataFrame(), True

            # Atualizar session_state
            st.session_state['uploaded_file_data'] = {
                "name": uploaded_file.name,
                "df": df
            }

            return df, True

        except Exception as e:
            st.error(f"❌ Erro ao ler arquivo: {e}")
            return pd.DataFrame(), True
    else:
        return uploaded_file_data['df'], False


def display_file_preview(df, uploaded_file):
    """Exibe prévia e informações do arquivo"""

    if df.empty:
        st.warning("⚠️ Nenhum dado válido para exibir.")
        return

    # Mostrar informações básicas
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("ℹ️ Nenhuma coluna numérica para exibir estatísticas.")

        # Mostrar colunas (limitando para não estourar layout)
        max_cols = 20
        cols = list(df.columns)
        if len(cols) > max_cols:
            cols_display = cols[:max_cols] + ["..."]
        else:
            cols_display = cols
        st.write(f"**Colunas ({len(cols)}):** {cols_display}")


def display_existing_analysis(analysis_data, filename):
    """Exibe análise existente da session_state e gera arquivos de exportação"""

    if analysis_data is None:
        st.warning("⚠️ Nenhum dado de análise disponível.")
        return

    temp_path, all_results = analysis_data

    if not all_results:
        st.warning("⚠️ Nenhuma ressonância foi analisada.")
        return

    try:
        results_df = pd.DataFrame(all_results)

        # Limpar diretório temporário antes de salvar (evita arquivos antigos no ZIP)
        for file in temp_path.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)

        # ATUALIZADO: Nome do arquivo adaptado para S11/S21
        s_type = st.session_state.get('s_param_type', 'S21')
        excel_path = temp_path / f"resultados_completos_{Path(filename).stem}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Resultados', index=False)

        # ATUALIZADO: Nome do ZIP adaptado para S11/S21
        zip_path = temp_path / f"resultados_analise_{s_type.lower()}_{Path(filename).stem}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(excel_path, excel_path.name)
            for txt_file in temp_path.glob("*.txt"):
                zip_file.write(txt_file, txt_file.name)

        st.success(f"✅ Análise concluída! Total de {len(all_results)} ressonâncias analisadas.")

        # ATUALIZADO: Botão de download com label adaptado
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        st.download_button(
            label=f"📥 Baixar Todos os Resultados {s_type} (ZIP)",
            data=zip_data,
            file_name=zip_path.name,
            mime="application/zip",
            use_container_width=True,
            key=f"download_{filename}"
        )

        # Mostrar resumo dos resultados
        with st.expander("📈 Resumo dos Resultados", expanded=True):
            st.dataframe(results_df, use_container_width=True)

            if len(results_df) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'Q_3db' in results_df.columns:
                        st.metric("Fator Q Médio (-3dB)", f"{results_df['Q_3db'].mean():.1f}")
                    else:
                        st.metric("Fator Q Médio", "N/A")
                with col2:
                    if 'frequencia_ressonancia_ghz' in results_df.columns:
                        st.metric("Freq. Ressonância Média", f"{results_df['frequencia_ressonancia_ghz'].mean():.3f} GHz")
                    else:
                        st.metric("Freq. Ressonância Média", "N/A")
                with col3:
                    st.metric("Total de Ressonâncias", len(results_df))

    except Exception as e:
        st.error(f"❌ Erro ao processar resultados: {e}")