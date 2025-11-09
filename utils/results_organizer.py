import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io

def create_results_organizer():
    """Cria a interface para organizar e enriquecer arquivos de resultados"""
    
    st.markdown("## 🗂️ Organizador de Resultados")
    st.markdown("""
    Faça upload de múltiplos arquivos Excel de resultados para:
    - **Organizar** os dados por altura/deslocamento da amostra, ressonância e permissividade
    - **Adicionar** novas colunas: variação de amplitude e Figura de Mérito (FoM)
    - **Exportar** todos os arquivos organizados em um ZIP
    """)
    
    # Upload de múltiplos arquivos Excel
    uploaded_files = st.file_uploader(
        "**Selecione os arquivos Excel para organizar**",
        type=['xlsx'],
        accept_multiple_files=True,
        key="results_organizer_uploader"
    )
    
    if not uploaded_files:
        st.info("👆 Faça upload de um ou mais arquivos Excel para organizar os resultados")
        return
    
    # Configurações
    st.markdown("---")
    st.markdown("### ⚙️ Configurações de Organização")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📋 Ordem de organização:**
        1. Altura/Deslocamento da amostra (crescente)
        2. Número da ressonância (crescente)  
        3. Permissividade (crescente)
        """)
    
    with col2:
        st.info("""
        **🆕 Novas colunas:**
        - **var_amplit**: Variação absoluta de amplitude (dB)
        - **FoM**: Q_3dB × sensibilidade × var_amplit
        """)
    
    # Processar arquivos
    if st.button("🚀 Processar e Organizar Arquivos", type="primary", use_container_width=True):
        with st.spinner("Processando arquivos..."):
            processed_files = process_and_organize_files(uploaded_files)
            
            if processed_files:
                # Criar arquivo ZIP com todos os resultados
                zip_buffer = create_zip_file(processed_files)
                
                # Botão de download
                st.success(f"✅ {len(processed_files)} arquivos processados com sucesso!")
                
                st.download_button(
                    label="📥 Baixar Todos os Arquivos Organizados (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="resultados_organizados.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
                # Mostrar estatísticas
                display_processing_stats(processed_files)
            else:
                st.error("❌ Nenhum arquivo foi processado com sucesso.")

def process_and_organize_files(uploaded_files):
    """Processa e organiza múltiplos arquivos Excel"""
    processed_files = {}
    
    for uploaded_file in uploaded_files:
        try:
            # Ler arquivo Excel
            df = pd.read_excel(uploaded_file, sheet_name='Resultados')
            
            # Detectar tipo S (S11 ou S21) baseado no nome do arquivo
            s_param_type = detect_s_param_type(uploaded_file.name)
            
            # Organizar dados
            df_organized = organize_dataframe(df, s_param_type)
            
            # Adicionar novas colunas
            df_enriched = add_new_columns(df_organized, s_param_type)
            
            # Gerar novo nome do arquivo
            new_filename = generate_new_filename(uploaded_file.name, s_param_type)
            
            processed_files[new_filename] = df_enriched
            
            st.success(f"✅ {uploaded_file.name} → {new_filename}")
            
        except Exception as e:
            st.error(f"❌ Erro ao processar {uploaded_file.name}: {e}")
    
    return processed_files

def detect_s_param_type(filename):
    """Detecta se o arquivo é S11 ou S21 baseado no nome"""
    filename_lower = filename.lower()
    if 's11' in filename_lower:
        return "S11"
    elif 's21' in filename_lower:
        return "S21"
    else:
        # Tentar detectar baseado nas colunas
        return "S21"  # Padrão

def detect_height_column(df):
    """Detecta qual coluna de altura/deslocamento está presente no DataFrame"""
    if 'sample_height [mm]' in df.columns:
        return 'sample_height [mm]'
    elif 'displacement [mm]' in df.columns:
        return 'displacement [mm]'
    else:
        # Tentar encontrar qualquer coluna que contenha 'height' ou 'displacement'
        for col in df.columns:
            col_lower = col.lower()
            if 'height' in col_lower or 'displacement' in col_lower:
                return col
        return None

def organize_dataframe(df, s_param_type):
    """Organiza o DataFrame na ordem especificada"""
    
    # Detectar coluna de altura/deslocamento
    height_col = detect_height_column(df)
    
    # Verificar colunas necessárias
    required_cols = [height_col, 'ressonancia_num', '$perm2 []']
    missing_cols = [col for col in required_cols if col is None or col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas ausentes para organização: {missing_cols}")
        return df
    
    # Ordenar dados
    df_sorted = df.sort_values([
        height_col, 
        'ressonancia_num', 
        '$perm2 []'
    ]).reset_index(drop=True)
    
    return df_sorted

def add_new_columns(df, s_param_type):
    """Adiciona as novas colunas var_amplit e FoM"""
    
    df_enriched = df.copy()
    
    # Adicionar coluna var_amplit
    df_enriched = calculate_amplitude_variation(df_enriched, s_param_type)
    
    # Adicionar coluna FoM
    df_enriched = calculate_figure_of_merit(df_enriched)
    
    return df_enriched

def calculate_amplitude_variation(df, s_param_type):
    """Calcula a variação de amplitude entre permissividades para mesma ressonância e altura/deslocamento"""
    
    # Coluna de amplitude baseada no tipo S
    amplitude_col = f'{s_param_type.lower()}_ressonancia_db'
    
    if amplitude_col not in df.columns:
        st.warning(f"⚠️ Coluna de amplitude '{amplitude_col}' não encontrada")
        df['var_amplit'] = np.nan
        return df
    
    # Detectar coluna de altura/deslocamento
    height_col = detect_height_column(df)
    if height_col is None:
        st.warning("⚠️ Nenhuma coluna de altura/deslocamento encontrada para calcular var_amplit")
        df['var_amplit'] = np.nan
        return df
    
    # Inicializar coluna
    df['var_amplit'] = np.nan
    
    # Agrupar por altura/deslocamento da amostra e número da ressonância
    grouped = df.groupby([height_col, 'ressonancia_num'])
    
    for (height, ressonance), group in grouped:
        if len(group) >= 2:
            # Ordenar por permissividade
            group_sorted = group.sort_values('$perm2 []')
            
            # Calcular variação entre a menor e maior permissividade
            min_perm_amplitude = group_sorted[amplitude_col].iloc[0]
            max_perm_amplitude = group_sorted[amplitude_col].iloc[-1]
            
            # Variação absoluta em dB
            amplitude_variation = abs(max_perm_amplitude - min_perm_amplitude)
            
            # Aplicar a todos do grupo
            df.loc[group.index, 'var_amplit'] = amplitude_variation
    
    return df

def calculate_figure_of_merit(df):
    """Calcula a Figura de Mérito: Q_3dB × sensibilidade × var_amplit"""
    
    # Verificar colunas necessárias
    required_cols = ['Q_3db', 'sensibilidade_mhz_sqrt_er', 'var_amplit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Colunas ausentes para FoM: {missing_cols}")
        df['FoM'] = np.nan
        return df
    
    # Calcular FoM
    df['FoM'] = df['Q_3db'] * df['sensibilidade_mhz_sqrt_er'] * df['var_amplit']
    
    return df

def generate_new_filename(original_filename, s_param_type):
    """Gera novo nome de arquivo para o resultado organizado"""
    
    original_path = Path(original_filename)
    stem = original_path.stem
    
    # Remover prefixos comuns
    if stem.startswith('resultados_completos_'):
        stem = stem.replace('resultados_completos_', '')
    
    # Adicionar sufixo de organização
    new_stem = f"resultados_organizados_{stem}"
    
    return f"{new_stem}.xlsx"

def create_zip_file(processed_files):
    """Cria arquivo ZIP com todos os arquivos processados"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, df in processed_files.items():
            # Criar arquivo Excel em memória
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Resultados', index=False)
            
            # Adicionar ao ZIP
            zip_file.writestr(filename, excel_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

def display_processing_stats(processed_files):
    """Exibe estatísticas do processamento"""
    
    st.markdown("---")
    st.markdown("### 📊 Estatísticas do Processamento")
    
    total_files = len(processed_files)
    total_rows = sum(len(df) for df in processed_files.values())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Arquivos Processados", total_files)
    
    with col2:
        st.metric("Total de Linhas", total_rows)
    
    with col3:
        if total_files > 0:
            avg_rows = total_rows // total_files
            st.metric("Média por Arquivo", avg_rows)
    
    # Mostrar prévia dos primeiros arquivos
    with st.expander("🔍 Detalhes dos Arquivos Processados"):
        for i, (filename, df) in enumerate(list(processed_files.items())[:3]):  # Mostrar apenas os 3 primeiros
            st.write(f"**{filename}**")
            st.write(f"- Linhas: {len(df)}")
            st.write(f"- Colunas: {len(df.columns)}")
            
            # Detectar qual coluna de altura está sendo usada
            height_col = detect_height_column(df)
            if height_col:
                st.write(f"- Coluna de altura: {height_col}")
            
            # Verificar se as novas colunas foram adicionadas
            new_cols = ['var_amplit', 'FoM']
            added_cols = [col for col in new_cols if col in df.columns]
            if added_cols:
                st.write(f"- Novas colunas: {', '.join(added_cols)}")
            
            # Mostrar prévia dos dados
            st.dataframe(df.head(3), use_container_width=True)
            
            if i < 2:  # Não adicionar linha após o último
                st.markdown("---")