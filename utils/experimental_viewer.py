# utils/experimental_viewer.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

def create_experimental_viewer():
    """Cria interface para visualização de dados experimentais"""
    
    st.header("🔬 Visualização de Dados Experimentais")
    
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader(
        "**Selecione o arquivo CSV com dados experimentais**",
        type=['csv'],
        key="experimental_file_upload"
    )
    
    if not uploaded_file:
        st.info("👆 Faça upload de um arquivo CSV experimental para iniciar a visualização")
        return
    
    try:
        # Carregar dados
        df, trace_names = load_experimental_data(uploaded_file)
        
        if df.empty:
            st.error("❌ Não foi possível carregar os dados do arquivo.")
            return
        
        st.success(f"✅ Dados carregados com sucesso! Encontrados {len(trace_names)} traços")
        
        # Mostrar informações sobre os traços
        with st.expander("📋 Informações dos Traços Encontrados"):
            for i, trace_name in enumerate(trace_names):
                st.write(f"**{i+1}. {trace_name}**")
        
        # Configurações de visualização
        st.markdown("---")
        st.subheader("⚙️ Configurações de Visualização")
        
        # Seleção de traços para mostrar
        col1, col2 = st.columns(2)
        
        with col1:
            selected_traces = st.multiselect(
                "Selecione os traços para visualizar:",
                options=trace_names,
                default=trace_names[:min(4, len(trace_names))],
                help="Escolha quais curvas mostrar no gráfico"
            )
        
        with col2:
            # Opções de normalização
            normalize_data = st.checkbox(
                "Normalizar magnitudes",
                value=False,
                help="Normalizar todas as curvas para mesma escala"
            )
            
            show_phase_wrap = st.checkbox(
                "Desembaralhar fase",
                value=True,
                help="Ajustar descontinuidades de fase"
            )
        
        # Controles de frequência
        st.markdown("---")
        st.subheader("📊 Controle de Faixa de Frequência")
        
        # Obter limites de frequência
        freq_min = float(df['freq_ghz'].min())
        freq_max = float(df['freq_ghz'].max())
        
        # Slider de frequência
        freq_range = st.slider(
            "Faixa de frequência (GHz):",
            min_value=freq_min,
            max_value=freq_max,
            value=(freq_min, freq_max),
            step=float((freq_max - freq_min) / 1000),
            key=f"freq_slider_{uploaded_file.name}"
        )
        
        # Extrair valores do slider
        min_freq_input, max_freq_input = freq_range
        
        # Filtrar dados pela faixa de frequência
        filtered_df = df[
            (df['freq_ghz'] >= min_freq_input) & 
            (df['freq_ghz'] <= max_freq_input)
        ].copy()
        
        # Gerar gráficos
        st.markdown("---")
        st.subheader("📈 Visualização dos Dados")
        
        if not selected_traces:
            st.warning("⚠️ Selecione pelo menos um traço para visualizar.")
            return
        
        # Criar gráficos
        fig_mag, fig_phase = create_s21_plots(filtered_df, selected_traces, normalize_data, show_phase_wrap)
        
        # Mostrar gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_mag, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_phase, use_container_width=True)
        
        # Estatísticas dos dados
        st.markdown("---")
        st.subheader("📊 Estatísticas dos Dados")
        
        display_data_statistics(filtered_df, selected_traces)
        
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo experimental: {e}")

def load_experimental_data(uploaded_file):
    """Carrega e processa dados experimentais do arquivo CSV"""
    
    # Ler arquivo ignorando linhas de comentário
    lines = []
    with uploaded_file as f:
        for line in f:
            line_str = line.decode('utf-8').strip()
            if not line_str.startswith('#') and line_str:
                lines.append(line_str)
    
    if not lines:
        return pd.DataFrame(), []
    
    # Processar header
    header_line = lines[0]
    data_lines = lines[1:]
    
    # Dividir header em colunas
    columns = header_line.split()
    
    # Processar dados
    data = []
    for line in data_lines:
        values = line.split()
        if len(values) == len(columns):
            try:
                row_data = [float(val) for val in values]
                data.append(row_data)
            except ValueError:
                continue
    
    if not data:
        return pd.DataFrame(), []
    
    # Criar DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Identificar traços S21
    trace_names = identify_s21_traces(columns)
    
    if not trace_names:
        return pd.DataFrame(), []
    
    # Processar cada traço
    processed_data = {'freq_ghz': df.iloc[:, 0] / 1e9}  # Converter Hz para GHz
    
    for trace_name in trace_names:
        # Encontrar colunas real e imaginária para este traço
        real_col = None
        imag_col = None
        
        for col in columns:
            if f're:{trace_name}_S21' in col:
                real_col = col
            elif f'im:{trace_name}_S21' in col:
                imag_col = col
        
        if real_col and imag_col and real_col in df.columns and imag_col in df.columns:
            # Calcular magnitude e fase
            real_data = df[real_col].values
            imag_data = df[imag_col].values
            
            # Magnitude em dB
            magnitude = np.sqrt(real_data**2 + imag_data**2)
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))  # Evitar log(0)
            
            # Fase em graus
            phase_rad = np.arctan2(imag_data, real_data)
            phase_deg = np.degrees(phase_rad)
            
            processed_data[f'{trace_name}_mag_db'] = magnitude_db
            processed_data[f'{trace_name}_phase_deg'] = phase_deg
    
    processed_df = pd.DataFrame(processed_data)
    
    return processed_df, trace_names

def identify_s21_traces(columns):
    """Identifica nomes dos traços S21 no header"""
    trace_names = set()
    
    for col in columns:
        # Procurar padrão: re:nome_S21 ou im:nome_S21
        if '_S21' in col and (col.startswith('re:') or col.startswith('im:')):
            # Extrair nome do traço
            parts = col.split(':')
            if len(parts) >= 2:
                trace_part = parts[1]
                trace_name = trace_part.replace('_S21', '')
                trace_names.add(trace_name)
    
    return sorted(list(trace_names))

def create_s21_plots(df, trace_names, normalize=False, phase_unwrap=True):
    """Cria gráficos de magnitude e fase S21"""
    
    # Cores para os traços
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Gráfico de magnitude
    fig_mag = go.Figure()
    
    for i, trace_name in enumerate(trace_names):
        mag_col = f'{trace_name}_mag_db'
        
        if mag_col not in df.columns:
            continue
        
        magnitude = df[mag_col].values
        
        # Normalizar se solicitado
        if normalize:
            magnitude = magnitude - np.max(magnitude)
        
        fig_mag.add_trace(go.Scatter(
            x=df['freq_ghz'],
            y=magnitude,
            name=f"{trace_name} (Mag)",
            line=dict(color=colors[i % len(colors)], width=2),
            mode='lines'
        ))
    
    fig_mag.update_layout(
        title="S21 - Magnitude (dB)",
        xaxis_title="Frequência (GHz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_white",
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Gráfico de fase
    fig_phase = go.Figure()
    
    for i, trace_name in enumerate(trace_names):
        phase_col = f'{trace_name}_phase_deg'
        
        if phase_col not in df.columns:
            continue
        
        phase = df[phase_col].values
        
        # Desembaralhar fase se solicitado
        if phase_unwrap:
            phase = np.unwrap(phase, period=360)
        
        fig_phase.add_trace(go.Scatter(
            x=df['freq_ghz'],
            y=phase,
            name=f"{trace_name} (Fase)",
            line=dict(color=colors[i % len(colors)], width=2),
            mode='lines'
        ))
    
    fig_phase.update_layout(
        title="S21 - Fase (Graus)",
        xaxis_title="Frequência (GHz)",
        yaxis_title="Fase (Graus)",
        template="plotly_white",
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig_mag, fig_phase

def display_data_statistics(df, trace_names):
    """Exibe estatísticas dos dados"""
    
    stats_data = []
    
    for trace_name in trace_names:
        mag_col = f'{trace_name}_mag_db'
        phase_col = f'{trace_name}_phase_deg'
        
        if mag_col in df.columns and phase_col in df.columns:
            mag_data = df[mag_col]
            phase_data = df[phase_col]
            
            stats_data.append({
                'Traço': trace_name,
                'Pontos': len(mag_data),
                'Mag Mín (dB)': f"{mag_data.min():.2f}",
                'Mag Máx (dB)': f"{mag_data.max():.2f}",
                'Fase Mín (°)': f"{phase_data.min():.1f}",
                'Fase Máx (°)': f"{phase_data.max():.1f}",
                'Freq Res (GHz)': find_resonance_frequency(df, trace_name)
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("ℹ️ Nenhuma estatística disponível para os traços selecionados.")

def find_resonance_frequency(df, trace_name):
    """Encontra frequência de ressonância aproximada (mínimo para S21)"""
    mag_col = f'{trace_name}_mag_db'
    
    if mag_col not in df.columns:
        return "N/A"
    
    mag_data = df[mag_col].values
    freq_data = df['freq_ghz'].values
    
    # Encontrar mínimo (ressonância para S21)
    min_idx = np.argmin(mag_data)
    resonance_freq = freq_data[min_idx]
    
    return f"{resonance_freq:.3f}"

def create_experimental_tab():
    """Função principal para criar a aba experimental"""
    try:
        create_experimental_viewer()
    except Exception as e:
        st.error(f"❌ Erro na visualização experimental: {e}")