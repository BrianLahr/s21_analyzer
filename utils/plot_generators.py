import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate

def plot_interactive_curve(df, param_id, params, param_cols, s_param_type="S21"):
    """Plota a curva S11/S21 interativa para visualização"""  # ATUALIZADO
    
    if df.empty or len(df) < 2:
        st.warning(f"⚠️ Dados insuficientes para plotar a curva {param_id}")
        return
    
    # ATUALIZADO: Colunas dinâmicas baseadas no tipo S
    s_db_col = f'{s_param_type.lower()}_db'
    s_linear_col = f'{s_param_type.lower()}_linear'
    
    required_cols = ['freq_ghz', s_db_col, s_linear_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colunas necessárias não encontradas: {missing_cols}")
        return
    
    try:
        interp_func_db = interpolate.interp1d(df['freq_ghz'], df[s_db_col], 
                                            kind='cubic', fill_value='extrapolate')
        interp_func_linear = interpolate.interp1d(df['freq_ghz'], df[s_linear_col], 
                                                kind='cubic', fill_value='extrapolate')
        
        freq_min, freq_max = df['freq_ghz'].min(), df['freq_ghz'].max()
        freq_interp = np.linspace(freq_min, freq_max, 10000)
        s_db_interp = interp_func_db(freq_interp)
        s_linear_interp = interp_func_linear(freq_interp)
        
        # ATUALIZADO: Títulos dinâmicos baseados no tipo S
        if s_param_type == "S21":
            subplot_title_db = 'S21 em dB - Identifique as ressonâncias no gráfico'
            subplot_title_linear = 'S21 em Unidades Lineares'
            trace_name_db = 'S21 (dB) - Interpolado'
            trace_name_linear = 'S21 (linear) - Interpolado'
            yaxis_title_db = 'S21 (dB)'
            yaxis_title_linear = 'S21 (linear)'
            data_name = 'Dados Originais'
        else:  # S11
            subplot_title_db = 'S11 em dB - Identifique as ressonâncias no gráfico'
            subplot_title_linear = 'S11 em Unidades Lineares'
            trace_name_db = 'S11 (dB) - Interpolado'
            trace_name_linear = 'S11 (linear) - Interpolado'
            yaxis_title_db = 'S11 (dB)'
            yaxis_title_linear = 'S11 (linear)'
            data_name = 'Dados Originais'
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(subplot_title_db, subplot_title_linear),  # ATUALIZADO
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=freq_interp, y=s_db_interp, mode='lines', 
                      name=trace_name_db, line=dict(color='blue')),  # ATUALIZADO
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['freq_ghz'], y=df[s_db_col], mode='markers',
                      name=data_name, marker=dict(color='lightblue', size=4, opacity=0.6)),  # ATUALIZADO
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=freq_interp, y=s_linear_interp, mode='lines',
                      name=trace_name_linear, line=dict(color='green')),  # ATUALIZADO
            row=2, col=1
        )
        
        # ATUALIZADO: Título dinâmico
        if param_cols[0] != '_dummy':
            param_title = " | ".join([f"{col}: {params[col]}" for col in param_cols])
            title_text = f"Análise {s_param_type} - {param_id}<br><sub>{param_title}</sub>"  # ATUALIZADO
        else:
            title_text = f"Análise {s_param_type} - {param_id}"  # ATUALIZADO
        
        fig.update_layout(height=700, title_text=title_text, showlegend=True)
        fig.update_xaxes(title_text="Frequência (GHz)", row=1, col=1)
        fig.update_yaxes(title_text=yaxis_title_db, row=1, col=1)  # ATUALIZADO
        fig.update_xaxes(title_text="Frequência (GHz)", row=2, col=1)
        fig.update_yaxes(title_text=yaxis_title_linear, row=2, col=1)  # ATUALIZADO
        
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{param_id}_{s_param_type}")  # ATUALIZADO
        
        # ATUALIZADO: Instruções específicas para S11/S21
        if s_param_type == "S21":
            st.info("""
            **Instruções:**
            1. Observe o gráfico e identifique as ressonâncias (mínimos na curva S21)
            2. Altere as frequências abaixo para calcular automaticamente os parâmetros
            """)
        else:  # S11
            st.info("""
            **Instruções:**
            1. Observe o gráfico e identifique as ressonâncias (picos na curva S11)
            2. Altere as frequências abaixo para calcular automaticamente os parâmetros
            """)
        
    except Exception as e:
        st.error(f"❌ Erro ao plotar a curva {param_id}: {e}")