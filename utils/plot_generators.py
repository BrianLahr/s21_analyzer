import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate

def plot_interactive_curve(df, param_id, params, param_cols):
    """Plota a curva S21 interativa para visualização"""
    
    # Interpolar dados para plotagem suave
    interp_func_db = interpolate.interp1d(df['freq_ghz'], df['s21_db'], 
                                        kind='cubic', fill_value='extrapolate')
    interp_func_linear = interpolate.interp1d(df['freq_ghz'], df['s21_linear'], 
                                            kind='cubic', fill_value='extrapolate')
    
    freq_min, freq_max = df['freq_ghz'].min(), df['freq_ghz'].max()
    freq_interp = np.linspace(freq_min, freq_max, 10000)
    s21_db_interp = interp_func_db(freq_interp)
    s21_linear_interp = interp_func_linear(freq_interp)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('S21 em dB - Identifique as ressonâncias no gráfico', 'S21 em Unidades Lineares'),
        vertical_spacing=0.1
    )
    
    # Gráfico em dB
    fig.add_trace(
        go.Scatter(x=freq_interp, y=s21_db_interp, mode='lines', 
                  name='S21 (dB) - Interpolado', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['freq_ghz'], y=df['s21_db'], mode='markers',
                  name='Dados Originais', marker=dict(color='lightblue', size=4, opacity=0.6)),
        row=1, col=1
    )
    
    # Gráfico linear
    fig.add_trace(
        go.Scatter(x=freq_interp, y=s21_linear_interp, mode='lines',
                  name='S21 (linear) - Interpolado', line=dict(color='green')),
        row=2, col=1
    )
    
    # Título com parâmetros
    if param_cols[0] != '_dummy':
        param_title = " | ".join([f"{col}: {params[col]}" for col in param_cols])
        title_text = f"Análise S21 - {param_id}<br><sub>{param_title}</sub>"
    else:
        title_text = f"Análise S21 - {param_id}"
    
    fig.update_layout(
        height=700, 
        title_text=title_text,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Frequência (GHz)", row=1, col=1)
    fig.update_yaxes(title_text="S21 (dB)", row=1, col=1)
    fig.update_xaxes(title_text="Frequência (GHz)", row=2, col=1)
    fig.update_yaxes(title_text="S21 (linear)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Instruções para o usuário
    st.info("""
    **📝 Instruções:**
    1. Observe o gráfico acima e identifique as ressonâncias (mínimos na curva S21)
    2. Na seção abaixo, insira a frequência de cada ressonância que deseja analisar
    3. O sistema calculará automaticamente os parâmetros para cada ressonância identificada
    """)