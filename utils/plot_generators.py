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


def plot_curves_comparison(df, s_param_type="S21"):
    """
    Gera gráfico comparativo de curvas S11/S21 agrupadas por sample_height 
    e com cores diferentes para cada permissividade.
    """
    
    if df.empty:
        st.warning("⚠️ Nenhum dado disponível para plotar.")
        return None
    
    # Verificar colunas necessárias
    required_cols = ['freq_ghz', f'{s_param_type.lower()}_db']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colunas necessárias não encontradas: {missing_cols}")
        return None
    
    # Verificar se temos as colunas de agrupamento
    has_sample_height = 'sample_height [mm]' in df.columns
    has_permittivity = '$perm2 []' in df.columns
    
    if not has_sample_height:
        st.warning("⚠️ Coluna 'sample_height [mm]' não encontrada para agrupamento.")
        return None
    
    # Criar figura
    fig = go.Figure()
    
    # Obter combinações únicas de sample_height e permissividade
    if has_permittivity:
        unique_combinations = df[['sample_height [mm]', '$perm2 []']].drop_duplicates()
        # Ordenar por sample_height e depois por permissividade
        unique_combinations = unique_combinations.sort_values(['sample_height [mm]', '$perm2 []'])
    else:
        unique_combinations = df[['sample_height [mm]']].drop_duplicates()
        unique_combinations = unique_combinations.sort_values('sample_height [mm]')
        unique_combinations['$perm2 []'] = 'N/A'
    
    # Definir paleta de cores por sample_height
    sample_heights = sorted(unique_combinations['sample_height [mm]'].unique())
    color_palettes = {
        'Vermelho': px.colors.sequential.Reds,
        'Azul': px.colors.sequential.Blues,
        'Verde': px.colors.sequential.Greens,
        'Roxo': px.colors.sequential.Purples,
        'Laranja': px.colors.sequential.Oranges,
        'Cinza': px.colors.sequential.Greys
    }
    
    # Mapear cada sample_height para uma paleta
    height_to_palette = {}
    available_palettes = list(color_palettes.keys())
    
    for i, height in enumerate(sample_heights):
        palette_name = available_palettes[i % len(available_palettes)]
        height_to_palette[height] = color_palettes[palette_name]
    
    # Plotar cada curva
    for _, combo in unique_combinations.iterrows():
        sample_height = combo['sample_height [mm]']
        permittivity = combo['$perm2 []']
        
        # Filtrar dados para esta combinação
        if has_permittivity:
            mask = (df['sample_height [mm]'] == sample_height) & (df['$perm2 []'] == permittivity)
        else:
            mask = (df['sample_height [mm]'] == sample_height)
        
        df_subset = df[mask].copy()
        
        if df_subset.empty:
            continue
        
        # Ordenar por frequência
        df_subset = df_subset.sort_values('freq_ghz')
        
        # Escolher cor baseada no sample_height e permissividade
        palette = height_to_palette[sample_height]
        perm_values = unique_combinations[unique_combinations['sample_height [mm]'] == sample_height]['$perm2 []'].unique()
        
        if len(perm_values) > 1:
            # Múltiplas permissividades para mesma altura - usar tons diferentes
            perm_index = list(perm_values).index(permittivity)
            color_index = min(perm_index * 2 + 2, len(palette) - 1)  # Pular tons muito claros
            line_color = palette[color_index]
        else:
            # Apenas uma permissividade - usar cor média da paleta
            line_color = palette[len(palette) // 2]
        
        # Criar label da legenda
        if has_permittivity:
            legend_label = f"Altura: {sample_height} mm, εr: {permittivity}"
        else:
            legend_label = f"Altura: {sample_height} mm"
        
        # Adicionar trace ao gráfico
        fig.add_trace(
            go.Scatter(
                x=df_subset['freq_ghz'],
                y=df_subset[f'{s_param_type.lower()}_db'],
                mode='lines',
                name=legend_label,
                line=dict(color=line_color, width=2),
                hovertemplate=(
                    f"<b>{s_param_type}</b><br>" +
                    "Freq: %{x:.3f} GHz<br>" +
                    f"{s_param_type}: %{{y:.2f}} dB<br>" +
                    f"Altura: {sample_height} mm<br>" +
                    (f"εr: {permittivity}<br>" if has_permittivity else "") +
                    "<extra></extra>"
                )
            )
        )
    
    # Configurar layout
    title = f"Comparação de Curvas {s_param_type} por Altura da Amostra"
    if has_permittivity:
        title += " e Permissividade"
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="Frequência (GHz)",
        yaxis_title=f"{s_param_type} (dB)",
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(l=50, r=200, t=50, b=50),
        height=600,
        showlegend=True
    )
    
    # Adicionar grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig