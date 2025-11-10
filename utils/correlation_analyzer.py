# utils/correlation_analyzer.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io

def create_correlation_analysis():
    """Cria interface para análise de correlação entre parâmetros e características de resposta"""
    
    st.header("📊 Análise de Correlação - Parâmetros vs Resposta")
    
    # Verificar se existem resultados para analisar
    all_results = []
    for key in st.session_state:
        if key.startswith("results_"):
            all_results.extend(st.session_state[key])
    
    if not all_results:
        st.info("ℹ️ Nenhum resultado de análise disponível. Execute primeiro a análise de ressonâncias na aba 'Análise de Dados S11/S21'.")
        return
    
    # Converter para DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Identificar colunas de parâmetros e características
    param_cols = identify_parameter_columns(results_df)
    response_cols = identify_response_columns(results_df)
    
    if not param_cols:
        st.error("❌ Não foram encontradas colunas de parâmetros de otimização no dataset.")
        return
    
    if not response_cols:
        st.error("❌ Não foram encontradas colunas de características de resposta no dataset.")
        return
    
    st.success(f"✅ Encontrados {len(param_cols)} parâmetros e {len(response_cols)} características de resposta")
    
    # Configurações da análise
    st.markdown("---")
    st.subheader("🔧 Configurações da Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        correlation_method = st.selectbox(
            "Método de correlação:",
            ["Pearson", "Spearman"],
            help="Pearson: correlação linear | Spearman: correlação monotônica"
        )
        
        min_correlation = st.slider(
            "Correlação mínima para destacar:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Valor absoluto mínimo para considerar correlação significativa"
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Tipo de análise:",
            ["Matriz de Correlação", "Análise Individual", "Análise PCA"],
            help="Matriz: visão geral | Individual: detalhes por parâmetro | PCA: componentes principais"
        )
        
        normalize_data = st.checkbox(
            "Normalizar dados",
            value=True,
            help="Normalizar parâmetros para mesma escala"
        )
    
    # Executar análise selecionada
    if analysis_type == "Matriz de Correlação":
        create_correlation_matrix(results_df, param_cols, response_cols, 
                                correlation_method, min_correlation, normalize_data)
    
    elif analysis_type == "Análise Individual":
        create_individual_analysis(results_df, param_cols, response_cols, 
                                 correlation_method, normalize_data)
    
    elif analysis_type == "Análise PCA":
        create_pca_analysis(results_df, param_cols, response_cols, normalize_data)

def identify_parameter_columns(df):
    """Identifica colunas que são parâmetros de otimização"""
    # Colunas que provavelmente são parâmetros geométricos
    parameter_indicators = ['[mm]', 'radius', 'width', 'height', 'length', 'gap', 
                           'distance', 'thickness', 'size', 'param', 'geometry']
    
    param_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Excluir colunas que são resultados
        if any(indicator in col_lower for indicator in ['freq', 'ressonancia', 'q_', 'fwhm', 
                                                       'sensibilidade', 'figura', 's11', 's21']):
            continue
        
        # Incluir colunas que parecem ser parâmetros
        if any(indicator in col_lower for indicator in parameter_indicators):
            param_cols.append(col)
        elif df[col].dtype in ['float64', 'int64'] and len(df[col].unique()) > 1:
            # Incluir colunas numéricas com variação (potenciais parâmetros)
            param_cols.append(col)
    
    return param_cols

def identify_response_columns(df):
    """Identifica colunas que são características de resposta"""
    response_cols = []
    
    # Características principais que calculamos
    target_cols = [
        'frequencia_ressonancia_ghz',
        's11_ressonancia_db', 's21_ressonancia_db',
        's11_ressonancia_linear', 's21_ressonancia_linear',
        'Q_3db', 'Q_linear',
        'fwhm_3db_ghz', 'fwhm_linear_ghz',
        'sensibilidade_mhz_sqrt_er',
        'figura_merito_3db', 'figura_merito_linear',
        'figura_merito_normal_3db', 'figura_merito_normal_linear'
    ]
    
    for col in target_cols:
        if col in df.columns and not df[col].isna().all():
            response_cols.append(col)
    
    return response_cols

def create_correlation_matrix(df, param_cols, response_cols, method, min_corr, normalize):
    """Cria matriz de correlação entre parâmetros e características"""
    
    st.subheader("📈 Matriz de Correlação")
    
    # Preparar dados
    analysis_df = df[param_cols + response_cols].copy()
    
    # Remover linhas com valores faltantes
    analysis_df = analysis_df.dropna()
    
    if analysis_df.empty:
        st.error("❌ Não há dados suficientes para análise de correlação (valores faltantes).")
        return
    
    if normalize:
        # Normalizar apenas os parâmetros
        scaler = StandardScaler()
        analysis_df[param_cols] = scaler.fit_transform(analysis_df[param_cols])
    
    # Calcular matriz de correlação
    if method == "Pearson":
        corr_matrix = analysis_df.corr(method='pearson')
    else:
        corr_matrix = analysis_df.corr(method='spearman')
    
    # Filtrar apenas correlações entre parâmetros e respostas
    param_response_corr = corr_matrix.loc[param_cols, response_cols]
    
    # Criar heatmap
    fig = go.Figure(data=go.Heatmap(
        z=param_response_corr.values,
        x=response_cols,
        y=param_cols,
        colorscale='RdBu_r',
        zmid=0,
        text=[[f'{val:.3f}' for val in row] for row in param_response_corr.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo="text",
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlação: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Matriz de Correlação ({method}) - Parâmetros vs Características",
        xaxis_title="Características de Resposta",
        yaxis_title="Parâmetros de Otimização",
        width=800,
        height=600,
        template="plotly_white"
    )
    
    # Rotacionar labels do eixo x
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise das correlações mais fortes
    st.subheader("🎯 Correlações Mais Significativas")
    
    strong_correlations = []
    for param in param_cols:
        for response in response_cols:
            corr_value = abs(param_response_corr.loc[param, response])
            if corr_value >= min_corr:
                strong_correlations.append({
                    'Parâmetro': param,
                    'Característica': response,
                    'Correlação': param_response_corr.loc[param, response],
                    '|Correlação|': corr_value
                })
    
    if strong_correlations:
        strong_df = pd.DataFrame(strong_correlations)
        strong_df = strong_df.sort_values('|Correlação|', ascending=False)
        
        # Classificar correlações
        def classify_correlation(corr):
            abs_corr = abs(corr)
            if abs_corr >= 0.7:
                return "Muito Forte"
            elif abs_corr >= 0.5:
                return "Forte" 
            elif abs_corr >= 0.3:
                return "Moderada"
            else:
                return "Fraca"
        
        strong_df['Classificação'] = strong_df['Correlação'].apply(classify_correlation)
        
        # Exibir tabela
        st.dataframe(strong_df[['Parâmetro', 'Característica', 'Correlação', 'Classificação']], 
                    use_container_width=True)
        
        # Download dos resultados
        csv_buffer = io.BytesIO()
        strong_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            "📥 Baixar Correlações Significativas (CSV)",
            csv_buffer,
            file_name="correlacoes_significativas.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info(f"ℹ️ Nenhuma correlação com magnitude ≥ {min_corr} encontrada.")

def create_individual_analysis(df, param_cols, response_cols, method, normalize):
    """Cria análise individual para cada parâmetro"""
    
    st.subheader("🔍 Análise Individual por Parâmetro")
    
    # Selecionar parâmetro para análise
    selected_param = st.selectbox(
        "Selecione o parâmetro para análise detalhada:",
        param_cols
    )
    
    if not selected_param:
        return
    
    # Preparar dados
    analysis_df = df[[selected_param] + response_cols].copy().dropna()
    
    if analysis_df.empty:
        st.error("❌ Não há dados suficientes para análise deste parâmetro.")
        return
    
    # Calcular correlações
    correlations = []
    for response in response_cols:
        if method == "Pearson":
            corr, p_value = pearsonr(analysis_df[selected_param], analysis_df[response])
        else:
            corr, p_value = spearmanr(analysis_df[selected_param], analysis_df[response])
        
        correlations.append({
            'Característica': response,
            'Correlação': corr,
            'p-valor': p_value,
            'Significativo': p_value < 0.05
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlação', key=abs, ascending=False)
    
    # Exibir resultados
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Correlações Calculadas")
        display_df = corr_df.copy()
        display_df['Correlação'] = display_df['Correlação'].round(4)
        display_df['p-valor'] = display_df['p-valor'].round(6)
        display_df['Significativo'] = display_df['Significativo'].map({True: '✅', False: '❌'})
        
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Estatísticas")
        
        # Estatísticas do parâmetro
        param_stats = analysis_df[selected_param].describe()
        st.metric("Média", f"{param_stats['mean']:.4f}")
        st.metric("Desvio Padrão", f"{param_stats['std']:.4f}")
        st.metric("Variação", f"{analysis_df[selected_param].var():.6f}")
        
        # Correlação mais forte
        strongest_corr = corr_df.iloc[0]
        st.metric(
            "Correlação Mais Forte", 
            f"{strongest_corr['Correlação']:.3f}",
            strongest_corr['Característica']
        )
    
    # Gráficos de dispersão para correlações mais fortes
    st.markdown("#### 📉 Gráficos de Dispersão")
    
    # Selecionar características para plotar
    significant_corrs = corr_df[corr_df['Significativo'] == True].head(4)
    
    if not significant_corrs.empty:
        cols = st.columns(2)
        
        for idx, (_, row) in enumerate(significant_corrs.iterrows()):
            with cols[idx % 2]:
                fig = px.scatter(
                    analysis_df, 
                    x=selected_param, 
                    y=row['Característica'],
                    trendline="ols",
                    title=f"{selected_param} vs {row['Característica']}",
                    labels={selected_param: selected_param, row['Característica']: row['Característica']}
                )
                
                # Adicionar informação de correlação
                fig.add_annotation(
                    x=0.05, y=0.95,
                    xref="paper", yref="paper",
                    text=f"r = {row['Correlação']:.3f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
                
                fig.update_layout(template="plotly_white", height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Nenhuma correlação significativa (p < 0.05) encontrada para este parâmetro.")

def create_pca_analysis(df, param_cols, response_cols, normalize):
    """Cria análise de componentes principais"""
    
    st.subheader("🔬 Análise de Componentes Principais (PCA)")
    
    # Preparar dados
    X = df[param_cols].copy().dropna()
    y = df[response_cols].copy().dropna()
    
    if X.empty or y.empty:
        st.error("❌ Dados insuficientes para análise PCA.")
        return
    
    # Ajustar índices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    if X.empty:
        st.error("❌ Não há dados comuns suficientes para análise PCA.")
        return
    
    # Normalizar
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Aplicar PCA
    n_components = min(5, len(param_cols), len(X))
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Criar DataFrame com componentes
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Adicionar características de resposta para análise
    for resp_col in response_cols[:3]:  # Limitar a 3 características para não poluir
        if resp_col in y.columns:
            pc_df[resp_col] = y[resp_col].values
    
    # Exibir variância explicada
    st.markdown("#### 📊 Variância Explicada")
    
    variance_df = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(n_components)],
        'Variância': pca.explained_variance_ratio_,
        'Variância Acumulada': np.cumsum(pca.explained_variance_ratio_)
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de variância
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=variance_df['Componente'],
            y=variance_df['Variância'],
            name='Variância Individual',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=variance_df['Componente'],
            y=variance_df['Variância Acumulada'],
            name='Variância Acumulada',
            marker=dict(color='red', size=8),
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Variância Explicada por Componente",
            xaxis_title="Componente Principal",
            yaxis_title="Variância Explicada",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(variance_df.round(4), use_container_width=True)
        
        total_variance = variance_df['Variência Acumulada'].iloc[-1]
        st.metric("Variância Total Explicada", f"{total_variance:.1%}")
    
    # Loadings dos componentes
    st.markdown("#### 🎯 Loadings dos Componentes Principais")
    
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=param_cols
    )
    
    # Heatmap de loadings
    fig = go.Figure(data=go.Heatmap(
        z=loadings_df.values,
        x=loadings_df.columns,
        y=loadings_df.index,
        colorscale='RdBu_r',
        zmid=0,
        text=[[f'{val:.3f}' for val in row] for row in loadings_df.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo="text",
        hovertemplate="<b>%{y}</b><br>Componente: %{x}<br>Loading: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Loadings dos Parâmetros nos Componentes Principais",
        xaxis_title="Componente Principal",
        yaxis_title="Parâmetro",
        width=700,
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretação dos componentes
    st.markdown("#### 📝 Interpretação dos Componentes")
    
    for i in range(min(2, n_components)):  # Interpretar apenas os 2 primeiros
        pc_name = f'PC{i+1}'
        pc_loadings = loadings_df[pc_name]
        
        # Parâmetros mais importantes (maiores loadings em valor absoluto)
        top_params = pc_loadings.abs().sort_values(ascending=False).head(3)
        
        st.write(f"**{pc_name}** (Variância: {pca.explained_variance_ratio_[i]:.1%}):")
        
        for param, loading in pc_loadings.loc[top_params.index].items():
            direction = "aumenta" if loading > 0 else "diminui"
            st.write(f"- {param} ({loading:.3f}): Quando {pc_name} {direction}, este parâmetro tende a {direction}")
        
        st.write("")

def create_correlation_tab():
    """Função principal para criar a aba de correlação"""
    try:
        create_correlation_analysis()
    except Exception as e:
        st.error(f"❌ Erro na análise de correlação: {e}")