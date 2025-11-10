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

from utils.data_processors import identify_columns

def create_correlation_analysis():
    """Cria interface para análise de correlação entre parâmetros e características de resposta"""
    
    st.header("📊 Análise de Correlação - Parâmetros vs Resposta")
    
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader(
        "**Selecione o arquivo CSV exportado do HFSS**",
        type=['csv'],
        key="correlation_file_upload"
    )
    
    if not uploaded_file:
        st.info("👆 Faça upload de um arquivo CSV para iniciar a análise de correlação")
        return
    
    try:
        # Carregar dados
        df = pd.read_csv(uploaded_file)
        
        # Identificar colunas de parâmetros (excluindo frequência e S-parameters)
        freq_cols, s_cols, param_cols, perm_cols = identify_columns(df)
        
        if not param_cols:
            st.error("❌ Não foram encontradas colunas de parâmetros de otimização no arquivo.")
            return
        
        st.success(f"✅ Encontradas {len(param_cols)} colunas de parâmetros")
        
        # Seleção da combinação ótima
        st.markdown("---")
        st.subheader("🎯 Seleção da Combinação Ótima")
        st.info("Selecione os valores dos parâmetros que correspondem à combinação ótima encontrada na otimização:")
        
        # Criar interface para seleção dos valores ótimos
        optimal_params = select_optimal_parameters(df, param_cols)
        
        if not optimal_params:
            st.warning("⚠️ Selecione valores para todos os parâmetros para continuar.")
            return
        
        # Processar dados para a combinação selecionada
        st.markdown("---")
        st.subheader("📈 Processamento dos Dados")
        
        # Filtrar dados próximos à combinação ótima
        analysis_df = filter_data_for_optimal_combination(df, optimal_params, param_cols)
        
        if analysis_df.empty:
            st.error("❌ Não foram encontrados dados suficientes para a combinação selecionada.")
            return
        
        st.success(f"✅ Encontrados {len(analysis_df)} pontos de dados para análise")
        
        # Calcular características de resposta
        response_df = calculate_response_characteristics(analysis_df, freq_cols[0] if freq_cols else df.columns[0], 
                                                        s_cols[0] if s_cols else df.columns[1])
        
        if response_df.empty:
            st.error("❌ Não foi possível calcular características de resposta.")
            return
        
        # Combinar dados
        combined_df = pd.concat([analysis_df[param_cols], response_df], axis=1)
        
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
        response_cols = response_df.columns.tolist()
        
        if analysis_type == "Matriz de Correlação":
            create_correlation_matrix(combined_df, param_cols, response_cols, 
                                    correlation_method, min_correlation, normalize_data)
        
        elif analysis_type == "Análise Individual":
            create_individual_analysis(combined_df, param_cols, response_cols, 
                                     correlation_method, normalize_data)
        
        elif analysis_type == "Análise PCA":
            create_pca_analysis(combined_df, param_cols, response_cols, normalize_data)
            
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {e}")

def identify_parameter_columns(df, freq_cols, s_cols):
    """Identifica colunas que são parâmetros de otimização"""
    # Colunas que provavelmente são parâmetros geométricos
    parameter_indicators = ['[mm]', 'radius', 'width', 'height', 'length', 'gap', 
                           'distance', 'thickness', 'size', 'param', 'geometry', 'optimization']
    
    param_cols = []
    excluded_cols = freq_cols + s_cols
    
    for col in df.columns:
        if col in excluded_cols:
            continue
            
        col_lower = col.lower()
        
        # Incluir colunas que parecem ser parâmetros
        if any(indicator in col_lower for indicator in parameter_indicators):
            param_cols.append(col)
        elif df[col].dtype in ['float64', 'int64'] and len(df[col].unique()) > 1:
            # Incluir colunas numéricas com variação (potenciais parâmetros)
            param_cols.append(col)
    
    return param_cols

def select_optimal_parameters(df, param_cols):
    """Cria interface para seleção dos valores ótimos dos parâmetros"""
    optimal_params = {}
    
    # Layout em colunas para melhor organização
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, param in enumerate(param_cols):
        with cols[i % num_cols]:
            unique_vals = sorted(df[param].unique())
            
            if len(unique_vals) <= 10:
                # Selectbox para poucos valores
                optimal_params[param] = st.selectbox(
                    f"{param}:",
                    options=unique_vals,
                    index=len(unique_vals)//2 if unique_vals else 0,
                    key=f"opt_{param}"
                )
            else:
                # Slider para muitos valores
                min_val = float(df[param].min())
                max_val = float(df[param].max())
                mean_val = float(df[param].mean())
                
                optimal_params[param] = st.slider(
                    f"{param}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    key=f"opt_{param}"
                )
    
    return optimal_params

def filter_data_for_optimal_combination(df, optimal_params, param_cols, tolerance=0.01):
    """Filtra dados próximos à combinação ótima selecionada"""
    mask = pd.Series([True] * len(df))
    
    for param, optimal_value in optimal_params.items():
        if param in df.columns:
            # Calcular tolerância baseada na variação do parâmetro
            param_range = df[param].max() - df[param].min()
            if param_range > 0:
                current_tolerance = tolerance * param_range
            else:
                current_tolerance = abs(optimal_value) * tolerance if optimal_value != 0 else 0.01
                
            mask &= (abs(df[param] - optimal_value) <= current_tolerance)
    
    return df[mask].copy()

def calculate_response_characteristics(df, freq_col, s_col):
    """Calcula características de resposta a partir dos dados S-parameters"""
    
    response_data = []
    
    # Agrupar por combinações únicas de parâmetros (se houver múltiplas frequências)
    group_cols = [col for col in df.columns if col not in [freq_col, s_col]]
    
    if not group_cols:
        # Se não há parâmetros para agrupar, processar cada linha individualmente
        for idx, row in df.iterrows():
            characteristics = calculate_single_response(row[freq_col], row[s_col])
            if characteristics:
                response_data.append(characteristics)
    else:
        # Agrupar por combinações de parâmetros e processar curva S completa
        groups = df.groupby(group_cols)
        
        for name, group in groups:
            if len(group) > 5:  # Mínimo de pontos para análise
                characteristics = analyze_s_curve(group, freq_col, s_col)
                if characteristics:
                    # Adicionar identificadores do grupo
                    if isinstance(name, tuple):
                        for i, col in enumerate(group_cols):
                            characteristics[col] = name[i]
                    else:
                        characteristics[group_cols[0]] = name
                    
                    response_data.append(characteristics)
    
    return pd.DataFrame(response_data) if response_data else pd.DataFrame()

def calculate_single_response(freq, s_db):
    """Calcula características para um único ponto (simplificado)"""
    try:
        # Para um único ponto, estimativas simplificadas
        return {
            'frequencia_ressonancia_ghz': float(freq),
            's_ressonancia_db': float(s_db),
            's_ressonancia_linear': 10**(float(s_db)/20),
            'Q_estimado': 1000,  # Valor padrão
            'largura_banda_estimada': 0.001  # Valor padrão
        }
    except:
        return None

def analyze_s_curve(group, freq_col, s_col):
    """Analisa curva S completa para extrair características"""
    try:
        # Ordenar por frequência
        group = group.sort_values(freq_col)
        freq = group[freq_col].values
        s_db = group[s_col].values
        
        if len(freq) < 5:
            return None
        
        # Encontrar ressonância (mínimo em S11 ou máximo em S21)
        if 's11' in s_col.lower():
            resonance_idx = np.argmin(s_db)
        else:  # S21
            resonance_idx = np.argmax(s_db)
        
        freq_resonance = freq[resonance_idx]
        s_resonance_db = s_db[resonance_idx]
        s_resonance_linear = 10**(s_resonance_db/20)
        
        # Estimativa simplificada de Q e largura de banda
        # Encontrar pontos -3dB
        if 's11' in s_col.lower():
            target_db = s_resonance_db + 3  # Para S11
        else:
            target_db = s_resonance_db - 3  # Para S21
        
        # Interpolação para encontrar bandwidth
        from scipy import interpolate
        try:
            interp_func = interpolate.interp1d(freq, s_db, kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Buscar pontos de cruzamento
            left_freq, right_freq = find_bandwidth_points(freq, s_db, freq_resonance, target_db)
            
            if left_freq and right_freq:
                bandwidth = right_freq - left_freq
                Q_factor = freq_resonance / bandwidth if bandwidth > 0 else 1000
            else:
                bandwidth = 0.001
                Q_factor = 1000
                
        except:
            bandwidth = 0.001
            Q_factor = 1000
        
        return {
            'frequencia_ressonancia_ghz': freq_resonance,
            's_ressonancia_db': s_resonance_db,
            's_ressonancia_linear': s_resonance_linear,
            'Q_3db': Q_factor,
            'fwhm_3db_ghz': bandwidth,
            'profundidade_ressonancia_db': abs(s_resonance_db) if 's11' in s_col.lower() else s_resonance_db
        }
        
    except Exception as e:
        st.warning(f"⚠️ Erro na análise da curva: {e}")
        return None

def find_bandwidth_points(freq, s_db, center_freq, target_db):
    """Encontra pontos de largura de banda"""
    try:
        # Encontrar cruzamentos com o valor target
        crossings = []
        for i in range(len(freq)-1):
            if (s_db[i] - target_db) * (s_db[i+1] - target_db) <= 0:
                # Interpolação linear
                x1, x2 = freq[i], freq[i+1]
                y1, y2 = s_db[i], s_db[i+1]
                if y2 != y1:
                    x_cross = x1 + (target_db - y1) * (x2 - x1) / (y2 - y1)
                    crossings.append(x_cross)
        
        if len(crossings) >= 2:
            # Encontrar os dois cruzamentos mais próximos da frequência central
            crossings_sorted = sorted(crossings, key=lambda x: abs(x - center_freq))
            return min(crossings_sorted[0], crossings_sorted[1]), max(crossings_sorted[0], crossings_sorted[1])
        else:
            return None, None
    except:
        return None, None

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
        
        total_variance = variance_df['Variância Acumulada'].iloc[-1]
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