import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import io

def create_results_visualizer():
    """Cria a interface para visualização e análise de resultados exportados"""
    
    st.markdown("## 📊 Visualização e Análise de Resultados")
    st.markdown("""
    Faça upload de arquivos Excel exportados (.xlsx) para visualizar e comparar resultados.
    Cada arquivo será plotado como uma curva diferente no gráfico.
    """)
    
    # Upload de múltiplos arquivos Excel
    uploaded_files = st.file_uploader(
        "**Selecione os arquivos Excel para análise**",
        type=['xlsx'],
        accept_multiple_files=True,
        key="excel_uploader"
    )
    
    if not uploaded_files:
        st.info("👆 Faça upload de um ou mais arquivos Excel para começar a análise")
        return
    
    # Processar arquivos carregados
    all_data = load_and_process_files(uploaded_files)
    
    # CORREÇÃO: Verificar se o DataFrame está vazio corretamente
    if all_data.empty:
        st.error("❌ Nenhum dado válido encontrado nos arquivos carregados.")
        return
    
    # NOVA FUNCIONALIDADE: Seletor de colunas para remover
    st.markdown("---")
    st.markdown("### 🗂️ Gerenciamento de Colunas e Linhas")
    
    # Mostrar tabela com opção de remover colunas e filtrar linhas
    filtered_data = display_data_table_with_filters(all_data)
    
    # Interface de configuração do gráfico
    st.markdown("---")
    st.markdown("### ⚙️ Configuração do Gráfico")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Seleção do eixo X
        x_axis = st.selectbox(
            "Eixo X",
            get_numeric_columns(filtered_data),
            key="x_axis_select"
        )
    
    with col2:
        # Seleção do eixo Y
        y_axis = st.selectbox(
            "Eixo Y", 
            get_numeric_columns(filtered_data),
            key="y_axis_select"
        )
    
    with col3:
        # Tipo de gráfico
        chart_type = st.selectbox(
            "Tipo de Gráfico",
            ["Linha", "Dispersão", "Linha + Pontos"],
            key="chart_type_select"
        )
    
    # Filtros adicionais
    st.markdown("#### 🔍 Filtros e Configurações")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Filtrar por número de ressonância
        ressonancia_filter = st.multiselect(
            "Filtrar por Ressonância",
            sorted(filtered_data['ressonancia_num'].unique()) if 'ressonancia_num' in filtered_data.columns else [],
            key="ressonancia_filter"
        )
    
    with col5:
        # Agrupar por
        group_by = st.selectbox(
            "Agrupar por",
            ["Nenhum"] + get_groupable_columns(filtered_data),
            key="group_by_select"
        )
    
    with col6:
        # Ordenar dados
        sort_data = st.checkbox("Ordenar dados por eixo X", value=True, key="sort_checkbox")
    
    # Aplicar filtros adicionais
    final_filtered_data = apply_filters(filtered_data, ressonancia_filter)
    
    if final_filtered_data.empty:
        st.warning("⚠️ Nenhum dado corresponde aos filtros aplicados.")
        return
    
    # Criar gráfico
    st.markdown("---")
    st.markdown("### 📈 Gráfico de Resultados")
    
    fig = create_interactive_plot(final_filtered_data, x_axis, y_axis, chart_type, group_by, sort_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas e informações
    st.markdown("---")
    st.markdown("### 📋 Estatísticas dos Dados")
    
    display_statistics(final_filtered_data, x_axis, y_axis)

def display_data_table_with_filters(df):
    """Exibe a tabela de dados com opção de selecionar colunas para remover e filtrar linhas"""
    
    st.markdown("#### 📊 Visualização dos Dados com Controle de Colunas e Linhas")
    
    # Criar duas abas para organização
    tab1, tab2 = st.tabs(["👁️ Controle de Colunas", "🎯 Filtro de Linhas"])
    
    with tab1:
        st.markdown("Selecione as colunas que deseja **remover** da visualização:")
        
        # Obter todas as colunas disponíveis
        all_columns = list(df.columns)
        
        # Remover colunas que não devem ser selecionáveis para remoção
        non_removable_columns = ['arquivo']  # Colunas essenciais que não podem ser removidas
        selectable_columns = [col for col in all_columns if col not in non_removable_columns]
        
        # Criar multiselect para escolher colunas a remover
        columns_to_remove = st.multiselect(
            "**Colunas para ocultar da tabela:**",
            options=selectable_columns,
            default=[],  # Nenhuma selecionada por padrão
            help="Selecione as colunas que deseja remover da visualização da tabela"
        )
        
        # Criar DataFrame filtrado (sem as colunas selecionadas para remover)
        display_columns = [col for col in all_columns if col not in columns_to_remove]
        column_filtered_df = df[display_columns]
    
    with tab2:
        st.markdown("Filtre linhas baseadas em valores de colunas específicas:")
        
        # Selecionar coluna para filtrar
        filter_column = st.selectbox(
            "**Coluna para filtrar:**",
            options=[col for col in df.columns if col != 'arquivo'],
            help="Selecione a coluna que deseja usar para filtrar as linhas"
        )
        
        # Mostrar informações sobre a coluna selecionada
        if filter_column in df.columns:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Tipo de dados", str(df[filter_column].dtype))
            with col_info2:
                st.metric("Valores únicos", df[filter_column].nunique())
            with col_info3:
                st.metric("Valores nulos", df[filter_column].isnull().sum())
            
            # Interface de filtro baseada no tipo de dados
            if pd.api.types.is_numeric_dtype(df[filter_column]):
                # Para colunas numéricas: permitir filtrar por range ou valor específico
                filter_method = st.radio(
                    "Método de filtro:",
                    ["Remover valores específicos", "Remover fora do intervalo"],
                    key="numeric_filter_method"
                )
                
                if filter_method == "Remover valores específicos":
                    # Multiselect para valores numéricos
                    unique_values = sorted(df[filter_column].dropna().unique())
                    values_to_remove = st.multiselect(
                        f"Valores em '{filter_column}' para **remover**:",
                        options=unique_values,
                        help="Selecione os valores que deseja remover do dataset"
                    )
                    
                    # Aplicar filtro
                    if values_to_remove:
                        row_filtered_df = df[~df[filter_column].isin(values_to_remove)]
                    else:
                        row_filtered_df = df
                        
                else:  # Remover fora do intervalo
                    min_val = float(df[filter_column].min())
                    max_val = float(df[filter_column].max())
                    
                    col_range1, col_range2 = st.columns(2)
                    with col_range1:
                        lower_bound = st.number_input(
                            "Valor mínimo (incluir):",
                            value=min_val,
                            min_value=min_val,
                            max_value=max_val,
                            key="lower_bound"
                        )
                    with col_range2:
                        upper_bound = st.number_input(
                            "Valor máximo (incluir):",
                            value=max_val,
                            min_value=min_val,
                            max_value=max_val,
                            key="upper_bound"
                        )
                    
                    # Aplicar filtro de intervalo
                    row_filtered_df = df[(df[filter_column] >= lower_bound) & (df[filter_column] <= upper_bound)]
                    
            else:
                # Para colunas categóricas/string: multiselect
                unique_values = sorted(df[filter_column].dropna().unique())
                values_to_remove = st.multiselect(
                    f"Valores em '{filter_column}' para **remover**:",
                    options=unique_values,
                    help="Selecione os valores que deseja remover do dataset"
                )
                
                # Aplicar filtro
                if values_to_remove:
                    row_filtered_df = df[~df[filter_column].isin(values_to_remove)]
                else:
                    row_filtered_df = df
            
            # Mostrar estatísticas do filtro
            removed_count = len(df) - len(row_filtered_df)
            st.info(f"🔍 Filtro aplicado: {len(row_filtered_df)} linhas restantes ({removed_count} linhas removidas)")
            
        else:
            row_filtered_df = df
            st.warning("Selecione uma coluna válida para filtrar")
    
    # Combinar filtros de colunas e linhas
    if 'column_filtered_df' in locals() and 'row_filtered_df' in locals():
        # Aplicar filtro de colunas ao DataFrame filtrado por linhas
        final_display_df = row_filtered_df[display_columns]
    else:
        final_display_df = df[display_columns] if 'display_columns' in locals() else df
    
    # Mostrar informações gerais sobre a seleção
    st.markdown("---")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Colunas totais", len(all_columns))
    with col_info2:
        st.metric("Colunas visíveis", len(display_columns))
    with col_info3:
        st.metric("Colunas ocultas", len(columns_to_remove))
    with col_info4:
        st.metric("Linhas totais", len(final_display_df))
    
    # Mostrar a tabela com os dados filtrados
    st.markdown(f"**Tabela de Dados ({len(final_display_df)} linhas × {len(display_columns)} colunas):**")
    
    # Adicionar opção para mostrar/ocultar a tabela completa
    show_full_table = st.checkbox("Mostrar tabela completa de dados", value=False)
    
    if show_full_table:
        # Mostrar tabela com paginação para melhor performance
        st.dataframe(
            final_display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
    else:
        # Mostrar apenas uma prévia
        st.dataframe(
            final_display_df.head(100),  # Limitar a 100 linhas para prévia
            use_container_width=True,
            height=300,
            hide_index=True
        )
        if len(final_display_df) > 100:
            st.info(f"📋 Mostrando 100 de {len(final_display_df)} linhas. Marque a opção acima para ver toda a tabela.")
    
    # Mostrar resumo dos filtros aplicados
    if columns_to_remove:
        st.warning(f"🚫 **Colunas ocultas:** {', '.join(columns_to_remove)}")
    
    if 'values_to_remove' in locals() and values_to_remove:
        st.warning(f"🚫 **Linhas removidas:** {len(df) - len(row_filtered_df)} linhas onde '{filter_column}' contém {values_to_remove}")
    
    # Retornar o DataFrame filtrado para uso no resto do aplicativo
    return row_filtered_df if 'row_filtered_df' in locals() else df

def load_and_process_files(uploaded_files):
    """Carrega e processa múltiplos arquivos Excel"""
    all_data = []
    
    for uploaded_file in uploaded_files:
        try:
            # Ler arquivo Excel
            df = pd.read_excel(uploaded_file, sheet_name='Resultados')
            
            # Adicionar coluna com nome do arquivo
            df['arquivo'] = uploaded_file.name
            
            # Converter colunas numéricas
            df = convert_numeric_columns(df)
            
            all_data.append(df)
            
        except Exception as e:
            st.error(f"❌ Erro ao processar {uploaded_file.name}: {e}")
    
    # CORREÇÃO: Retornar DataFrame vazio se não houver dados
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def convert_numeric_columns(df):
    """Converte colunas para numérico quando possível"""
    for col in df.columns:
        if col not in ['parametros', 'arquivo']:
            try:
                # CORREÇÃO: Remover errors='ignore' deprecated
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Manter como string se não puder converter
                pass
    return df

def get_numeric_columns(df):
    """Retorna lista de colunas numéricas para eixos"""
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in ['ressonancia_num', 'arquivo']:
            numeric_cols.append(col)
    return numeric_cols

def get_groupable_columns(df):
    """Retorna colunas que podem ser usadas para agrupamento"""
    groupable = []
    for col in df.columns:
        if col not in ['parametros', 'arquivo'] and df[col].nunique() < 20:  # Limitar a colunas com poucos valores únicos
            groupable.append(col)
    return groupable

def apply_filters(df, ressonancia_filter):
    """Aplica filtros aos dados"""
    filtered_df = df.copy()
    
    if ressonancia_filter and 'ressonancia_num' in df.columns:
        filtered_df = filtered_df[filtered_df['ressonancia_num'].isin(ressonancia_filter)]
    
    return filtered_df

def create_interactive_plot(df, x_axis, y_axis, chart_type, group_by, sort_data):
    """Cria gráfico interativo com Plotly"""
    
    # Ordenar dados se solicitado
    if sort_data and x_axis in df.columns:
        df = df.sort_values(by=[x_axis, 'ressonancia_num'])
    
    # Configurar cores por arquivo
    color_col = 'arquivo'
    if group_by != "Nenhum":
        color_col = group_by
    
    # Criar figura base
    if chart_type == "Linha":
        fig = px.line(df, x=x_axis, y=y_axis, color=color_col,
                     hover_data=get_hover_columns(df),
                     title=f"{y_axis} vs {x_axis}")
    
    elif chart_type == "Dispersão":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col,
                        hover_data=get_hover_columns(df),
                        title=f"{y_axis} vs {x_axis}")
    
    else:  # Linha + Pontos
        fig = px.line(df, x=x_axis, y=y_axis, color=color_col,
                     hover_data=get_hover_columns(df),
                     title=f"{y_axis} vs {x_axis}")
        # CORREÇÃO: Adicionar pontos de forma correta
        scatter_trace = px.scatter(df, x=x_axis, y=y_axis, color=color_col).data[0]
        fig.add_trace(scatter_trace)
    
    # Melhorar layout
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=50, r=150, t=50, b=50)
    )
    
    return fig

def get_hover_columns(df):
    """Retorna colunas para mostrar no hover"""
    hover_cols = {}
    
    # Sempre incluir arquivo e ressonância
    if 'arquivo' in df.columns:
        hover_cols['arquivo'] = True
    if 'ressonancia_num' in df.columns:
        hover_cols['ressonancia_num'] = True
    
    # Adicionar outras colunas importantes
    important_cols = ['s21_ressonancia_db', 'Q_3db', 'sensibilidade_ghz_sqrt_er', 
                     'figura_merito_normal_3db', 'sample_height [mm]', '$perm2 []']
    
    for col in important_cols:
        if col in df.columns:
            hover_cols[col] = True
    
    return hover_cols

def display_statistics(df, x_axis, y_axis):
    """Exibe estatísticas dos dados"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Pontos", len(df))
        st.metric("Arquivos Carregados", df['arquivo'].nunique())
    
    with col2:
        if x_axis in df.columns and pd.api.types.is_numeric_dtype(df[x_axis]):
            st.metric(f"Média {x_axis}", f"{df[x_axis].mean():.4f}")
            st.metric(f"Desvio Padrão {x_axis}", f"{df[x_axis].std():.4f}")
    
    with col3:
        if y_axis in df.columns and pd.api.types.is_numeric_dtype(df[y_axis]):
            st.metric(f"Média {y_axis}", f"{df[y_axis].mean():.4f}")
            st.metric(f"Desvio Padrão {y_axis}", f"{df[y_axis].std():.4f}")
    
    # Tabela resumo por arquivo
    st.markdown("#### 📄 Resumo por Arquivo")
    summary_data = []
    
    for arquivo in df['arquivo'].unique():
        arquivo_data = df[df['arquivo'] == arquivo]
        summary_row = {'Arquivo': arquivo, 'Pontos': len(arquivo_data)}
        
        if x_axis in df.columns and pd.api.types.is_numeric_dtype(df[x_axis]):
            summary_row[f'{x_axis} (média)'] = arquivo_data[x_axis].mean()
            summary_row[f'{x_axis} (std)'] = arquivo_data[x_axis].std()
        
        if y_axis in df.columns and pd.api.types.is_numeric_dtype(df[y_axis]):
            summary_row[f'{y_axis} (média)'] = arquivo_data[y_axis].mean()
            summary_row[f'{y_axis} (std)'] = arquivo_data[y_axis].std()
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data).round(4)
    st.dataframe(summary_df, use_container_width=True)