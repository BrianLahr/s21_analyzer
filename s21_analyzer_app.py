import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate
import zipfile
import io
import tempfile
import os
from pathlib import Path

def main():
    st.set_page_config(
        page_title="Analisador S21", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS customizado
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .ressonance-input {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 Analisador de Parâmetros S21</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar com informações
    st.sidebar.title("ℹ️ Sobre")
    st.sidebar.info(
        "Esta aplicação analisa ressonâncias em dados S21 de arquivos CSV. "
        "Plote os gráficos e identifique manualmente as ressonâncias para cálculo dos parâmetros."
    )
    
    # Inicializar session_state
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "**Selecione o arquivo CSV**", 
        type=['csv'],
        help="Arquivo CSV contendo dados de frequência e S21"
    )
    
    if uploaded_file is not None:
        try:
            # Verificar se o arquivo mudou ou se é a primeira vez
            if (st.session_state.uploaded_file_data is None or 
                st.session_state.uploaded_file_data['name'] != uploaded_file.name):
                
                # Ler o arquivo CSV
                df = pd.read_csv(uploaded_file)
                
                # Armazenar dados na session_state
                st.session_state.uploaded_file_data = {
                    'name': uploaded_file.name,
                    'df': df,
                    'freq_col': None,
                    's21_col': None,
                    'param_cols': None,
                    'perm_col': None
                }
                
                # Resetar análise quando o arquivo muda
                st.session_state.analysis_started = False
                st.session_state.analysis_results = None
            else:
                # Usar dados da session_state
                df = st.session_state.uploaded_file_data['df']
            
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
                    st.dataframe(df.describe(), use_container_width=True)
                
                st.write(f"**Colunas:** {list(df.columns)}")
            
            # Identificar colunas (apenas se ainda não foram identificadas)
            if st.session_state.uploaded_file_data['freq_col'] is None:
                freq_col, s21_col, param_cols, perm_col = identify_columns(df)
                st.session_state.uploaded_file_data.update({
                    'freq_col': freq_col,
                    's21_col': s21_col,
                    'param_cols': param_cols,
                    'perm_col': perm_col
                })
            else:
                freq_col = st.session_state.uploaded_file_data['freq_col']
                s21_col = st.session_state.uploaded_file_data['s21_col']
                param_cols = st.session_state.uploaded_file_data['param_cols']
                perm_col = st.session_state.uploaded_file_data['perm_col']
            
            if freq_col and s21_col:
                st.success(f"✅ Colunas identificadas: Frequência='{freq_col}', S21='{s21_col}'")
                
                # Botão para iniciar análise
                if not st.session_state.analysis_started:
                    if st.button(
                        "🚀 Plotar Gráficos e Identificar Ressonâncias", 
                        type="primary",
                        use_container_width=True,
                        key="start_analysis"
                    ):
                        st.session_state.analysis_started = True
                        st.rerun()
                
                # Se a análise já foi iniciada, mostrar os resultados
                if st.session_state.analysis_started:
                    with st.spinner("🔬 Processando dados... Isso pode levar alguns segundos"):
                        if st.session_state.analysis_results is None:
                            st.session_state.analysis_results = analyze_data(
                                df, uploaded_file.name, freq_col, s21_col, param_cols, perm_col
                            )
                        else:
                            # Reutilizar resultados existentes
                            display_existing_analysis(st.session_state.analysis_results, uploaded_file.name)
                
            else:
                st.error("❌ Não foi possível identificar colunas de frequência e S21 automaticamente.")
                if not freq_col:
                    st.error("Coluna de frequência não encontrada. Procure por 'freq', 'frequency' no nome das colunas.")
                if not s21_col:
                    st.error("Coluna S21 não encontrada. Procure por 's21', 'S(2,1)', 'dB(S(2,1))' no nome das colunas.")
                
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
            st.info("💡 Dica: Verifique se o arquivo é um CSV válido e se está corretamente formatado.")
    else:
        # Tela inicial quando não há arquivo
        st.info("👆 Faça upload de um arquivo CSV para iniciar a análise")
        
        # Resetar session_state quando não há arquivo
        st.session_state.analysis_started = False
        st.session_state.uploaded_file_data = None
        st.session_state.analysis_results = None
        
        # Exemplo de formato esperado
        with st.expander("📋 Exemplo de formato do arquivo CSV"):
            st.code("""
Freq [GHz], dB(S(2,1)), permittivity, other_param
1.0, -2.5, 4.0, 1.0
1.1, -1.8, 4.0, 1.0
1.2, -0.5, 4.0, 1.0
1.3, -4.2, 4.0, 1.0
...            """, language="csv")

def display_existing_analysis(analysis_data, filename):
    """Exibe análise existente da session_state"""
    temp_path, all_results = analysis_data
    
    # Criar dataframe com todos os resultados
    if all_results:
        results_df = pd.DataFrame(all_results)
        
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
            use_container_width=True
        )
        
        # Mostrar resumo dos resultados
        with st.expander("📈 Resumo dos Resultados", expanded=True):
            st.dataframe(results_df, use_container_width=True)
            
            # Estatísticas básicas
            if len(results_df) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_q = results_df['Q_3db'].mean()
                    st.metric("Fator Q Médio (-3dB)", f"{avg_q:.1f}")
                with col2:
                    avg_freq = results_df['frequencia_ressonancia_ghz'].mean()
                    st.metric("Freq. Ressonância Média", f"{avg_freq:.3f} GHz")
                with col3:
                    total_ressonances = len(results_df)
                    st.metric("Total de Ressonâncias", total_ressonances)
    else:
        st.warning("⚠️ Nenhuma ressonância foi analisada.")

def identify_columns(df):
    """Identifica automaticamente as colunas no dataframe"""
    freq_col = None
    s21_col = None
    param_cols = []
    perm_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['freq', 'frequency', 'ghz', 'mhz']):
            freq_col = col
        elif any(x in col_lower for x in ['s21', 's(2,1)', 'db(s(2,1))', 'insertion']):
            s21_col = col
        elif any(x in col_lower for x in ['perm', 'permittivity', 'epsilon', 'dielectric']):
            perm_col = col
            param_cols.append(col)
        elif col not in [freq_col, s21_col]:
            param_cols.append(col)
    
    # Se não encontrou S21, usar a primeira coluna numérica após frequência
    if s21_col is None and freq_col is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        other_numeric = [col for col in numeric_cols if col != freq_col]
        if other_numeric:
            s21_col = other_numeric[0]
    
    return freq_col, s21_col, param_cols, perm_col

def analyze_data(df, filename, freq_col, s21_col, param_cols, perm_col):
    """Função principal de análise"""
    
    # Criar diretório temporário para resultados
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Renomear colunas para facilitar
        df = df.rename(columns={freq_col: 'freq_ghz', s21_col: 's21_db'})
        
        # Converter para unidades lineares
        df['s21_linear'] = 10 ** (df['s21_db'] / 20)
        
        # Identificar combinações únicas de parâmetros
        if param_cols:
            unique_combinations = df[param_cols].drop_duplicates()
            st.write(f"**📊 Combinações de parâmetros encontradas:** {len(unique_combinations)}")
            
            # Mostrar combinações
            with st.expander("Ver combinações de parâmetros"):
                st.dataframe(unique_combinations, use_container_width=True)
        else:
            unique_combinations = pd.DataFrame({'_dummy': [1]})
            param_cols = ['_dummy']
        
        # Processar cada combinação
        all_results = []
        
        for idx, (_, combo) in enumerate(unique_combinations.iterrows()):
            if param_cols[0] != '_dummy':
                mask = pd.Series([True] * len(df))
                for col in param_cols:
                    mask = mask & (df[col] == combo[col])
                df_subset = df[mask].copy()
                param_id = "_".join([f"{col}_{combo[col]}" for col in param_cols])
                param_id = "".join(c for c in param_id if c.isalnum() or c in ('_', '-'))
            else:
                df_subset = df.copy()
                param_id = "single_curve"
                combo = {}
            
            # Processar esta curva
            st.markdown(f"## 📈 Análise: {param_id}")
            
            # Plotar gráfico primeiro
            plot_interactive_curve(df_subset, param_id, params=combo, param_cols=param_cols)
            
            # Seção para identificação manual de ressonâncias
            results = manual_ressonance_identification(
                df_subset, filename, param_id, combo, param_cols, perm_col, 
                unique_combinations, temp_path
            )
            all_results.extend(results)
        
        return temp_path, all_results

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

def manual_ressonance_identification(df, filename, param_id, params, param_cols, perm_col, 
                                   unique_combinations, temp_path):
    """Permite ao usuário identificar manualmente as ressonâncias"""
    
    st.markdown("### 🔬 Identificação Manual de Ressonâncias")
    
    # Interpolar dados para cálculos precisos
    interp_func_db = interpolate.interp1d(df['freq_ghz'], df['s21_db'], 
                                        kind='cubic', fill_value='extrapolate')
    interp_func_linear = interpolate.interp1d(df['freq_ghz'], df['s21_linear'], 
                                            kind='cubic', fill_value='extrapolate')
    
    freq_min, freq_max = df['freq_ghz'].min(), df['freq_ghz'].max()
    freq_interp = np.linspace(freq_min, freq_max, 10000)
    s21_db_interp = interp_func_db(freq_interp)
    s21_linear_interp = interp_func_linear(freq_interp)
    
    results = []
    
    # Interface para adicionar múltiplas ressonâncias
    st.markdown("#### ➕ Adicionar Ressonâncias")
    
    # Número de ressonâncias a analisar
    num_ressonances = st.number_input(
        f"Quantas ressonâncias deseja analisar em {param_id}?",
        min_value=0,
        max_value=50,
        value=1,
        key=f"num_{param_id}"
    )
    
    for i in range(num_ressonances):
        st.markdown(f'<div class="ressonance-input">', unsafe_allow_html=True)
        st.markdown(f"##### Ressonância {i+1}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_ressonancia = st.number_input(
                f"Frequência de ressonância (GHz)",
                min_value=float(freq_min),
                max_value=float(freq_max),
                value=float((freq_min + freq_max) / 2),
                step=0.01,
                format="%.4f",
                key=f"freq_{param_id}_{i}"
            )
        
        with col2:
            st.write("**Informações da ressonância:**")
            # Encontrar o ponto mais próximo nos dados interpolados
            idx_ressonancia = (freq_interp - freq_ressonancia).argmin()
            s21_ressonancia_db = s21_db_interp[idx_ressonancia]
            s21_ressonancia_linear = s21_linear_interp[idx_ressonancia]
            
            st.write(f"Frequência: {freq_interp[idx_ressonancia]:.4f} GHz")
            st.write(f"S21: {s21_ressonancia_db:.4f} dB")
            st.write(f"S21 (linear): {s21_ressonancia_linear:.4f}")
        
        # Botão para calcular parâmetros
        if st.button(f"📊 Calcular Parâmetros - Ressonância {i+1}", key=f"calc_{param_id}_{i}"):
            # Análise -3dB
            bandwidth_result_db = find_bandwidth_points(
                freq_interp, s21_db_interp, freq_ressonancia, s21_ressonancia_db, 
                s21_ressonancia_db - 3.0)
            
            # Análise 1/√2
            target_linear = s21_ressonancia_linear / np.sqrt(2)
            bandwidth_result_linear = find_bandwidth_points(
                freq_interp, s21_linear_interp, freq_ressonancia, s21_ressonancia_linear, 
                target_linear, is_linear=True)
            
            if bandwidth_result_db is None or bandwidth_result_linear is None:
                st.error(f"❌ Não foi possível calcular largura de banda para esta ressonância. Verifique se a curva possui pontos de cruzamento adequados.")
            else:
                freq_left_db, freq_right_db, fwhm_db, Q_db = bandwidth_result_db
                freq_left_linear, freq_right_linear, fwhm_linear, Q_linear = bandwidth_result_linear
                
                # Calcular sensibilidade se aplicável
                sensitivity, figure_of_merit_db, figure_of_merit_linear = calculate_sensitivity(
                    freq_ressonancia, params, perm_col, unique_combinations, Q_db, Q_linear, 
                    s21_ressonancia_linear)
                
                # Armazenar resultados
                result = {
                    'parametros': param_id,
                    'ressonancia_num': i + 1,
                    'frequencia_ressonancia_ghz': freq_ressonancia,
                    's21_ressonancia_db': s21_ressonancia_db,
                    's21_ressonancia_linear': s21_ressonancia_linear,
                    'fwhm_3db_ghz': fwhm_db,
                    'Q_3db': Q_db,
                    'fwhm_linear_ghz': fwhm_linear,
                    'Q_linear': Q_linear,
                    'sensibilidade_ghz_sqrt_er': sensitivity,
                    'figura_merito_3db': figure_of_merit_db,
                    'figura_merito_linear': figure_of_merit_linear
                }
                
                # Adicionar parâmetros específicos
                for col in param_cols:
                    if col != '_dummy':
                        result[col] = params[col]
                
                results.append(result)
                
                # Mostrar resultados
                st.success(f"✅ Parâmetros calculados para ressonância {i+1}!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📉 Método -3dB:**")
                    st.markdown(f"""
                    <div class="result-card">
                    <b>Largura de banda:</b> {fwhm_db:.6f} GHz<br>
                    <b>Fator Q:</b> {Q_db:.2f}<br>
                    <b>Frequência esquerda:</b> {freq_left_db:.6f} GHz<br>
                    <b>Frequência direita:</b> {freq_right_db:.6f} GHz
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📐 Método 1/√2:**")
                    st.markdown(f"""
                    <div class="result-card">
                    <b>Largura de banda:</b> {fwhm_linear:.6f} GHz<br>
                    <b>Fator Q:</b> {Q_linear:.2f}<br>
                    <b>Frequência esquerda:</b> {freq_left_linear:.6f} GHz<br>
                    <b>Frequência direita:</b> {freq_right_linear:.6f} GHz
                    </div>
                    """, unsafe_allow_html=True)
                
                if sensitivity is not None:
                    st.markdown("**🎯 Sensibilidade:**")
                    st.markdown(f"""
                    <div class="result-card">
                    <b>Sensibilidade:</b> {sensitivity:.6f} GHz/√εr<br>
                    <b>Figura de mérito (-3dB):</b> {figure_of_merit_db:.6f}<br>
                    <b>Figura de mérito (1/√2):</b> {figure_of_merit_linear:.6f}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Salvar resultados individuais em txt
                txt_content = generate_txt_result(result, param_cols, params)
                txt_filename = f"resultado_{Path(filename).stem}_{param_id}_ressonancia_{i+1}.txt"
                txt_path = temp_path / txt_filename
                with open(txt_path, 'w') as f:
                    f.write(txt_content)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return results

# [As funções find_bandwidth_points, calculate_sensitivity, e generate_txt_result permanecem EXATAMENTE iguais]

def find_bandwidth_points(freq, y_values, center_freq, center_value, target, is_linear=False):
    """Encontra os pontos de largura de banda"""
    
    def find_crossings(freq, y, target_val):
        crossings = []
        for i in range(len(freq) - 1):
            if (y[i] - target_val) * (y[i+1] - target_val) < 0:
                # Interpolação linear para encontrar o ponto exato
                x1, x2 = freq[i], freq[i+1]
                y1, y2 = y[i], y[i+1]
                x_cross = x1 + (target_val - y1) * (x2 - x1) / (y2 - y1)
                crossings.append(x_cross)
        return crossings
    
    crossings = find_crossings(freq, y_values, target)
    
    if len(crossings) < 2:
        st.warning(f"❌ Apenas {len(crossings)} ponto(s) de cruzamento encontrado(s). Necessário 2 pontos para calcular largura de banda.")
        return None
    
    # Encontrar os dois cruzamentos mais próximos da frequência de ressonância
    crossings_sorted = sorted(crossings, key=lambda x: abs(x - center_freq))
    
    # Pegar os dois mais próximos (pode haver mais de 2 cruzamentos em curvas complexas)
    if len(crossings_sorted) >= 2:
        freq_left = min(crossings_sorted[0], crossings_sorted[1])
        freq_right = max(crossings_sorted[0], crossings_sorted[1])
    else:
        return None
    
    bandwidth = abs(freq_right - freq_left)
    
    # Evitar divisão por zero
    if bandwidth > 0:
        Q = center_freq / bandwidth
    else:
        Q = float('inf')
        st.warning("⚠️ Largura de banda zero detectada - verifique os dados")
    
    return freq_left, freq_right, bandwidth, Q

def calculate_sensitivity(freq_ressonancia, params, perm_col, unique_combinations, 
                         Q_db, Q_linear, amplitude_linear):
    """Calcula sensibilidade e figura de mérito"""
    
    # Verificar se temos todos os valores necessários
    if (Q_db is None or Q_linear is None or amplitude_linear is None or 
        not perm_col or perm_col not in params):
        return None, None, None
    
    try:
        current_perm = params[perm_col]
        unique_perms = sorted(unique_combinations[perm_col].unique())
        
        if len(unique_perms) < 2:
            return None, None, None
        
        current_index = unique_perms.index(current_perm)
        
        if current_index < len(unique_perms) - 1:
            next_perm = unique_perms[current_index + 1]
            sqrt_diff = abs(np.sqrt(next_perm) - np.sqrt(current_perm))
            
            if sqrt_diff > 0:
                sensitivity = freq_ressonancia / sqrt_diff
            else:
                sensitivity = 0
            
            figure_of_merit_db = Q_db * sensitivity * amplitude_linear
            figure_of_merit_linear = Q_linear * sensitivity * amplitude_linear
            
            return sensitivity, figure_of_merit_db, figure_of_merit_linear
        
        return None, None, None
    except (ValueError, IndexError, TypeError) as e:
        st.warning(f"⚠️ Não foi possível calcular sensibilidade: {e}")
        return None, None, None

def generate_txt_result(result, param_cols, params):
    """Gera conteúdo TXT com resultados"""
    
    content = "RESULTADOS DA ANÁLISE S21 - IDENTIFICAÇÃO MANUAL\n"
    content += "=" * 60 + "\n\n"
    
    content += f"Ressonância: {result['ressonancia_num']}\n"
    content += f"Frequência de ressonância: {result['frequencia_ressonancia_ghz']:.6f} GHz\n"
    content += f"S21 na ressonância: {result['s21_ressonancia_db']:.6f} dB\n"
    content += f"S21 na ressonância (linear): {result['s21_ressonancia_linear']:.6f}\n\n"
    
    # Parâmetros
    if param_cols[0] != '_dummy':
        content += "Parâmetros:\n"
        for col in param_cols:
            content += f"  {col}: {params[col]}\n"
        content += "\n"
    
    content += "MÉTODO -3dB:\n"
    content += f"  Largura de banda (FWHM): {result['fwhm_3db_ghz']:.6f} GHz\n"
    content += f"  Fator de qualidade (Q): {result['Q_3db']:.2f}\n\n"
    
    content += "MÉTODO 1/√2:\n"
    content += f"  Largura de banda: {result['fwhm_linear_ghz']:.6f} GHz\n"
    content += f"  Fator de qualidade (Q): {result['Q_linear']:.2f}\n\n"
    
    if result['sensibilidade_ghz_sqrt_er'] is not None:
        content += "SENSIBILIDADE:\n"
        content += f"  Sensibilidade: {result['sensibilidade_ghz_sqrt_er']:.6f} GHz/√εr\n"
        content += f"  Figura de mérito (-3dB): {result['figura_merito_3db']:.6f}\n"
        content += f"  Figura de mérito (1/√2): {result['figura_merito_linear']:.6f}\n"
    
    content += f"\nArquivo gerado automaticamente em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return content

if __name__ == "__main__":
    main()