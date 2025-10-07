import streamlit as st
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from pathlib import Path

def detect_auto_ressonances(df, min_depth_db=-3.0, min_prominence=1.0, min_distance_ghz=0.01):
    """
    Detecta automaticamente ressonâncias baseado em mínimos locais com critérios.
    
    Args:
        df: DataFrame com colunas 'freq_ghz', 's21_db', 's21_linear'
        min_depth_db: Profundidade mínima em dB (default: -3dB)
        min_prominence: Prominência mínima do pico em dB
        min_distance_ghz: Distância mínima entre ressonâncias em GHz
    
    Returns:
        Lista de dicionários com ressonâncias detectadas
    """
    try:
        # Extrair arrays
        freq = df['freq_ghz'].values
        s21_db = df['s21_db'].values
        s21_linear = df['s21_linear'].values
        
        # Criar interpolação de alta precisão para refinamento
        interp_func_db = interpolate.interp1d(freq, s21_db, kind='cubic', fill_value='extrapolate')
        interp_func_linear = interpolate.interp1d(freq, s21_linear, kind='cubic', fill_value='extrapolate')
        
        # Encontrar mínimos (inverter sinal pois find_peaks busca máximos)
        inverted_s21 = -s21_db
        
        # Calcular distância em pontos
        freq_range = freq[-1] - freq[0]
        min_distance_points = int(min_distance_ghz * len(freq) / freq_range)
        min_distance_points = max(5, min_distance_points)  # Mínimo de 5 pontos
        
        # Encontrar picos (mínimos no S21 original)
        peaks, properties = find_peaks(
            inverted_s21,
            prominence=min_prominence,
            distance=min_distance_points,
            height=-min_depth_db  # Altura mínima (invertida)
        )
        
        # Filtrar ressonâncias válidas
        valid_ressonances = []
        for i, peak_idx in enumerate(peaks):
            freq_ress = freq[peak_idx]
            s21_val_db = s21_db[peak_idx]
            s21_val_linear = s21_linear[peak_idx]
            prominence_val = properties['prominences'][i]
            
            # Critérios de validação
            if (s21_val_db < min_depth_db and  # Deve ser mais profundo que -3dB
                prominence_val >= min_prominence):  # Prominência suficiente
                
                # ESTRATÉGIA HÍBRIDA: usar dados originais E interpolação
                refined_freq, refined_s21_db, use_original = refine_minimum_location_hybrid(
                    freq, s21_db, peak_idx, interp_func_db
                )
                
                # Calcular S21 linear usando a frequência refinada
                if use_original:
                    # Usar valor original se refinamento não foi aplicado
                    refined_s21_linear = s21_linear[peak_idx]
                else:
                    # Usar interpolação para S21 linear
                    refined_s21_linear = float(interp_func_linear(refined_freq))
                
                valid_ressonances.append({
                    'frequencia': refined_freq,
                    's21_db': refined_s21_db,
                    's21_linear': refined_s21_linear,
                    'prominence': prominence_val,
                    'index': peak_idx,
                    'use_original': use_original
                })
        
        # Ordenar por frequência
        return sorted(valid_ressonances, key=lambda x: x['frequencia'])
        
    except Exception as e:
        st.warning(f"⚠️ Erro na detecção automática: {e}")
        return []

def refine_minimum_location_hybrid(freq, s21_db, peak_idx, interp_func_db, window_size=15):
    """
    Refina a localização do mínimo usando estratégia híbrida.
    Retorna: (frequência, s21_db, use_original)
    """
    n = len(freq)
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(n, peak_idx + window_size + 1)
    
    # Pegar região ao redor do pico
    freq_region = freq[start_idx:end_idx]
    s21_region = s21_db[start_idx:end_idx]
    
    # Encontrar índice do mínimo absoluto na região (dados originais)
    min_idx_original = np.argmin(s21_region)
    freq_min_original = freq_region[min_idx_original]
    s21_min_original = s21_region[min_idx_original]
    
    # ESTRATÉGIA 1: Ajuste quadrático nos dados originais
    if len(freq_region) >= 5:  # Mais pontos para melhor ajuste
        try:
            # Ajuste quadrático para encontrar mínimo exato
            coeffs = np.polyfit(freq_region, s21_region, 2)
            a, b, c = coeffs
            
            # Mínimo da parábola: x = -b/(2a)
            if a > 0.001:  # Concavidade para cima significativa
                exact_freq = -b / (2 * a)
                exact_s21 = a * exact_freq**2 + b * exact_freq + c
                
                # Verificar se está dentro da região e é melhor que o original
                if (freq_region[0] <= exact_freq <= freq_region[-1] and
                    exact_s21 <= s21_min_original + 0.1):  # Não pode ser pior
                    
                    # VALIDAÇÃO: Verificar com interpolação cúbica
                    interp_s21 = float(interp_func_db(exact_freq))
                    
                    # Se a interpolação confirma, usar valor refinado
                    if abs(exact_s21 - interp_s21) < 0.5:  # Diferença pequena
                        return exact_freq, interp_s21, False
        
        except:
            pass
    
    # ESTRATÉGIA 2: Busca por mínimos na interpolação de alta resolução
    try:
        # Criar grid fino na região
        freq_fine = np.linspace(freq_region[0], freq_region[-1], 1000)
        s21_fine = interp_func_db(freq_fine)
        
        # Encontrar mínimo no grid fino
        min_idx_fine = np.argmin(s21_fine)
        freq_min_fine = freq_fine[min_idx_fine]
        s21_min_fine = s21_fine[min_idx_fine]
        
        # Só usar se for significativamente melhor que o original
        if s21_min_fine < s21_min_original - 0.01:  # Pelo menos 0.01dB melhor
            return freq_min_fine, s21_min_fine, False
    
    except:
        pass
    
    # ESTRATÉGIA 3: Fallback para o melhor ponto original
    return freq_min_original, s21_min_original, True

def manual_ressonance_identification(df, filename, param_id, params, param_cols, perm_col, 
                                   unique_combinations, temp_path, all_results_combined=None):
    """Permite ao usuário identificar manualmente as ressonâncias com cálculo automático."""
    
    st.markdown("### 🔬 Identificação Manual de Ressonâncias")
    
    # Chave única para resultados desta curva
    results_key = f"results_{param_id}"
    if results_key not in st.session_state:
        st.session_state[results_key] = []
    
    # CORREÇÃO: Coletar todos os resultados disponíveis
    if all_results_combined is None:
        all_results_combined = []
        for key in st.session_state:
            if key.startswith("results_"):
                all_results_combined.extend(st.session_state[key])
    
    
    # Interpolar dados para cálculos precisos
    interp_func_db = interpolate.interp1d(df['freq_ghz'], df['s21_db'], kind='cubic', fill_value='extrapolate')
    interp_func_linear = interpolate.interp1d(df['freq_ghz'], df['s21_linear'], kind='cubic', fill_value='extrapolate')
    
    freq_min, freq_max = df['freq_ghz'].min(), df['freq_ghz'].max()
    freq_interp = np.linspace(freq_min, freq_max, 10000)
    s21_db_interp = interp_func_db(freq_interp)
    s21_linear_interp = interp_func_linear(freq_interp)
    
    current_results = st.session_state[results_key].copy()
    
    # SEÇÃO DE DETECÇÃO AUTOMÁTICA
    st.markdown("#### 🤖 Detecção Automática")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_depth = st.number_input(
            "Profundidade mínima (dB)",
            min_value=-50.0,
            max_value=0.0,
            value=-3.0,
            step=0.5,
            key=f"min_depth_{param_id}"
        )
    with col2:
        min_prominence = st.number_input(
            "Prominência mínima (dB)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=f"min_prom_{param_id}"
        )
    with col3:
        min_distance = st.number_input(
            "Distância mínima (GHz)",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            key=f"min_dist_{param_id}"
        )
    
    # Adicionar informações sobre a estratégia de detecção
    with st.expander("🔧 Configurações Avançadas de Detecção"):
        st.info("""
        **Estratégia Híbrida de Detecção:**
        - **Fase 1**: Encontra mínimos nos dados originais
        - **Fase 2**: Refina com ajuste quadrático  
        - **Fase 3**: Valida com interpolação cúbica
        - **Fase 4**: Seleciona o melhor resultado
        """)
        
        auto_method = st.selectbox(
            "Método de refinamento",
            ["Automático (Recomendado)", "Apenas dados originais", "Interpolação máxima"],
            key=f"method_{param_id}"
        )
    
    # CORREÇÃO: Usar uma chave única para controlar o estado da detecção
    detection_key = f"detection_done_{param_id}"
    
    if st.button("🔍 Executar Detecção Automática", key=f"auto_detect_{param_id}"):
        # Marcar que a detecção foi iniciada
        st.session_state[detection_key] = True
        
        with st.spinner("Procurando ressonâncias..."):
            auto_ressonances = detect_auto_ressonances(
                df, min_depth, min_prominence, min_distance
            )
            
            # CORREÇÃO: Armazenar resultados temporariamente
            st.session_state[f"auto_results_{param_id}"] = auto_ressonances
    
    # CORREÇÃO: Mostrar resultados APÓS a detecção, sem recarregar imediatamente
    if st.session_state.get(detection_key, False):
        auto_ressonances = st.session_state.get(f"auto_results_{param_id}", [])
        
        if auto_ressonances:
            # Mostrar estatísticas da detecção
            original_count = sum(1 for r in auto_ressonances if r['use_original'])
            refined_count = len(auto_ressonances) - original_count
            
            st.success(f"✅ Encontradas {len(auto_ressonances)} ressonâncias!")
            st.info(f"📊 Estatísticas: {refined_count} refinadas, {original_count} originais")
            
            # Mostrar detalhes das ressonâncias detectadas
            with st.expander("📋 Detalhes das Ressonâncias Detectadas"):
                for i, ress in enumerate(auto_ressonances):
                    method = "Original" if ress['use_original'] else "Refinada"
                    st.write(f"**Ressonância {i+1}:** {ress['frequencia']:.6f} GHz "
                           f"(S21: {ress['s21_db']:.3f} dB, Método: {method})")
            
            # Botão para aplicar os resultados
            if st.button("🔄 Aplicar Ressonâncias Detectadas", key=f"apply_{param_id}"):
                # Atualizar número de ressonâncias
                st.session_state[f"num_{param_id}"] = len(auto_ressonances)
                
                # Atualizar resultados com as ressonâncias detectadas
                current_results = []
                for i, ress in enumerate(auto_ressonances):
                    result = {
                        'parametros': param_id,
                        'ressonancia_num': i + 1,
                        'frequencia_ressonancia_ghz': ress['frequencia'],
                        's21_ressonancia_db': ress['s21_db'],
                        's21_ressonancia_linear': ress['s21_linear'],
                        'fwhm_3db_ghz': None,
                        'Q_3db': None,
                        'fwhm_linear_ghz': None,
                        'Q_linear': None,
                        'sensibilidade_ghz_sqrt_er': None,
                        'figura_merito_3db': None,
                        'figura_merito_linear': None,
                        'figura_merito_normal_3db': None,
                        'figura_merito_normal_linear': None,
                        'detection_method': 'original' if ress['use_original'] else 'refined'
                    }
                    
                    for col in param_cols:
                        if col != '_dummy':
                            result[col] = params[col]
                    
                    current_results.append(result)
                
                st.session_state[results_key] = current_results
                
                # Limpar estado temporário
                st.session_state[detection_key] = False
                if f"auto_results_{param_id}" in st.session_state:
                    del st.session_state[f"auto_results_{param_id}"]
                
                st.rerun()
            
            # Botão para descartar resultados
            if st.button("❌ Descartar Resultados", key=f"discard_{param_id}"):
                st.session_state[detection_key] = False
                if f"auto_results_{param_id}" in st.session_state:
                    del st.session_state[f"auto_results_{param_id}"]
                st.rerun()
                
        elif auto_ressonances == []:  # Lista vazia significa que não encontrou nada
            st.warning("❌ Nenhuma ressonância encontrada com os critérios atuais.")
            st.info("💡 Tente ajustar: Reduzir profundidade mínima ou prominência")
            
            # Botão para tentar novamente
            if st.button("🔄 Tentar Novamente", key=f"retry_{param_id}"):
                st.session_state[detection_key] = False
                if f"auto_results_{param_id}" in st.session_state:
                    del st.session_state[f"auto_results_{param_id}"]
                st.rerun()
    
    st.markdown("---")
    
    # Número de ressonâncias a analisar
    num_ressonances = st.number_input(
        f"Quantas ressonâncias deseja analisar em {param_id}?",
        min_value=0,
        max_value=50,
        value=len(current_results),  # CORREÇÃO: usar sempre o length atual
        key=f"num_{param_id}"
    )
    
    # Ajustar lista de resultados se necessário
    if len(current_results) > num_ressonances:
        current_results = current_results[:num_ressonances]
    elif len(current_results) < num_ressonances:
        for i in range(len(current_results), num_ressonances):
            # CORREÇÃO: Usar valor mais razoável para nova ressonância
            default_freq = freq_min + (i + 1) * (freq_max - freq_min) / (num_ressonances + 1)
            current_results.append({
                'parametros': param_id,
                'ressonancia_num': i + 1,
                'frequencia_ressonancia_ghz': default_freq,
                's21_ressonancia_db': None,
                's21_ressonancia_linear': None,
                'fwhm_3db_ghz': None,
                'Q_3db': None,
                'fwhm_linear_ghz': None,
                'Q_linear': None,
                'sensibilidade_ghz_sqrt_er': None,
                'figura_merito_3db': None,
                'figura_merito_linear': None,
                'figura_merito_normal_3db': None,
                'figura_merito_normal_linear': None
            })
    
    # Processar cada ressonância com cálculo automático
    for i in range(num_ressonances):
        st.markdown(f"##### Ressonância {i+1}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CORREÇÃO: Garantir que o valor mostrado seja atualizado
            current_freq = current_results[i]['frequencia_ressonancia_ghz']
            
            # Campo de frequência - cálculo automático ao alterar
            freq_ressonancia = st.number_input(
                f"Frequência de ressonância (GHz)",
                min_value=float(freq_min),
                max_value=float(freq_max),
                value=float(current_freq),
                step=0.001,  # Passo menor para mais precisão
                format="%.6f",  # Mais casas decimais
                key=f"freq_{param_id}_{i}"
            )
            
            # Calcular S21 atualizado para a frequência selecionada
            s21_ressonancia_db = float(interp_func_db(freq_ressonancia))
            s21_ressonancia_linear = float(interp_func_linear(freq_ressonancia))
            
            st.write(f"**Frequência:** {freq_ressonancia:.6f} GHz")
            st.write(f"**S21:** {s21_ressonancia_db:.4f} dB")
            st.write(f"**S21 (linear):** {s21_ressonancia_linear:.6f}")
        
        with col2:
            # Cálculo automático dos parâmetros usando a lógica correta
            bandwidth_result_db = find_bandwidth_points_corrected(
                freq_interp, s21_db_interp, freq_ressonancia, interp_func_db, is_db=True
            )
            
            bandwidth_result_linear = find_bandwidth_points_corrected(
                freq_interp, s21_linear_interp, freq_ressonancia, interp_func_linear, is_db=False
            )
            
            # Dentro do loop de processamento de cada ressonância:
            if bandwidth_result_db and bandwidth_result_linear:
                freq_left_db, freq_right_db, fwhm_db, Q_db = bandwidth_result_db
                freq_left_linear, freq_right_linear, fwhm_linear, Q_linear = bandwidth_result_linear
                
                # CORREÇÃO: Tentar calcular sensibilidade, mas não bloquear se não conseguir
                sensitivity_data = None
                try:
                    sensitivity_data = calculate_sensitivity_corrected(
                        freq_ressonancia, params, perm_col, unique_combinations, Q_db, Q_linear, s21_ressonancia_linear,
                        param_id, all_results_combined
                    )
                except Exception as e:
                    # Não mostrar erro, apenas não calcular sensibilidade nesta fase
                    pass
                
                if sensitivity_data and all(x is not None for x in sensitivity_data):
                    sensitivity, figure_of_merit_db, figure_of_merit_linear, figure_of_merit_normal_db, figure_of_merit_normal_linear = sensitivity_data
                else:
                    # Se não conseguiu calcular, deixar como None (será calculado depois)
                    sensitivity, figure_of_merit_db, figure_of_merit_linear, figure_of_merit_normal_db, figure_of_merit_normal_linear = None, None, None, None, None
                
                # Atualizar resultado
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
                    'sensibilidade_mhz_sqrt_er': sensitivity,
                    'figura_merito_3db': figure_of_merit_db,
                    'figura_merito_linear': figure_of_merit_linear,
                    'figura_merito_normal_3db': figure_of_merit_normal_db,
                    'figura_merito_normal_linear': figure_of_merit_normal_linear
                }
                
                for col in param_cols:
                    if col != '_dummy':
                        result[col] = params[col]
                
                current_results[i] = result
                
                # Mostrar resultados calculados automaticamente
                st.markdown("**📊 Parâmetros Calculados:**")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"""
                    **Método -3dB:**
                    - Largura de banda: {fwhm_db:.6f} GHz
                    - Fator Q: {Q_db:.2f}
                    - Frequências: {freq_left_db:.6f} - {freq_right_db:.6f} GHz
                    """)
                
                with col_res2:
                    st.markdown(f"""
                    **Método 1/√2:**
                    - Largura de banda: {fwhm_linear:.6f} GHz  
                    - Fator Q: {Q_linear:.2f}
                    - Frequências: {freq_left_linear:.6f} - {freq_right_linear:.6f} GHz
                    """)
                
                if sensitivity is not None:
                    st.markdown(f"""
                    **Sensibilidade:**
                    - Sensibilidade: {sensitivity:.6f} MHz/√εr
                    """)
                    
                    st.markdown(f"""
                    **Figura de Mérito (Normal):**
                    - Método -3dB: {figure_of_merit_normal_db:.6f} (Q × Sensibilidade)
                    - Método 1/√2: {figure_of_merit_normal_linear:.6f} (Q × Sensibilidade)
                    """)
                    
                    st.markdown(f"""
                    **Figura de Mérito (Com Amplitude):**
                    - Método -3dB: {figure_of_merit_db:.6f} (Q × Sensibilidade × (1-Amplitude))
                    - Método 1/√2: {figure_of_merit_linear:.6f} (Q × Sensibilidade × (1-Amplitude))
                    """)
                
                # Salvar resultado automaticamente em TXT
                txt_content = generate_txt_result_corrected(result, param_cols, params)
                txt_filename = f"resultado_{Path(filename).stem}_{param_id}_ressonancia_{i+1}.txt"
                txt_path = temp_path / txt_filename
                with open(txt_path, 'w') as f:
                    f.write(txt_content)
                    
            else:
                st.warning("⚠️ Não foi possível calcular os parâmetros para esta frequência.")
        
        st.markdown("---")
    
    # Atualizar session_state com resultados
    st.session_state[results_key] = current_results
    
    return current_results

def find_bandwidth_points_corrected(freq, y_values, center_freq, interp_func, is_db=True):
    """Encontra os pontos de largura de banda usando a lógica correta"""
    
    def find_crossing_points(freq_array, y_array, target_val):
        """Encontra os pontos onde a curva cruza o valor target"""
        crossings = []
        for i in range(len(freq_array) - 1):
            if (y_array[i] - target_val) * (y_array[i+1] - target_val) < 0:
                # Interpolação linear para encontrar o ponto exato
                x1, x2 = freq_array[i], freq_array[i+1]
                y1, y2 = y_array[i], y_array[i+1]
                if y2 - y1 != 0:
                    x_cross = x1 + (target_val - y1) * (x2 - x1) / (y2 - y1)
                    crossings.append(x_cross)
        return crossings
    
    # ANÁLISE 1: Pontos de -3dB (em dB)
    if is_db:
        target_db = -3.0  # VALOR ABSOLUTO -3dB
        
        # Encontrar todos os cruzamentos com -3dB
        crossings_db = find_crossing_points(freq, y_values, target_db)
        
        if len(crossings_db) < 2:
            st.warning(f"❌ Apenas {len(crossings_db)} ponto(s) de cruzamento com -3dB encontrado(s). Necessário 2 pontos.")
            return None
        
        # Encontrar os dois cruzamentos mais próximos da frequência de ressonância
        crossings_db_sorted = sorted(crossings_db, key=lambda x: abs(x - center_freq))
        freq_left_db = min(crossings_db_sorted[0], crossings_db_sorted[1])
        freq_right_db = max(crossings_db_sorted[0], crossings_db_sorted[1])
        
        # Calcular largura de banda (FWHM) e fator de qualidade
        fwhm_db = abs(freq_right_db - freq_left_db)
        Q_db = center_freq / fwhm_db if fwhm_db > 0 else float('inf')
        
        return freq_left_db, freq_right_db, fwhm_db, Q_db
    
    # ANÁLISE 2: Pontos onde amplitude linear = 1/√2
    else:
        target_linear = 1 / np.sqrt(2)  # ≈ 0.7071 (VALOR ABSOLUTO)
        
        # Encontrar todos os cruzamentos com 1/√2
        crossings_linear = find_crossing_points(freq, y_values, target_linear)
        
        if len(crossings_linear) < 2:
            st.warning(f"❌ Apenas {len(crossings_linear)} ponto(s) de cruzamento com 1/√2 encontrado(s). Necessário 2 pontos.")
            return None
        
        # Encontrar os dois cruzamentos mais próximos da frequência de ressonância
        crossings_linear_sorted = sorted(crossings_linear, key=lambda x: abs(x - center_freq))
        freq_left_linear = min(crossings_linear_sorted[0], crossings_linear_sorted[1])
        freq_right_linear = max(crossings_linear_sorted[0], crossings_linear_sorted[1])
        
        # Calcular largura de banda e fator de qualidade
        fwhm_linear = abs(freq_right_linear - freq_left_linear)
        Q_linear = center_freq / fwhm_linear if fwhm_linear > 0 else float('inf')
        
        return freq_left_linear, freq_right_linear, fwhm_linear, Q_linear

def calculate_sensitivity_corrected(freq_ressonancia, params, perm_col, unique_combinations, 
                                  Q_db, Q_linear, amplitude_linear, current_param_id, all_results):
    """Calcula sensibilidade e ambas as figuras de mérito - VERSÃO CORRIGIDA"""
    
    # Verificar se temos todos os valores necessários
    if (Q_db is None or Q_linear is None or amplitude_linear is None or 
        not perm_col or perm_col not in params):
        return None, None, None, None, None
    
    try:
        current_perm = params[perm_col]
        unique_perms = sorted(unique_combinations[perm_col].unique())
        
        if len(unique_perms) < 2:
            return None, None, None, None, None
        
        current_index = unique_perms.index(current_perm)
        
        # CORREÇÃO: Precisamos encontrar a próxima permissividade e suas ressonâncias correspondentes
        if current_index < len(unique_perms) - 1:
            next_perm = unique_perms[current_index + 1]
            
            # CORREÇÃO: Encontrar todas as ressonâncias para a permissividade atual e próxima
            current_ressonances = get_ressonances_for_permittivity(all_results, params, perm_col, current_perm)
            next_ressonances = get_ressonances_for_permittivity(all_results, params, perm_col, next_perm)
            
            if not current_ressonances or not next_ressonances:
                return None, None, None, None, None
            
            # CORREÇÃO: Encontrar o par correspondente para esta ressonância específica
            paired_freq_next = find_corresponding_ressonance(
                freq_ressonancia, current_ressonances, next_ressonances
            )
            
            if paired_freq_next is None:
                return None, None, None, None, None
            
            # CORREÇÃO: Calcular diferença de frequência (deve ser positiva pois freq_current > freq_next)
            freq_diff = 1000.0*abs(freq_ressonancia - paired_freq_next)
            
            # Calcular sqrt_diff das permissividades
            sqrt_diff = abs(np.sqrt(next_perm) - np.sqrt(current_perm))
            
            if sqrt_diff > 0 and freq_diff > 0:
                sensitivity = freq_diff / sqrt_diff
            else:
                sensitivity = 0
            
            # Figura de mérito normal (Q × Sensibilidade)
            figure_of_merit_normal_db = Q_db * sensitivity
            figure_of_merit_normal_linear = Q_linear * sensitivity
            
            # Figura de mérito com amplitude (Q × Sensibilidade × (1-Amplitude))
            figure_of_merit_db = Q_db * sensitivity * (1 - amplitude_linear)
            figure_of_merit_linear = Q_linear * sensitivity * (1 - amplitude_linear)
            
            return sensitivity, figure_of_merit_db, figure_of_merit_linear, figure_of_merit_normal_db, figure_of_merit_normal_linear
        
        return None, None, None, None, None
    except (ValueError, IndexError, TypeError) as e:
        st.warning(f"⚠️ Não foi possível calcular sensibilidade: {e}")
        return None, None, None, None, None

def get_ressonances_for_permittivity(all_results, current_params, perm_col, target_perm):
    """
    Encontra todas as ressonâncias para uma determinada permissividade,
    mantendo os mesmos valores dos outros parâmetros.
    """
    try:
        target_ressonances = []
        
        for result in all_results:
            # Verificar se é a mesma permissividade
            if result.get(perm_col) != target_perm:
                continue
            
            # Verificar se os outros parâmetros são iguais (exceto perm_col)
            params_match = True
            for key, value in current_params.items():
                if key != perm_col and key != '_dummy' and key in result:
                    if result[key] != value:
                        params_match = False
                        break
            
            if params_match:
                target_ressonances.append({
                    'freq': result['frequencia_ressonancia_ghz'],
                    's21_db': result['s21_ressonancia_db'],
                    'ressonancia_num': result['ressonancia_num']
                })
        
        # Ordenar por frequência (mais baixa para mais alta)
        return sorted(target_ressonances, key=lambda x: x['freq'])
    
    except Exception as e:
        st.warning(f"⚠️ Erro ao buscar ressonâncias para permissividade {target_perm}: {e}")
        return []

def find_corresponding_ressonance(current_freq, current_ressonances, next_ressonances):
    """
    Encontra a ressonância correspondente na próxima permissividade.
    Usa matching por ordem e validação por proximidade de frequência.
    """
    try:
        # Ordenar ambas as listas por frequência
        current_sorted = sorted(current_ressonances, key=lambda x: x['freq'])
        next_sorted = sorted(next_ressonances, key=lambda x: x['freq'])
        
        # Encontrar o índice da ressonância atual na lista ordenada
        current_index = None
        for i, ress in enumerate(current_sorted):
            if abs(ress['freq'] - current_freq) < 0.001:  # Tolerância de 1MHz
                current_index = i
                break
        
        if current_index is None or current_index >= len(next_sorted):
            return None
        
        # CORREÇÃO: Validar se a correspondência faz sentido fisicamente
        # A frequência deve diminuir com o aumento da permissividade
        corresponding_freq = next_sorted[current_index]['freq']
        
        if corresponding_freq < current_freq:  # Fisicamente correto
            return corresponding_freq
        else:
            # Tentar encontrar a melhor correspondência por proximidade
            best_match = None
            min_diff = float('inf')
            
            for next_ress in next_sorted:
                freq_diff = abs(next_ress['freq'] - current_freq)
                # Preferir matches onde next_freq < current_freq (fisicamente correto)
                if next_ress['freq'] < current_freq and freq_diff < min_diff:
                    min_diff = freq_diff
                    best_match = next_ress['freq']
            
            if best_match is not None and min_diff < 0.1:  # Tolerância de 100MHz
                return best_match
            
            return None
    
    except Exception as e:
        st.warning(f"⚠️ Erro ao encontrar ressonância correspondente: {e}")
        return None

def generate_txt_result_corrected(result, param_cols, params):
    """Gera conteúdo TXT com resultados incluindo ambas as figuras de mérito"""
    
    content = "RESULTADOS DA ANÁLISE S21 - IDENTIFICAÇÃO MANUAL\n"
    content += "=" * 60 + "\n\n"
    
    content += f"Ressonância: {result['ressonancia_num']}\n"
    content += f"Frequência de ressonância: {result['frequencia_ressonancia_ghz']:.6f} GHz\n"
    content += f"S21 na ressonância: {result['s21_ressonancia_db']:.6f} dB\n"
    content += f"S21 na ressonância (linear): {result['s21_ressonancia_linear']:.6f}\n\n"
    content += f"Profundidade de S21 na ressonância (linear): {(1-result['s21_ressonancia_linear']):.6f}\n\n"
    
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
        content += f"  Sensibilidade: {result['sensibilidade_mhz_sqrt_er']:.6f} MHz/√εr\n\n"
        
        content += "FIGURAS DE MÉRITO:\n"
        content += f"  Figura de mérito normal (-3dB): {result['figura_merito_normal_3db']:.6f} (Q × Sensibilidade)\n"
        content += f"  Figura de mérito normal (1/√2): {result['figura_merito_normal_linear']:.6f} (Q × Sensibilidade)\n"
        content += f"  Figura de mérito com amplitude (-3dB): {result['figura_merito_3db']:.6f} (Q × Sensibilidade × Amplitude)\n"
        content += f"  Figura de mérito com amplitude (1/√2): {result['figura_merito_linear']:.6f} (Q × Sensibilidade × Amplitude)\n"
    
    content += f"\nArquivo gerado automaticamente em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return content

# Funções antigas mantidas para compatibilidade
def find_bandwidth_points(freq, y_values, center_freq, center_value, target, is_linear=False):
    """Função antiga mantida para compatibilidade - NÃO USAR"""
    return find_bandwidth_points_corrected(freq, y_values, center_freq, None, is_linear)

def calculate_sensitivity(freq_ressonancia, params, perm_col, unique_combinations, 
                         Q_db, Q_linear, amplitude_linear):
    """Função antiga mantida para compatibilidade"""
    result = calculate_sensitivity_corrected(freq_ressonancia, params, perm_col, unique_combinations, Q_db, Q_linear, amplitude_linear)
    if result:
        return result[0], result[1], result[2]  # Retorna apenas os 3 primeiros para compatibilidade
    return None, None, None

def generate_txt_result(result, param_cols, params):
    """Função antiga mantida para compatibilidade"""
    return generate_txt_result_corrected(result, param_cols, params)