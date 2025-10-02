import streamlit as st
import pandas as pd
import numpy as np
from scipy import interpolate
from pathlib import Path

def manual_ressonance_identification(df, filename, param_id, params, param_cols, perm_col, 
                                   unique_combinations, temp_path):
    """Permite ao usuário identificar manualmente as ressonâncias com cálculo automático."""
    
    st.markdown("### 🔬 Identificação Manual de Ressonâncias")
    
    # Chave única para resultados desta curva
    results_key = f"results_{param_id}"
    if results_key not in st.session_state:
        st.session_state[results_key] = []
    
    # Interpolar dados para cálculos precisos
    interp_func_db = interpolate.interp1d(df['freq_ghz'], df['s21_db'], kind='cubic', fill_value='extrapolate')
    interp_func_linear = interpolate.interp1d(df['freq_ghz'], df['s21_linear'], kind='cubic', fill_value='extrapolate')
    
    freq_min, freq_max = df['freq_ghz'].min(), df['freq_ghz'].max()
    freq_interp = np.linspace(freq_min, freq_max, 10000)
    s21_db_interp = interp_func_db(freq_interp)
    s21_linear_interp = interp_func_linear(freq_interp)
    
    current_results = st.session_state[results_key].copy()
    
    # Número de ressonâncias a analisar
    num_ressonances = st.number_input(
        f"Quantas ressonâncias deseja analisar em {param_id}?",
        min_value=0,
        max_value=50,
        value=max(1, len(current_results)),
        key=f"num_{param_id}"
    )
    
    # Ajustar lista de resultados se necessário
    if len(current_results) > num_ressonances:
        current_results = current_results[:num_ressonances]
    elif len(current_results) < num_ressonances:
        for i in range(len(current_results), num_ressonances):
            current_results.append({
                'parametros': param_id,
                'ressonancia_num': i + 1,
                'frequencia_ressonancia_ghz': (freq_min + freq_max) / 2,
                's21_ressonancia_db': None,
                's21_ressonancia_linear': None,
                'fwhm_3db_ghz': None,
                'Q_3db': None,
                'fwhm_linear_ghz': None,
                'Q_linear': None,
                'sensibilidade_ghz_sqrt_er': None,
                'figura_merito_3db': None,
                'figura_merito_linear': None,
                'figura_merito_normal_3db': None,  # Nova coluna
                'figura_merito_normal_linear': None  # Nova coluna
            })
    
    # Processar cada ressonância com cálculo automático
    for i in range(num_ressonances):
        st.markdown(f"##### Ressonância {i+1}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Campo de frequência - cálculo automático ao alterar
            freq_ressonancia = st.number_input(
                f"Frequência de ressonância (GHz)",
                min_value=float(freq_min),
                max_value=float(freq_max),
                value=float(current_results[i]['frequencia_ressonancia_ghz']),
                step=0.01,
                format="%.4f",
                key=f"freq_{param_id}_{i}"
            )
            
            # Calcular S21 atualizado para a frequência selecionada
            s21_ressonancia_db = float(interp_func_db(freq_ressonancia))
            s21_ressonancia_linear = float(interp_func_linear(freq_ressonancia))
            
            st.write(f"**Frequência:** {freq_ressonancia:.4f} GHz")
            st.write(f"**S21:** {s21_ressonancia_db:.4f} dB")
            st.write(f"**S21 (linear):** {s21_ressonancia_linear:.4f}")
        
        with col2:
            # Cálculo automático dos parâmetros usando a lógica correta
            bandwidth_result_db = find_bandwidth_points_corrected(
                freq_interp, s21_db_interp, freq_ressonancia, interp_func_db, is_db=True
            )
            
            bandwidth_result_linear = find_bandwidth_points_corrected(
                freq_interp, s21_linear_interp, freq_ressonancia, interp_func_linear, is_db=False
            )
            
            if bandwidth_result_db and bandwidth_result_linear:
                freq_left_db, freq_right_db, fwhm_db, Q_db = bandwidth_result_db
                freq_left_linear, freq_right_linear, fwhm_linear, Q_linear = bandwidth_result_linear
                
                # CORREÇÃO: Calcular ambas as figuras de mérito
                sensitivity, figure_of_merit_db, figure_of_merit_linear, figure_of_merit_normal_db, figure_of_merit_normal_linear = calculate_sensitivity_corrected(
                    freq_ressonancia, params, perm_col, unique_combinations, Q_db, Q_linear, s21_ressonancia_linear
                )
                
                # Atualizar resultado automaticamente
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
                    'figura_merito_linear': figure_of_merit_linear,
                    'figura_merito_normal_3db': figure_of_merit_normal_db,  # Nova coluna
                    'figura_merito_normal_linear': figure_of_merit_normal_linear  # Nova coluna
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
                    - Frequências: {freq_left_db:.4f} - {freq_right_db:.4f} GHz
                    """)
                
                with col_res2:
                    st.markdown(f"""
                    **Método 1/√2:**
                    - Largura de banda: {fwhm_linear:.6f} GHz  
                    - Fator Q: {Q_linear:.2f}
                    - Frequências: {freq_left_linear:.4f} - {freq_right_linear:.4f} GHz
                    """)
                
                if sensitivity is not None:
                    st.markdown(f"""
                    **Sensibilidade:**
                    - Sensibilidade: {sensitivity:.6f} GHz/√εr
                    """)
                    
                    st.markdown(f"""
                    **Figura de Mérito (Normal):**
                    - Método -3dB: {figure_of_merit_normal_db:.6f} (Q × Sensibilidade)
                    - Método 1/√2: {figure_of_merit_normal_linear:.6f} (Q × Sensibilidade)
                    """)
                    
                    st.markdown(f"""
                    **Figura de Mérito (Com Amplitude):**
                    - Método -3dB: {figure_of_merit_db:.6f} (Q × Sensibilidade × Amplitude)
                    - Método 1/√2: {figure_of_merit_linear:.6f} (Q × Sensibilidade × Amplitude)
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
                                  Q_db, Q_linear, amplitude_linear):
    """Calcula sensibilidade e ambas as figuras de mérito"""
    
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
        
        if current_index < len(unique_perms) - 1:
            next_perm = unique_perms[current_index + 1]
            sqrt_diff = abs(np.sqrt(next_perm) - np.sqrt(current_perm))
            
            if sqrt_diff > 0:
                sensitivity = freq_ressonancia / sqrt_diff
            else:
                sensitivity = 0
            
            # Figura de mérito normal (Q × Sensibilidade)
            figure_of_merit_normal_db = Q_db * sensitivity
            figure_of_merit_normal_linear = Q_linear * sensitivity
            
            # Figura de mérito com amplitude (Q × Sensibilidade × Amplitude)
            figure_of_merit_db = Q_db * sensitivity * amplitude_linear
            figure_of_merit_linear = Q_linear * sensitivity * amplitude_linear
            
            return sensitivity, figure_of_merit_db, figure_of_merit_linear, figure_of_merit_normal_db, figure_of_merit_normal_linear
        
        return None, None, None, None, None
    except (ValueError, IndexError, TypeError) as e:
        st.warning(f"⚠️ Não foi possível calcular sensibilidade: {e}")
        return None, None, None, None, None

def generate_txt_result_corrected(result, param_cols, params):
    """Gera conteúdo TXT com resultados incluindo ambas as figuras de mérito"""
    
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
        content += f"  Sensibilidade: {result['sensibilidade_ghz_sqrt_er']:.6f} GHz/√εr\n\n"
        
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