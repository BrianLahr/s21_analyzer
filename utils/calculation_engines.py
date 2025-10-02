import streamlit as st
import pandas as pd
import numpy as np
from scipy import interpolate
from pathlib import Path

import streamlit as st
import numpy as np
from scipy import interpolate
from pathlib import Path

def manual_ressonance_identification(df, filename, param_id, params, param_cols, perm_col, 
                                   unique_combinations, temp_path):
    """Permite ao usuário identificar manualmente as ressonâncias com preservação de resultados."""
    
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
        value=max(1, len(current_results)),  # mantém quantidade atual
        key=f"num_{param_id}"
    )
    
    # Ajustar lista de resultados se necessário
    if len(current_results) > num_ressonances:
        current_results = current_results[:num_ressonances]
    elif len(current_results) < num_ressonances:
        # Preenche com placeholders
        for _ in range(num_ressonances - len(current_results)):
            current_results.append({
                'ressonancia_num': len(current_results)+1,
                'frequencia_ressonancia_ghz': (freq_min+freq_max)/2,
                's21_ressonancia_db': None,
                's21_ressonancia_linear': None,
                'fwhm_3db_ghz': None,
                'Q_3db': None,
                'fwhm_linear_ghz': None,
                'Q_linear': None,
                'sensibilidade_ghz_sqrt_er': None,
                'figura_merito_3db': None,
                'figura_merito_linear': None
            })
    
    st.session_state[results_key] = current_results
    calculated_this_run = [False]*num_ressonances
    
    for i in range(num_ressonances):
        st.markdown(f'<div class="ressonance-input">', unsafe_allow_html=True)
        st.markdown(f"##### Ressonância {i+1}")
        
        col1, col2 = st.columns(2)
        with col1:
            default_freq = current_results[i]['frequencia_ressonancia_ghz'] or (freq_min+freq_max)/2
            freq_ressonancia = st.number_input(
                f"Frequência de ressonância (GHz)",
                min_value=float(freq_min),
                max_value=float(freq_max),
                value=float(default_freq),
                step=0.01,
                format="%.4f",
                key=f"freq_{param_id}_{i}"
            )
        
        with col2:
            # Informações aproximadas da ressonância
            idx_ressonancia = (freq_interp - freq_ressonancia).argmin()
            s21_ressonancia_db = s21_db_interp[idx_ressonancia]
            s21_ressonancia_linear = s21_linear_interp[idx_ressonancia]
            
            st.write(f"Frequência: {freq_interp[idx_ressonancia]:.4f} GHz")
            st.write(f"S21: {s21_ressonancia_db:.4f} dB")
            st.write(f"S21 (linear): {s21_ressonancia_linear:.4f}")
        
        # Botão de cálculo
        calculate_col1, calculate_col2 = st.columns([2,1])
        with calculate_col2:
            if st.button(f"📊 Calcular - Ressonância {i+1}", key=f"calc_{param_id}_{i}"):
                calculated_this_run[i] = True
                
                # Cálculo das larguras de banda
                bandwidth_result_db = find_bandwidth_points(
                    freq_interp, s21_db_interp, freq_ressonancia, s21_ressonancia_db, s21_ressonancia_db-3.0
                )
                target_linear = s21_ressonancia_linear / np.sqrt(2)
                bandwidth_result_linear = find_bandwidth_points(
                    freq_interp, s21_linear_interp, freq_ressonancia, s21_ressonancia_linear, target_linear, is_linear=True
                )
                
                if bandwidth_result_db is None or bandwidth_result_linear is None:
                    st.error(f"❌ Não foi possível calcular largura de banda para esta ressonância.")
                    continue
                
                freq_left_db, freq_right_db, fwhm_db, Q_db = bandwidth_result_db
                freq_left_linear, freq_right_linear, fwhm_linear, Q_linear = bandwidth_result_linear
                
                sensitivity, figure_of_merit_db, figure_of_merit_linear = calculate_sensitivity(
                    freq_ressonancia, params, perm_col, unique_combinations, Q_db, Q_linear, s21_ressonancia_linear
                )
                
                result = {
                    'parametros': param_id,
                    'ressonancia_num': i+1,
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
                
                for col in param_cols:
                    if col != '_dummy':
                        result[col] = params[col]
                
                current_results[i] = result
                st.session_state[results_key] = current_results
                
                # Salvar resultado individual em TXT
                txt_content = generate_txt_result(result, param_cols, params)
                txt_filename = f"resultado_{Path(filename).stem}_{param_id}_ressonancia_{i+1}.txt"
                txt_path = temp_path / txt_filename
                with open(txt_path, 'w') as f:
                    f.write(txt_content)
                
                st.success(f"✅ Parâmetros calculados para ressonância {i+1}!")
        
        # Mostrar resultados anteriores mesmo sem recalcular
        if current_results[i]['fwhm_3db_ghz'] is not None and not calculated_this_run[i]:
            st.info("📊 **Resultados Calculados Anteriormente**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="result-card">
                <b>Largura de banda (-3dB):</b> {current_results[i]['fwhm_3db_ghz']:.6f} GHz<br>
                <b>Fator Q (-3dB):</b> {current_results[i]['Q_3db']:.2f}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="result-card">
                <b>Largura de banda (1/√2):</b> {current_results[i]['fwhm_linear_ghz']:.6f} GHz<br>
                <b>Fator Q (1/√2):</b> {current_results[i]['Q_linear']:.2f}
                </div>
                """, unsafe_allow_html=True)
            if current_results[i]['sensibilidade_ghz_sqrt_er'] is not None:
                st.markdown(f"""
                <div class="result-card">
                <b>Sensibilidade:</b> {current_results[i]['sensibilidade_ghz_sqrt_er']:.6f} GHz/√εr
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return current_results


# [As funções find_bandwidth_points, calculate_sensitivity, e generate_txt_result permanecem IGUAIS]

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