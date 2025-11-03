import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

def identify_columns(df: pd.DataFrame, s_param_type: str = "S21"):
    """Identifica automaticamente colunas de frequência, S11/S21, parâmetros e permissividade."""
    freq_col = None
    s_col = None  # ATUALIZADO: nome mais genérico
    param_cols = []
    perm_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['freq', 'frequency', 'ghz', 'mhz']):
            freq_col = col
        elif s_param_type == "S21" and any(x in col_lower for x in ['s21', 's(2,1)', 'db(s(2,1))', 'insertion']):
            s_col = col
        elif s_param_type == "S11" and any(x in col_lower for x in ['s11', 's(1,1)', 'db(s(1,1))', 'reflection', 'return']):
            s_col = col
        elif any(x in col_lower for x in ['perm', 'permittivity', 'epsilon', 'dielectric']):
            perm_col = col
            param_cols.append(col)
        elif col not in [freq_col, s_col]:
            param_cols.append(col)

    # Se não encontrou S11/S21, usar a primeira coluna numérica diferente de freq
    if s_col is None and freq_col is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        other_numeric = [col for col in numeric_cols if col != freq_col]
        if other_numeric:
            s_col = other_numeric[0]
            st.warning(f"⚠️ Coluna {s_param_type} não identificada automaticamente. Usando '{s_col}'")

    return freq_col, s_col, param_cols, perm_col

def process_data(df: pd.DataFrame, filename: str, freq_col: str, s_col: str, param_cols: list, perm_col: str, s_param_type: str = "S21"):
    """Processa dados e mostra gráfico + campos de análise juntos para cada combinação"""
    
    # ATUALIZADO: Nome do diretório temporário adaptado
    temp_path = Path(tempfile.mkdtemp(prefix=f"{s_param_type.lower()}_analysis_"))
    
    df_processed = df.copy()
    df_processed = df_processed.rename(columns={freq_col: 'freq_ghz', s_col: f'{s_param_type.lower()}_db'})  # ATUALIZADO
    
    # ATUALIZADO: Converter para linear baseado no tipo S
    if s_param_type == "S21":
        df_processed[f'{s_param_type.lower()}_linear'] = 10 ** (df_processed[f'{s_param_type.lower()}_db'] / 20)
    elif s_param_type == "S11":
        # Para S11, o valor em linear é diferente (magnitude da reflexão)
        df_processed[f'{s_param_type.lower()}_linear'] = 10 ** (df_processed[f'{s_param_type.lower()}_db'] / 20)
    
    if param_cols:
        unique_combinations = df_processed[param_cols].drop_duplicates()
        st.write(f"**📊 Combinações de parâmetros encontradas:** {len(unique_combinations)}")
        
        with st.expander("Ver combinações de parâmetros"):
            st.dataframe(unique_combinations, use_container_width=True)
    else:
        unique_combinations = pd.DataFrame({'_dummy': [1]})
        param_cols = ['_dummy']

    all_results = []

    for idx, (_, combo) in enumerate(unique_combinations.iterrows()):
        if param_cols[0] != '_dummy':
            mask = pd.Series(True, index=df_processed.index)
            for col in param_cols:
                mask &= (df_processed[col] == combo[col])
            df_subset = df_processed[mask].copy()
            param_id = "_".join([f"{col}_{combo[col]}" for col in param_cols])
            param_id = "".join(c for c in param_id if c.isalnum() or c in ('_', '-'))
        else:
            df_subset = df_processed.copy()
            param_id = "single_curve"
            combo = {}

        st.markdown("---")
        st.markdown(f"## 📈 Análise {s_param_type}: {param_id}")  # ATUALIZADO
        
        from utils.plot_generators import plot_interactive_curve
        plot_interactive_curve(df_subset, param_id, params=combo, param_cols=param_cols, s_param_type=s_param_type)  # ATUALIZADO
        
        from utils.calculation_engines import manual_ressonance_identification
        
        # ATUALIZADO: Passar s_param_type para a função de identificação
        results = manual_ressonance_identification(
            df_subset, filename, param_id, combo, param_cols, perm_col, 
            unique_combinations, temp_path, None, s_param_type  # ATUALIZADO
        )
        
        results_key = f"results_{param_id}"
        st.session_state[results_key] = results
        all_results.extend(results)

    # CORREÇÃO: Segunda passada para calcular sensibilidade com todos os resultados disponíveis
    if len(all_results) > 0:
        st.markdown("---")
        st.markdown("## 🔄 Recalculando Sensibilidades com Todos os Dados")
        
        # Re-processar cada combinação com todos os resultados disponíveis
        updated_all_results = []
        for idx, (_, combo) in enumerate(unique_combinations.iterrows()):
            if param_cols[0] != '_dummy':
                param_id = "_".join([f"{col}_{combo[col]}" for col in param_cols])
                param_id = "".join(c for c in param_id if c.isalnum() or c in ('_', '-'))
            else:
                param_id = "single_curve"
            
            results_key = f"results_{param_id}"
            if results_key in st.session_state:
                current_results = st.session_state[results_key].copy()
                
                # Recalcular sensibilidade para cada ressonância
                for i, result in enumerate(current_results):
                    if result['frequencia_ressonancia_ghz'] is not None:
                        # Recalcular sensibilidade com todos os resultados
                        from utils.calculation_engines import calculate_sensitivity_corrected
                        
                        sensitivity, figure_of_merit_db, figure_of_merit_linear, figure_of_merit_normal_db, figure_of_merit_normal_linear = calculate_sensitivity_corrected(
                            result['frequencia_ressonancia_ghz'], 
                            combo, perm_col, unique_combinations, 
                            result['Q_3db'], result['Q_linear'], result[f'{s_param_type.lower()}_ressonancia_linear'],  # ATUALIZADO
                            param_id, all_results, s_param_type  # ATUALIZADO
                        )
                        
                        # Atualizar resultado
                        result.update({
                            'sensibilidade_mhz_sqrt_er': sensitivity,
                            'figura_merito_3db': figure_of_merit_db,
                            'figura_merito_linear': figure_of_merit_linear,
                            'figura_merito_normal_3db': figure_of_merit_normal_db,
                            'figura_merito_normal_linear': figure_of_merit_normal_linear
                        })
                
                st.session_state[results_key] = current_results
                updated_all_results.extend(current_results)
        
        all_results = updated_all_results

    return temp_path, all_results

def reset_analysis_state(filename: str):
    """Reseta o estado da análise para um arquivo específico"""
    # Remover todas as chaves de resultados
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('results_')]
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Remover chave de análise
    analysis_key = f"analysis_done_{filename}"
    if analysis_key in st.session_state:
        del st.session_state[analysis_key]