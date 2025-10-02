import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

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
            from utils.plot_generators import plot_interactive_curve
            plot_interactive_curve(df_subset, param_id, params=combo, param_cols=param_cols)
            
            # Seção para identificação manual de ressonâncias
            from utils.calculation_engines import manual_ressonance_identification
            results = manual_ressonance_identification(
                df_subset, filename, param_id, combo, param_cols, perm_col, 
                unique_combinations, temp_path
            )
            all_results.extend(results)
        
        return temp_path, all_results