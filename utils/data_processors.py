import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

def identify_columns(df: pd.DataFrame):
    """Identifica automaticamente colunas de frequência, S21, parâmetros e permissividade."""
    freq_col = None
    s21_col = None
    param_cols = []
    perm_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['freq', 'frequency', 'ghz', 'mhz']):
            freq_col = col
        elif any(x in col_lower for x in ['s21', 's(2,1)', 'db(s(2,1))', 'insertion']):
            s21_col = col
        elif any(x in col_lower for x in ['perm', 'permittivity', 'epsilon', 'dielectric']):
            perm_col = col
            param_cols.append(col)
        elif col not in [freq_col, s21_col]:
            param_cols.append(col)

    # Se não encontrou S21, usar a primeira coluna numérica diferente de freq
    if s21_col is None and freq_col is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        other_numeric = [col for col in numeric_cols if col != freq_col]
        if other_numeric:
            s21_col = other_numeric[0]

    return freq_col, s21_col, param_cols, perm_col


import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

def process_data(df: pd.DataFrame, filename: str, freq_col: str, s21_col: str, param_cols: list, perm_col: str):
    """Processa dados e plota gráficos, preservando resultados anteriores.
    Retorna sempre (temp_path, all_results) para compatibilidade com main.py
    """

    # Padronizar colunas principais
    df = df.rename(columns={freq_col: 'freq_ghz', s21_col: 's21_db'})
    df['s21_linear'] = 10 ** (df['s21_db'] / 20)

    if param_cols:
        unique_combinations = df[param_cols].drop_duplicates()
    else:
        unique_combinations = pd.DataFrame({'_dummy': [1]})
        param_cols = ['_dummy']

    all_results = []

    temp_path = Path(tempfile.gettempdir()) / "s21_analysis"
    temp_path.mkdir(exist_ok=True)

    for _, combo in unique_combinations.iterrows():
        if param_cols[0] != '_dummy':
            mask = pd.Series(True, index=df.index)
            for col in param_cols:
                mask &= df[col] == combo[col]
            df_subset = df[mask].copy()
            param_id = "_".join([f"{col}_{combo[col]}" for col in param_cols])
            param_id = "".join(c for c in param_id if c.isalnum() or c in ('_', '-'))
        else:
            df_subset = df.copy()
            param_id = "single_curve"
            combo = {}

        results_key = f"results_{param_id}"
        if results_key not in st.session_state:
            st.session_state[results_key] = []

        # Plotar gráfico interativo
        from utils.plot_generators import plot_interactive_curve
        plot_interactive_curve(df_subset, param_id, params=combo, param_cols=param_cols)

        # Identificação manual de ressonâncias
        from utils.calculation_engines import manual_ressonance_identification
        results = manual_ressonance_identification(
            df_subset, filename, param_id, combo, param_cols, perm_col,
            unique_combinations, temp_path
        )

        st.session_state[results_key] = results
        all_results.extend(results)

    return temp_path, all_results





def reset_analysis_state(filename: str):
    """Reseta o estado da análise para um arquivo específico"""
    analysis_key = f"analysis_done_{filename}"
    if analysis_key in st.session_state:
        del st.session_state[analysis_key]
    if "analysis_data" in st.session_state:
        del st.session_state["analysis_data"]
