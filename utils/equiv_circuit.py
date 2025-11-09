# utils/equiv_circuit.py
import numpy as np
import pandas as pd
import math
from scipy import signal
from scipy.optimize import least_squares

Z0 = 50.0

# ---------------------------
# Funções para extração
# ---------------------------
def find_f0_fc(freq, s21_db, smooth_window=11, polyorder=3, prominence_db=1.0):
    """Encontra f0 (mínimo de S21) e fc (ponto -3dB partindo do passband).
       freq: GHz
       s21_db: dB
    """
    freq = np.asarray(freq)
    s21_db = np.asarray(s21_db)

    # Suaviza (protege para arrays pequenos)
    if len(s21_db) >= smooth_window:
        s_smooth = signal.savgol_filter(s21_db, smooth_window, polyorder)
    else:
        s_smooth = s21_db

    # encontra mínimos (peaks em -s21)
    peaks, props = signal.find_peaks(-s_smooth, prominence=prominence_db)
    if peaks.size == 0:
        return None, None, s_smooth

    # escolher pico mais profundo
    peak_idx = peaks[np.argmin(s_smooth[peaks])]
    f0 = float(freq[peak_idx])

    # estimar nível de passband: média nos primeiros 5% (ou 3 pontos)
    n = max(3, int(len(freq) * 0.05))
    passband_level = np.mean(s_smooth[:n])
    thresh = passband_level - 3.0
    # localizar primeiro ponto onde cai abaixo do threshold
    idxs = np.where(s_smooth <= thresh)[0]
    fc = float(freq[idxs[0]]) if idxs.size else None

    return f0, fc, s_smooth

def compute_LC_from_fc_f0(fc, f0):
    """Aplicação direta das fórmulas do capítulo (fc,f0 em GHz) -> C em pF, L em nH."""
    if fc is None or f0 is None or f0 <= fc:
        return None, None
    C_pF = (5.0 * fc) / (math.pi * (f0**2 - fc**2))
    if C_pF <= 0:
        return None, None
    L_nH = 250.0 / ((math.pi * f0)**2 * C_pF)
    return float(L_nH), float(C_pF)

# ---------------------------
# Circuito: paralelo R||L||C shunt entre duas metades de linha Z0/2
# montagem via ABCD
# ---------------------------
def Y_RLC(freq_ghz, R_ohm, L_nH, C_pF):
    f = np.asarray(freq_ghz) * 1e9
    w = 2 * np.pi * f
    L = L_nH * 1e-9
    C = C_pF * 1e-12
    # cuidado com w=0
    adm = np.ones_like(w, dtype=complex) * (1.0 / R_ohm)
    adm += 1j * (w * C - 1.0 / (w * L))
    return adm

def abcd_series(Z):
    """ABCD matrix for series impedance Z"""
    A = 1.0
    B = Z
    C = 0.0
    D = 1.0
    return np.array([[A, B], [C, D]], dtype=complex)

def abcd_shunt(Y):
    """ABCD for shunt admittance Y (1x1 complex or array)"""
    # [ [1,0], [Y,1] ]
    if np.ndim(Y) == 0:
        return np.array([[1.0, 0.0], [Y, 1.0]], dtype=complex)
    else:
        # return arrays shaped (2,2,N)
        A = np.ones_like(Y, dtype=complex)
        B = np.zeros_like(Y, dtype=complex)
        C = Y
        D = np.ones_like(Y, dtype=complex)
        return np.array([ [A, B], [C, D] ])

def abcd_multiply(M1, M2):
    """Multiply two ABCD matrices; M may be arrays"""
    return np.einsum('ij...,jk...->ik...', M1, M2)

def abcd_to_s21(ABCD, Z0=50.0):
    """Converte ABCD para S21 (módulo complexo). ABCD pode ser 2x2xN arrays."""
    A = ABCD[0,0]
    B = ABCD[0,1]
    C = ABCD[1,0]
    D = ABCD[1,1]
    denom = (A + B / Z0 + C * Z0 + D)
    S21 = 2.0 / denom
    return S21

def s21_db_from_RLC(freq_ghz, R_ohm, L_nH, C_pF, Z0=50.0):
    """Calcula S21[dB] para a rede: series(Z0/2) - shunt(R||L||C) - series(Z0/2)."""
    freq_ghz = np.asarray(freq_ghz)
    Zs = Z0 / 2.0
    Y = Y_RLC(freq_ghz, R_ohm, L_nH, C_pF)  # complex array
    # ABCD matrices
    M1 = abcd_series(Zs)
    Ms = abcd_series(Zs)
    # shunt returns arrays
    Msh = abcd_shunt(Y)
    # multiply: M_total = M1 * Msh * M1
    Mtmp = abcd_multiply(M1[..., None] if M1.ndim==2 else M1, Msh)
    Mtot = abcd_multiply(Mtmp, Ms[..., None] if Ms.ndim==2 else Ms)
    # Mtot shape: 2x2xN
    S21 = abcd_to_s21(Mtot, Z0=Z0)
    s21_db = 20.0 * np.log10(np.abs(S21) + 1e-20)
    return s21_db

# ---------------------------
# Ajuste simples: ajustar R mantendo L/C analíticos
# ---------------------------
def fit_R_only(freq_ghz, s21_db_meas, L_nH, C_pF, r0=1e3):
    """Ajusta R para aproximar s21_db_meas, mantendo L/C fixos."""
    freq_ghz = np.asarray(freq_ghz)
    s21_db_meas = np.asarray(s21_db_meas)

    def resid(logR):
        R = 10**logR[0]
        s21_sim = s21_db_from_RLC(freq_ghz, R, L_nH, C_pF)
        return s21_sim - s21_db_meas

    logR0 = [math.log10(max(1.0, r0))]
    res = least_squares(resid, logR0, bounds=([math.log10(1e-3)], [math.log10(1e8)]))
    R_est = 10**res.x[0]
    return float(R_est), res

# ---------------------------
# Pipeline: agrupar e extrair
# ---------------------------
def analyze_equiv_circuits_from_df(df,
                                   geom_cols=('g [mm]',
                                              'outer_ring_radius [mm]',
                                              'outer_ring_width [mm]',
                                              'split_width [mm]'),
                                   freq_col='Freq [GHz]',
                                   s11_col='dB(S(1,1)) []',
                                   s21_col='dB(S(2,1)) []'):
    """
    df: pandas DataFrame com colunas geom_cols + freq_col + s11_col + s21_col
    Retorna: resultados_df, per-group time series dict
    """
    # sanity check
    for c in list(geom_cols) + [freq_col, s11_col, s21_col]:
        if c not in df.columns:
            raise ValueError(f"Coluna esperada não encontrada: {c}")

    # convert numeric
    df2 = df.copy()
    df2[freq_col] = pd.to_numeric(df2[freq_col], errors='coerce')
    df2[s11_col] = pd.to_numeric(df2[s11_col], errors='coerce')
    df2[s21_col] = pd.to_numeric(df2[s21_col], errors='coerce')

    group_keys = list(geom_cols)
    grouped = df2.groupby(group_keys)

    results = []
    series_store = {}  # para plotar por caso
    for gvals, gdf in grouped:
        gdf_sorted = gdf.sort_values(freq_col)
        freq = gdf_sorted[freq_col].values
        s21_db = gdf_sorted[s21_col].values
        s11_db = gdf_sorted[s11_col].values

        # extração f0, fc
        f0, fc, s_smooth = find_f0_fc(freq, s21_db)
        L_nH, C_pF = (None, None)
        if f0 is not None and fc is not None:
            L_nH, C_pF = compute_LC_from_fc_f0(fc, f0)
        # ajustar R (apenas R) para melhorar ajuste com L,C (se obtidos)
        R_est = None
        fit_success = False
        if L_nH is not None and C_pF is not None:
            try:
                R_est, res = fit_R_only(freq, s21_db, L_nH, C_pF, r0=1e3)
                fit_success = res.success if hasattr(res, 'success') else True
            except Exception:
                R_est = None
        # Simula com esses parâmetros (se tiver L,C)
        s21_model_db = None
        if L_nH is not None and C_pF is not None:
            R_for_sim = R_est if R_est is not None else 1e6
            s21_model_db = s21_db_from_RLC(freq, R_for_sim, L_nH, C_pF)

        # montar output
        row = dict(zip(group_keys, gvals if isinstance(gvals, tuple) else (gvals,)))
        row.update({
            'f0 [GHz]': f0,
            'fc [GHz]': fc,
            'L_nH': L_nH,
            'C_pF': C_pF,
            'R_ohm_fit': R_est,
            'fit_success': fit_success,
            'n_points': len(freq)
        })
        results.append(row)
        series_store[str(gvals)] = {
            'freq': freq,
            's21_db': s21_db,
            's21_smooth': s_smooth,
            's21_model_db': s21_model_db,
            's11_db': s11_db
        }

    results_df = pd.DataFrame(results)
    # ordena por f0 (se presente)
    if 'f0 [GHz]' in results_df.columns:
        results_df = results_df.sort_values('f0 [GHz]')
    return results_df.reset_index(drop=True), series_store

# ---------------------------
# Correlação simples entre geométricos e L/C
# ---------------------------
def geom_to_LC_correlations(results_df, geom_cols=('g [mm]',
                                                  'outer_ring_radius [mm]',
                                                  'outer_ring_width [mm]',
                                                  'split_width [mm]')):
    """
    Computa correlação Pearson entre cada geom_col e L_nH / C_pF.
    Retorna dict de {col: {'r_L':..., 'p_L':..., 'r_C':..., 'p_C':...}} (p via np.corrcoef aproximação).
    """
    corr = {}
    for col in geom_cols:
        if col not in results_df.columns:
            continue
        x = pd.to_numeric(results_df[col], errors='coerce')
        L = pd.to_numeric(results_df['L_nH'], errors='coerce')
        C = pd.to_numeric(results_df['C_pF'], errors='coerce')
        mask = x.notna() & L.notna()
        if mask.sum() < 3:
            corr[col] = {'r_L': None, 'r_C': None}
            continue
        rL = np.corrcoef(x[mask], L[mask])[0,1]
        rC = np.corrcoef(x[mask], C[mask])[0,1]
        corr[col] = {'r_L': float(rL), 'r_C': float(rC)}
    return corr

# ---------------------------
# Plot helpers (compatível com Plotly)
# ---------------------------
def get_comparison_data(series):
    """Retorna dicionário pronto para plotar no Plotly"""
    return {
        "freq": series["freq"],
        "s21_db": series["s21_db"],
        "s21_smooth": series["s21_smooth"],
        "s21_model_db": series.get("s21_model_db", None),
    }


# ---------------------------
# Utility: export results to excel
# ---------------------------
def export_results_df_to_excel(results_df, path):
    results_df.to_excel(path, index=False)

# ---------------------------
# End of file
# ---------------------------
