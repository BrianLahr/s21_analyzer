import numpy as np
import pandas as pd
import math
from scipy import interpolate
from scipy import signal
from scipy.optimize import least_squares
from scipy.signal import find_peaks

Z0 = 50.0

# --- NOVO: MODELO PI (Seção 2.3.1 do PDF) ---
def s_to_abcd(s11, s12, s21, s22, z0=Z0):
    """
    Converte S-parameters complexos em parâmetros ABCD.
    Baseado nas equações (2.15) - (2.18) do PDF. [cite: 338, 339, 340, 341]
    sXX são arrays complexos.
    """
    # Denominador comum (2 * S21)
    denom = 2 * s21
    
    # Eq (2.15) [cite: 338]
    A = ((1 + s11) * (1 - s22) + s12 * s21) / denom
    
    # Eq (2.16) [cite: 339] - NOTA: O PDF parece ter um erro de digitação (é (1+S11)(1+S22)).
    # A fórmula padrão de conversão S-para-T (e depois para B) usa Z0.
    # B = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / denom
    # Vamos usar a fórmula de B derivada de Y_a = 1/B e A = (1+Y_b/Y_a)
    # Vamos seguir a Eq (2.16) do PDF por enquanto, mas ela é Z0 * (A + ...).
    # A fórmula padrão é:
    B = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / denom
    
    # Eq (2.17) [cite: 340]
    C = (1 / z0) * ((1 - s11) * (1 - s22) - s12 * s21) / denom
    
    # Eq (2.18) [cite: 341]
    D = ((1 - s11) * (1 + s22) + s12 * s21) / denom
    
    return A, B, C, D

def abcd_to_pi_admittance(A, B, C, D):
    """
    Converte parâmetros ABCD de uma rede Pi em admitâncias Ya e Yb.
    Baseado nas equações (2.19) - (2.22).
    """
    # Eq (2.20) -> Ya = 1 / B [cite: 350, 357]
    Ya = 1.0 / B
    
    # Eq (2.19) e (2.20) -> Yb = (A - 1) * Ya = (A - 1) / B [cite: 348, 350, 358]
    Yb = (A - 1.0) / B
    
    return Ya, Yb

def extract_pi_model_params(freq_ghz, s11_c, s12_c, s21_c, s22_c, f0, fc, z0=Z0):
    """
    Extrai os 5 parâmetros do modelo Pi (Req, Leq, Ceq, Rp, Cp)
    usando a metodologia da Seção 2.3.1. [cite: 312]
    
    sXX_c: S-parameters complexos (arrays)
    f0, fc: Frequências de ressonância e corte (em GHz)
    """
    
    # 1. Converter S -> ABCD -> Pi Admittance para TODAS as frequências
    A, B, C, D = s_to_abcd(s11_c, s12_c, s21_c, s22_c, z0)
    Ya, Yb = abcd_to_pi_admittance(A, B, C, D)
    
    # 2. Encontrar índices das frequências de interesse
    idx_f0 = np.argmin(np.abs(freq_ghz - f0))
    idx_fc = np.argmin(np.abs(freq_ghz - fc))
    
    w0_rad = 2 * np.pi * f0 * 1e9
    wc_rad = 2 * np.pi * fc * 1e9
    
    # 3. Extrair parâmetros de Yb (Rp, Cp) - Fringing fields 
    #    Extraídos na frequência de corte (fc), onde o efeito é mais claro
    Yb_at_fc = Yb[idx_fc]
    Rp = 1.0 / np.real(Yb_at_fc)       # [cite: 367]
    Bp_at_fc = np.imag(Yb_at_fc)       # [cite: 358]
    Cp_pF = (Bp_at_fc / wc_rad) * 1e12  # Eq (2.27) [cite: 369]
    
    # 4. Extrair parâmetros de Ya (Req, Leq, Ceq) - Ressonância
    
    # Req é extraído no pico da ressonância (f0) [cite: 367]
    Ya_at_f0 = Ya[idx_f0]
    Req = 1.0 / np.real(Ya_at_f0)
    
    # Leq e Ceq são extraídos usando f0 e fc
    # Usamos o sistema de equações (2.25) e (2.26) [cite: 362, 365]
    
    # Primeiro, pegamos a susceptância Ba em fc
    Ya_at_fc = Ya[idx_fc]
    Ba_at_fc = np.imag(Ya_at_fc)       # [cite: 357]
    
    # Reorganizando Eq (2.25) [cite: 362] e (2.26) [cite: 365] (como derivamos na outra conversa)
    # Ceq = (Ba(wc) * wc) / (wc^2 - w0^2)
    Ceq_F = (Ba_at_fc * wc_rad) / (wc_rad**2 - w0_rad**2)
    Ceq_pF = Ceq_F * 1e12
    
    # Eq (2.26) [cite: 365]
    Leq_H = 1.0 / (w0_rad**2 * Ceq_F)
    Leq_nH = Leq_H * 1e9
    
    return {
        "Req_ohm": Req,
        "Leq_nH": Leq_nH,
        "Ceq_pF": Ceq_pF,
        "Rp_ohm": Rp,
        "Cp_pF": Cp_pF
    }
# --- FIM DO NOVO BLOCO ---


# ---------------------------
# Funções para extração - (Seu código, sem alterações)
# ---------------------------
def find_f0_fc(freq, s21_db, prominence_db=1.0, min_distance_ghz=0.01):
    """Encontra f0 (mínimo de S21) e fc (ponto -3dB partindo do passband).
       Usa a MESMA LÓGICA da detecção de ressonâncias do código anterior.
       freq: GHz, s21_db: dB
    """
    freq = np.asarray(freq)
    s21_db = np.asarray(s21_db)
    
    # MESMA LÓGICA: Criar interpolação de alta precisão
    interp_func_db = interpolate.interp1d(freq, s21_db, kind='cubic', fill_value='extrapolate')
    
    # MESMA LÓGICA: Encontrar mínimos (inverter sinal pois find_peaks busca máximos)
    inverted_s = -s21_db
    
    # MESMA LÓGICA: Calcular distância em pontos
    freq_range = freq[-1] - freq[0]
    min_distance_points = int(min_distance_ghz * len(freq) / freq_range)
    min_distance_points = max(5, min_distance_points)  # Mínimo de 5 pontos
    
    # MESMA LÓGICA: Encontrar picos (mínimos no S original)
    peaks, properties = find_peaks(
        inverted_s,
        prominence=prominence_db,
        distance=min_distance_points
    )
    
    if peaks.size == 0:
        return None, None, s21_db  # Retorna dados originais, não suavizados

    # MESMA LÓGICA: Escolher pico mais profundo
    peak_idx = peaks[np.argmin(s21_db[peaks])]
    
    # MESMA LÓGICA: Refinar localização do mínimo
    refined_f0, refined_s_db, use_original = refine_minimum_location_hybrid(
        freq, s21_db, peak_idx, interp_func_db
    )
    
    f0 = refined_f0
    
    # NOVA LÓGICA: Encontrar fc (frequência de corte -3dB) usando a MESMA abordagem
    fc = find_cutoff_frequency(freq, s21_db, f0, interp_func_db)
    
    return f0, fc, s21_db  # Retorna dados originais, não suavizados

def refine_minimum_location_hybrid(freq, s_db, peak_idx, interp_func_db, window_size=15):
    """
    Refina a localização do mínimo usando estratégia híbrida.
    MESMA LÓGICA do código anterior.
    Retorna: (frequência, s_db, use_original)
    """
    n = len(freq)
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(n, peak_idx + window_size + 1)
    
    # Pegar região ao redor do pico
    freq_region = freq[start_idx:end_idx]
    s_region = s_db[start_idx:end_idx]
    
    # Encontrar índice do mínimo absoluto na região (dados originais)
    min_idx_original = np.argmin(s_region)
    freq_min_original = freq_region[min_idx_original]
    s_min_original = s_region[min_idx_original]
    
    # ESTRATÉGIA 1: Ajuste quadrático nos dados originais
    if len(freq_region) >= 5:  # Mais pontos para melhor ajuste
        try:
            # Ajuste quadrático para encontrar mínimo exato
            coeffs = np.polyfit(freq_region, s_region, 2)
            a, b, c = coeffs
            
            # Mínimo da parábola: x = -b/(2a)
            if a > 0.001:  # Concavidade para cima significativa
                exact_freq = -b / (2 * a)
                exact_s = a * exact_freq**2 + b * exact_freq + c
                
                # Verificar se está dentro da região e é melhor que o original
                if (freq_region[0] <= exact_freq <= freq_region[-1] and
                    exact_s <= s_min_original + 0.1):  # Não pode ser pior
                    
                    # VALIDAÇÃO: Verificar com interpolação cúbica
                    interp_s = float(interp_func_db(exact_freq))
                    
                    # Se a interpolação confirma, usar valor refinado
                    if abs(exact_s - interp_s) < 0.5:  # Diferença pequena
                        return exact_freq, interp_s, False
        
        except:
            pass
    
    # ESTRATÉGIA 2: Busca por mínimos na interpolação de alta resolução
    try:
        # Criar grid fino na região
        freq_fine = np.linspace(freq_region[0], freq_region[-1], 1000)
        s_fine = interp_func_db(freq_fine)
        
        # Encontrar mínimo no grid fino
        min_idx_fine = np.argmin(s_fine)
        freq_min_fine = freq_fine[min_idx_fine]
        s_min_fine = s_fine[min_idx_fine]
        
        # Só usar se for significativamente melhor que o original
        if s_min_fine < s_min_original - 0.01:  # Pelo menos 0.01dB melhor
            return freq_min_fine, s_min_fine, False
    
    except:
        pass
    
    # ESTRATÉGIA 3: Fallback para o melhor ponto original
    return freq_min_original, s_min_original, True

def find_cutoff_frequency(freq, s_db, f0, interp_func_db):
    """
    Encontra a frequência de corte fc (ponto -3dB partindo do passband).
    Usa a MESMA LÓGICA da função find_bandwidth_points_corrected.
    """
    def find_crossing_points(freq_array, y_array, target_val):
        """Encontra os pontos onde a curva cruza o valor target - MESMA LÓGICA"""
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
    
    # Encontrar nível do passband (máximo de S21 antes da ressonância)
    # Buscar apenas na região antes da ressonância
    mask_pre_resonance = freq < f0
    if np.sum(mask_pre_resonance) == 0:
        return None
    
    freq_pre = freq[mask_pre_resonance]
    s_pre = s_db[mask_pre_resonance]
    
    if len(freq_pre) == 0:
        return None
    
    # Usar o valor máximo como nível do passband
    passband_level = np.max(s_pre)
    cutoff_threshold = passband_level - 3.0  # -3dB do passband
    
    # Encontrar todos os cruzamentos com o threshold de corte
    crossings = find_crossing_points(freq, s_db, cutoff_threshold)
    
    if len(crossings) == 0:
        return None
    
    # Encontrar o cruzamento mais próximo antes da ressonância
    crossings_before = [x for x in crossings if x < f0]
    if len(crossings_before) == 0:
        return None
    
    # Pegar o último cruzamento antes da ressonância (mais próximo de f0)
    fc = max(crossings_before)
    
    return fc

def compute_LC_from_fc_f0(fc, f0):
    """Aplicação direta das fórmulas do capítulo (fc,f0 em GHz) -> C em pF, L em nH."""
    if fc is None or f0 is None or f0 <= fc:
        return None, None
    # Eq (2.10) [cite: 153]
    C_pF = (5.0 * fc) / (math.pi * (f0**2 - fc**2))
    if C_pF <= 0:
        return None, None
    # Eq (2.11) [cite: 155]
    L_nH = 250.0 / ((math.pi * f0)**2 * C_pF)
    return float(L_nH), float(C_pF)

# ---------------------------
# Circuito: RLC simples (Seu código, sem alterações)
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
    # NOTA: Este é o seu modelo, que eu havia comentado ser inconsistente
    # com as fórmulas de extração (2.10) e (2.11).
    # Mantendo seu código original.
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
# Ajuste simples: RLC (Seu código, sem alterações)
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
# Pipeline: agrupar e extrair (MODIFICADO)
# ---------------------------
def analyze_equiv_circuits_from_df(df,
                                   geom_cols=None,
                                   freq_col='Freq [GHz]',
                                   # --- MODIFICADO: Adicionar todos os S-params ---
                                   s11_col='dB(S(1,1)) []',
                                   ang_s11_col='ang_deg(S(1,1)) [deg]',
                                   s12_col='dB(S(1,2)) []',
                                   ang_s12_col='ang_deg(S(1,2)) [deg]',
                                   s21_col='dB(S(2,1)) []',
                                   ang_s21_col='ang_deg(S(2,1)) [deg]',
                                   s22_col='dB(S(2,2)) []',
                                   ang_s22_col='ang_deg(S(2,2)) [deg]'):
    """
    df: pandas DataFrame com colunas geom_cols + freq_col + todos S-params
    Retorna: resultados_df, per-group time series dict, correlations
    """
    # Definir colunas geométricas padrão se não especificadas
    if geom_cols is None:
        geom_cols = ['g [mm]', 'outer_ring_radius [mm]', 'outer_ring_width [mm]', 'split_width [mm]']
    
    # Filtrar apenas colunas geométricas que existem no DataFrame
    available_geom_cols = [col for col in geom_cols if col in df.columns]
    if not available_geom_cols:
        raise ValueError("Nenhuma coluna geométrica encontrada no DataFrame")
    
    # --- MODIFICADO: Checar todas as colunas S-params ---
    required_cols = [freq_col, s11_col, s21_col, s12_col, s22_col, 
                     ang_s11_col, ang_s12_col, ang_s21_col, ang_s22_col]
    
    # Checar quais colunas realmente existem
    # O modelo RLC Simples só precisa de s21_col
    if s21_col not in df.columns:
        raise ValueError(f"Coluna esperada não encontrada: {s21_col}")
    
    # Checar se podemos rodar o modelo Pi
    pi_model_ready = True
    for c in required_cols:
        if c not in df.columns:
            pi_model_ready = False
            # print(f"Aviso: Coluna {c} não encontrada. Modelo Pi será desativado.")
            
    # Converter numérico
    df2 = df.copy()
    for col in required_cols: # Tenta converter todas as colunas necessárias
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
    # Remover linhas inválidas
    df2 = df2.dropna(subset=[freq_col, s21_col]) # S21 é o mínimo

    group_keys = available_geom_cols
    grouped = df2.groupby(group_keys)

    results = []
    series_store = {}
    
    for gvals, gdf in grouped:
        gdf_sorted = gdf.sort_values(freq_col)
        freq = gdf_sorted[freq_col].values
        s21_db = gdf_sorted[s21_col].values
        s11_db = gdf_sorted[s11_col].values if s11_col in gdf_sorted else np.full_like(freq, np.nan)

        # --- MODIFICADO: Extrair TODOS os 4 S-params complexos ---
        def get_complex_s(s_db_col, s_ang_col):
            if (s_db_col in gdf_sorted and s_ang_col in gdf_sorted and
                not gdf_sorted[s_db_col].isna().all() and
                not gdf_sorted[s_ang_col].isna().all()):
                
                s_db = gdf_sorted[s_db_col].values
                s_ang_rad = np.deg2rad(gdf_sorted[s_ang_col].values)
                return 10**(s_db / 20.0) * np.exp(1j * s_ang_rad)
            return None

        S11_complex = get_complex_s(s11_col, ang_s11_col)
        S12_complex = get_complex_s(s12_col, ang_s12_col)
        S21_complex = get_complex_s(s21_col, ang_s21_col)
        S22_complex = get_complex_s(s22_col, ang_s22_col)
        # --- Fim da Modificação ---

        # --- Análise RLC Simples (Seu código, sem alterações) ---
        f0, fc, s_original = find_f0_fc(freq, s21_db)
        L_nH, C_pF = (None, None)
        if f0 is not None and fc is not None:
            L_nH, C_pF = compute_LC_from_fc_f0(fc, f0)
        
        R_est = None
        fit_success = False
        if L_nH is not None and C_pF is not None:
            try:
                R_est, res = fit_R_only(freq, s21_db, L_nH, C_pF, r0=1e3)
                fit_success = res.success if hasattr(res, 'success') else True
            except Exception:
                R_est = None
                fit_success = False
        
        s21_model_db = None
        if L_nH is not None and C_pF is not None and R_est is not None:
            s21_model_db = s21_db_from_RLC(freq, R_est, L_nH, C_pF)
        
        # --- NOVO: Análise do Modelo Pi ---
        pi_model_params = {}
        all_s_params_available = (
            S11_complex is not None and S12_complex is not None and
            S21_complex is not None and S22_complex is not None
        )
        
        if pi_model_ready and all_s_params_available and f0 is not None and fc is not None:
            try:
                pi_model_params = extract_pi_model_params(
                    freq, S11_complex, S12_complex, S21_complex, S22_complex,
                    f0, fc, Z0
                )
            except Exception as e:
                pi_model_params = {"pi_error": str(e)}
        elif not (pi_model_ready and all_s_params_available):
             pi_model_params = {"pi_error": "Colunas S-params complexas ausentes."}
        else:
             pi_model_params = {"pi_error": "f0 ou fc não encontrado."}
        # --- Fim do Novo Bloco ---

        
        row = dict(zip(group_keys, gvals if isinstance(gvals, tuple) else (gvals,)))
        # Atualização do RLC Simples
        row.update({
            'f0 [GHz]': f0,
            'fc [GHz]': fc,
            'L_nH': L_nH,
            'C_pF': C_pF,
            'R_ohm_fit': R_est,
            'fit_success': fit_success,
            'n_points': len(freq)
        })
        # --- NOVO: Adicionar resultados do Modelo Pi ---
        row.update(pi_model_params)
        results.append(row)
        
        group_id = str(gvals) if isinstance(gvals, tuple) else f"({gvals})"
        series_store[group_id] = {
            'freq': freq,
            's21_db': s21_db,
            's21_smooth': s_original,
            's21_model_db': s21_model_db,
            's11_db': s11_db,
            'S11_complex': S11_complex,
            'S12_complex': S12_complex,
            'S21_complex': S21_complex,
            'S22_complex': S22_complex,
            'group_values': gvals,
            'params': {
                'L_nH': L_nH,
                'C_pF': C_pF,
                'R_ohm': R_est,
                'f0': f0,
                'fc': fc,
                'pi_model': pi_model_params  # Adiciona o dict do modelo Pi
            }
        }

    results_df = pd.DataFrame(results)

    correlations = {}
    if not results_df.empty:
        correlations = geom_to_LC_correlations(results_df, available_geom_cols)
        
    # --- NOVO: Calcular correlações para o modelo Pi ---
    pi_corr = {}
    if not results_df.empty and 'Leq_nH' in results_df.columns:
        # Pega apenas colunas geométricas
        pi_cols = ['Leq_nH', 'Ceq_pF', 'Req_ohm', 'Cp_pF', 'Rp_ohm']
        pi_corr = geom_to_LC_correlations(results_df, available_geom_cols, pi_cols)
    
    # Combina os dicionários de correlação
    correlations['simple_RLC'] = correlations.pop('simple_RLC', correlations) # Renomeia o antigo
    correlations['pi_model'] = pi_corr
    
    return results_df.reset_index(drop=True), series_store, correlations


# ---------------------------
# Correlação (MODIFICADO)
# ---------------------------
def geom_to_LC_correlations(results_df, geom_cols, target_cols=None):
    """
    Computa correlação Pearson entre cada geom_col e as target_cols.
    """
    if target_cols is None:
        target_cols = ['L_nH', 'C_pF', 'R_ohm_fit'] # Padrão antigo
        
    corr = {}
    for g_col in geom_cols:
        if g_col not in results_df.columns:
            continue
        
        corr[g_col] = {}
        x = pd.to_numeric(results_df[g_col], errors='coerce')
        
        for t_col in target_cols:
            if t_col not in results_df.columns:
                corr[g_col][f'r_{t_col}'] = None
                continue
                
            y = pd.to_numeric(results_df[t_col], errors='coerce')
            
            mask = x.notna() & y.notna()
            if mask.sum() >= 2:
                try:
                    r = np.corrcoef(x[mask], y[mask])[0, 1]
                except:
                    r = None
            else:
                r = None
            
            corr[g_col][f'r_{t_col}'] = r
            
    return corr

# ---------------------------
# Plot helpers (Seu código, sem alterações)
# ---------------------------
def get_comparison_data(series):
    """Retorna dicionário pronto para plotar no Plotly"""
    return {
        "freq": series["freq"],
        "s21_db": series["s21_db"],
        "s21_smooth": series["s21_smooth"],
        "s21_model_db": series.get("s21_model_db", None),
        "s11_db": series.get("s11_db", None),
    }

# ---------------------------
# Utility: export (Seu código, sem alterações)
# ---------------------------
def export_results_df_to_excel(results_df, path):
    """Exporta resultados para Excel"""
    results_df.to_excel(path, index=False)

# ---------------------------
# Streamlit UI (Seu código, sem alterações)
# ---------------------------
def create_circuit_analysis_interface():
    """Cria interface para análise de circuito equivalente no Streamlit"""
    import streamlit as st
    
    st.markdown("## 🔌 Análise de Circuito Equivalente")
    st.markdown("""
    Extrai automaticamente parâmetros RLC do circuito equivalente a partir dos dados S21.
    - **Identifica** f0 (frequência de ressonância) e fc (frequência de corte -3dB)
    - **Modelo RLC Simples:** Calcula L/C analiticamente e ajusta R.
    - **Modelo Pi (Seção 2.3.1):** Extrai 5 parâmetros (Req, Leq, Ceq, Rp, Cp) se todos os dados S-complexos estiverem presentes.
    - **Mostra** correlações entre parâmetros geométricos e L/C
    """)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "**Selecione o arquivo CSV com dados de simulação**",
        type=['csv'],
        key="circuit_analysis_uploader"
    )
    
    if not uploaded_file:
        st.info("👆 Faça upload de um arquivo CSV para analisar o circuito equivalente")
        return None, None, None
    
    try:
        # Ler arquivo
        df = pd.read_csv(uploaded_file)
        
        # Detectar colunas geométricas automaticamente
        geometric_cols = [col for col in df.columns if any(x in col.lower() for x in 
                         ['g [mm]', 'radius', 'width', 'height', 'displacement', 'split'])]
        
        if not geometric_cols:
            st.warning("⚠️ Não foram encontradas colunas geométricas típicas no arquivo")
            default_non_geom = [
                'Freq [GHz]', 'dB(S(1,1)) []', 'ang_deg(S(1,1)) [deg]',
                'dB(S(1,2)) []', 'ang_deg(S(1,2)) [deg]',
                'dB(S(2,1)) []', 'ang_deg(S(2,1)) [deg]',
                'dB(S(2,2)) []', 'ang_deg(S(2,2)) [deg]'
            ]
            geometric_cols = [col for col in df.columns if col not in default_non_geom][:4]
        
        # Processar automaticamente (sem botão)
        with st.spinner("Analisando circuito equivalente..."):
            
            # --- MODIFICADO: Passar os nomes das colunas S-params ---
            # Tentar adivinhar os nomes das colunas S22 e S12
            s12_col = next((c for c in df.columns if 'S(1,2)' in c and 'dB' in c), 'dB(S(1,2)) []')
            ang_s12_col = next((c for c in df.columns if 'S(1,2)' in c and 'ang' in c), 'ang_deg(S(1,2)) [deg]')
            s22_col = next((c for c in df.columns if 'S(2,2)' in c and 'dB' in c), 'dB(S(2,2)) []')
            ang_s22_col = next((c for c in df.columns if 'S(2,2)' in c and 'ang' in c), 'ang_deg(S(2,2)) [deg]')
            
            results_df, series_store, correlations = analyze_equiv_circuits_from_df(
                df, 
                geom_cols=geometric_cols,
                # Passa os nomes das colunas encontradas
                s12_col=s12_col, ang_s12_col=ang_s12_col,
                s22_col=s22_col, ang_s22_col=ang_s22_col
            )
        
        st.success("✅ Análise concluída!")
        
        # --- NOVO: Mostrar resultados do Modelo Pi ---
        st.markdown("### Resultados do Modelo RLC Simples")
        st.dataframe(results_df.drop(columns=[c for c in results_df.columns if 'pi_' in str(c) or 'Req_' in str(c) or 'Leq_' in str(c) or 'Ceq_' in str(c) or 'Rp_' in str(c) or 'Cp_' in str(c)], errors='ignore'))
        
        st.markdown("### Resultados do Modelo Pi (Seção 2.3.1)")
        pi_cols = available_geom_cols + [c for c in results_df.columns if 'pi_' in str(c) or 'Req_' in str(c) or 'Leq_' in str(c) or 'Ceq_' in str(c) or 'Rp_' in str(c) or 'Cp_' in str(c)]
        st.dataframe(results_df[pi_cols])
        
        st.markdown("### Correlações (Parâmetros Geométricos vs. Circuito)")
        st.json(correlations, expanded=False)
        
        return results_df, series_store, correlations
        
    except Exception as e:
        st.error(f"❌ Erro na análise do circuito equivalente: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None