from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


GRID_SIZE = 6
DEFAULT_PITCH_MM = 2.1
SEGMENT_RE = re.compile(
    r"(?<!\d)(?P<a>[1-6][1-6])a(?P<b>[1-6][1-6])(?!\d)",
    re.IGNORECASE,
)

METRIC_OPTIONS = [
    "Magnitude mínima (dB)",
    "Frequência do mínimo (GHz)",
    "Magnitude máxima (dB)",
    "Frequência do máximo (GHz)",
    "Magnitude em frequência escolhida (dB)",
    "Fase em frequência escolhida (graus)",
    "RMSE da magnitude vs referência (dB)",
    "Correlação da magnitude vs referência",
    "Δ frequência do mínimo vs referência (MHz)",
]


def _expand_segment(start: str, end: str) -> List[str]:
    """Expand matrix notation such as 16a66 or 65a45."""
    r1, c1 = int(start[0]), int(start[1])
    r2, c2 = int(end[0]), int(end[1])

    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [f"{r1}{c}" for c in range(c1, c2 + step, step)]

    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [f"{r}{c1}" for r in range(r1, r2 + step, step)]

    raise ValueError(
        f"Trecho {start}a{end} não é horizontal nem vertical no grid."
    )


def parse_scan_positions(filename: str) -> List[str]:
    """Decode all scan segments embedded in a filename.

    Examples
    --------
    ReT_16a66_65a45_16092026.csv
      -> 16, 26, 36, 46, 56, 66, 65, 55, 45
    3_ReT_63a13_12a32_16092026.csv
      -> 63, 53, 43, 33, 23, 13, 12, 22, 32
    """
    positions: List[str] = []

    for match in SEGMENT_RE.finditer(filename):
        expanded = _expand_segment(match.group("a"), match.group("b"))
        if positions and expanded and expanded[0] == positions[-1]:
            expanded = expanded[1:]
        positions.extend(expanded)

    return positions


def _trace_index_map(trace_names, prefix: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    for trace in trace_names:
        match = pattern.match(trace)
        if match:
            result[int(match.group(1))] = trace
    return result


def build_grid_records(
    parsed_files: List[dict], pitch_mm: float = DEFAULT_PITCH_MM
) -> Tuple[List[dict], List[str]]:
    """Map Rn/Tn traces to 6x6 matrix positions using filename scan notation."""
    records: List[dict] = []
    warnings: List[str] = []
    seen_positions: Dict[str, str] = {}

    for item in parsed_files:
        filename = item["name"]
        positions = parse_scan_positions(filename)

        if not positions:
            warnings.append(
                f"{filename}: não encontrei trechos como 16a66 no nome; "
                "o arquivo foi ignorado na análise XY."
            )
            continue

        s11_indices = _trace_index_map(item["trace_map"].get("S11", {}), "R")
        s21_indices = _trace_index_map(item["trace_map"].get("S21", {}), "T")
        measured_indices = sorted(set(s11_indices) | set(s21_indices))

        if not measured_indices:
            warnings.append(
                f"{filename}: não encontrei traços R1…Rn/T1…Tn; "
                "Trc1/Trc2 são ignorados de propósito."
            )
            continue

        if len(positions) != len(measured_indices):
            warnings.append(
                f"{filename}: o nome descreve {len(positions)} posições, mas "
                f"foram encontrados {len(measured_indices)} índices R/T. "
                "O mapeamento será feito na ordem até o menor comprimento."
            )

        pair_count = min(len(positions), len(measured_indices))
        for scan_order in range(pair_count):
            position = positions[scan_order]
            trace_index = measured_indices[scan_order]
            row = int(position[0])
            col = int(position[1])

            if position in seen_positions:
                warnings.append(
                    f"Posição {position} apareceu em {seen_positions[position]} e "
                    f"novamente em {filename}; a ocorrência mais recente também será mostrada."
                )
            seen_positions[position] = filename

            records.append(
                {
                    "position": position,
                    "row": row,
                    "col": col,
                    "x_mm": (col - 1) * pitch_mm,
                    "y_mm": (GRID_SIZE - row) * pitch_mm,
                    "scan_order": scan_order + 1,
                    "trace_index": trace_index,
                    "file_name": filename,
                    "df": item["df"],
                    "S11": item["trace_map"]
                    .get("S11", {})
                    .get(s11_indices.get(trace_index, ""), {}),
                    "S21": item["trace_map"]
                    .get("S21", {})
                    .get(s21_indices.get(trace_index, ""), {}),
                }
            )

    return records, warnings


def _sorted_position_labels(records: List[dict]) -> List[str]:
    return [
        record["position"]
        for record in sorted(records, key=lambda r: (r["row"], r["col"]))
    ]


def _frequency_overlap(records: List[dict]) -> Optional[Tuple[float, float]]:
    if not records:
        return None
    lower = max(float(r["df"]["frequency_hz"].min()) for r in records)
    upper = min(float(r["df"]["frequency_hz"].max()) for r in records)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def _frequency_step_ghz(records: List[dict]) -> float:
    steps = []
    for record in records:
        freq = np.sort(record["df"]["frequency_hz"].dropna().to_numpy(dtype=float))
        if freq.size > 1:
            diffs = np.diff(freq)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                steps.append(float(np.median(diffs) / 1e9))
    if not steps:
        return 0.001
    return max(min(steps), 1e-9)


def _curve(
    record: dict,
    sparam: str,
    kind: str,
    fmin_hz: float,
    fmax_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    column = record.get(sparam, {}).get(kind)
    if not column:
        return np.array([]), np.array([])

    df = record["df"]
    freq = df["frequency_hz"].to_numpy(dtype=float)
    values = df[column].to_numpy(dtype=float)
    finite = np.isfinite(freq) & np.isfinite(values)
    finite &= (freq >= fmin_hz) & (freq <= fmax_hz)

    freq = freq[finite]
    values = values[finite]
    if freq.size == 0:
        return freq, values

    order = np.argsort(freq)
    return freq[order], values[order]


def _value_at_frequency(
    record: dict,
    sparam: str,
    kind: str,
    target_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> float:
    freq, values = _curve(record, sparam, kind, fmin_hz, fmax_hz)
    if freq.size == 0 or target_hz < freq[0] or target_hz > freq[-1]:
        return np.nan

    if kind == "ang":
        values = np.rad2deg(np.unwrap(np.deg2rad(values)))
    return float(np.interp(target_hz, freq, values))


def _extreme(
    record: dict,
    sparam: str,
    which: str,
    fmin_hz: float,
    fmax_hz: float,
) -> Tuple[float, float]:
    freq, mag = _curve(record, sparam, "db", fmin_hz, fmax_hz)
    if mag.size == 0:
        return np.nan, np.nan
    index = int(np.nanargmin(mag) if which == "min" else np.nanargmax(mag))
    return float(mag[index]), float(freq[index])


def _aligned_magnitude_pair(
    record: dict,
    reference: dict,
    sparam: str,
    fmin_hz: float,
    fmax_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    f_a, y_a = _curve(record, sparam, "db", fmin_hz, fmax_hz)
    f_b, y_b = _curve(reference, sparam, "db", fmin_hz, fmax_hz)
    if f_a.size < 2 or f_b.size < 2:
        return np.array([]), np.array([])

    low = max(f_a[0], f_b[0])
    high = min(f_a[-1], f_b[-1])
    if high <= low:
        return np.array([]), np.array([])

    mask = (f_b >= low) & (f_b <= high)
    common_f = f_b[mask]
    if common_f.size < 2:
        return np.array([]), np.array([])

    return np.interp(common_f, f_a, y_a), y_b[mask]


def _compute_metric(
    record: dict,
    metric: str,
    sparam: str,
    fmin_hz: float,
    fmax_hz: float,
    target_hz: Optional[float],
    reference: Optional[dict],
) -> float:
    if metric == "Magnitude mínima (dB)":
        value, _ = _extreme(record, sparam, "min", fmin_hz, fmax_hz)
        return value

    if metric == "Frequência do mínimo (GHz)":
        _, freq = _extreme(record, sparam, "min", fmin_hz, fmax_hz)
        return freq / 1e9

    if metric == "Magnitude máxima (dB)":
        value, _ = _extreme(record, sparam, "max", fmin_hz, fmax_hz)
        return value

    if metric == "Frequência do máximo (GHz)":
        _, freq = _extreme(record, sparam, "max", fmin_hz, fmax_hz)
        return freq / 1e9

    if metric == "Magnitude em frequência escolhida (dB)":
        if target_hz is None:
            return np.nan
        return _value_at_frequency(
            record, sparam, "db", target_hz, fmin_hz, fmax_hz
        )

    if metric == "Fase em frequência escolhida (graus)":
        if target_hz is None:
            return np.nan
        return _value_at_frequency(
            record, sparam, "ang", target_hz, fmin_hz, fmax_hz
        )

    if metric in {
        "RMSE da magnitude vs referência (dB)",
        "Correlação da magnitude vs referência",
    }:
        if reference is None:
            return np.nan
        y, y_ref = _aligned_magnitude_pair(
            record, reference, sparam, fmin_hz, fmax_hz
        )
        if y.size < 2:
            return np.nan
        if metric.startswith("RMSE"):
            return float(np.sqrt(np.mean((y - y_ref) ** 2)))
        if np.nanstd(y) == 0 or np.nanstd(y_ref) == 0:
            return np.nan
        return float(np.corrcoef(y, y_ref)[0, 1])

    if metric == "Δ frequência do mínimo vs referência (MHz)":
        if reference is None:
            return np.nan
        _, freq = _extreme(record, sparam, "min", fmin_hz, fmax_hz)
        _, freq_ref = _extreme(reference, sparam, "min", fmin_hz, fmax_hz)
        return (freq - freq_ref) / 1e6

    return np.nan


def _metric_axis_label(metric: str) -> str:
    labels = {
        "Magnitude mínima (dB)": "dB",
        "Frequência do mínimo (GHz)": "GHz",
        "Magnitude máxima (dB)": "dB",
        "Frequência do máximo (GHz)": "GHz",
        "Magnitude em frequência escolhida (dB)": "dB",
        "Fase em frequência escolhida (graus)": "graus",
        "RMSE da magnitude vs referência (dB)": "dB",
        "Correlação da magnitude vs referência": "r",
        "Δ frequência do mínimo vs referência (MHz)": "MHz",
    }
    return labels.get(metric, "")


def _metric_dataframe(
    records: List[dict],
    metric: str,
    sparam: str,
    fmin_hz: float,
    fmax_hz: float,
    target_hz: Optional[float],
    reference: Optional[dict],
) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "Posição": record["position"],
                "X (mm)": record["x_mm"],
                "Y (mm)": record["y_mm"],
                "Linha": record["row"],
                "Coluna": record["col"],
                "Arquivo": record["file_name"],
                "Índice": record["trace_index"],
                "Valor": _compute_metric(
                    record,
                    metric,
                    sparam,
                    fmin_hz,
                    fmax_hz,
                    target_hz,
                    reference,
                ),
            }
        )
    return pd.DataFrame(rows)


def _heatmap_figure(
    metric_df: pd.DataFrame,
    metric: str,
    sparam: str,
    pitch_mm: float,
) -> go.Figure:
    x_values = np.arange(GRID_SIZE, dtype=float) * pitch_mm
    y_values = np.arange(GRID_SIZE, dtype=float) * pitch_mm
    z = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    labels = np.full((GRID_SIZE, GRID_SIZE), "", dtype=object)
    files = np.full((GRID_SIZE, GRID_SIZE), "", dtype=object)

    for _, row in metric_df.iterrows():
        x_index = int(row["Coluna"]) - 1
        y_index = GRID_SIZE - int(row["Linha"])
        z[y_index, x_index] = row["Valor"]
        labels[y_index, x_index] = str(row["Posição"])
        files[y_index, x_index] = str(row["Arquivo"])

    unit = _metric_axis_label(metric)
    custom = np.dstack([labels, files])

    heatmap_kwargs = {}
    if metric == "Correlação da magnitude vs referência":
        heatmap_kwargs.update(zmin=-1.0, zmax=1.0, zmid=0.0)

    fig = go.Figure(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=z,
            text=labels,
            customdata=custom,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title=unit),
            hovertemplate=(
                "posição %{customdata[0]}<br>"
                "X=%{x:.2f} mm<br>"
                "Y=%{y:.2f} mm<br>"
                f"{metric}: %{{z:.5g}} {unit}<br>"
                "arquivo: %{customdata[1]}"
                "<extra></extra>"
            ),
            **heatmap_kwargs,
        )
    )
    fig.update_layout(
        title=f"{sparam} — {metric}",
        template="plotly_white",
        height=600,
        margin=dict(l=55, r=35, t=70, b=50),
    )
    fig.update_xaxes(
        title="X (mm) — coluna da matriz",
        tickmode="array",
        tickvals=x_values,
        constrain="domain",
    )
    fig.update_yaxes(
        title="Y (mm) — linha 1 no topo",
        tickmode="array",
        tickvals=y_values,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def _surface_figure(
    metric_df: pd.DataFrame,
    metric: str,
    sparam: str,
    pitch_mm: float,
) -> go.Figure:
    x_values = np.arange(GRID_SIZE, dtype=float) * pitch_mm
    y_values = np.arange(GRID_SIZE, dtype=float) * pitch_mm
    z = np.full((GRID_SIZE, GRID_SIZE), np.nan)

    for _, row in metric_df.iterrows():
        z[GRID_SIZE - int(row["Linha"]), int(row["Coluna"]) - 1] = row["Valor"]

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_values,
                y=y_values,
                z=z,
                colorbar=dict(title=_metric_axis_label(metric)),
                connectgaps=False,
            )
        ]
    )
    fig.update_layout(
        title=f"Superfície XY — {sparam} — {metric}",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title=_metric_axis_label(metric),
        ),
    )
    return fig


def _curve_comparison(
    records: List[dict],
    selected_positions: List[str],
    sparam: str,
    fmin_hz: float,
    fmax_hz: float,
) -> go.Figure:
    fig = go.Figure()
    by_position = {record["position"]: record for record in records}

    for position in selected_positions:
        record = by_position.get(position)
        if not record:
            continue
        freq, mag = _curve(record, sparam, "db", fmin_hz, fmax_hz)
        if freq.size:
            fig.add_trace(
                go.Scatter(
                    x=freq / 1e9,
                    y=mag,
                    mode="lines",
                    name=position,
                    hovertemplate=(
                        f"posição {position}<br>"
                        "%{x:.6f} GHz<br>%{y:.4f} dB<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"{sparam} — comparação por posição",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        xaxis_title="Frequência (GHz)",
        yaxis_title="Magnitude (dB)",
        legend_title="Posição",
    )
    return fig


def create_spatial_xy_analysis(parsed_files: List[dict]) -> None:
    """Spatial analysis for the 6x6 snake scan used in the VNA experiment."""
    st.header("🗺️ Análise espacial XY — grid 6×6")
    st.caption(
        "Reconstrói automaticamente a varredura pelos nomes dos arquivos "
        "(ex.: 16a66_65a45), ignora Trc1/Trc2 e associa Rn/Tn às posições "
        "na ordem de aquisição."
    )

    pitch_mm = st.number_input(
        "Passo entre posições do grid (mm)",
        min_value=0.001,
        value=float(DEFAULT_PITCH_MM),
        step=0.1,
        format="%.3f",
        key="xy_pitch_mm",
    )

    records, warnings = build_grid_records(parsed_files, pitch_mm=pitch_mm)
    for warning in warnings:
        st.warning(warning)

    if not records:
        st.info(
            "Nenhuma posição XY pôde ser reconstruída. "
            "Use nomes contendo trechos como 16a66_65a45."
        )
        return

    unique_positions = sorted(set(r["position"] for r in records))
    detected = len(unique_positions)
    missing = [
        f"{row}{col}"
        for row in range(1, GRID_SIZE + 1)
        for col in range(1, GRID_SIZE + 1)
        if f"{row}{col}" not in unique_positions
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Posições detectadas", f"{detected}/36")
    c2.metric("Arquivos mapeados", len(set(r["file_name"] for r in records)))
    c3.metric("Passo XY", f"{pitch_mm:.3f} mm")

    if missing:
        st.info("Posições ainda ausentes: " + ", ".join(missing))
    else:
        st.success("Grid completo: 36/36 posições reconstruídas automaticamente.")

    with st.expander("🔎 Conferir mapeamento arquivo → posição → Rn/Tn"):
        mapping = pd.DataFrame(
            [
                {
                    "Posição": r["position"],
                    "X (mm)": r["x_mm"],
                    "Y (mm)": r["y_mm"],
                    "Arquivo": r["file_name"],
                    "Ordem no arquivo": r["scan_order"],
                    "R/T índice": r["trace_index"],
                    "S11": f"R{r['trace_index']}" if r["S11"] else "—",
                    "S21": f"T{r['trace_index']}" if r["S21"] else "—",
                }
                for r in sorted(records, key=lambda x: (x["row"], x["col"]))
            ]
        )
        st.dataframe(mapping, use_container_width=True, hide_index=True)

    overlap = _frequency_overlap(records)
    if overlap is None:
        st.error("Os arquivos não possuem uma faixa de frequência comum.")
        return

    global_min_ghz, global_max_ghz = overlap[0] / 1e9, overlap[1] / 1e9
    st.subheader("📐 Métrica espacial")

    p1, p2 = st.columns(2)
    with p1:
        sparam = st.radio(
            "Parâmetro",
            ["S11", "S21"],
            horizontal=True,
            key="xy_sparam",
        )
    with p2:
        metric = st.selectbox(
            "Métrica do mapa",
            METRIC_OPTIONS,
            index=1,
            key="xy_metric",
        )

    frequency_step_ghz = _frequency_step_ghz(records)
    analysis_range = st.slider(
        "Faixa usada na análise (GHz)",
        min_value=float(global_min_ghz),
        max_value=float(global_max_ghz),
        value=(float(global_min_ghz), float(global_max_ghz)),
        step=float(frequency_step_ghz),
        key="xy_analysis_frequency_range",
        help=(
            "Importante para mínimos/máximos: restrinja a faixa à ressonância "
            "de interesse para não selecionar um extremo na borda da varredura."
        ),
    )
    fmin_hz, fmax_hz = analysis_range[0] * 1e9, analysis_range[1] * 1e9

    target_hz: Optional[float] = None
    if metric in {
        "Magnitude em frequência escolhida (dB)",
        "Fase em frequência escolhida (graus)",
    }:
        target_ghz = st.slider(
            "Frequência do mapa (GHz)",
            min_value=float(analysis_range[0]),
            max_value=float(analysis_range[1]),
            value=float((analysis_range[0] + analysis_range[1]) / 2.0),
            step=float(frequency_step_ghz),
            key="xy_target_frequency",
        )
        target_hz = target_ghz * 1e9

    reference: Optional[dict] = None
    if "referência" in metric:
        position_labels = _sorted_position_labels(records)
        reference_position = st.selectbox(
            "Posição de referência",
            position_labels,
            index=0,
            key="xy_reference_position",
            help=(
                "A curva de cada posição é comparada com esta assinatura dentro "
                "da faixa de frequência selecionada."
            ),
        )
        reference = next(
            (r for r in records if r["position"] == reference_position), None
        )

    metric_df = _metric_dataframe(
        records,
        metric,
        sparam,
        fmin_hz,
        fmax_hz,
        target_hz,
        reference,
    )

    st.plotly_chart(
        _heatmap_figure(metric_df, metric, sparam, pitch_mm),
        use_container_width=True,
    )

    show_surface = st.checkbox(
        "Mostrar também superfície 3D",
        value=False,
        key="xy_show_surface",
    )
    if show_surface:
        st.plotly_chart(
            _surface_figure(metric_df, metric, sparam, pitch_mm),
            use_container_width=True,
        )

    finite_values = metric_df[np.isfinite(metric_df["Valor"])].copy()
    if not finite_values.empty:
        min_row = finite_values.loc[finite_values["Valor"].idxmin()]
        max_row = finite_values.loc[finite_values["Valor"].idxmax()]
        e1, e2 = st.columns(2)
        e1.metric(
            "Menor valor no mapa",
            f"{min_row['Valor']:.6g} {_metric_axis_label(metric)}",
            help=f"Posição {min_row['Posição']}",
        )
        e2.metric(
            "Maior valor no mapa",
            f"{max_row['Valor']:.6g} {_metric_axis_label(metric)}",
            help=f"Posição {max_row['Posição']}",
        )

    table = metric_df[
        ["Posição", "X (mm)", "Y (mm)", "Valor", "Arquivo", "Índice"]
    ].sort_values(["Y (mm)", "X (mm)"], ascending=[False, True])
    with st.expander("📋 Valores numéricos do mapa"):
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar mapa como CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"mapa_xy_{sparam.lower()}.csv",
            mime="text/csv",
            key="xy_download_metric_csv",
        )

    st.subheader("📈 Comparar posições específicas")
    positions = _sorted_position_labels(records)
    default_positions = positions[: min(4, len(positions))]
    selected_positions = st.multiselect(
        "Posições da matriz",
        positions,
        default=default_positions,
        key="xy_compare_positions",
    )
    if selected_positions:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(
                _curve_comparison(
                    records,
                    selected_positions,
                    "S11",
                    fmin_hz,
                    fmax_hz,
                ),
                use_container_width=True,
            )
        with cc2:
            st.plotly_chart(
                _curve_comparison(
                    records,
                    selected_positions,
                    "S21",
                    fmin_hz,
                    fmax_hz,
                ),
                use_container_width=True,
            )
