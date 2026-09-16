from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


TRACE_RE = re.compile(
    r"^(?P<kind>db|ang|re|im):(?P<trace>.+)_(?P<sparam>S11|S21)$",
    re.IGNORECASE,
)

FREQ_UNIT_FACTORS = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


def _read_non_comment_lines(file_bytes: bytes) -> List[str]:
    text = file_bytes.decode("utf-8-sig", errors="replace")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _frequency_factor(column_name: str) -> float:
    match = re.search(r"\[\s*(Hz|kHz|MHz|GHz)\s*\]", column_name, re.IGNORECASE)
    if not match:
        return 1.0
    return FREQ_UNIT_FACTORS[match.group(1).lower()]


def _trace_sort_key(name: str) -> Tuple[int, int, str]:
    low = name.lower()
    if low.startswith("trc"):
        match = re.search(r"(\d+)", low)
        return (0, int(match.group(1)) if match else 0, low)
    if low.startswith(("r", "t")):
        match = re.search(r"(\d+)", low)
        return (1, int(match.group(1)) if match else 0, low)
    return (2, 0, low)


def _display_name(trace: str, sparam: str) -> str:
    return f"{trace}_{sparam}"


@st.cache_data(show_spinner=False)
def load_experimental_data(
    file_bytes: bytes, filename: str
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, str]]]]:
    """Load VNA CSV data in either db/ang or re/im format.

    Supported examples:
      - semicolon export: freq[Hz];db:Trc1_S11;ang:Trc1_S11;...
      - whitespace export: freq[Hz] re:Trace_S21 im:Trace_S21 ...

    Returns a dataframe with ``frequency_hz`` plus a trace map:
    ``trace_map['S11']['Trc1']['db'] -> dataframe column``.
    """
    lines = _read_non_comment_lines(file_bytes)
    if not lines:
        raise ValueError(f"{filename}: nenhum dado encontrado.")

    separator = ";" if ";" in lines[0] else r"\s+"
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=separator, engine="python")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        raise ValueError(f"{filename}: arquivo sem linhas de dados.")

    freq_candidates = [column for column in df.columns if column.lower().startswith("freq")]
    if not freq_candidates:
        raise ValueError(f"{filename}: coluna de frequência não encontrada.")

    freq_col = freq_candidates[0]
    factor = _frequency_factor(freq_col)
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce") * factor
    df = df.dropna(subset=[freq_col]).copy()
    df = df.rename(columns={freq_col: "frequency_hz"})

    raw_map: Dict[str, Dict[str, Dict[str, str]]] = {"S11": {}, "S21": {}}
    for column in list(df.columns):
        match = TRACE_RE.match(column)
        if not match:
            continue
        kind = match.group("kind").lower()
        trace = match.group("trace")
        sparam = match.group("sparam").upper()
        raw_map[sparam].setdefault(trace, {})[kind] = column
        df[column] = pd.to_numeric(df[column], errors="coerce")

    trace_map: Dict[str, Dict[str, Dict[str, str]]] = {"S11": {}, "S21": {}}

    for sparam, traces in raw_map.items():
        for trace, columns in traces.items():
            resolved: Dict[str, str] = {}

            if "db" in columns:
                resolved["db"] = columns["db"]
            if "ang" in columns:
                resolved["ang"] = columns["ang"]

            if "re" in columns and "im" in columns:
                real = df[columns["re"]].to_numpy(dtype=float)
                imag = df[columns["im"]].to_numpy(dtype=float)

                if "db" not in resolved:
                    db_col = f"__db__{trace}_{sparam}"
                    magnitude = np.hypot(real, imag)
                    df[db_col] = 20.0 * np.log10(np.maximum(magnitude, 1e-15))
                    resolved["db"] = db_col

                if "ang" not in resolved:
                    ang_col = f"__ang__{trace}_{sparam}"
                    df[ang_col] = np.degrees(np.arctan2(imag, real))
                    resolved["ang"] = ang_col

            if resolved:
                trace_map[sparam][trace] = resolved

    if not trace_map["S11"] and not trace_map["S21"]:
        raise ValueError(
            f"{filename}: nenhuma curva S11/S21 reconhecida. "
            "São aceitas colunas db:/ang: ou re:/im:."
        )

    return df, trace_map


def _available_trace_labels(parsed_files: Iterable[dict]) -> List[str]:
    labels = set()
    for item in parsed_files:
        for sparam, traces in item["trace_map"].items():
            for trace in traces:
                labels.add(_display_name(trace, sparam))
    return sorted(
        labels,
        key=lambda label: (
            0 if label.endswith("_S11") else 1,
            _trace_sort_key(label.rsplit("_", 1)[0]),
        ),
    )


def _default_trace_labels(labels: Iterable[str]) -> List[str]:
    labels = list(labels)
    preferred = [
        label
        for label in labels
        if label.lower() in {"trc1_s11", "trc2_s21"}
    ]
    return preferred if preferred else labels[: min(2, len(labels))]


def _frequency_scale(unit: str) -> Tuple[float, str]:
    return {
        "Hz": (1.0, "Hz"),
        "MHz": (1e6, "MHz"),
        "GHz": (1e9, "GHz"),
    }[unit]


def _add_curve(
    fig: go.Figure,
    df: pd.DataFrame,
    x: pd.Series,
    y_col: str,
    legend_name: str,
    quantity: str,
    unwrap_phase: bool,
    normalize: bool,
) -> None:
    y = df[y_col].to_numpy(dtype=float)

    if quantity == "Fase" and unwrap_phase:
        y = np.rad2deg(np.unwrap(np.deg2rad(y)))
    elif quantity == "Magnitude" and normalize and np.any(np.isfinite(y)):
        y = y - np.nanmax(y)

    hover = (
        "%{x:.6g}<br>%{y:.4f} dB<extra>%{fullData.name}</extra>"
        if quantity == "Magnitude"
        else "%{x:.6g}<br>%{y:.3f}°<extra>%{fullData.name}</extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=legend_name,
            hovertemplate=hover,
        )
    )


def _style_figure(fig: go.Figure, title: str, freq_unit: str, quantity: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        margin=dict(l=30, r=20, t=60, b=35),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(title=f"Frequência ({freq_unit})", showgrid=True)
    fig.update_yaxes(
        title="Magnitude (dB)" if quantity == "Magnitude" else "Fase (graus)",
        showgrid=True,
    )


def _draw_overlay(
    parsed_files: List[dict],
    selected_labels: List[str],
    quantity: str,
    unit: str,
    unwrap_phase: bool,
    normalize: bool,
) -> None:
    scale, unit_label = _frequency_scale(unit)
    kind = "db" if quantity == "Magnitude" else "ang"
    fig = go.Figure()

    for item in parsed_files:
        df = item["df"]
        x = df["frequency_hz"] / scale
        for label in selected_labels:
            trace, sparam = label.rsplit("_", 1)
            column = item["trace_map"].get(sparam, {}).get(trace, {}).get(kind)
            if column:
                _add_curve(
                    fig,
                    df,
                    x,
                    column,
                    f"{item['name']} · {label}",
                    quantity,
                    unwrap_phase,
                    normalize,
                )

    _style_figure(fig, f"{quantity} — curvas sobrepostas", unit_label, quantity)
    st.plotly_chart(fig, use_container_width=True)


def _draw_by_sparameter(
    parsed_files: List[dict],
    selected_labels: List[str],
    quantity: str,
    unit: str,
    unwrap_phase: bool,
    normalize: bool,
) -> None:
    scale, unit_label = _frequency_scale(unit)
    kind = "db" if quantity == "Magnitude" else "ang"

    for sparam in ("S11", "S21"):
        labels = [label for label in selected_labels if label.endswith(f"_{sparam}")]
        if not labels:
            continue

        fig = go.Figure()
        for item in parsed_files:
            df = item["df"]
            x = df["frequency_hz"] / scale
            for label in labels:
                trace, _ = label.rsplit("_", 1)
                column = item["trace_map"].get(sparam, {}).get(trace, {}).get(kind)
                if column:
                    _add_curve(
                        fig,
                        df,
                        x,
                        column,
                        f"{item['name']} · {label}",
                        quantity,
                        unwrap_phase,
                        normalize,
                    )

        _style_figure(fig, f"{sparam} — {quantity}", unit_label, quantity)
        st.plotly_chart(fig, use_container_width=True)


def _draw_by_file(
    parsed_files: List[dict],
    selected_labels: List[str],
    quantity: str,
    unit: str,
    unwrap_phase: bool,
    normalize: bool,
) -> None:
    scale, unit_label = _frequency_scale(unit)
    kind = "db" if quantity == "Magnitude" else "ang"

    for item in parsed_files:
        df = item["df"]
        x = df["frequency_hz"] / scale
        fig = go.Figure()

        for label in selected_labels:
            trace, sparam = label.rsplit("_", 1)
            column = item["trace_map"].get(sparam, {}).get(trace, {}).get(kind)
            if column:
                _add_curve(
                    fig,
                    df,
                    x,
                    column,
                    label,
                    quantity,
                    unwrap_phase,
                    normalize,
                )

        _style_figure(fig, f"{item['name']} — {quantity}", unit_label, quantity)
        st.plotly_chart(fig, use_container_width=True)


def _display_file_details(parsed_files: List[dict]) -> None:
    rows = []
    for item in parsed_files:
        df = item["df"]
        s11 = ", ".join(sorted(item["trace_map"]["S11"], key=_trace_sort_key)) or "—"
        s21 = ", ".join(sorted(item["trace_map"]["S21"], key=_trace_sort_key)) or "—"
        rows.append(
            {
                "Arquivo": item["name"],
                "Pontos": len(df),
                "f inicial (GHz)": df["frequency_hz"].min() / 1e9,
                "f final (GHz)": df["frequency_hz"].max() / 1e9,
                "S11": s11,
                "S21": s21,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _display_statistics(
    parsed_files: List[dict], selected_labels: List[str]
) -> None:
    rows = []
    for item in parsed_files:
        df = item["df"]
        for label in selected_labels:
            trace, sparam = label.rsplit("_", 1)
            columns = item["trace_map"].get(sparam, {}).get(trace, {})
            mag_col = columns.get("db")
            if not mag_col or df.empty:
                continue

            values = df[mag_col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue

            valid_indices = np.flatnonzero(finite)
            local_index = int(np.argmin(values[finite]))
            row_index = valid_indices[local_index]
            rows.append(
                {
                    "Arquivo": item["name"],
                    "Traço": label,
                    "Pontos": int(np.sum(finite)),
                    "Mag. mín. (dB)": float(np.nanmin(values)),
                    "Mag. máx. (dB)": float(np.nanmax(values)),
                    "Freq. do mínimo (GHz)": float(df["frequency_hz"].iloc[row_index] / 1e9),
                }
            )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def create_experimental_viewer() -> None:
    """Interactive viewer for experimental VNA S11/S21 CSV files."""
    st.header("🔬 Visualização de Dados Experimentais do VNA")
    st.caption(
        "Carregue um ou vários CSVs, escolha S11/S21 e compare magnitude ou fase "
        "com as curvas sobrepostas ou separadas."
    )

    uploaded_files = st.file_uploader(
        "Selecione um ou mais arquivos CSV do VNA",
        type=["csv"],
        accept_multiple_files=True,
        key="experimental_vna_files",
        help="Aceita o formato atual db:/ang: e o formato antigo re:/im:.",
    )

    if not uploaded_files:
        st.info("👆 Selecione um ou mais arquivos CSV para iniciar a visualização.")
        return

    parsed_files = []
    errors = []
    for uploaded in uploaded_files:
        try:
            df, trace_map = load_experimental_data(uploaded.getvalue(), uploaded.name)
            parsed_files.append({"name": uploaded.name, "df": df, "trace_map": trace_map})
        except Exception as exc:
            errors.append(str(exc))

    for error in errors:
        st.error(f"❌ {error}")

    if not parsed_files:
        return

    all_labels = _available_trace_labels(parsed_files)

    st.markdown("---")
    st.subheader("⚙️ Visualização")

    top1, top2, top3 = st.columns(3)
    with top1:
        s_choice = st.radio(
            "Parâmetro S",
            ["S11", "S21", "S11 + S21"],
            index=2,
            horizontal=True,
            key="experimental_sparam_choice",
        )
    with top2:
        quantity_choice = st.radio(
            "Grandeza",
            ["Magnitude", "Fase", "Magnitude + Fase"],
            index=0,
            horizontal=True,
            key="experimental_quantity_choice",
        )
    with top3:
        freq_unit = st.selectbox(
            "Unidade de frequência",
            ["GHz", "MHz", "Hz"],
            index=0,
            key="experimental_freq_unit",
        )

    allowed_sparams = {
        "S11": {"S11"},
        "S21": {"S21"},
        "S11 + S21": {"S11", "S21"},
    }[s_choice]
    compatible_labels = [
        label for label in all_labels if label.rsplit("_", 1)[1] in allowed_sparams
    ]

    option1, option2 = st.columns([2, 1])
    with option1:
        selected_labels = st.multiselect(
            "Traços",
            options=compatible_labels,
            default=_default_trace_labels(compatible_labels),
            key=f"experimental_traces_{s_choice}",
            help=(
                "As curvas principais Trc1_S11 e Trc2_S21 são selecionadas por padrão "
                "quando existem; R1…R9 e T1…T9 ficam disponíveis também."
            ),
        )
    with option2:
        layout_choice = st.selectbox(
            "Organização dos gráficos",
            ["Mesmo gráfico", "Separar S11 e S21", "Um gráfico por arquivo"],
            key="experimental_layout_choice",
        )

    check1, check2 = st.columns(2)
    with check1:
        normalize = st.checkbox(
            "Normalizar magnitudes pelo máximo de cada curva",
            value=False,
            key="experimental_normalize",
        )
    with check2:
        unwrap_phase = st.checkbox(
            "Desembrulhar fase",
            value=False,
            key="experimental_unwrap_phase",
        )

    if not selected_labels:
        st.warning("⚠️ Selecione pelo menos um traço para visualizar.")
        return

    scale, unit_label = _frequency_scale(freq_unit)
    global_min = min(item["df"]["frequency_hz"].min() for item in parsed_files) / scale
    global_max = max(item["df"]["frequency_hz"].max() for item in parsed_files) / scale

    if np.isfinite(global_min) and np.isfinite(global_max) and global_max > global_min:
        freq_range = st.slider(
            f"Faixa de frequência ({unit_label})",
            min_value=float(global_min),
            max_value=float(global_max),
            value=(float(global_min), float(global_max)),
            key=f"experimental_freq_range_{freq_unit}",
        )
        fmin, fmax = freq_range
    else:
        fmin, fmax = global_min, global_max

    filtered_files = []
    for item in parsed_files:
        df = item["df"]
        x = df["frequency_hz"] / scale
        mask = (x >= fmin) & (x <= fmax)
        filtered_files.append({**item, "df": df.loc[mask].copy()})

    st.markdown("---")
    st.subheader("📈 Curvas")

    quantities = (
        ["Magnitude", "Fase"]
        if quantity_choice == "Magnitude + Fase"
        else [quantity_choice]
    )

    for quantity in quantities:
        if layout_choice == "Mesmo gráfico":
            _draw_overlay(
                filtered_files,
                selected_labels,
                quantity,
                freq_unit,
                unwrap_phase,
                normalize,
            )
        elif layout_choice == "Separar S11 e S21":
            _draw_by_sparameter(
                filtered_files,
                selected_labels,
                quantity,
                freq_unit,
                unwrap_phase,
                normalize,
            )
        else:
            _draw_by_file(
                filtered_files,
                selected_labels,
                quantity,
                freq_unit,
                unwrap_phase,
                normalize,
            )

    with st.expander("📋 Detalhes dos arquivos carregados"):
        _display_file_details(parsed_files)

    with st.expander("📊 Estatísticas dos traços selecionados"):
        _display_statistics(filtered_files, selected_labels)


def create_experimental_tab() -> None:
    """Entry point used by main.py."""
    try:
        create_experimental_viewer()
    except Exception as exc:
        st.error(f"❌ Erro na visualização experimental: {exc}")
