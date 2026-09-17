from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


TRACE_RE = re.compile(
    r"^(?P<kind>db|ang|re|im):(?P<trace>.+)_(?P<sparam>S11|S21)$",
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False)
def parse_vna_csv(file_bytes: bytes, filename: str):
    text = file_bytes.decode("utf-8-sig", errors="replace")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{filename}: arquivo sem dados.")

    header = lines[0]
    if ";" in header:
        sep = ";"
        fmt = "separado por ponto e vírgula"
    elif "," in header:
        sep = ","
        fmt = "separado por vírgula"
    else:
        sep = r"\s+"
        fmt = "separado por espaços"

    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    freq_candidates = [c for c in df.columns if c.lower().startswith("freq")]
    if not freq_candidates:
        raise ValueError(f"{filename}: coluna de frequência não encontrada.")

    freq_col = freq_candidates[0]
    factor = 1.0
    low = freq_col.lower()
    if "ghz" in low:
        factor = 1e9
    elif "mhz" in low:
        factor = 1e6
    elif "khz" in low:
        factor = 1e3

    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce") * factor
    df = df.dropna(subset=[freq_col]).copy()
    df = df.rename(columns={freq_col: "frequency_hz"})

    raw: Dict[str, Dict[str, Dict[str, str]]] = {"S11": {}, "S21": {}}
    for col in df.columns:
        match = TRACE_RE.match(col)
        if not match:
            continue
        kind = match.group("kind").lower()
        trace = match.group("trace")
        sparam = match.group("sparam").upper()
        raw[sparam].setdefault(trace, {})[kind] = col
        df[col] = pd.to_numeric(df[col], errors="coerce")

    traces: Dict[str, Dict[str, Dict[str, str]]] = {"S11": {}, "S21": {}}
    for sparam, trace_dict in raw.items():
        for trace, cols in trace_dict.items():
            resolved = dict(cols)
            if "re" in cols and "im" in cols:
                real = df[cols["re"]].to_numpy(dtype=float)
                imag = df[cols["im"]].to_numpy(dtype=float)

                if "db" not in resolved:
                    db_col = f"__db__{trace}_{sparam}"
                    df[db_col] = 20.0 * np.log10(
                        np.maximum(np.hypot(real, imag), 1e-15)
                    )
                    resolved["db"] = db_col

                if "ang" not in resolved:
                    ang_col = f"__ang__{trace}_{sparam}"
                    df[ang_col] = np.degrees(np.arctan2(imag, real))
                    resolved["ang"] = ang_col

            if resolved:
                traces[sparam][trace] = resolved

    if not traces["S11"] and not traces["S21"]:
        raise ValueError(
            f"{filename}: nenhuma coluna S11/S21 reconhecida em db/ang ou re/im."
        )

    return df, traces, fmt


def _expand_uploads(uploaded_files) -> Tuple[List[Tuple[str, bytes]], List[str]]:
    csv_files: List[Tuple[str, bytes]] = []
    errors: List[str] = []

    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        name = uploaded.name
        if name.lower().endswith(".csv"):
            csv_files.append((name, data))
            continue

        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    members = [
                        member
                        for member in archive.namelist()
                        if member.lower().endswith(".csv") and not member.endswith("/")
                    ]
                    for member in members:
                        csv_files.append((os.path.basename(member), archive.read(member)))
                if not members:
                    errors.append(f"{name}: o ZIP não contém arquivos CSV.")
            except zipfile.BadZipFile:
                errors.append(f"{name}: ZIP inválido ou corrompido.")

    return csv_files, errors


def _trace_labels(parsed: List[dict]) -> List[str]:
    labels = set()
    for item in parsed:
        for sparam, trace_dict in item["traces"].items():
            for trace in trace_dict:
                labels.add(f"{trace}_{sparam}")

    def key(label: str):
        trace, sparam = label.rsplit("_", 1)
        match = re.search(r"(\d+)", trace)
        number = int(match.group(1)) if match else 999999
        family = 0 if trace.lower().startswith("trc") else 1
        return (0 if sparam == "S11" else 1, family, number, trace.lower())

    return sorted(labels, key=key)


def _default_labels(labels: List[str]) -> List[str]:
    preferred = [
        label
        for label in labels
        if label.lower() in {"trc7_s21", "trc8_s11"}
    ]
    return preferred if preferred else labels[: min(4, len(labels))]


def _plot(
    parsed: List[dict],
    selected_files: List[str],
    labels: List[str],
    quantity: str,
    fmin_ghz: float,
    fmax_ghz: float,
    unwrap_phase: bool,
    split_by_file: bool,
):
    kind = "db" if quantity == "Magnitude" else "ang"
    selected = [item for item in parsed if item["name"] in selected_files]

    figures = []
    groups = [[item] for item in selected] if split_by_file else [selected]

    for group in groups:
        fig = go.Figure()
        for item in group:
            df = item["df"]
            freq_ghz = df["frequency_hz"] / 1e9
            mask = (freq_ghz >= fmin_ghz) & (freq_ghz <= fmax_ghz)

            for label in labels:
                trace, sparam = label.rsplit("_", 1)
                column = item["traces"].get(sparam, {}).get(trace, {}).get(kind)
                if not column:
                    continue

                x = freq_ghz.loc[mask]
                y = df.loc[mask, column].to_numpy(dtype=float)
                if quantity == "Fase" and unwrap_phase:
                    y = np.rad2deg(np.unwrap(np.deg2rad(y)))

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{item['name']} · {label}",
                    )
                )

        title_file = f" — {group[0]['name']}" if split_by_file and group else ""
        fig.update_layout(
            title=f"{quantity}{title_file}",
            template="plotly_white",
            height=520,
            hovermode="x unified",
            xaxis_title="Frequência (GHz)",
            yaxis_title="Magnitude (dB)" if quantity == "Magnitude" else "Fase (graus)",
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=35, r=25, t=65, b=40),
        )
        figures.append(fig)

    return figures


def create_legacy_csv_viewer() -> None:
    st.header("📂 Visualizador de CSVs VNA adicionais")
    st.caption(
        "Página isolada para arquivos antigos/externos. Aceita CSV separado por vírgula, "
        "ponto e vírgula ou espaços, com dados db/ang ou re/im. Também aceita ZIP com vários CSVs."
    )

    uploads = st.file_uploader(
        "Carregue CSVs ou um ZIP contendo CSVs",
        type=["csv", "zip"],
        accept_multiple_files=True,
        key="legacy_vna_uploads",
    )
    if not uploads:
        st.info("Selecione os arquivos para começar.")
        return

    expanded, expansion_errors = _expand_uploads(uploads)
    for error in expansion_errors:
        st.error(error)

    parsed = []
    for name, data in expanded:
        try:
            df, traces, fmt = parse_vna_csv(data, name)
            parsed.append({"name": name, "df": df, "traces": traces, "format": fmt})
        except Exception as exc:
            st.error(f"{name}: {exc}")

    if not parsed:
        return

    file_names = [item["name"] for item in parsed]
    selected_files = st.multiselect(
        "Arquivos a visualizar",
        file_names,
        default=file_names,
        key="legacy_selected_files",
    )
    if not selected_files:
        st.warning("Selecione pelo menos um arquivo.")
        return

    selected_parsed = [item for item in parsed if item["name"] in selected_files]
    labels = _trace_labels(selected_parsed)

    c1, c2, c3 = st.columns(3)
    with c1:
        s_choice = st.radio(
            "Parâmetro S",
            ["S11", "S21", "S11 + S21"],
            index=2,
            horizontal=True,
            key="legacy_s_choice",
        )
    with c2:
        quantity = st.radio(
            "Grandeza",
            ["Magnitude", "Fase", "Magnitude + Fase"],
            horizontal=True,
            key="legacy_quantity",
        )
    with c3:
        split_by_file = st.checkbox(
            "Um gráfico por arquivo",
            value=False,
            key="legacy_split_by_file",
        )

    allowed = {"S11": {"S11"}, "S21": {"S21"}, "S11 + S21": {"S11", "S21"}}[s_choice]
    compatible = [label for label in labels if label.rsplit("_", 1)[1] in allowed]
    chosen_labels = st.multiselect(
        "Traços",
        compatible,
        default=_default_labels(compatible),
        key=f"legacy_trace_labels_{s_choice}",
    )
    if not chosen_labels:
        st.warning("Selecione pelo menos um traço.")
        return

    freq_min = min(float(item["df"]["frequency_hz"].min()) for item in selected_parsed) / 1e9
    freq_max = max(float(item["df"]["frequency_hz"].max()) for item in selected_parsed) / 1e9
    if freq_max > freq_min:
        fmin, fmax = st.slider(
            "Faixa de frequência (GHz)",
            min_value=freq_min,
            max_value=freq_max,
            value=(freq_min, freq_max),
            key="legacy_frequency_range",
        )
    else:
        fmin, fmax = freq_min, freq_max

    unwrap_phase = st.checkbox(
        "Desembrulhar fase",
        value=False,
        key="legacy_unwrap_phase",
    )

    quantities = ["Magnitude", "Fase"] if quantity == "Magnitude + Fase" else [quantity]
    for q in quantities:
        for fig in _plot(
            selected_parsed,
            selected_files,
            chosen_labels,
            q,
            fmin,
            fmax,
            unwrap_phase,
            split_by_file,
        ):
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Resumo dos arquivos detectados"):
        rows = []
        for item in parsed:
            df = item["df"]
            rows.append(
                {
                    "Arquivo": item["name"],
                    "Formato": item["format"],
                    "Pontos": len(df),
                    "f inicial (GHz)": float(df["frequency_hz"].min() / 1e9),
                    "f final (GHz)": float(df["frequency_hz"].max() / 1e9),
                    "S11": ", ".join(item["traces"]["S11"].keys()) or "—",
                    "S21": ", ".join(item["traces"]["S21"].keys()) or "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
