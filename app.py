from __future__ import annotations

import base64
import importlib
import mimetypes
import sys
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src import utils as app_utils


def _module_with_attrs(module, required_attrs: tuple[str, ...]):
    if all(hasattr(module, attr) for attr in required_attrs):
        return module
    name = getattr(module, "__name__", "")
    if name and sys.modules.get(name) is module:
        return importlib.reload(module)
    if name:
        return importlib.import_module(name)
    return module


app_utils = _module_with_attrs(app_utils, ("mode_key", "sensitivity_mode_title"))

from src import plotting as app_plotting
from src import predict as app_predict
from src import preprocess as app_preprocess
from src import sensitivity as app_sensitivity

app_sensitivity = _module_with_attrs(app_sensitivity, ("generate_sensitivity_heatmap", "mode_key"))

plot_heatmap = app_plotting.plot_heatmap
load_model_bundle = app_predict.load_model_bundle
predict_single = app_predict.predict_single
generate_sensitivity_heatmap = app_sensitivity.generate_sensitivity_heatmap

ANALYSIS_MODES = app_utils.ANALYSIS_MODES
BOUNDARY_OPTIONS = app_utils.BOUNDARY_OPTIONS
LOGO_PATH = app_utils.LOGO_PATH
MECHANICS_WORKFLOW_PATH = app_utils.MECHANICS_WORKFLOW_PATH
OUTPUT_LABELS = app_utils.OUTPUT_LABELS
REPRESENTATIVE_WALL_CASES_3D_PATH = app_utils.REPRESENTATIVE_WALL_CASES_3D_PATH
REPRESENTATIVE_WALL_CONFIGURATION_PATH = app_utils.ASSETS_DIR / "representative_wall_configuration.png"
WALL_PARAMETER_ICONS_DIR = app_utils.ASSETS_DIR / "wall_parameter_icons"
WALL_MECHANICS_PANELS_PATH = app_utils.WALL_MECHANICS_PANELS_PATH
case_from_quality = app_utils.case_from_quality
constraint_violations = app_utils.constraint_violations
house_image_for_boundary = app_utils.house_image_for_boundary
normalize_case = app_utils.normalize_case
quality_level_table = app_utils.quality_level_table
mode_key = getattr(
    app_utils,
    "mode_key",
    lambda mode: {
        "BC vs Wall case index": "bc_vs_wall_case",
        "Z vs Wall case index": "z_vs_wall_case",
        "L vs Wall case index": "l_vs_wall_case",
        "Nf vs Wall case index": "nf_vs_wall_case",
    }.get(str(mode).strip(), str(mode).strip()),
)
valid_quality_levels_for_cases = app_utils.valid_quality_levels_for_cases

MODEL_CACHE_VERSION = "2026-06-27-hgb-beta-l4"
HEATMAP_OUTPUT_METRICS = ["mu", "beta", "mu_over_z", "probability", "probability_relative"]


st.set_page_config(
    page_title="Machine Learning Based Masonry Wall Sensitivity Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-size: clamp(1.65rem, 2.4vw, 2.35rem) !important;
        line-height: 1.1 !important;
        margin-bottom: 0.45rem !important;
    }
    h2, h3 {
        margin-top: 0.75rem;
    }
    div[data-testid="stImageContainer"] img {
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain;
    }
    div[data-testid="stMetric"] {
        padding: 0.25rem 0;
    }
    div[data-testid="stExpander"] {
        margin: 0.4rem 0 1rem;
    }
    .full-width-figure-wrap,
    .mechanics-image-wrap {
        width: 100%;
        margin: 0.35rem 0 1rem;
        padding: 0.35rem;
        border: 1px solid rgba(22, 51, 95, 0.18);
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.04);
    }
    .full-width-figure-img,
    .mechanics-img {
        display: block;
        width: 100%;
        height: auto;
    }
    .image-caption {
        color: #4b5563;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 0.3rem;
    }
    .config-summary-card {
        border: 1px solid rgba(22, 51, 95, 0.28);
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.05);
        margin: 0.75rem 0 0.65rem;
    }
    .config-summary-title {
        background: #0d376d;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0;
        padding: 0.55rem 0.8rem;
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .wall-case-chip {
        border: 1px solid rgba(255, 255, 255, 0.42);
        border-radius: 999px;
        padding: 0.16rem 0.52rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ffffff;
        background: rgba(255, 255, 255, 0.12);
    }
    .config-summary-body {
        display: grid;
        grid-template-columns: minmax(0, 2.15fr) minmax(220px, 0.85fr);
        gap: 0;
    }
    .config-table-scroll {
        overflow-x: auto;
        border-right: 1px solid rgba(22, 51, 95, 0.16);
    }
    .config-summary-table {
        width: 100%;
        min-width: 560px;
        border-collapse: collapse;
        table-layout: fixed;
    }
    .config-summary-table th,
    .config-summary-table td {
        text-align: center;
        padding: 0.65rem 0.45rem;
        border-right: 1px solid rgba(22, 51, 95, 0.14);
    }
    .config-summary-table th:last-child,
    .config-summary-table td:last-child {
        border-right: 0;
    }
    .config-summary-table th {
        color: #111827;
        font-size: 0.98rem;
        font-weight: 700;
        border-bottom: 1px solid rgba(22, 51, 95, 0.20);
        background: #f8fafc;
    }
    .config-summary-table td {
        font-size: 0.98rem;
        color: #111827;
    }
    .config-symbol {
        display: block;
        font-style: italic;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.15;
        margin-top: 0.04rem;
    }
    .config-symbol sub,
    .config-legend-symbol sub,
    .field-label sub {
        font-size: 0.72em;
        line-height: 0;
        vertical-align: sub;
    }
    .config-icon {
        width: 40px;
        height: 40px;
        object-fit: contain;
        display: block;
        margin: 0 auto 0.18rem;
        opacity: 0.92;
    }
    .config-unit {
        display: block;
        color: #4b5563;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 0.12rem;
    }
    .config-legend {
        padding: 0.62rem 0.78rem;
        background: #fbfdff;
    }
    .config-legend-title {
        color: #0d376d;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .config-legend-row {
        display: grid;
        grid-template-columns: 2.15rem 1fr;
        gap: 0.5rem;
        align-items: baseline;
        margin: 0.14rem 0;
        font-size: 0.82rem;
        line-height: 1.22;
    }
    .config-legend-symbol {
        color: #111827;
        font-style: italic;
        font-weight: 700;
        text-align: right;
    }
    .field-label {
        color: rgb(49, 51, 63);
        font-size: 0.875rem;
        font-weight: 400;
        line-height: 1.35;
        margin: 0 0 0.22rem;
    }
    .current-config-summary-body {
        align-items: stretch;
    }
    .current-config-table-panel {
        display: flex;
        align-items: stretch;
        min-height: 100%;
    }
    .current-config-grid {
        width: 100%;
        min-width: 560px;
        height: 100%;
        min-height: 222px;
        display: grid;
        grid-template-rows: minmax(0, 0.68fr) minmax(0, 0.32fr);
    }
    .current-config-icon-row,
    .current-config-value-row {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
    }
    .current-config-cell {
        text-align: center;
        border-right: 1px solid rgba(22, 51, 95, 0.14);
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 0;
    }
    .current-config-cell:last-child {
        border-right: 0;
    }
    .current-config-icon-row .current-config-cell {
        flex-direction: column;
        padding: 0.62rem 0.4rem 0.56rem;
        border-bottom: 1px solid rgba(22, 51, 95, 0.20);
        background: #f8fafc;
        color: #111827;
        font-size: 0.98rem;
        font-weight: 700;
    }
    .current-config-value-row .current-config-cell {
        padding: 0.48rem 0.45rem 0.52rem;
        font-size: 0.94rem;
        line-height: 1.22;
        font-weight: 500;
        color: #111827;
    }
    .current-config-grid .config-icon {
        width: 42px;
        height: 42px;
        margin-bottom: 0.2rem;
    }
    .symbol-definitions-panel {
        height: 100%;
        box-sizing: border-box;
    }
    @media (max-width: 1000px) {
        .config-summary-body {
            grid-template-columns: 1fr;
        }
        .config-table-scroll {
            border-right: 0;
            border-bottom: 1px solid rgba(22, 51, 95, 0.16);
        }
        .current-config-table-panel {
            display: block;
        }
        .current-config-grid {
            min-height: 218px;
        }
        .symbol-definitions-panel {
            height: auto;
        }
    }
    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        h1 {
            font-size: 1.45rem !important;
            line-height: 1.12 !important;
        }
        .config-summary-table {
            min-width: 520px;
        }
        .current-config-grid {
            min-height: 205px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_model_bundle(cache_version: str):
    return load_model_bundle()


def show_image(path: Path, caption: str | None = None, *, width: int | None = None) -> None:
    if path.exists():
        # Embed assets so rendering does not depend on the cloud media endpoint.
        st.image(image_data_uri(path), caption=caption, width=width or "stretch")
    else:
        st.warning(f"Missing image: {path}")


def show_centered_image(path: Path, caption: str | None = None, *, width: int = 900) -> None:
    left, center, right = st.columns([0.08, 0.84, 0.08])
    with center:
        show_image(path, caption=caption, width=width)


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    media_type, _ = mimetypes.guess_type(path.name)
    return f"data:{media_type or 'application/octet-stream'};base64,{encoded}"


def show_full_width_figure(path: Path, caption: str | None = None) -> None:
    if not path.exists():
        st.warning(f"Missing image: {path}")
        return
    caption_html = f'<div class="image-caption">{escape(caption)}</div>' if caption else ""
    st.markdown(
        f"""
<div class="full-width-figure-wrap">
    <img class="full-width-figure-img" src="{image_data_uri(path)}" alt="{escape(caption or '')}">
    {caption_html}
</div>
""",
        unsafe_allow_html=True,
    )


def format_display_number(value: object, *, decimals: int = 2, trim: bool = True) -> str:
    if pd.isna(value):
        return "N/A"
    text = f"{float(value):.{decimals}f}"
    if trim:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def render_field_label(label_html: str) -> None:
    st.markdown(f'<div class="field-label">{label_html}</div>', unsafe_allow_html=True)


def _safe_number_token(value: float, *, decimals: int = 2, keep_trailing: bool = False) -> str:
    text = f"{float(value):.{decimals}f}"
    if not keep_trailing:
        text = text.rstrip("0").rstrip(".")
        if "." not in text:
            text = f"{text}.0"
    return text.replace("-", "minus_").replace(".", "p").lower()


def _safe_text_token(value: str) -> str:
    text = str(value).strip()
    if "." in text:
        prefix, rest = text.split(".", 1)
        if prefix.strip().isdigit():
            text = rest.strip()
    text = text.lower().replace("/", " over ").replace("-", " ")
    chars: list[str] = []
    previous_is_separator = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_is_separator = False
        elif not previous_is_separator:
            chars.append("_")
            previous_is_separator = True
    return "".join(chars).strip("_") or "sensitivity"


def heatmap_csv_filename(
    output_key: str,
    *,
    mode: str,
    flood_depth: float,
    relative_flood_depth: float,
) -> str:
    mode_token = _safe_text_token(mode)
    if output_key == "probability":
        token = f"probability_h_{_safe_number_token(flood_depth)}m"
    elif output_key == "probability_relative":
        token = f"probability_h_over_z_{_safe_number_token(relative_flood_depth, keep_trailing=True)}"
    else:
        token = output_key.lower().replace("/", "_").replace(" ", "_")
    return f"strength_heatmap_{mode_token}_{token}.csv"


def display_frame(
    frame: pd.DataFrame,
    *,
    decimals: int = 2,
    probability_columns: set[str] | None = None,
    integer_columns: set[str] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    probability_columns = probability_columns or set()
    integer_columns = integer_columns or set()
    for col in out.columns:
        if col in integer_columns:
            out[col] = out[col].map(lambda value: "N/A" if pd.isna(value) else f"{int(value)}")
        elif pd.api.types.is_numeric_dtype(out[col]):
            if col in probability_columns:
                out[col] = out[col].map(lambda value: format_display_number(value, decimals=3, trim=False))
            else:
                out[col] = out[col].map(lambda value: format_display_number(value, decimals=decimals, trim=True))
    return out


def render_configuration_summary(case: dict[str, object]) -> None:
    normalized = normalize_case(case)
    icon_paths = {
        "t": WALL_PARAMETER_ICONS_DIR / "01_thickness_t.png",
        "Z": WALL_PARAMETER_ICONS_DIR / "02_wall_height_Z.png",
        "rho": WALL_PARAMETER_ICONS_DIR / "03_density_rho.png",
        "O": WALL_PARAMETER_ICONS_DIR / "04_opening_ratio_o.png",
        "L": WALL_PARAMETER_ICONS_DIR / "05_span_length_L.png",
        "BC": WALL_PARAMETER_ICONS_DIR / "06_boundary_condition_BC.png",
        "Nf": WALL_PARAMETER_ICONS_DIR / "07_upper_floors_Nf.png",
    }
    columns = [
        ("t", "(m)", format_display_number(normalized["thickness"]), icon_paths["t"]),
        ("Z", "(m)", format_display_number(normalized["Z"]), icon_paths["Z"]),
        ("&rho;", "(kg/m&sup3;)", format_display_number(normalized["density"]), icon_paths["rho"]),
        ("O", "(&ndash;)", format_display_number(normalized["opening"]), icon_paths["O"]),
        ("L", "(m)", format_display_number(normalized["length"]), icon_paths["L"]),
        ("BC", "(&ndash;)", str(normalized["boundary"]), icon_paths["BC"]),
        ("N<sub>f</sub>", "(&ndash;)", str(normalized["n_upper"]), icon_paths["Nf"]),
    ]
    legend = [
        ("t", "Wall thickness"),
        ("Z", "Wall height"),
        ("&rho;", "Wall density"),
        ("O", "Opening ratio"),
        ("L", "Wall length/span"),
        ("BC", "Boundary condition"),
        ("N<sub>f</sub>", "Number of upper floors"),
        ("h", "Flood depth"),
        ("&mu;", "Predicted fragility median"),
        ("&beta;", "Predicted fragility dispersion"),
    ]
    wall_case_index = normalized.get("level")
    wall_case_chip = (
        f'<span class="wall-case-chip">Wall case index: {escape(str(int(wall_case_index)))}</span>'
        if wall_case_index is not None
        else ""
    )
    header_cells = "".join(
        (
            '<div class="current-config-cell">'
            f'<img class="config-icon" src="{image_data_uri(icon_path)}" alt="">'
            f'<span class="config-symbol">{symbol}</span>'
            f'<span class="config-unit">{unit}</span>'
            "</div>"
        )
        for symbol, unit, _, icon_path in columns
    )
    value_cells = "".join(f'<div class="current-config-cell">{escape(value)}</div>' for _, _, value, _ in columns)
    legend_rows = "".join(
        (
            '<div class="config-legend-row">'
            f'<span class="config-legend-symbol">{symbol}</span>'
            f"<span>{escape(description)}</span>"
            "</div>"
        )
        for symbol, description in legend
    )

    st.markdown(
        f"""
<div class="config-summary-card">
    <div class="config-summary-title"><span>CURRENT CONFIGURATION SUMMARY</span>{wall_case_chip}</div>
    <div class="config-summary-body current-config-summary-body">
        <div class="config-table-scroll current-config-table-panel">
            <div class="current-config-grid">
                <div class="current-config-icon-row">{header_cells}</div>
                <div class="current-config-value-row">{value_cells}</div>
            </div>
        </div>
        <div class="config-legend symbol-definitions-panel">
            <div class="config-legend-title">Symbol Definitions</div>
            {legend_rows}
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def compact_model_frame() -> pd.DataFrame:
    frame = bundle.summary_frame().loc[:, ["Index", "Model", "output", "R2"]].copy()
    frame["output"] = frame["output"].map({"mu": "\u03bc", "beta": "\u03b2"}).fillna(frame["output"])
    return frame.rename(columns={"output": "Output", "R2": "R\u00b2"})


def render_representative_wall_cases_table() -> None:
    frame = quality_level_table()
    columns = [
        ("Wall case index", "", "", lambda row: str(int(row["wall_case_index"]))),
        ("&rho;", "(kg/m&sup3;)", image_data_uri(WALL_PARAMETER_ICONS_DIR / "03_density_rho.png"), lambda row: format_display_number(row["density"])),
        ("O", "(&ndash;)", image_data_uri(WALL_PARAMETER_ICONS_DIR / "04_opening_ratio_o.png"), lambda row: format_display_number(row["opening"])),
        ("t", "(m)", image_data_uri(WALL_PARAMETER_ICONS_DIR / "01_thickness_t.png"), lambda row: format_display_number(row["thickness"])),
    ]
    header_cells = "".join(
        (
            "<th>"
            + (f'<img class="config-icon" src="{icon}" alt="">' if icon else "")
            + f'<span class="config-symbol">{symbol}</span>'
            + (f'<span class="config-unit">{unit}</span>' if unit else "")
            + "</th>"
        )
        for symbol, unit, icon, _ in columns
    )
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{escape(value_getter(row))}</td>" for _, _, _, value_getter in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"""
<div class="config-summary-card">
    <div class="config-summary-title"><span>REPRESENTATIVE WALL CASES</span></div>
    <div class="config-table-scroll">
        <table class="config-summary-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_matrix_table(frame: pd.DataFrame, *, probability_columns: set[str] | None = None) -> None:
    display = display_frame(frame, probability_columns=probability_columns)
    index_label = display.index.name or ""
    if index_label == "$N_f$":
        index_label = "N<sub>f</sub>"
    else:
        index_label = escape(str(index_label))
    header_cells = f"<th>{index_label}</th>" + "".join(f"<th>{escape(str(col))}</th>" for col in display.columns)
    body_rows = []
    for idx, row in display.iterrows():
        cells = f"<td>{escape(str(idx))}</td>" + "".join(f"<td>{escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"""
<div class="config-summary-card">
    <div class="config-table-scroll">
        <table class="config-summary-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def model_column_config() -> dict[str, object]:
    return {"R\u00b2": st.column_config.NumberColumn("R\u00b2", format="%.3f")}


def format_duration(seconds: float) -> str:
    total = int(round(float(seconds)))
    if total < 60:
        return f"{format_display_number(seconds, decimals=2)} s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {sec} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def uses_p1(boundaries: list[str]) -> bool:
    return any(str(boundary).startswith("P1") for boundary in boundaries)


try:
    bundle = cached_model_bundle(MODEL_CACHE_VERSION)
except Exception as exc:
    st.error("The ML pretrained models could not be loaded.")
    st.exception(exc)
    st.stop()


logo_col, intro_col = st.columns([0.3, 0.7])
with logo_col:
    show_image(LOGO_PATH, width=330)
with intro_col:
    st.title("Machine Learning Based Masonry Wall Sensitivity Analysis")
    st.markdown(
        r"""
This web app uses pretrained machine learning models to rapidly estimate flood fragility for masonry walls. The user selects the wall geometry, material properties, boundary condition, and number of upper floors, and the app predicts the median collapse flood depth $\mu$, the dispersion parameter $\beta$, and collapse probabilities for selected flood-depth scenarios.

The app includes two main sections. **Single Prediction** evaluates one wall configuration at a time, while **Sensitivity Heatmap** compares many configurations at once. In the heatmap, the user can choose which result to display: median collapse flood depth, fragility dispersion, normalized median collapse depth, collapse probability at a fixed flood depth, or collapse probability at a flood depth defined as a fraction of wall height. The heatmap results can also be exported as a CSV file.
"""
    )

if st.toggle("Project workflow and model context", value=False):
    show_full_width_figure(MECHANICS_WORKFLOW_PATH)

with st.sidebar:
    st.header("Model")
    st.success("Loaded pretrained machine learning surrogate models")
    st.dataframe(
        compact_model_frame(),
        hide_index=True,
        width="stretch",
        column_config=model_column_config(),
    )
    st.info(
        "This web app is intended for research and demonstration purposes. Predictions are "
        "valid only within the parameter ranges and assumptions considered in the underlying study "
        "and training dataset."
    )

tab_single, tab_sensitivity, tab_reference, tab_model = st.tabs(
    ["Single case prediction", "Sensitivity heatmap", "Wall case indices and images", "Model details"]
)

with tab_single:
    visual_col, input_col = st.columns([1.2, 0.8])
    with visual_col:
        show_image(REPRESENTATIVE_WALL_CONFIGURATION_PATH, width=760)
    with input_col:
        st.markdown("**Configuration Inputs**")
        single_boundary = st.selectbox("Boundary condition, BC", BOUNDARY_OPTIONS, index=2, key="single_boundary")
        single_is_p1 = single_boundary.startswith("P1")
        if single_is_p1 and st.session_state.get("single_n_upper", 0) != 0:
            st.session_state["single_n_upper"] = 0
        render_field_label("Number of upper floors, <em>N</em><sub>f</sub>")
        single_n_upper = st.number_input(
            "Number of upper floors",
            min_value=0,
            max_value=0 if single_is_p1 else 2,
            value=0 if single_is_p1 else 2,
            step=1,
            key="single_n_upper",
            disabled=single_is_p1,
            help="P1 constrained cases require zero upper floors.",
            label_visibility="collapsed",
        )
        single_length = st.number_input("Wall length/span, L (m)", min_value=3.0, max_value=12.0, value=5.0, step=0.5, format="%g", key="single_length")
        single_floor_height = st.number_input("Wall height, Z (m)", min_value=2.4, max_value=4.0, value=3.0, step=0.1, format="%g", key="single_floor_height")
        single_flood_depth = st.number_input(
            "Flood depth, h (m)",
            min_value=0.05,
            value=1.5,
            step=0.05,
            format="%g",
            key="single_flood_depth",
        )
        single_effective_n_upper = int(single_n_upper)
        valid_single_levels = valid_quality_levels_for_cases(
            [
                {
                    "Z": single_floor_height,
                    "length": single_length,
                    "n_upper": single_effective_n_upper,
                    "boundary": single_boundary,
                }
            ]
        )
        if not valid_single_levels:
            st.error("No wall case index satisfies the constrained model rules for this single case setup.")
            st.stop()
        single_level = st.selectbox("Wall case index", valid_single_levels, index=min(3, len(valid_single_levels) - 1), key="single_level")
        if len(valid_single_levels) < 10:
            st.caption(f"Allowed wall case indices for this setup: {', '.join(map(str, valid_single_levels))}")

    single_case = case_from_quality(
        single_level,
        Z=single_floor_height,
        length=single_length,
        n_upper=single_effective_n_upper,
        boundary=single_boundary,
    )
    errors = constraint_violations(single_case)
    if errors:
        st.error("\n".join(f"- {message}" for message in errors))
        st.stop()

    prediction = predict_single(bundle, single_case, flood_depth=single_flood_depth)
    render_configuration_summary(single_case)

    st.markdown("**Prediction Results**")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted \u03bc (m)", format_display_number(prediction["mu"]))
    r2.metric("Predicted \u03b2", format_display_number(prediction["beta"]))
    r3.metric(f"P(collapse | h={format_display_number(single_flood_depth)} m)", f"{prediction['probability']:.3f}")

with tab_sensitivity:
    st.subheader("Sensitivity Heatmap")

    top_left, top_right = st.columns([2.3, 0.9])
    with top_left:
        mode = st.selectbox("Sensitivity mode", ANALYSIS_MODES, index=0)
        sensitivity_mode_key = mode_key(mode)
        output_key = st.selectbox(
            "Heatmap output metric",
            HEATMAP_OUTPUT_METRICS,
            format_func=lambda key: OUTPUT_LABELS[key],
        )
        st.markdown(
            r"""
The median collapse flood depth $\mu$ is expressed in metres. It represents the flood depth at which the selected wall configuration reaches 50% probability of collapse. When comparing walls with different heights $Z$, $\mu$ should be interpreted together with the normalized value $\mu/Z$, which expresses the median collapse depth as a fraction of wall height.

The dispersion parameter $\beta$ describes the spread of the lognormal fragility curve. It is not a direct measure of wall capacity. Larger $\beta$ values indicate a more gradual transition from low to high collapse probability, meaning greater uncertainty or variability in the predicted fragility response.

The probability metrics provide the most direct risk interpretation. $P(C \mid h)$ answers: "What is the probability of collapse at a selected flood depth?" $P(C \mid h/Z)$ answers: "What is the probability of collapse when the floodwater reaches a selected fraction of the wall height?" For example, $P(C \mid h/Z=0.50)$ evaluates the collapse probability when the water depth is equal to 50% of the wall height.
"""
        )
        heatmap_flood_depth = 2.0
        heatmap_relative_depth = 0.50
        if output_key == "probability":
            heatmap_flood_depth = st.slider(
                "Selected flood depth, h (m)",
                min_value=0.05,
                max_value=4.0,
                value=2.0,
                step=0.05,
                format="%g",
            )
        elif output_key == "probability_relative":
            heatmap_relative_depth = st.slider(
                "Selected relative flood depth, h/Z (-)",
                min_value=0.10,
                max_value=1.20,
                value=0.50,
                step=0.05,
                format="%.2f",
            )
        annotate_heatmap = st.checkbox("Show numeric values in heatmap cells", value=False)
    with top_right:
        show_image(WALL_MECHANICS_PANELS_PATH, width=300)

    hm_boundary = BOUNDARY_OPTIONS[2]
    hm_n_upper = 2
    hm_length = 5.0
    hm_floor = 3.0

    fixed_controls = []
    if sensitivity_mode_key != "bc_vs_wall_case":
        fixed_controls.append("boundary")
    if sensitivity_mode_key != "nf_vs_wall_case":
        fixed_controls.append("n_upper")
    if sensitivity_mode_key != "l_vs_wall_case":
        fixed_controls.append("length")
    if sensitivity_mode_key != "z_vs_wall_case":
        fixed_controls.append("floor")

    if fixed_controls:
        st.markdown("**Fixed parameters**")
        fixed_columns = st.columns(len(fixed_controls))
        for control_name, column in zip(fixed_controls, fixed_columns):
            with column:
                if control_name == "boundary":
                    hm_boundary = st.selectbox("Fixed boundary condition, BC", BOUNDARY_OPTIONS, index=2, key="hm_boundary")
                elif control_name == "n_upper":
                    hm_is_p1_for_upper_floor = hm_boundary.startswith("P1")
                    render_field_label("Fixed <em>N</em><sub>f</sub>")
                    hm_n_upper = st.number_input(
                        "Fixed number of upper floors",
                        min_value=0,
                        max_value=0 if hm_is_p1_for_upper_floor else 2,
                        value=0 if hm_is_p1_for_upper_floor else 2,
                        step=1,
                        key="hm_fixed_n_upper_p1" if hm_is_p1_for_upper_floor else "hm_fixed_n_upper",
                        disabled=hm_is_p1_for_upper_floor,
                        help="P1 constrained cases require zero upper floors.",
                        label_visibility="collapsed",
                    )
                elif control_name == "length":
                    hm_length = st.number_input(
                        "Fixed L (m)",
                        min_value=3.0,
                        max_value=12.0,
                        value=5.0,
                        step=0.5,
                        format="%g",
                        key="hm_fixed_length",
                    )
                elif control_name == "floor":
                    hm_floor = st.number_input(
                        "Fixed Z (m)",
                        min_value=2.4,
                        max_value=4.0,
                        value=3.0,
                        step=0.1,
                        format="%g",
                        key="hm_fixed_z",
                    )
    varied_notes = {
        "bc_vs_wall_case": "Varied by the selected mode: BC and wall case index.",
        "z_vs_wall_case": "Varied by the selected mode: Z and wall case index.",
        "l_vs_wall_case": "Varied by the selected mode: L and wall case index.",
        "nf_vs_wall_case": "Varied by the selected mode: $N_f$ and wall case index.",
    }
    st.caption(varied_notes.get(sensitivity_mode_key, "Wall case index is varied by the selected heatmap columns."))

    hm_is_p1 = hm_boundary.startswith("P1")
    floor_min = 2.5
    floor_max = 3.5
    length_min = 3.0
    length_max = 10.0
    length_step = 0.5
    n_upper_min = 0
    n_upper_max = 2
    selected_boundaries = BOUNDARY_OPTIONS

    if sensitivity_mode_key == "bc_vs_wall_case":
        boundary_default = BOUNDARY_OPTIONS
        selected_boundaries = st.multiselect("BC set", BOUNDARY_OPTIONS, default=boundary_default)
        if int(hm_n_upper) > 0 and uses_p1(selected_boundaries):
            st.info(
                "For BC vs wall case index heatmaps, P1 is included with zero upper floors; "
                f"the other selected BC values use $N_f$ = {int(hm_n_upper)}."
            )
    elif sensitivity_mode_key == "z_vs_wall_case":
        r1, r2 = st.columns(2)
        with r1:
            floor_min = st.number_input("Z range min (m)", min_value=2.4, max_value=4.0, value=2.5, step=0.1, format="%g")
        with r2:
            floor_max = st.number_input("Z range max (m)", min_value=2.4, max_value=4.0, value=3.5, step=0.1, format="%g")
    elif sensitivity_mode_key == "l_vs_wall_case":
        r1, r2, r3 = st.columns(3)
        with r1:
            length_min = st.number_input("L range min (m)", min_value=3.0, max_value=12.0, value=3.0, step=0.5, format="%g")
        with r2:
            length_max = st.number_input("L range max (m)", min_value=3.0, max_value=12.0, value=10.0, step=0.5, format="%g")
        with r3:
            length_step = st.number_input("L range step (m)", min_value=0.1, value=0.5, step=0.1, format="%g")
    elif sensitivity_mode_key == "nf_vs_wall_case":
        upper_floor_max = 0 if hm_is_p1 else 2
        r1, r2 = st.columns(2)
        with r1:
            render_field_label("<em>N</em><sub>f</sub> min")
            n_upper_min = st.number_input(
                "Minimum number of upper floors",
                min_value=0,
                max_value=upper_floor_max,
                value=0,
                step=1,
                key="hm_upper_min_p1" if hm_is_p1 else "hm_upper_min",
                disabled=hm_is_p1,
                help="P1 constrained cases require zero upper floors.",
                label_visibility="collapsed",
            )
        with r2:
            render_field_label("<em>N</em><sub>f</sub> max")
            n_upper_max = st.number_input(
                "Maximum number of upper floors",
                min_value=0,
                max_value=upper_floor_max,
                value=upper_floor_max,
                step=1,
                key="hm_upper_max_p1" if hm_is_p1 else "hm_upper_max",
                disabled=hm_is_p1,
                help="P1 constrained cases require zero upper floors.",
                label_visibility="collapsed",
            )

    heatmap_templates = []
    if sensitivity_mode_key == "bc_vs_wall_case":
        heatmap_templates = [
            {"Z": hm_floor, "length": hm_length, "n_upper": 0 if b.startswith("P1") else int(hm_n_upper), "boundary": b}
            for b in selected_boundaries
        ]
    elif sensitivity_mode_key == "z_vs_wall_case":
        y_values = [round(x * 0.1, 10) for x in range(int(round(floor_min * 10)), int(round(floor_max * 10)) + 1)]
        heatmap_templates = [{"Z": z, "length": hm_length, "n_upper": int(hm_n_upper), "boundary": hm_boundary} for z in y_values]
    elif sensitivity_mode_key == "l_vs_wall_case":
        n_steps = int(round((length_max - length_min) / length_step)) if length_step > 0 else 0
        y_values = [round(length_min + idx * length_step, 10) for idx in range(n_steps + 1)]
        heatmap_templates = [{"Z": hm_floor, "length": length, "n_upper": int(hm_n_upper), "boundary": hm_boundary} for length in y_values]
    elif sensitivity_mode_key == "nf_vs_wall_case":
        heatmap_templates = [{"Z": hm_floor, "length": hm_length, "n_upper": nup, "boundary": hm_boundary} for nup in range(int(n_upper_min), int(n_upper_max) + 1)]

    valid_heatmap_levels = valid_quality_levels_for_cases(heatmap_templates)
    if not valid_heatmap_levels:
        st.error("No wall case indices satisfy the constrained model rules for this heatmap setup.")
        st.stop()
    selected_levels = st.multiselect(
        "Selected wall case indices",
        valid_heatmap_levels,
        default=valid_heatmap_levels,
    )
    if len(valid_heatmap_levels) < 10:
        st.caption(f"Allowed wall case indices for this heatmap setup: {', '.join(map(str, valid_heatmap_levels))}")

    try:
        result = generate_sensitivity_heatmap(
            bundle,
            mode=sensitivity_mode_key,
            levels=selected_levels,
            output_key=output_key,
            flood_depth=heatmap_flood_depth,
            relative_flood_depth=heatmap_relative_depth,
            length=hm_length,
            n_upper=int(hm_n_upper),
            boundary=hm_boundary,
            floor_height=hm_floor,
            floor_min=floor_min,
            floor_max=floor_max,
            length_min=length_min,
            length_max=length_max,
            length_step=length_step,
            n_upper_min=int(n_upper_min),
            n_upper_max=int(n_upper_max),
            selected_boundaries=selected_boundaries,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        if result.warnings:
            st.warning("\n".join(f"- {message}" for message in result.warnings))

        fig = plot_heatmap(
            matrix=result.matrix,
            y_values=result.y_values,
            levels=result.levels,
            y_label=result.y_label,
            value_label=result.value_label,
            title=result.title,
            output_key=result.output_key,
            annotate=annotate_heatmap,
        )
        with BytesIO() as image_buffer:
            fig.savefig(image_buffer, format="png", dpi=200, bbox_inches="tight")
            encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        st.image(f"data:image/png;base64,{encoded}", width="stretch")
        fig.clear()

        st.subheader("Numerical Heatmap Values")
        matrix_df = result.matrix_dataframe()
        st.caption(f"Displayed metric: {result.value_label}")
        render_matrix_table(matrix_df)
        long_df = result.long_dataframe()
        st.download_button(
            "Download heatmap values as CSV",
            data=long_df.to_csv(index=False).encode("utf-8"),
            file_name=heatmap_csv_filename(
                result.output_key,
                mode=result.mode,
                flood_depth=heatmap_flood_depth,
                relative_flood_depth=heatmap_relative_depth,
            ),
            mime="text/csv",
        )

with tab_reference:
    st.subheader("Representative Wall Cases")
    render_representative_wall_cases_table()
    show_full_width_figure(REPRESENTATIVE_WALL_CASES_3D_PATH, caption="Representative wall cases")

with tab_model:
    st.subheader("Model Details")
    st.markdown(
        r"""
### 1. Web app purpose

This web app is designed to rapidly estimate the flood fragility of masonry walls using pretrained machine learning models. Instead of running a time-consuming mechanics based Limit Analysis model each time, the app uses surrogate models that have already learned from a large set of simulated wall cases.

The user defines the main characteristics of a masonry wall, including wall thickness, wall height, wall length, masonry density, opening ratio, boundary condition, and number of upper floors. The app then predicts the main fragility parameters: the median collapse flood depth $\mu$ and the fragility dispersion $\beta$.

### 2. How to use the app

#### Single Prediction

The **Single Prediction** section evaluates one wall configuration at a time. The user changes the wall parameters, checks the current configuration summary, and obtains the predicted fragility response. The app can also estimate the probability of collapse for a selected flood depth. This helps answer practical questions such as: "What is the probability of collapse if the floodwater reaches 2 m?" or "What is the probability of collapse when the floodwater reaches 50% of the wall height?"

#### Sensitivity Heatmap

The **Sensitivity Heatmap** section compares many wall configurations at once. The user selects the sensitivity-analysis mode, for example boundary condition versus wall case index, wall height versus wall case index, wall length versus wall case index, or number of upper floors versus wall case index. The app then generates a heatmap showing how the selected output changes across the chosen range of values.

The heatmap can display the median collapse flood depth, the fragility dispersion, the normalized median collapse depth, the collapse probability at a fixed flood depth, or the collapse probability at a flood depth defined as a fraction of wall height. This makes the heatmap easier to read than listing only symbols.

#### Figures and visual aids

The figures and icons in the app help users understand the input parameters and model outputs. The representative wall image illustrates the meaning of the wall geometry and flood loading variables. The icon-based configuration summary visually shows the selected wall thickness, height, density, opening ratio, wall length, boundary condition, and number of upper floors. The representative wall-case image/table explains the predefined wall cases used by the model.

#### CSV export

After generating a sensitivity heatmap, the user can export the displayed numerical results as a CSV file. The downloaded file includes the selected sensitivity-analysis mode and selected output metric in the filename.
"""
    )
    st.markdown(
        """
### 3. Reference study

This web app was designed within the study:

Georgiadis, S., Nale, M., Faggioli, M., Nikolić, Ž., and Benvenuti, E.  
*Machine learning surrogates for predicting and interpreting flood fragility parameters of masonry walls*.

AI in Civil Engineering.

DOI: [10.1007/s43503-026-00112-x](https://doi.org/10.1007/s43503-026-00112-x).
"""
    )
    st.markdown(
        """
### 4. Model basis and computational efficiency

The app uses pretrained machine learning surrogate models for immediate inference. Compared with repeated evaluations of the mechanics based Limit Analysis model, the surrogate models provide substantially faster batch prediction while preserving the constrained input rules used to generate the training data.
"""
    )
    st.dataframe(
        compact_model_frame(),
        hide_index=True,
        width="stretch",
        column_config=model_column_config(),
    )
    efficiency_df = pd.DataFrame(
        [
            {
                "Number of wall configurations": 100,
                "Machine learning surrogate time": format_duration(0.03982695),
                "Limit Analysis model time": format_duration(135.19873432960586),
            },
            {
                "Number of wall configurations": 1000,
                "Machine learning surrogate time": format_duration(0.16877150),
                "Limit Analysis model time": format_duration(1351.9873432960585),
            },
            {
                "Number of wall configurations": 10000,
                "Machine learning surrogate time": format_duration(1.35431820),
                "Limit Analysis model time": format_duration(13519.873432960585),
            },
        ]
    )
    st.table(efficiency_df)
    st.markdown(
        r"""
### 5. Output interpretation

The app predicts the parameters of a lognormal flood-fragility function for the selected wall configuration.

The median collapse flood depth $\mu$ is expressed in metres. It represents the flood depth at which the selected wall configuration reaches 50% probability of collapse. Larger values of $\mu$ generally indicate higher median flood-depth capacity.

The dispersion parameter $\beta$ describes the spread of the lognormal fragility curve. It is not a direct measure of wall capacity. Larger values of $\beta$ indicate a more gradual transition from low to high collapse probability, meaning greater uncertainty or variability in the predicted fragility response.

For a given flood depth $h$, the probability of collapse is evaluated as:

$$
P(C \mid h)=\Phi\left(\frac{\ln(h)-\ln(\mu)}{\beta}\right)
$$

where $\Phi(\cdot)$ is the standard normal cumulative distribution function.

The probability-based outputs are useful for practical interpretation. $P(C \mid h)$ answers: "What is the probability of collapse at a selected flood depth?" For example, this can be used to evaluate collapse probability when the floodwater reaches 2 m.

$P(C \mid h/Z)$ answers: "What is the probability of collapse when the floodwater reaches a selected fraction of the wall height?" For example, $P(C \mid h/Z=0.50)$ evaluates the collapse probability when the water depth is equal to 50% of the wall height.

When comparing walls with different heights $Z$, the absolute median collapse depth $\mu$ should be interpreted together with the normalized value $\mu/Z$, which expresses the median collapse depth as a fraction of wall height.
"""
    )

st.caption(
    "Disclaimer: This web app is intended for research and demonstration purposes. "
    "Predictions are valid only within the parameter ranges and assumptions considered in the underlying study and training dataset."
)
