from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = APP_ROOT / "assets"
MODELS_DIR = APP_ROOT / "models"

LOGO_PATH = ASSETS_DIR / "logo_interreg_strength.jpg"
MASONRY_WALL_PATH = ASSETS_DIR / "MasonryWall.jpg"
WALL_3D_PATH = ASSETS_DIR / "3D wall.png"
HOUSE_PANELS_PATH = ASSETS_DIR / "house_panels_4subplots.png"
WALL_MECHANICS_PANELS_PATH = ASSETS_DIR / "wall_mechanics_panels.png"
REPRESENTATIVE_WALL_CASES_3D_PATH = ASSETS_DIR / "representative_wall_cases_3d.png"
MECHANICS_WORKFLOW_PATH = ASSETS_DIR / "mechanics_based_collapse_modelling.png"

OPENING_STD = np.array([0.29, 0.27, 0.25, 0.24, 0.23, 0.22, 0.22, 0.21, 0.21, 0.20], dtype=float)
DENSITY_STD = np.array([1600, 1620, 1650, 1675, 1700, 1750, 1800, 1850, 1900, 2000], dtype=float)
THICK_STD = np.array([0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42], dtype=float)

QUALITY_LEVELS = list(range(1, 11))
BOUNDARY_OPTIONS = ["P1 (1-edge)", "P2 (2-edges)", "P3 (3-edges)", "P4 (4-edges)"]
HOUSE_BOUNDARY_IMAGE_PATHS = {
    "P1": ASSETS_DIR / "House_c1_color.jpg",
    "P2": ASSETS_DIR / "House_c2_color.jpg",
    "P3": ASSETS_DIR / "House_c3_color.jpg",
    "P4": ASSETS_DIR / "House_c4_color.jpg",
}

ANALYSIS_MODES = [
    "1. Z - Wall case index",
    "2. BC - Wall case index",
    "3. L - Wall case index",
    "4. Number of upper floors - Wall case index",
]

OUTPUT_LABELS = {
    "mu": "Median collapse flood depth, \u03bc (m)",
    "beta": "Fragility dispersion, \u03b2 (-)",
    "mu_over_z": "Normalized median collapse depth, \u03bc/Z (-)",
    "probability": "Collapse probability, P(C | h) (%)",
    "probability_relative": "Collapse probability, P(C | h/Z) (%)",
}

MODEL_DOMAIN = {
    "thickness": (0.20, 0.50),
    "Z": (2.40, 4.00),
    "density": (1200.0, 2200.0),
    "opening": (0.10, 0.40),
    "length": (3.0, 12.0),
    "area": (9.0, 144.0),
    "n_upper": (0, 2),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def quality_level_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wall_case_index": QUALITY_LEVELS,
            "density": DENSITY_STD,
            "opening": OPENING_STD,
            "thickness": THICK_STD,
        }
    )


def level_properties(level: int) -> dict[str, float]:
    level_int = int(level)
    if level_int not in QUALITY_LEVELS:
        raise ValueError("Wall case index must be between 1 and 10.")
    idx = level_int - 1
    return {
        "level": level_int,
        "opening": float(OPENING_STD[idx]),
        "density": float(DENSITY_STD[idx]),
        "thickness": float(THICK_STD[idx]),
    }


def normalize_boundary(boundary: Any) -> str:
    value = str(boundary).strip()
    if not value:
        raise ValueError("Boundary condition is required.")
    for option in BOUNDARY_OPTIONS:
        if value == option or value.upper() == option[:2]:
            return option
        if value.upper().startswith(option[:2]):
            return option
    raise ValueError(f"Unknown boundary condition: {boundary!r}")


def boundary_key(boundary: Any) -> str:
    return normalize_boundary(boundary)[:2]


def house_image_for_boundary(boundary: Any) -> Path:
    return HOUSE_BOUNDARY_IMAGE_PATHS[boundary_key(boundary)]


def normalize_case(case: dict[str, Any], *, coerce_p1_n_upper: bool = False) -> dict[str, Any]:
    out = dict(case)
    out["boundary"] = normalize_boundary(out.get("boundary", ""))

    if "area" not in out and "length" in out:
        out["area"] = float(out["length"]) ** 2
    if "length" not in out and "area" in out:
        out["length"] = math.sqrt(float(out["area"]))

    for col in ["density", "opening", "thickness", "Z", "area", "length"]:
        if col not in out:
            raise ValueError(f"Missing required input: {col}")
        out[col] = float(out[col])

    if "n_upper" not in out:
        raise ValueError("Missing required input: number of upper floors")
    out["n_upper"] = int(round(float(out["n_upper"])))

    if coerce_p1_n_upper and boundary_key(out["boundary"]) == "P1":
        out["n_upper"] = 0

    return out


def case_from_quality(
    level: int,
    *,
    Z: float,
    length: float,
    n_upper: int,
    boundary: str,
) -> dict[str, Any]:
    props = level_properties(level)
    return normalize_case(
        {
            "density": props["density"],
            "opening": props["opening"],
            "thickness": props["thickness"],
            "Z": float(Z),
            "length": float(length),
            "area": float(length) ** 2,
            "n_upper": int(n_upper),
            "boundary": boundary,
            "level": int(level),
        }
    )


def max_upper_floors_for_thickness(thickness: float) -> int:
    t = float(thickness)
    if t < 0.22:
        return 0
    if t < 0.24:
        return 1
    return 2


def constraint_violations(case: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    try:
        normalized = normalize_case(case)
    except Exception as exc:
        return [str(exc)]

    t = normalized["thickness"]
    z = normalized["Z"]
    rho = normalized["density"]
    opening = normalized["opening"]
    length = normalized["length"]
    n_upper = normalized["n_upper"]
    bc = boundary_key(normalized["boundary"])
    x1 = length / z if z > 0 else math.inf
    x2 = t / z if z > 0 else math.inf

    checks = [
        ("\u03c1", rho, MODEL_DOMAIN["density"], "kg/m3"),
        ("O", opening, MODEL_DOMAIN["opening"], ""),
        ("t", t, MODEL_DOMAIN["thickness"], "m"),
        ("Z", z, MODEL_DOMAIN["Z"], "m"),
        ("L", length, MODEL_DOMAIN["length"], "m"),
        ("$N_f$", n_upper, MODEL_DOMAIN["n_upper"], ""),
    ]
    for name, value, (lo, hi), unit in checks:
        if not (lo <= value <= hi):
            suffix = f" {unit}" if unit else ""
            violations.append(f"{name} = {value:g}{suffix} is outside the constrained model range [{lo:g}, {hi:g}]{suffix}.")

    if bc == "P1" and n_upper != 0:
        violations.append("P1 constrained cases require $N_f$ = 0.")
    if not (0.05 <= x2 <= 0.35):
        violations.append(f"t/Z = {x2:g} is outside [0.05, 0.35].")
    if bc == "P3" and x1 > 4.0:
        violations.append(f"P3 L/Z = {x1:g} exceeds 4.0.")
    if bc == "P4" and x1 > 2.75:
        violations.append(f"P4 L/Z = {x1:g} exceeds 2.75.")
    if t < 0.22 and n_upper != 0:
        violations.append("t below 0.22 m is allowed only for $N_f$ = 0.")
    if t < 0.24 and n_upper > 1:
        violations.append("t below 0.24 m is allowed only for $N_f$ <= 1.")
    if n_upper >= 2 and t < 0.24:
        violations.append("$N_f$ >= 2 requires t >= 0.24 m.")
    if length > 10.0 and t < 0.28:
        violations.append("L > 10 m requires t >= 0.28 m.")
    if length > 10.0 and n_upper >= 1 and t < 0.30:
        violations.append("Loaded walls with L > 10 m require t >= 0.30 m.")
    if length > 10.0 and opening > 0.35:
        violations.append("L > 10 m requires O <= 0.35.")
    if opening > 0.35 and t < 0.24:
        violations.append("O > 0.35 requires t >= 0.24 m.")

    return violations


def valid_quality_levels_for_cases(case_templates: Iterable[dict[str, Any]]) -> list[int]:
    valid: list[int] = []
    templates = list(case_templates)
    for level in QUALITY_LEVELS:
        level_is_valid = True
        for template in templates:
            case = dict(template)
            case.update(level_properties(level))
            if constraint_violations(case):
                level_is_valid = False
                break
        if level_is_valid:
            valid.append(level)
    return valid


def mode_number(mode: str) -> str:
    return str(mode).split(".", 1)[0].strip()


def float_grid(start: float, stop: float, step: float) -> np.ndarray:
    start_f = float(start)
    stop_f = float(stop)
    step_f = float(step)
    if step_f <= 0:
        raise ValueError("Step must be positive.")
    if start_f > stop_f:
        raise ValueError("Range minimum must be less than or equal to range maximum.")
    decimals = max(0, int(math.ceil(-math.log10(step_f))) + 2) if step_f < 1 else 2
    return np.round(np.arange(start_f, stop_f + step_f * 0.5, step_f), decimals)


def format_axis_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value_f = float(value)
    text = f"{value_f:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def domain_warnings(case: dict[str, Any]) -> list[str]:
    raw = dict(case)
    warnings: list[str] = []
    try:
        normalized = normalize_case(case)
    except Exception as exc:
        return [str(exc)]

    raw_boundary = normalize_boundary(raw.get("boundary", normalized["boundary"]))
    raw_n_upper = int(round(float(raw.get("n_upper", normalized["n_upper"]))))
    if boundary_key(raw_boundary) == "P1" and raw_n_upper != 0:
        warnings.append("P1 constrained cases are valid only with $N_f$ = 0.")

    t = normalized["thickness"]
    z = normalized["Z"]
    rho = normalized["density"]
    opening = normalized["opening"]
    length = normalized["length"]
    n_upper = normalized["n_upper"]
    bc = boundary_key(normalized["boundary"])
    x1 = length / z if z > 0 else math.inf
    x2 = t / z if z > 0 else math.inf

    checks = [
        ("\u03c1", rho, MODEL_DOMAIN["density"], "kg/m3"),
        ("O", opening, MODEL_DOMAIN["opening"], ""),
        ("t", t, MODEL_DOMAIN["thickness"], "m"),
        ("Z", z, MODEL_DOMAIN["Z"], "m"),
        ("L", length, MODEL_DOMAIN["length"], "m"),
    ]
    for name, value, (lo, hi), unit in checks:
        if not (lo <= value <= hi):
            suffix = f" {unit}" if unit else ""
            warnings.append(f"{name} = {value:g}{suffix} is outside the trained range [{lo:g}, {hi:g}]{suffix}.")

    if not (MODEL_DOMAIN["n_upper"][0] <= n_upper <= MODEL_DOMAIN["n_upper"][1]):
        warnings.append("$N_f$ is outside the final $N_f$ <= 2 model range.")
    if not (0.05 <= x2 <= 0.35):
        warnings.append(f"t/Z = {x2:g} is outside the constrained training range [0.05, 0.35].")
    if bc == "P3" and x1 > 4.0:
        warnings.append(f"P3 L/Z = {x1:g} exceeds the constrained training limit 4.0.")
    if bc == "P4" and x1 > 2.75:
        warnings.append(f"P4 L/Z = {x1:g} exceeds the constrained training limit 2.75.")
    if t < 0.22 and n_upper != 0:
        warnings.append("For t < 0.22 m, constrained training cases used $N_f$ = 0.")
    if t < 0.24 and n_upper > 1:
        warnings.append("For t < 0.24 m, constrained training cases used $N_f$ <= 1.")
    if n_upper >= 2 and t < 0.24:
        warnings.append("For $N_f$ >= 2, constrained training cases used t >= 0.24 m.")
    if length > 10.0 and t < 0.28:
        warnings.append("For L > 10 m, constrained training cases used t >= 0.28 m.")
    if length > 10.0 and n_upper >= 1 and t < 0.30:
        warnings.append("For loaded walls with L > 10 m, constrained training cases used t >= 0.30 m.")
    if length > 10.0 and opening > 0.35:
        warnings.append("For L > 10 m, constrained training cases used O <= 0.35.")
    if opening > 0.35 and t < 0.24:
        warnings.append("For O > 0.35, constrained training cases used t >= 0.24 m.")

    return warnings


def aggregate_domain_warnings(cases: Iterable[dict[str, Any]], *, limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        counter.update(domain_warnings(case))
    return [message for message, _ in counter.most_common(limit)]
