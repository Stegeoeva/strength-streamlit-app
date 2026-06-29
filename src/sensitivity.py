from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .predict import ModelBundle, effective_beta, predict_cases
from .utils import (
    ANALYSIS_MODES,
    BOUNDARY_OPTIONS,
    OUTPUT_LABELS,
    aggregate_domain_warnings,
    case_from_quality,
    constraint_violations,
    float_grid,
    format_axis_value,
    mode_number,
    normalize_boundary,
)


@dataclass
class SensitivityResult:
    mode: str
    title: str
    y_label: str
    y_values: list[Any]
    levels: list[int]
    output_key: str
    value_label: str
    matrix: np.ndarray
    records: list[dict[str, Any]]
    warnings: list[str]

    def long_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def matrix_dataframe(self) -> pd.DataFrame:
        columns = [f"Wall case index {level}" for level in self.levels]
        index = [format_axis_value(value) for value in self.y_values]
        frame = pd.DataFrame(self.matrix, index=index, columns=columns)
        frame.index.name = self.y_label
        return frame


def sensitivity_axis(
    *,
    mode: str,
    floor_min: float,
    floor_max: float,
    length_min: float,
    length_max: float,
    length_step: float,
    n_upper_min: int,
    n_upper_max: int,
    selected_boundaries: list[str],
) -> tuple[list[Any], str]:
    number = mode_number(mode)
    if number == "1":
        return list(float_grid(floor_min, floor_max, 0.1)), "Z (m)"
    if number == "2":
        if not selected_boundaries:
            raise ValueError("Select at least one boundary condition.")
        return [normalize_boundary(boundary) for boundary in selected_boundaries], "BC"
    if number == "3":
        return list(float_grid(length_min, length_max, length_step)), "L (m)"
    if number == "4":
        if int(n_upper_min) > int(n_upper_max):
            raise ValueError("Upper-floor range minimum must be <= maximum.")
        return list(range(int(n_upper_min), int(n_upper_max) + 1)), "$N_f$"
    raise ValueError(f"Unknown sensitivity mode: {mode}")


def _case_for_axis_value(
    *,
    mode: str,
    axis_value: Any,
    level: int,
    length: float,
    n_upper: int,
    boundary: str,
    floor_height: float,
) -> dict[str, Any]:
    number = mode_number(mode)
    case_length = float(length)
    case_n_upper = int(n_upper)
    case_boundary = normalize_boundary(boundary)
    case_z = float(floor_height)

    if number == "1":
        case_z = float(axis_value)
    elif number == "2":
        case_boundary = normalize_boundary(axis_value)
        if case_boundary.startswith("P1"):
            case_n_upper = 0
    elif number == "3":
        case_length = float(axis_value)
    elif number == "4":
        case_n_upper = int(axis_value)

    return case_from_quality(
        level,
        Z=case_z,
        length=case_length,
        n_upper=case_n_upper,
        boundary=case_boundary,
    )


def _depth_label(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return "0" if text == "-0" else text


def _relative_depth_label(value: float) -> str:
    return f"{float(value):.2f}"


def _fragility_probability_at_depths(
    mu: np.ndarray,
    beta: np.ndarray,
    flood_depths: np.ndarray,
) -> np.ndarray:
    mu_safe = np.asarray(mu, dtype=float)
    beta_safe = np.asarray(beta, dtype=float)
    depth_safe = np.asarray(flood_depths, dtype=float)
    invalid = (mu_safe <= 0) | (depth_safe <= 0) | ~np.isfinite(mu_safe) | ~np.isfinite(beta_safe) | ~np.isfinite(depth_safe)
    beta_eff = effective_beta(beta_safe)
    z_score = (np.log(np.maximum(depth_safe, 1e-12)) - np.log(np.maximum(mu_safe, 1e-12))) / beta_eff
    probabilities = norm.cdf(z_score)
    probabilities[invalid] = np.nan
    return probabilities


def _heatmap_values(
    bundle: ModelBundle,
    cases: list[dict[str, Any]],
    *,
    output_key: str,
    flood_depth: float,
    relative_flood_depth: float,
) -> tuple[np.ndarray, str]:
    if output_key == "mu_over_z":
        mu = predict_cases(bundle, cases, output_key="mu")
        heights = np.asarray([float(case["Z"]) for case in cases], dtype=float)
        return mu / heights, OUTPUT_LABELS[output_key]
    if output_key == "probability":
        probability = predict_cases(bundle, cases, output_key="probability", flood_depth=flood_depth)
        label = f"Collapse probability, P(C | h = {_depth_label(flood_depth)} m) (%)"
        return probability * 100.0, label
    if output_key == "probability_relative":
        mu = predict_cases(bundle, cases, output_key="mu")
        beta = predict_cases(bundle, cases, output_key="beta")
        heights = np.asarray([float(case["Z"]) for case in cases], dtype=float)
        flood_depths = float(relative_flood_depth) * heights
        probability = _fragility_probability_at_depths(mu, beta, flood_depths)
        label = f"Collapse probability, P(C | h/Z = {_relative_depth_label(relative_flood_depth)}) (%)"
        return probability * 100.0, label
    values = predict_cases(bundle, cases, output_key=output_key, flood_depth=flood_depth)
    return values, OUTPUT_LABELS.get(output_key, output_key)


def generate_sensitivity_heatmap(
    bundle: ModelBundle,
    *,
    mode: str = ANALYSIS_MODES[0],
    levels: list[int] | None = None,
    output_key: str = "mu",
    flood_depth: float = 1.5,
    relative_flood_depth: float = 0.50,
    length: float = 4.0,
    n_upper: int = 2,
    boundary: str = "P3 (3-edges)",
    floor_height: float = 3.0,
    floor_min: float = 2.5,
    floor_max: float = 3.5,
    length_min: float = 3.0,
    length_max: float = 10.0,
    length_step: float = 0.5,
    n_upper_min: int = 0,
    n_upper_max: int = 2,
    selected_boundaries: list[str] | None = None,
) -> SensitivityResult:
    selected_levels = sorted([int(level) for level in (levels or list(range(1, 11)))])
    if not selected_levels:
        raise ValueError("Select at least one wall case index.")

    y_values, y_label = sensitivity_axis(
        mode=mode,
        floor_min=floor_min,
        floor_max=floor_max,
        length_min=length_min,
        length_max=length_max,
        length_step=length_step,
        n_upper_min=n_upper_min,
        n_upper_max=n_upper_max,
        selected_boundaries=selected_boundaries or BOUNDARY_OPTIONS,
    )

    cases: list[dict[str, Any]] = []
    metadata: list[tuple[Any, int]] = []
    for y_value in y_values:
        for level in selected_levels:
            case = _case_for_axis_value(
                mode=mode,
                axis_value=y_value,
                level=level,
                length=length,
                n_upper=n_upper,
                boundary=boundary,
                floor_height=floor_height,
            )
            cases.append(case)
            metadata.append((y_value, level))

    invalid = [(case, constraint_violations(case)) for case in cases]
    invalid = [(case, messages) for case, messages in invalid if messages]
    if invalid:
        first_case, messages = invalid[0]
        level = first_case.get("level", "?")
        raise ValueError(
            "Selected representative wall cases include cases outside the constrained training rules. "
            f"First invalid wall case: {level}. " + " ".join(messages[:2])
        )

    values, value_label = _heatmap_values(
        bundle,
        cases,
        output_key=output_key,
        flood_depth=flood_depth,
        relative_flood_depth=relative_flood_depth,
    )
    matrix = np.asarray(values, dtype=float).reshape(len(y_values), len(selected_levels))

    title_map = {
        "1": "Z vs Wall case index",
        "2": "BC vs Wall case index",
        "3": "L vs Wall case index",
        "4": "$N_f$ vs Wall case index",
    }

    records: list[dict[str, Any]] = []
    for case, (y_value, level), value in zip(cases, metadata, values):
        demand_fields: dict[str, Any] = {}
        if output_key == "probability":
            demand_fields["Selected flood depth h (m)"] = float(flood_depth)
        elif output_key == "probability_relative":
            demand_fields["Selected relative flood depth h/Z (-)"] = float(relative_flood_depth)
            demand_fields["Cell flood depth h (m)"] = float(relative_flood_depth) * float(case["Z"])
        records.append(
            {
                y_label: y_value,
                "Wall case index": level,
                value_label: float(value),
                "Heatmap metric": value_label,
                **demand_fields,
                "\u03c1 (kg/m3)": case["density"],
                "O (-)": case["opening"],
                "t (m)": case["thickness"],
                "Z (m)": case["Z"],
                "L (m)": case["length"],
                "Number of upper floors": case["n_upper"],
                "BC (-)": case["boundary"],
            }
        )

    return SensitivityResult(
        mode=mode,
        title=f"{value_label}\n{title_map.get(mode_number(mode), 'Sensitivity Heatmap')}",
        y_label=y_label,
        y_values=y_values,
        levels=selected_levels,
        output_key=output_key,
        value_label=value_label,
        matrix=matrix,
        records=records,
        warnings=aggregate_domain_warnings(cases),
    )
