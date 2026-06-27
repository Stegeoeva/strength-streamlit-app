from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import normalize_boundary, normalize_case


PRIMITIVE_NUMERIC_FEATURES = ["density", "opening", "thickness", "Z", "area", "n_upper"]
PRIMITIVE_CATEGORICAL_FEATURES = ["boundary"]
PRIMITIVE_FEATURES = PRIMITIVE_NUMERIC_FEATURES + PRIMITIVE_CATEGORICAL_FEATURES
PHYSICS_LITE_FEATURES = ["sqrt_area", "open_len", "tz"]


def add_physics_lite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sqrt_area"] = np.sqrt(out["area"].astype(float))
    out["open_len"] = out["opening"].astype(float) * out["sqrt_area"]
    out["tz"] = out["thickness"].astype(float) * out["Z"].astype(float)
    return out


def prepare_input_frame(cases: dict[str, Any] | Iterable[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(cases, dict):
        rows = [normalize_case(cases)]
    else:
        rows = [normalize_case(case) for case in cases]
    df = pd.DataFrame(rows)

    for col in PRIMITIVE_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["n_upper"] = df["n_upper"].astype(int)
    df["boundary"] = df["boundary"].map(normalize_boundary).astype(str)
    return df


def prepare_model_frame(
    cases: dict[str, Any] | Iterable[dict[str, Any]],
    feature_names: list[str],
) -> pd.DataFrame:
    df = prepare_input_frame(cases)
    if any(feature in feature_names for feature in PHYSICS_LITE_FEATURES):
        df = add_physics_lite(df)

    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    return df.loc[:, feature_names]

