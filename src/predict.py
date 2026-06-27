from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from .preprocess import prepare_model_frame
from .utils import MODELS_DIR, constraint_violations, normalize_case, read_json


@dataclass(frozen=True)
class ModelArtifact:
    target: str
    path: Path
    estimator: Any
    feature_names: list[str]
    target_column: str
    model_name: str
    use_phys_lite: bool
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ModelBundle:
    mu: ModelArtifact
    beta: ModelArtifact
    manifest: dict[str, Any]
    feature_metadata: dict[str, Any]

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for index, artifact in enumerate([self.mu, self.beta], start=1):
            rows.append(
                {
                    "Index": index,
                    "Model": _display_model_name(artifact.model_name),
                    "output": artifact.target,
                    "R2": artifact.metrics.get("r2"),
                    "Target column": artifact.target_column,
                    "Features": ", ".join(artifact.feature_names),
                    "MAE": artifact.metrics.get("mae"),
                    "Uses physics-lite features": artifact.use_phys_lite,
                    "File": artifact.path.name,
                }
            )
        return pd.DataFrame(rows)


def _display_model_name(model_name: str) -> str:
    key = str(model_name).upper()
    if key.startswith("SVR") or "SVM" in key or "SUPPORT" in key:
        return "Support Vector Machine"
    if key.startswith("RF") or "RANDOM" in key:
        return "Random Forest"
    if key.startswith("HGB") or "HIST" in key:
        return "Histogram Gradient Boosting"
    return str(model_name)


def _normalize_valid_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        normalized = normalize_case(case)
        messages = constraint_violations(normalized)
        if messages:
            raise ValueError(" ".join(messages))
        rows.append(normalized)
    return rows


def _artifact_from_manifest(
    *,
    target: str,
    filename: str,
    models_dir: Path,
    manifest: dict[str, Any],
) -> ModelArtifact:
    path = models_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained model artifact: {path}")

    estimator = joblib.load(path)
    _force_single_thread_inference(estimator)
    entry = manifest.get(target, {})
    detected = getattr(estimator, "feature_names_in_", None)
    if detected is not None:
        feature_names = [str(name) for name in detected]
    else:
        feature_names = [str(name) for name in entry.get("features", [])]
    if not feature_names:
        raise ValueError(f"Could not determine feature order for {path.name}.")

    return ModelArtifact(
        target=target,
        path=path,
        estimator=estimator,
        feature_names=feature_names,
        target_column=str(entry.get("target_column", target)),
        model_name=str(entry.get("model", estimator.__class__.__name__)),
        use_phys_lite=bool(entry.get("use_phys_lite", False)),
        metrics=dict(entry.get("holdout_metrics", {})),
    )


def _force_single_thread_inference(estimator: Any) -> None:
    """Avoid Windows sandbox pipe/thread-pool errors from saved n_jobs=-1 models."""
    candidates: list[Any] = [estimator]
    if hasattr(estimator, "named_steps"):
        candidates.extend(estimator.named_steps.values())
    for candidate in candidates:
        if hasattr(candidate, "n_jobs"):
            try:
                candidate.n_jobs = 1
            except Exception:
                pass


def load_model_bundle(models_dir: Path | None = None) -> ModelBundle:
    model_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    manifest_path = model_dir / "best_models_manifest.json"
    feature_metadata_path = model_dir / "feature_names.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    feature_metadata = read_json(feature_metadata_path) if feature_metadata_path.exists() else {}

    mu = _artifact_from_manifest(
        target="mu",
        filename="final_model_mu.joblib",
        models_dir=model_dir,
        manifest=manifest,
    )
    beta = _artifact_from_manifest(
        target="beta",
        filename="final_model_beta.joblib",
        models_dir=model_dir,
        manifest=manifest,
    )
    return ModelBundle(mu=mu, beta=beta, manifest=manifest, feature_metadata=feature_metadata)


def _predict_artifact(artifact: ModelArtifact, cases: Iterable[dict[str, Any]]) -> np.ndarray:
    rows = _normalize_valid_cases(cases)
    X = prepare_model_frame(rows, artifact.feature_names)
    return np.asarray(artifact.estimator.predict(X), dtype=float)


def predict_cases(
    bundle: ModelBundle,
    cases: Iterable[dict[str, Any]],
    *,
    output_key: str = "mu",
    flood_depth: float = 1.5,
) -> np.ndarray:
    rows = _normalize_valid_cases(cases)
    if output_key == "mu":
        return _predict_artifact(bundle.mu, rows)
    if output_key == "beta":
        return _predict_artifact(bundle.beta, rows)
    if output_key == "probability":
        mu = _predict_artifact(bundle.mu, rows)
        beta = _predict_artifact(bundle.beta, rows)
        return fragility_probability(mu, beta, flood_depth)
    raise ValueError(f"Unknown output key: {output_key}")


def predict_single(bundle: ModelBundle, case: dict[str, Any], *, flood_depth: float = 1.5) -> dict[str, float]:
    row = _normalize_valid_cases([case])
    mu = float(_predict_artifact(bundle.mu, row)[0])
    beta = float(_predict_artifact(bundle.beta, row)[0])
    probability = float(fragility_probability(np.array([mu]), np.array([beta]), flood_depth)[0])
    return {"mu": mu, "beta": beta, "probability": probability}


def fragility_probability(mu: np.ndarray, beta: np.ndarray, flood_depth: float) -> np.ndarray:
    depth = float(flood_depth)
    if depth <= 0:
        raise ValueError("Flood depth must be positive for fragility probability.")
    mu_safe = np.asarray(mu, dtype=float)
    beta_safe = np.asarray(beta, dtype=float)
    invalid = (mu_safe <= 0) | (beta_safe <= 0)
    z = (np.log(depth) - np.log(np.maximum(mu_safe, 1e-12))) / np.maximum(beta_safe, 1e-12)
    probs = norm.cdf(z)
    probs[invalid] = np.nan
    return probs
