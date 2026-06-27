from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .utils import format_axis_value


def _trimmed_number(value: float, decimals: int = 2) -> str:
    text = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _tick_positions_and_labels(values: list[Any]) -> tuple[np.ndarray, list[str]]:
    positions = np.arange(len(values))
    labels = [format_axis_value(value) for value in values]
    if len(labels) > 16:
        keep = max(1, int(np.ceil(len(labels) / 12)))
        labels = [label if idx % keep == 0 else "" for idx, label in enumerate(labels)]
    return positions, labels


def plot_heatmap(
    *,
    matrix: np.ndarray,
    y_values: list[Any],
    levels: list[int],
    y_label: str,
    value_label: str,
    title: str,
    output_key: str,
    annotate: bool = False,
) -> plt.Figure:
    values = np.asarray(matrix, dtype=float)
    is_probability = output_key in {"probability", "probability_relative"}
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=125, constrained_layout=True)

    if output_key == "mu":
        cmap = mcolors.LinearSegmentedColormap.from_list("risk_fixed", ["red", "orange", "yellow", "green"])
        display_values = np.clip(values, 1.0, 3.0)
        vmin, vmax = 1.0, 3.0
    elif output_key == "beta":
        cmap = plt.get_cmap("cividis")
        display_values = values
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmin -= 0.05
            vmax += 0.05
    elif output_key == "mu_over_z":
        cmap = mcolors.LinearSegmentedColormap.from_list("risk_normalized", ["red", "orange", "yellow", "green"])
        display_values = values
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmin -= 0.05
            vmax += 0.05
    elif is_probability:
        cmap = mcolors.LinearSegmentedColormap.from_list("fragility_probability", ["green", "yellow", "orange", "red"])
        display_values = np.clip(values, 0.0, 100.0)
        vmin, vmax = 0.0, 100.0
    else:
        cmap = plt.get_cmap("viridis")
        display_values = values
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    im = ax.imshow(
        display_values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Wall case index")
    ax.set_ylabel(y_label)

    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([str(level) for level in levels])
    y_positions, y_labels = _tick_positions_and_labels(y_values)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.invert_xaxis()
    ax.invert_yaxis()

    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                label = f"{values[row, col]:.1f}" if is_probability else _trimmed_number(values[row, col])
                ax.text(col, row, label, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_label)
    if output_key == "mu":
        cbar.ax.set_ylim(3.0, 1.0)
        cbar.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0])
        cbar.set_ticklabels(["1 (High risk)", "1.5", "2 (Moderate risk)", "2.5", "3 (Low risk)"])
    elif is_probability:
        cbar.set_ticks([0.0, 25.0, 50.0, 75.0, 100.0])
        cbar.set_ticklabels(["0", "25", "50", "75", "100"])
    else:
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([_trimmed_number(tick) for tick in ticks])

    return fig
