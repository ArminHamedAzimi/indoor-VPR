from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from .core import VPRResults


BACKGROUND = "#10131A"
MUTED = "#697386"
QUERY_COLOR = "#7C3AED"
MATCH_COLOR = "#0F9D8A"


def _load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _image_card(axis, path, label: str, subtitle: str, color: str) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.imshow(_load_rgb(path))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2)
    axis.text(
        0.025,
        0.975,
        label,
        transform=axis.transAxes,
        color="white",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "none", "alpha": 0.92},
    )
    axis.text(
        0.025,
        0.025,
        subtitle,
        transform=axis.transAxes,
        color="white",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.62},
    )


def show_matches(results: VPRResults, query_index: int = 0, top_k: int = 5) -> Figure:
    """Show one query and its ranked database matches as compact image cards."""

    query_count = len(results.dataset.query_paths)
    if not 0 <= query_index < query_count:
        raise IndexError(f"query_index must be between 0 and {query_count - 1}")

    indices = results.top_indices(query_index, top_k)
    columns = len(indices) + 1
    figure, axes = plt.subplots(1, columns, figsize=(2.3 * columns, 4.1), facecolor=BACKGROUND)
    axes = np.atleast_1d(axes)

    query_path = results.dataset.query_paths[query_index]
    _image_card(axes[0], query_path, "QUERY", query_path.name, QUERY_COLOR)
    for rank, (axis, database_index) in enumerate(zip(axes[1:], indices), start=1):
        path = results.dataset.database_paths[int(database_index)]
        score = results.similarity[query_index, database_index]
        _image_card(axis, path, f"#{rank}  ·  {score:.3f}", path.name, MATCH_COLOR)

    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.035)
    return figure

def show_similarity_matrix(results: VPRResults, figsize=(12, 6.5)) -> Figure:
    """Plot the query-to-database cosine similarities as a polished heatmap."""

    figure, axis = plt.subplots(figsize=figsize, facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    matrix = results.similarity
    lower, upper = np.percentile(matrix, [2, 98])
    if np.isclose(lower, upper):
        lower, upper = float(matrix.min()), float(matrix.max() + 1e-6)
    heatmap = axis.imshow(
        matrix,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=lower,
        vmax=upper,
    )
    axis.scatter(
        results.best_indices,
        np.arange(len(results.dataset.query_paths)),
        s=16,
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
        alpha=0.9,
        label="Best match",
    )
    axis.set_title("Query ↔ database similarity", loc="left", color="white", fontsize=14, fontweight="bold", pad=8)
    axis.set_xlabel("Database image index", color="#C4CAD6", labelpad=8)
    axis.set_ylabel("Query image index", color="#C4CAD6", labelpad=8)
    axis.tick_params(colors=MUTED, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(heatmap, ax=axis, pad=0.02, fraction=0.025)
    colorbar.set_label("Cosine similarity", color=MUTED)
    colorbar.ax.tick_params(colors=MUTED, length=0)
    axis.legend(frameon=False, labelcolor=MUTED, loc="upper right")
    figure.subplots_adjust(left=0.07, right=0.94, top=0.92, bottom=0.11)
    return figure
