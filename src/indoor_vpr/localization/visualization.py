from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PIL import Image

from .arkit import read_frame_manifest
from .benchmark import benchmark_hloc


BACKGROUND = "#10131A"
PANEL = "#171B24"
TEXT = "#F4F7FB"
MUTED = "#8B95A7"
REFERENCE = "#21C7A8"
GROUND_TRUTH = "#8B5CF6"
ESTIMATE = "#FFB547"
FAILURE = "#EF5B5B"


def _style_axis(axis) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(color="white", alpha=0.07, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.12)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(TEXT)


def _equal_xy(axis, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) == 0:
        return
    center_x = (float(np.min(x)) + float(np.max(x))) / 2
    center_y = (float(np.min(y)) + float(np.max(y))) / 2
    radius = max(float(np.ptp(x)), float(np.ptp(y)), 1e-3) * 0.55
    axis.set_xlim(center_x - radius, center_x + radius)
    axis.set_ylim(center_y - radius, center_y + radius)
    axis.set_aspect("equal", adjustable="box")


def plot_hloc_trajectories(
    results_path: str | Path,
    reference_manifest: str | Path,
    query_manifest: str | Path,
    *,
    alignment_mode: str = "raw",
) -> Figure:
    """Plot reference, ARKit ground truth, and HLoc estimated camera paths."""

    benchmark = benchmark_hloc(results_path, reference_manifest, query_manifest)
    if alignment_mode not in benchmark:
        raise ValueError(
            f"alignment_mode must be one of: {', '.join(benchmark)}"
        )
    errors = benchmark[alignment_mode]["errors"]
    references = read_frame_manifest(reference_manifest)
    reference_xyz = np.stack([record.translation for record in references])
    ground_truth_xyz = np.stack(
        [error.ground_truth_world_from_camera[:3, 3] for error in errors]
    )
    valid_errors = [error for error in errors if error.estimated_world_from_camera is not None]
    estimated_xyz = (
        np.stack([error.estimated_world_from_camera[:3, 3] for error in valid_errors])
        if valid_errors
        else np.empty((0, 3))
    )
    local_estimated_xyz = np.full((len(errors), 3), np.nan)
    outlier_ground_truth = []
    for index, error in enumerate(errors):
        if error.estimated_world_from_camera is None:
            continue
        if error.translation_error_m is not None and error.translation_error_m <= 2.0:
            local_estimated_xyz[index] = error.estimated_world_from_camera[:3, 3]
        else:
            outlier_ground_truth.append(error.ground_truth_world_from_camera[:3, 3])

    figure = plt.figure(figsize=(14, 6.2), facecolor=BACKGROUND)
    top_down = figure.add_subplot(1, 2, 1)
    three_d = figure.add_subplot(1, 2, 2, projection="3d")
    _style_axis(top_down)
    three_d.set_facecolor(PANEL)

    top_down.plot(reference_xyz[:, 0], reference_xyz[:, 2], color=REFERENCE, linewidth=2.2, label="Reference ARKit")
    top_down.plot(ground_truth_xyz[:, 0], ground_truth_xyz[:, 2], color=GROUND_TRUTH, linewidth=2.0, label="Query ground truth")
    if np.isfinite(local_estimated_xyz).any():
        top_down.plot(local_estimated_xyz[:, 0], local_estimated_xyz[:, 2], color=ESTIMATE, linewidth=1.4, alpha=0.9, label="HLoc estimate (≤2 m)")
    if outlier_ground_truth:
        outliers = np.stack(outlier_ground_truth)
        top_down.scatter(outliers[:, 0], outliers[:, 2], marker="x", s=38, color=FAILURE, linewidth=1.4, label="HLoc outlier (>2 m)", zorder=6)
    top_down.scatter(reference_xyz[0, 0], reference_xyz[0, 2], marker="*", s=130, color=REFERENCE, edgecolor="white", linewidth=0.7, zorder=5)
    top_down.scatter(ground_truth_xyz[0, 0], ground_truth_xyz[0, 2], marker="*", s=130, color=GROUND_TRUTH, edgecolor="white", linewidth=0.7, zorder=5)
    combined = np.vstack([reference_xyz[:, [0, 2]], ground_truth_xyz[:, [0, 2]]])
    _equal_xy(top_down, combined[:, 0], combined[:, 1])
    top_down.set_title(f"Top-down trajectory · {alignment_mode.replace('_', ' ')}", loc="left", fontweight="bold", color=TEXT)
    top_down.set_xlabel("world X (m)")
    top_down.set_ylabel("world Z (m)")
    top_down.legend(frameon=False, labelcolor=TEXT, loc="best")

    for xyz, color, label, width in (
        (reference_xyz, REFERENCE, "Reference ARKit", 2.0),
        (ground_truth_xyz, GROUND_TRUTH, "Query ground truth", 1.8),
        (estimated_xyz, ESTIMATE, "HLoc estimate", 1.2),
    ):
        if len(xyz):
            three_d.plot(xyz[:, 0], xyz[:, 2], xyz[:, 1], color=color, linewidth=width, label=label)
    three_d.set_title("Metric 3D camera path", color=TEXT, fontweight="bold")
    three_d.set_xlabel("X (m)", color=MUTED)
    three_d.set_ylabel("Z (m)", color=MUTED)
    three_d.set_zlabel("Y / height (m)", color=MUTED)
    three_d.tick_params(colors=MUTED)
    three_d.grid(color="white", alpha=0.08)
    three_d.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    figure.suptitle("HLoc localization trajectories", color=TEXT, fontsize=16, fontweight="bold", x=0.04, ha="left")
    figure.subplots_adjust(left=0.055, right=0.98, top=0.88, bottom=0.1, wspace=0.16)
    return figure


def plot_hloc_error_timeline(
    results_path: str | Path,
    reference_manifest: str | Path,
    query_manifest: str | Path,
    *,
    alignment_mode: str = "raw",
) -> Figure:
    """Plot per-frame translation, rotation, and PnP inlier diagnostics."""

    benchmark = benchmark_hloc(results_path, reference_manifest, query_manifest)
    if alignment_mode not in benchmark:
        raise ValueError(f"Unknown alignment mode: {alignment_mode}")
    errors = benchmark[alignment_mode]["errors"]
    frame_indices = np.asarray([error.frame_index for error in errors])
    translation = np.asarray([
        np.nan if error.translation_error_m is None else error.translation_error_m
        for error in errors
    ])
    rotation = np.asarray([
        np.nan if error.rotation_error_deg is None else error.rotation_error_deg
        for error in errors
    ])
    inliers = np.asarray([
        0 if error.num_inliers is None else error.num_inliers for error in errors
    ])

    figure, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True, facecolor=BACKGROUND)
    for axis in axes:
        _style_axis(axis)
    translation_plot = np.maximum(translation, 1e-4)
    axes[0].plot(frame_indices, translation_plot, color=ESTIMATE, linewidth=1.35)
    axes[0].axhline(0.25, color=REFERENCE, linestyle="--", linewidth=1, alpha=0.8, label="0.25 m")
    axes[0].axhline(0.50, color=GROUND_TRUTH, linestyle="--", linewidth=1, alpha=0.8, label="0.50 m")
    axes[0].set_ylabel("translation (m)")
    axes[0].set_yscale("log")
    axes[0].set_title("Translation error · log scale", loc="left", fontweight="bold", color=TEXT)
    axes[0].legend(frameon=False, labelcolor=TEXT, ncol=2, loc="upper right")

    rotation_plot = np.maximum(rotation, 1e-4)
    axes[1].plot(frame_indices, rotation_plot, color=GROUND_TRUTH, linewidth=1.35)
    axes[1].axhline(2.0, color=REFERENCE, linestyle="--", linewidth=1, alpha=0.8, label="2°")
    axes[1].axhline(5.0, color=ESTIMATE, linestyle="--", linewidth=1, alpha=0.8, label="5°")
    axes[1].set_ylabel("rotation (deg)")
    axes[1].set_yscale("log")
    axes[1].set_title("Rotation error · log scale", loc="left", fontweight="bold", color=TEXT)
    axes[1].legend(frameon=False, labelcolor=TEXT, ncol=2, loc="upper right")

    axes[2].bar(frame_indices, inliers, width=max(1, int(np.median(np.diff(frame_indices))) * 0.7), color=REFERENCE, alpha=0.8)
    axes[2].set_ylabel("PnP inliers")
    axes[2].set_xlabel("source video frame index")
    axes[2].set_title("Geometrically verified correspondences", loc="left", fontweight="bold", color=TEXT)
    summary = benchmark[alignment_mode]["summary"]
    figure.suptitle(
        f"HLoc error timeline · {alignment_mode.replace('_', ' ')} · "
        f"PnP {summary['pnp_success_count']}/{summary['query_count']}",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        x=0.055,
        ha="left",
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.91, bottom=0.075, hspace=0.25)
    return figure


def _pair_group(handle, name0: str, name1: str):
    def encoded(left: str, right: str, separator: str) -> str:
        return separator.join((left.replace("/", "-"), right.replace("/", "-")))

    for left, right, reverse in ((name0, name1, False), (name1, name0, True)):
        for separator in ("/", "_"):
            key = encoded(left, right, separator)
            if key in handle:
                return handle[key], reverse
    raise KeyError(f"No stored HLoc matches for {name0} and {name1}.")


def _display_image(path: Path, max_height: int) -> tuple[np.ndarray, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = min(1.0, max_height / image.height)
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(image), scale


def show_hloc_correspondences(
    query_image: str,
    image_root: str | Path,
    pairs_path: str | Path,
    features_path: str | Path,
    matches_path: str | Path,
    *,
    max_reference_images: int = 3,
    max_matches: int = 120,
    max_image_height: int = 650,
) -> list[Figure]:
    """Visualize the strongest local correspondences for one query image."""

    import h5py

    query_name = query_image if query_image.startswith("query/") else f"query/{query_image}"
    references = []
    with Path(pairs_path).open() as handle:
        for line in handle:
            query, reference = line.split()
            if query == query_name:
                references.append(reference)
                if len(references) >= max_reference_images:
                    break
    if not references:
        raise ValueError(f"Query is not present in the retrieval pairs: {query_name}")

    figures = []
    with h5py.File(features_path, "r") as features, h5py.File(matches_path, "r") as matches_file:
        query_keypoints = features[query_name]["keypoints"][:]
        for rank, reference_name in enumerate(references, start=1):
            reference_keypoints = features[reference_name]["keypoints"][:]
            group, reverse = _pair_group(matches_file, query_name, reference_name)
            matches0 = group["matches0"][:]
            scores0 = group["matching_scores0"][:] if "matching_scores0" in group else np.ones_like(matches0, dtype=float)
            source_indices = np.flatnonzero(matches0 >= 0)
            target_indices = matches0[source_indices].astype(int)
            scores = scores0[source_indices]
            if reverse:
                source_indices, target_indices = target_indices, source_indices
            order = np.argsort(-scores)[:max_matches]
            source_indices, target_indices, scores = source_indices[order], target_indices[order], scores[order]

            query_rgb, query_scale = _display_image(Path(image_root) / query_name, max_image_height)
            reference_rgb, reference_scale = _display_image(Path(image_root) / reference_name, max_image_height)
            height = max(query_rgb.shape[0], reference_rgb.shape[0])
            gap = 18
            canvas = np.full((height, query_rgb.shape[1] + gap + reference_rgb.shape[1], 3), 16, dtype=np.uint8)
            canvas[: query_rgb.shape[0], : query_rgb.shape[1]] = query_rgb
            offset = query_rgb.shape[1] + gap
            canvas[: reference_rgb.shape[0], offset:] = reference_rgb
            query_points = query_keypoints[source_indices] * query_scale
            reference_points = reference_keypoints[target_indices] * reference_scale
            reference_points[:, 0] += offset
            segments = np.stack([query_points, reference_points], axis=1)

            figure, axis = plt.subplots(figsize=(15, 6), facecolor=BACKGROUND)
            axis.set_facecolor(BACKGROUND)
            axis.imshow(canvas)
            if len(segments):
                normalized = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-8)
                colors = plt.get_cmap("viridis")(normalized)
                axis.add_collection(LineCollection(segments, colors=colors, linewidths=0.75, alpha=0.72))
                axis.scatter(query_points[:, 0], query_points[:, 1], c=colors, s=7, edgecolors="none")
                axis.scatter(reference_points[:, 0], reference_points[:, 1], c=colors, s=7, edgecolors="none")
            axis.axvspan(query_rgb.shape[1], offset, color=BACKGROUND)
            axis.text(12, 24, "QUERY", color="white", fontsize=10, fontweight="bold", bbox={"boxstyle": "round,pad=0.35", "facecolor": GROUND_TRUTH, "edgecolor": "none"})
            axis.text(offset + 12, 24, f"REFERENCE #{rank}", color="white", fontsize=10, fontweight="bold", bbox={"boxstyle": "round,pad=0.35", "facecolor": REFERENCE, "edgecolor": "none"})
            axis.set_title(
                f"{query_name}  ↔  {reference_name}  ·  {len(source_indices)} strongest local matches shown",
                loc="left",
                color=TEXT,
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            axis.set_axis_off()
            figure.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01)
            figures.append(figure)
    return figures
