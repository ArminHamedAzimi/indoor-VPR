"""ARKit and HLoc helpers for metric 6-DoF camera localization."""

from .arkit import (
    ARKitFrame,
    first_pose_alignment,
    prepare_localization_frames,
    read_arkit_poses,
    read_frame_manifest,
    rotation_error_degrees,
)
from .benchmark import benchmark_hloc, export_hloc_benchmark_xlsx
from .hloc_pipeline import (
    HLocConfig,
    export_anyloc_retrieval_xlsx,
    run_hloc,
)
from .visualization import (
    plot_hloc_error_timeline,
    plot_hloc_trajectories,
    show_hloc_correspondences,
)

__all__ = [
    "ARKitFrame",
    "HLocConfig",
    "benchmark_hloc",
    "export_anyloc_retrieval_xlsx",
    "export_hloc_benchmark_xlsx",
    "first_pose_alignment",
    "prepare_localization_frames",
    "plot_hloc_error_timeline",
    "plot_hloc_trajectories",
    "read_arkit_poses",
    "read_frame_manifest",
    "rotation_error_degrees",
    "run_hloc",
    "show_hloc_correspondences",
]
