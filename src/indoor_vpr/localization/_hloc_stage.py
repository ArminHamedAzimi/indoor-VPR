"""Isolated HLoc stages.

On macOS, PyTorch and PyCOLMAP wheels can bundle incompatible OpenMP runtimes.
Keeping neural and geometry stages in different processes avoids loading both
runtimes into one address space.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import types
from pathlib import Path


def _status_from_logs(results: Path) -> None:
    logs_path = Path(f"{results}_logs.pkl")
    if not logs_path.is_file():
        return
    with logs_path.open("rb") as handle:
        logs = pickle.load(handle)
    with Path(f"{results}_status.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pnp_success", "num_inliers"])
        for name, log in logs.get("loc", {}).items():
            if log.get("covisibility_clustering"):
                cluster = log.get("best_cluster")
                pnp = None if cluster is None else log["log_clusters"][cluster].get("PnP_ret")
            else:
                pnp = log.get("PnP_ret")
            writer.writerow(
                [name, pnp is not None, "" if pnp is None else int(pnp.get("num_inliers", 0))]
            )


def _mask_pycolmap_for_neural_stage() -> None:
    """Provide the type-only symbols imported by HLoc's I/O helpers."""

    stub = types.ModuleType("pycolmap")
    stub.__version__ = "dev"
    stub.Rigid3d = object
    sys.modules["pycolmap"] = stub


def _force_single_process_dataloader() -> None:
    """Prevent spawned workers from re-importing the real PyCOLMAP module."""

    import torch

    original = torch.utils.data.DataLoader

    def data_loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original(*args, **kwargs)

    torch.utils.data.DataLoader = data_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("known-model", "extract", "match", "triangulate", "localize")
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conf")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--reference-model", type=Path)
    parser.add_argument("--reference-sfm", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--ransac-threshold", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stage == "known-model":
        from .arkit import read_frame_manifest
        from .hloc_pipeline import build_known_pose_reference_model

        build_known_pose_reference_model(read_frame_manifest(args.manifest), args.output)
    elif args.stage == "extract":
        # HLoc's I/O module imports PyCOLMAP for a type annotation even though
        # feature extraction never calls it. Keep this process PyTorch-only.
        _mask_pycolmap_for_neural_stage()
        from hloc import extract_features

        _force_single_process_dataloader()

        extract_features.main(
            extract_features.confs[args.conf],
            args.image_root,
            feature_path=args.features,
            image_list=args.image_list,
            overwrite=args.overwrite,
        )
    elif args.stage == "match":
        _mask_pycolmap_for_neural_stage()
        from hloc import match_features

        _force_single_process_dataloader()

        match_features.main(
            match_features.confs[args.conf],
            args.pairs,
            args.features,
            matches=args.matches,
            overwrite=args.overwrite,
        )
    elif args.stage == "triangulate":
        from hloc import triangulation

        triangulation.main(
            args.output,
            args.reference_model,
            args.image_root,
            args.pairs,
            args.features,
            args.matches,
        )
    else:
        from hloc import localize_sfm

        localize_sfm.main(
            args.reference_sfm,
            args.queries,
            args.pairs,
            args.features,
            args.matches,
            args.results,
            ransac_thresh=args.ransac_threshold,
            prepend_camera_name=True,
        )
        _status_from_logs(args.results)


if __name__ == "__main__":
    main()
