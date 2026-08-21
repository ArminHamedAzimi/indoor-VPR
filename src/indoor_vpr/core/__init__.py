"""Stable interfaces and shared services used by every VPR algorithm."""

from .algorithm import VPRAlgorithm
from .data import DatasetConfig, ImageDataset
from .pipeline import SimilarityCSVView, StreamedVPRResults, VPRResults, run_vpr, stream_vpr_similarity_csv
from .registry import ALGORITHM_REGISTRY, create_algorithm, list_algorithms, register_algorithm

__all__ = [
    "ALGORITHM_REGISTRY", "DatasetConfig", "ImageDataset", "VPRAlgorithm",
    "SimilarityCSVView", "StreamedVPRResults", "VPRResults", "create_algorithm",
    "list_algorithms", "register_algorithm", "run_vpr", "stream_vpr_similarity_csv",
]
