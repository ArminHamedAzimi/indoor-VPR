"""Stable interfaces and shared services used by every VPR algorithm."""

from .algorithm import VPRAlgorithm
from .data import DatasetConfig, ImageDataset
from .pipeline import VPRResults, run_vpr
from .registry import ALGORITHM_REGISTRY, create_algorithm, list_algorithms, register_algorithm

__all__ = [
    "ALGORITHM_REGISTRY", "DatasetConfig", "ImageDataset", "VPRAlgorithm",
    "VPRResults", "create_algorithm", "list_algorithms", "register_algorithm", "run_vpr",
]
