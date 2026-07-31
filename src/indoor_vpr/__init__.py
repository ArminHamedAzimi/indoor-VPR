"""Notebook-friendly building blocks for visual place recognition."""

from . import algorithms as algorithms
from .core import (
    ALGORITHM_REGISTRY, DatasetConfig, ImageDataset, VPRAlgorithm, VPRResults,
    create_algorithm, list_algorithms, register_algorithm, run_vpr,
)

__all__ = [
    "ALGORITHM_REGISTRY", "DatasetConfig", "ImageDataset", "VPRAlgorithm",
    "VPRResults", "create_algorithm", "list_algorithms", "register_algorithm", "run_vpr",
]
