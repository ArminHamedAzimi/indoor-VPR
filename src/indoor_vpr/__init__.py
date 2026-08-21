"""Notebook-friendly building blocks for visual place recognition."""

from . import algorithms as algorithms
from .core import (
    ALGORITHM_REGISTRY, DatasetConfig, ImageDataset, SimilarityCSVView,
    StreamedVPRResults, VPRAlgorithm, VPRResults, create_algorithm,
    list_algorithms, register_algorithm, run_vpr, stream_vpr_similarity_csv,
)

__all__ = [
    "ALGORITHM_REGISTRY", "DatasetConfig", "ImageDataset", "SimilarityCSVView",
    "StreamedVPRResults", "VPRAlgorithm", "VPRResults", "create_algorithm",
    "list_algorithms", "register_algorithm", "run_vpr", "stream_vpr_similarity_csv",
]
