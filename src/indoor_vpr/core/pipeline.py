from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithm import VPRAlgorithm
from .data import ImageDataset


@dataclass(frozen=True)
class VPRResults:
    dataset: ImageDataset
    similarity: np.ndarray
    ranked_database_indices: np.ndarray

    @property
    def best_indices(self) -> np.ndarray:
        return self.ranked_database_indices[:, 0]

    @property
    def best_scores(self) -> np.ndarray:
        return self.similarity[np.arange(len(self.dataset.query_paths)), self.best_indices]

    def top_indices(self, query_index: int, top_k: int = 5) -> np.ndarray:
        return self.ranked_database_indices[query_index, :top_k]


def _l2_normalize(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2:
        raise ValueError("Descriptors must have shape (number_of_images, descriptor_size).")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-12)


def run_vpr(dataset: ImageDataset, algorithm: VPRAlgorithm) -> VPRResults:
    """Encode both sets and rank database images using cosine similarity."""

    database_descriptors = _l2_normalize(algorithm.encode(dataset.database_paths))
    query_descriptors = _l2_normalize(algorithm.encode(dataset.query_paths))
    if database_descriptors.shape[1] != query_descriptors.shape[1]:
        raise ValueError("Database and query descriptor sizes do not match.")
    similarity = query_descriptors @ database_descriptors.T
    ranking = np.argsort(-similarity, axis=1)
    return VPRResults(dataset=dataset, similarity=similarity, ranked_database_indices=ranking)
