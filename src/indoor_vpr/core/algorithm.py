from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class VPRAlgorithm(ABC):
    """Contract implemented by every global-descriptor VPR algorithm."""

    name = "base"

    @abstractmethod
    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Return one descriptor row per image."""

    def encode_database(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Prepare on, then encode, a database.

        Stateless algorithms can use this default. Stateful methods such as
        AnyLoc can override it to fit an aggregator without extracting twice.
        """

        return self.encode(image_paths)

    def similarity(
        self,
        query_descriptors: np.ndarray,
        database_descriptors: np.ndarray,
    ) -> np.ndarray:
        """Return pairwise query-to-database similarity scores.

        Most algorithms emit one L2-normalized descriptor per image and use a
        dot product. Algorithms with multiple views per image can override
        this method while keeping the retrieval pipeline unchanged.
        """

        return query_descriptors @ database_descriptors.T
