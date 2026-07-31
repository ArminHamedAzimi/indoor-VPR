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
