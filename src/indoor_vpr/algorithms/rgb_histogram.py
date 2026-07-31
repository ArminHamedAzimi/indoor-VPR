from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm


class RGBHistogram(VPRAlgorithm):
    """Fast, dependency-light baseline useful for checking an experiment setup."""

    name = "rgb_histogram"

    def __init__(self, bins: int = 32) -> None:
        self.bins = bins

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        descriptors = []
        for path in image_paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            channel_histograms = [
                np.histogram(pixels[..., channel], bins=self.bins, range=(0, 256))[0]
                for channel in range(3)
            ]
            descriptors.append(np.concatenate(channel_histograms).astype(np.float32))
        return np.stack(descriptors)


register_algorithm(RGBHistogram.name, RGBHistogram)
