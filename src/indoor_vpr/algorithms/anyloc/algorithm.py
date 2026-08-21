from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm

from .extractor import DINOv2PatchExtractor, Facet
from .vlad import VLAD

if TYPE_CHECKING:
    import torch


class AnyLoc(VPRAlgorithm):
    """AnyLoc-style DINOv2 patch descriptors with VLAD aggregation.

    Supply an official AnyLoc ``c_centers.pt`` file for reproducible inference.
    If no vocabulary is supplied, a codebook is fitted from the active database.
    """

    name = "anyloc"

    def __init__(
        self,
        model_name: str = "dinov2_vitg14",
        layer: int = 31,
        facet: Facet = "value",
        num_clusters: int = 32,
        vocabulary_path: str | Path | None = None,
        max_image_size: int = 1024,
        max_vocabulary_descriptors: int = 50_000,
        kmeans_iterations: int = 20,
        device: str | None = None,
    ) -> None:
        import torch
        import torchvision.transforms as transforms
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform_functional

        self.torch = torch
        self.transforms = transforms
        self.resize = transform_functional.resize
        self.center_crop = transform_functional.center_crop
        self.interpolation_mode = InterpolationMode
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.max_image_size = max_image_size
        self.max_vocabulary_descriptors = max_vocabulary_descriptors
        self.kmeans_iterations = kmeans_iterations
        self.extractor = DINOv2PatchExtractor(model_name, layer, facet, self.device)
        self.vlad = VLAD(num_clusters, self.device)
        self.to_tensor = cast(
            "Callable[[Image.Image], torch.Tensor]",
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            ),
        )
        if vocabulary_path is not None:
            path = Path(vocabulary_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"AnyLoc vocabulary not found: {path}")
            self.vlad.load(path)

    def _local_descriptors(self, path: Path):
        with Image.open(path) as image:
            tensor = self.to_tensor(image.convert("RGB")).to(self.device)
        height, width = tensor.shape[-2:]
        scale = min(1.0, self.max_image_size / max(height, width))
        if scale < 1.0:
            height, width = round(height * scale), round(width * scale)
            tensor = self.resize(
                tensor,
                [height, width],
                interpolation=self.interpolation_mode.BICUBIC,
                antialias=True,
            )
        height, width = tensor.shape[-2:]
        patch_height, patch_width = (height // 14) * 14, (width // 14) * 14
        if patch_height == 0 or patch_width == 0:
            raise ValueError(f"Image is too small for DINOv2 patches: {path}")
        tensor = self.center_crop(tensor, [patch_height, patch_width])
        return self.extractor(tensor.unsqueeze(0)).squeeze(0)

    def _sample_vocabulary_descriptors(self, image_paths: Sequence[Path]):
        if not image_paths:
            raise ValueError("AnyLoc needs at least one image to fit a vocabulary.")

        generator = self.torch.Generator().manual_seed(42)
        per_image_cap = max(1, math.ceil(self.max_vocabulary_descriptors / len(image_paths)))
        samples = []
        collected = 0

        for path in image_paths:
            descriptors = self._local_descriptors(path).detach().cpu()
            if len(descriptors) == 0:
                continue
            take = min(len(descriptors), per_image_cap, self.max_vocabulary_descriptors - collected)
            if take <= 0:
                break
            if take < len(descriptors):
                indices = self.torch.randperm(len(descriptors), generator=generator)[:take]
                descriptors = descriptors[indices]
            elif take > len(descriptors):
                take = len(descriptors)
            samples.append(descriptors[:take])
            collected += take
            if collected >= self.max_vocabulary_descriptors:
                break

        if not samples:
            raise ValueError("AnyLoc could not sample descriptors for vocabulary fitting.")
        return self.torch.cat(samples, dim=0)

    def _aggregate_paths(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("AnyLoc needs at least one image to encode.")

        descriptors = []
        for path in image_paths:
            local_descriptors = self._local_descriptors(path)
            descriptors.append(self.vlad.generate(local_descriptors).cpu())
        return self.torch.stack(descriptors).numpy().astype(np.float32)

    def encode_database(self, image_paths: Sequence[Path]) -> np.ndarray:
        if self.vlad.centers is None:
            vocabulary_descriptors = self._sample_vocabulary_descriptors(image_paths)
            self.vlad.fit(vocabulary_descriptors, iterations=self.kmeans_iterations)
        return self._aggregate_paths(image_paths)

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        if self.vlad.centers is None:
            raise RuntimeError("Encode the database first or provide an AnyLoc vocabulary file.")
        return self._aggregate_paths(image_paths)


register_algorithm(AnyLoc.name, AnyLoc)
