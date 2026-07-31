from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm


class DINOv2(VPRAlgorithm):
    """DINOv2 global descriptors loaded lazily through torch.hub."""

    name = "dinov2"

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
        except ImportError as error:
            raise ImportError("DINOv2 requires torch and torchvision.") from error

        self.torch = torch
        self.batch_size = batch_size
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        # torch.hub.load can return arbitrary objects. DINOv2 returns a module.
        self.model = cast(
            torch.nn.Module,
            torch.hub.load("facebookresearch/dinov2", model_name),
        )
        self.model.eval().to(self.device)

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        batches = []
        for start in range(0, len(image_paths), self.batch_size):
            tensors = []
            for path in image_paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                batches.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)


register_algorithm(DINOv2.name, DINOv2)
