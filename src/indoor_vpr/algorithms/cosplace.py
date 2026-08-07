from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm


_AVAILABLE_MODELS = {
    "VGG16": [64, 128, 256, 512],
    "ResNet18": [32, 64, 128, 256, 512],
    "ResNet50": [32, 64, 128, 256, 512, 1024, 2048],
    "ResNet101": [32, 64, 128, 256, 512, 1024, 2048],
    "ResNet152": [32, 64, 128, 256, 512, 1024, 2048],
}

_CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


def _make_model(torch: Any, torchvision: Any, backbone: str, fc_output_dim: int):
    """Build the inference-time CosPlace network used by the official release."""

    nn = torch.nn
    functional = torch.nn.functional

    class L2Norm(nn.Module):
        def __init__(self, dim: int = 1) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return functional.normalize(x, p=2.0, dim=self.dim)

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return functional.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1)),
            ).pow(1.0 / self.p)

    class Flatten(nn.Module):
        def forward(self, x):
            if x.shape[2] != 1 or x.shape[3] != 1:
                raise ValueError(f"Expected pooled features of shape (B, C, 1, 1), got {tuple(x.shape)}")
            return x[:, :, 0, 0]

    def get_backbone() -> tuple[torch.nn.Module, int]:
        if backbone.startswith("ResNet"):
            model = getattr(torchvision.models, backbone.lower())(weights=None)
            layers = list(model.children())[:-2]
        elif backbone == "VGG16":
            model = torchvision.models.vgg16(weights=None)
            layers = list(model.features.children())[:-2]
        else:
            raise ValueError(f"Unsupported CosPlace backbone: {backbone}")

        return torch.nn.Sequential(*layers), _CHANNELS_NUM_IN_LAST_CONV[backbone]

    class GeoLocalizationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone, features_dim = get_backbone()
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm(),
            )

        def forward(self, x):
            return self.aggregation(self.backbone(x))

    return GeoLocalizationNet()


class CosPlace(VPRAlgorithm):
    """Official CosPlace descriptors pretrained on SF-XL."""

    name = "cosplace"

    def __init__(
        self,
        backbone: str = "ResNet50",
        fc_output_dim: int = 2048,
        checkpoint_path: str | Path | None = None,
        image_size: int = 512,
        device: str | None = None,
        batch_size: int = 8,
        pretrained: bool = True,
    ) -> None:
        if backbone not in _AVAILABLE_MODELS:
            raise ValueError(f"backbone must be one of: {', '.join(_AVAILABLE_MODELS)}.")
        if fc_output_dim not in _AVAILABLE_MODELS[backbone]:
            raise ValueError(
                f"fc_output_dim must be one of {', '.join(str(dim) for dim in _AVAILABLE_MODELS[backbone])} for {backbone}."
            )
        if image_size < 1:
            raise ValueError("image_size must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        try:
            import torch
            import torchvision
            from torchvision.transforms import v2
        except ImportError as error:
            raise ImportError("CosPlace requires torch and torchvision.") from error

        self.torch = torch
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
        self.batch_size = batch_size
        self.model = _make_model(torch, torchvision, backbone, fc_output_dim)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if pretrained:
            checkpoint = self._resolve_checkpoint(backbone, fc_output_dim, checkpoint_path)
            try:
                self._load_checkpoint(checkpoint)
            except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as error:
                if checkpoint_path is not None:
                    raise ValueError(f"CosPlace checkpoint is corrupted: {checkpoint}") from error
                print(f"Cached CosPlace checkpoint is incomplete; downloading it again: {checkpoint}")
                checkpoint.unlink(missing_ok=True)
                self._download_checkpoint(backbone, fc_output_dim, checkpoint)
                self._load_checkpoint(checkpoint)
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path cannot be used with pretrained=False.")

        self.model.eval().to(self.device)

    def _checkpoint_url(self, backbone: str, fc_output_dim: int) -> str:
        return f"https://github.com/gmberton/CosPlace/releases/download/v1.0/{backbone}_{fc_output_dim}_cosplace.pth"

    def _resolve_checkpoint(self, backbone: str, fc_output_dim: int, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is not None:
            path = Path(checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"CosPlace checkpoint not found: {path}")
            return path

        filename = f"{backbone}_{fc_output_dim}_cosplace.pth"
        path = Path(self.torch.hub.get_dir()) / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            self._download_checkpoint(backbone, fc_output_dim, path)
        return path

    def _download_checkpoint(self, backbone: str, fc_output_dim: int, path: Path) -> None:
        url = self._checkpoint_url(backbone, fc_output_dim)
        partial_path = path.with_suffix(path.suffix + ".part")
        partial_path.unlink(missing_ok=True)
        print(f"Downloading official CosPlace checkpoint to {path}")
        try:
            self.torch.hub.download_url_to_file(url, str(partial_path), progress=True)
            self.torch.load(partial_path, map_location="cpu", weights_only=True)
            partial_path.replace(path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = self.torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid CosPlace checkpoint: {checkpoint_path}")
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(f"Checkpoint is incompatible with the selected CosPlace settings: {checkpoint_path}") from error

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("CosPlace needs at least one image to encode.")

        descriptors = []
        for start in range(0, len(image_paths), self.batch_size):
            images = []
            for path in image_paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
            batch = self.torch.stack(images).to(self.device)
            with self.torch.inference_mode():
                descriptors.append(self.model(batch).cpu())
        return self.torch.cat(descriptors).numpy().astype(np.float32)


register_algorithm(CosPlace.name, CosPlace)
