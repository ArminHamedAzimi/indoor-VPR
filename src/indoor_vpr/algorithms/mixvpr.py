from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm


_OFFICIAL_CHECKPOINTS = {
    4096: ("https://drive.usercontent.google.com/download?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L&export=download&confirm=t", "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt", 1024, 4),
    512: ("https://drive.usercontent.google.com/download?id=1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj&export=download&confirm=t", "resnet50_MixVPR_512_channels(256)_rows(2).ckpt", 256, 2),
    128: ("https://drive.usercontent.google.com/download?id=1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz&export=download&confirm=t", "resnet50_MixVPR_128_channels(64)_rows(2).ckpt", 64, 2),
}


def _make_model(torch: Any, torchvision: Any, output_dim: int):
    """Build the inference-only equivalent of the official MixVPR model."""

    nn = torch.nn
    functional = torch.nn.functional
    _, _, out_channels, out_rows = _OFFICIAL_CHECKPOINTS[output_dim]

    class FeatureMixerLayer(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.mix = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim)
            )

        def forward(self, features):
            return features + self.mix(features)

    class MixVPRAggregator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            spatial_size = 20 * 20
            self.mix = nn.Sequential(*(FeatureMixerLayer(spatial_size) for _ in range(4)))
            self.channel_proj = nn.Linear(1024, out_channels)
            self.row_proj = nn.Linear(spatial_size, out_rows)

        def forward(self, features):
            features = self.mix(features.flatten(2))
            features = self.channel_proj(features.permute(0, 2, 1)).permute(0, 2, 1)
            features = self.row_proj(features)
            return functional.normalize(features.flatten(1), p=2, dim=-1)

    class ResNetBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torchvision.models.resnet50(weights=None)
            self.model.avgpool = None
            self.model.fc = None
            self.model.layer4 = None

        def forward(self, image):
            model = self.model
            image = model.maxpool(model.relu(model.bn1(model.conv1(image))))
            image = model.layer1(image)
            image = model.layer2(image)
            return model.layer3(image)

    class MixVPRModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = ResNetBackbone()
            self.aggregator = MixVPRAggregator()

        def forward(self, image):
            return self.aggregator(self.backbone(image))

    return MixVPRModel()


class MixVPR(VPRAlgorithm):
    """Official ResNet-50 MixVPR descriptors trained on GSV-Cities."""

    name = "mixvpr"

    def __init__(self, output_dim: int = 4096, checkpoint_path: str | Path | None = None,
                 device: str | None = None, batch_size: int = 8, pretrained: bool = True) -> None:
        if output_dim not in _OFFICIAL_CHECKPOINTS:
            raise ValueError("output_dim must be one of: 128, 512, 4096.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        try:
            import torch
            import torchvision
            from torchvision.transforms import v2
        except ImportError as error:
            raise ImportError("MixVPR requires torch and torchvision.") from error

        self.torch = torch
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.transform = v2.Compose([
            v2.ToImage(), v2.Resize((320, 320), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = _make_model(torch, torchvision, output_dim)
        if pretrained:
            checkpoint = self._resolve_checkpoint(output_dim, checkpoint_path)
            try:
                self._load_checkpoint(checkpoint)
            except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as error:
                if checkpoint_path is not None:
                    raise ValueError(f"MixVPR checkpoint is corrupted: {checkpoint}") from error
                print(f"Cached MixVPR checkpoint is incomplete; downloading it again: {checkpoint}")
                checkpoint.unlink(missing_ok=True)
                self._download_checkpoint(output_dim, checkpoint)
                self._load_checkpoint(checkpoint)
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path cannot be used with pretrained=False.")
        self.model.eval().to(self.device)

    def _resolve_checkpoint(self, output_dim: int, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is not None:
            path = Path(checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"MixVPR checkpoint not found: {path}")
            return path
        url, filename, _, _ = _OFFICIAL_CHECKPOINTS[output_dim]
        path = Path(self.torch.hub.get_dir()) / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            self._download_checkpoint(output_dim, path)
        return path

    def _download_checkpoint(self, output_dim: int, path: Path) -> None:
        url, _, _, _ = _OFFICIAL_CHECKPOINTS[output_dim]
        partial_path = path.with_suffix(path.suffix + ".part")
        partial_path.unlink(missing_ok=True)
        print(f"Downloading official MixVPR checkpoint to {path}")
        try:
            self.torch.hub.download_url_to_file(url, str(partial_path), progress=True)
            # A checkpoint is a zip archive in current PyTorch releases. Opening
            # it here prevents an interrupted or HTML response from entering cache.
            try:
                with ZipFile(partial_path) as archive:
                    if not archive.namelist():
                        raise BadZipFile("checkpoint archive is empty")
            except BadZipFile as error:
                raise RuntimeError("The MixVPR download is not a valid PyTorch checkpoint.") from error
            partial_path.replace(path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = self.torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid MixVPR checkpoint: {checkpoint_path}")
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(f"Checkpoint is incompatible with the selected MixVPR output_dim: {checkpoint_path}") from error

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("MixVPR needs at least one image to encode.")
        descriptors = []
        for start in range(0, len(image_paths), self.batch_size):
            images = []
            for path in image_paths[start:start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
            batch = self.torch.stack(images).to(self.device)
            with self.torch.inference_mode():
                descriptors.append(self.model(batch).cpu())
        return self.torch.cat(descriptors).numpy().astype(np.float32)


register_algorithm(MixVPR.name, MixVPR)
