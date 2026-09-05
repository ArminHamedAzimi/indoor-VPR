from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm


# AnyLoc bundles only the strongest public MixVPR model: ResNet-50 cropped
# before layer4, followed by the 4096-dimensional MixVPR aggregator.
_ANYLOC_CHECKPOINT = (
    "https://drive.usercontent.google.com/download?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L&export=download&confirm=t",
    "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
)


def _make_model(torch: Any, torchvision: Any):
    """Direct inference-only port of AnyLoc/MixVPR's model classes."""

    nn = torch.nn
    functional = torch.nn.functional

    class ResNet(nn.Module):
        """AnyLoc's ResNet-50 backbone with residual block 4 cropped."""

        def __init__(self) -> None:
            super().__init__()
            self.model_name = "resnet50"
            self.layers_to_freeze = 1
            # The checkpoint contains the complete trained backbone, so loading
            # ImageNet weights first would only add an unnecessary download.
            self.model = torchvision.models.resnet50(weights=None)
            self.model.avgpool = None
            self.model.fc = None
            self.model.layer4 = None
            self.out_channels = 1024

        def forward(self, image):
            image = self.model.conv1(image)
            image = self.model.bn1(image)
            image = self.model.relu(image)
            image = self.model.maxpool(image)
            image = self.model.layer1(image)
            image = self.model.layer2(image)
            if self.model.layer3 is not None:
                image = self.model.layer3(image)
            if self.model.layer4 is not None:
                image = self.model.layer4(image)
            return image

    class FeatureMixerLayer(nn.Module):
        def __init__(self, in_dim: int, mlp_ratio: int = 1) -> None:
            super().__init__()
            self.mix = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, int(in_dim * mlp_ratio)),
                nn.ReLU(),
                nn.Linear(int(in_dim * mlp_ratio), in_dim),
            )

        def forward(self, features):
            return features + self.mix(features)

    class MixVPRAggregator(nn.Module):
        """AnyLoc's bundled 1024 x 4 MixVPR aggregator."""

        def __init__(self) -> None:
            super().__init__()
            self.in_h = 20
            self.in_w = 20
            self.in_channels = 1024
            self.out_channels = 1024
            self.out_rows = 4
            self.mix_depth = 4
            self.mlp_ratio = 1

            spatial_dim = self.in_h * self.in_w
            self.mix = nn.Sequential(
                *(
                    FeatureMixerLayer(spatial_dim, self.mlp_ratio)
                    for _ in range(self.mix_depth)
                )
            )
            self.channel_proj = nn.Linear(self.in_channels, self.out_channels)
            self.row_proj = nn.Linear(spatial_dim, self.out_rows)

        def forward(self, features):
            features = features.flatten(2)
            features = self.mix(features)
            features = features.permute(0, 2, 1)
            features = self.channel_proj(features)
            features = features.permute(0, 2, 1)
            features = self.row_proj(features)
            return functional.normalize(features.flatten(1), p=2, dim=-1)

    class VPRModel(nn.Module):
        """AnyLoc/MixVPR VPRModel stripped of training-only fields."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = ResNet()
            self.aggregator = MixVPRAggregator()

        def forward(self, image):
            return self.aggregator(self.backbone(image))

    return VPRModel()


class MixVPR(VPRAlgorithm):
    """AnyLoc's bundled pretrained ResNet-50 MixVPR baseline."""

    name = "mixvpr"

    def __init__(
        self,
        output_dim: int = 4096,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 8,
        pretrained: bool = True,
        preprocessing: str = "stretch",
        backbone: str = "ResNet50",
    ) -> None:
        if backbone.casefold() != "resnet50":
            raise ValueError("The official 4096-D MixVPR checkpoint requires backbone='ResNet50'.")
        if output_dim != 4096:
            raise ValueError(
                "AnyLoc bundles only the 4096-dimensional MixVPR checkpoint; "
                "output_dim must be 4096."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if preprocessing == "anyloc":
            # Backward-compatible name used by earlier versions of the notebook.
            preprocessing = "stretch"
        if preprocessing not in {"stretch", "letterbox", "two_crops"}:
            raise ValueError(
                "preprocessing must be 'stretch', 'letterbox', or 'two_crops'."
            )

        try:
            import torch
            import torchvision
        except ImportError as error:
            raise ImportError("MixVPR requires torch and torchvision.") from error

        self.torch = torch
        self.torchvision = torchvision
        self.batch_size = batch_size
        self.preprocessing = preprocessing
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

        # AnyLoc normalizes the tensor before resizing. Keeping the geometric
        # operation separate lets us compare the official square stretch with
        # aspect-preserving letterbox and two square crops.
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.model = _make_model(torch, torchvision)

        if pretrained:
            checkpoint = self._resolve_checkpoint(checkpoint_path)
            try:
                self._load_checkpoint(checkpoint)
            except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as error:
                if checkpoint_path is not None:
                    raise ValueError(f"MixVPR checkpoint is corrupted: {checkpoint}") from error
                print(f"Cached MixVPR checkpoint is incomplete; downloading it again: {checkpoint}")
                checkpoint.unlink(missing_ok=True)
                self._download_checkpoint(checkpoint)
                self._load_checkpoint(checkpoint)
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path cannot be used with pretrained=False.")

        self.model.eval().to(self.device)

    def _resolve_checkpoint(self, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is not None:
            path = Path(checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"MixVPR checkpoint not found: {path}")
            return path

        _, filename = _ANYLOC_CHECKPOINT
        path = Path(self.torch.hub.get_dir()) / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            self._download_checkpoint(path)
        return path

    def _download_checkpoint(self, path: Path) -> None:
        url, _ = _ANYLOC_CHECKPOINT
        partial_path = path.with_suffix(path.suffix + ".part")
        partial_path.unlink(missing_ok=True)
        print(f"Downloading the MixVPR checkpoint bundled by AnyLoc to {path}")
        try:
            self.torch.hub.download_url_to_file(url, str(partial_path), progress=True)
            try:
                with ZipFile(partial_path) as archive:
                    if not archive.namelist():
                        raise BadZipFile("checkpoint archive is empty")
            except BadZipFile as error:
                raise RuntimeError("The MixVPR download is not a valid checkpoint.") from error
            partial_path.replace(path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = self.torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid MixVPR checkpoint: {checkpoint_path}")
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value for key, value in state_dict.items()
            }
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint is incompatible with AnyLoc's 4096-dimensional MixVPR: {checkpoint_path}"
            ) from error

    @property
    def views_per_image(self) -> int:
        return 2 if self.preprocessing == "two_crops" else 1

    def _resize_square(self, image: Image.Image):
        tensor = self.normalize(self.to_tensor(image))
        return self.torchvision.transforms.functional.resize(
            tensor,
            [320, 320],
            interpolation=self.torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _image_views(self, image: Image.Image):
        image = image.convert("RGB")
        if self.preprocessing == "stretch":
            return [self._resize_square(image)]

        width, height = image.size
        if self.preprocessing == "letterbox":
            tensor = self.normalize(self.to_tensor(image))
            scale = 320 / max(height, width)
            resized_height = max(1, round(height * scale))
            resized_width = max(1, round(width * scale))
            tensor = self.torchvision.transforms.functional.resize(
                tensor,
                [resized_height, resized_width],
                interpolation=self.torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            horizontal_padding = 320 - resized_width
            vertical_padding = 320 - resized_height
            padding = [
                horizontal_padding // 2,
                vertical_padding // 2,
                horizontal_padding - horizontal_padding // 2,
                vertical_padding - vertical_padding // 2,
            ]
            # Zero after ImageNet normalization is the ImageNet mean color,
            # which is less out-of-distribution than black padding.
            return [
                self.torchvision.transforms.functional.pad(tensor, padding, fill=0)
            ]

        side = min(width, height)
        if height >= width:
            crop_boxes = [(0, 0, side, side), (0, height - side, side, height)]
        else:
            crop_boxes = [(0, 0, side, side), (width - side, 0, width, side)]
        return [self._resize_square(image.crop(box)) for box in crop_boxes]

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("MixVPR needs at least one image to encode.")

        descriptor_size = 4096 * self.views_per_image
        descriptors = np.empty((len(image_paths), descriptor_size), dtype=np.float32)
        image_batch_size = max(1, self.batch_size // self.views_per_image)
        for start in range(0, len(image_paths), image_batch_size):
            paths = image_paths[start : start + image_batch_size]
            views = []
            for path in paths:
                with Image.open(path) as image:
                    views.extend(self._image_views(image))

            view_descriptors = []
            for view_start in range(0, len(views), self.batch_size):
                batch = self.torch.stack(
                    views[view_start : view_start + self.batch_size]
                ).to(self.device)
                with self.torch.inference_mode():
                    view_descriptors.append(self.model(batch).cpu().numpy())
            batch_descriptors = np.concatenate(view_descriptors, axis=0)
            batch_descriptors = batch_descriptors.reshape(len(paths), descriptor_size)
            descriptors[start : start + len(paths)] = batch_descriptors
        return descriptors

    def similarity(
        self,
        query_descriptors: np.ndarray,
        database_descriptors: np.ndarray,
    ) -> np.ndarray:
        if self.preprocessing != "two_crops":
            return super().similarity(query_descriptors, database_descriptors)

        expected_size = self.views_per_image * 4096
        if (
            query_descriptors.shape[1] != expected_size
            or database_descriptors.shape[1] != expected_size
        ):
            raise ValueError(
                f"Two-crop MixVPR descriptors must have {expected_size} values per image."
            )
        queries = query_descriptors.reshape(-1, self.views_per_image, 4096)
        database = database_descriptors.reshape(-1, self.views_per_image, 4096)
        queries = queries / np.maximum(
            np.linalg.norm(queries, axis=2, keepdims=True), 1e-12
        )
        database = database / np.maximum(
            np.linalg.norm(database, axis=2, keepdims=True), 1e-12
        )

        scores = queries[:, 0] @ database[:, 0].T
        for query_view in range(self.views_per_image):
            for database_view in range(self.views_per_image):
                if query_view == 0 and database_view == 0:
                    continue
                candidate = queries[:, query_view] @ database[:, database_view].T
                np.maximum(scores, candidate, out=scores)
        return scores


register_algorithm(MixVPR.name, MixVPR)
