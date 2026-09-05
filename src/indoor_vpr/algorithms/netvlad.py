from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from indoor_vpr.core import VPRAlgorithm, register_algorithm


# Official checkpoints published by the authors of CosPlace with their
# deep-visual-geo-localization-benchmark.  The AnyLoc benchmark vendors that
# same project for its NetVLAD baseline.
_CHECKPOINT_IDS = {
    "vgg16": ("14s7OZor6wrlGBKeXr0vKbPfTzlW9preM", "1dwai3uNudjvns58JIyaf5CBRg4ojcWIW"),
    "resnet18conv4": ("1KFwonDQYdvzTAIILsOMjmLRUR76jXXvB", "1_Ozq2TdvwLAJUwy7YH9l69GsfOU-MlFZ"),
    "resnet50conv4": ("1KL8HoAApOjJFETin7Q7u7IcsOvroKlSj", "1krf0A6CeW8GqLqHWZ7dlSNJ9aTJ4dotF"),
    "resnet101conv4": ("1064kDJ0LPyWoU7J4bMvAa0lTNEhAEi8v", "1rtPfsgfJ2Zoxs5uu7Ph1_qc7q-hIxJek"),
    "cct384": ("1Rx0oG4PG9bEraIg4y7e6Z24Q6b_TGr5u", "1wDZ6XRVYz6bcGe_p3Iiz2NfIe9MmZZMN"),
}
_BACKBONE_ALIASES = {"vgg16": "vgg16", "resnet18-conv4": "resnet18conv4", "resnet18conv4": "resnet18conv4", "resnet50-conv4": "resnet50conv4", "resnet50conv4": "resnet50conv4", "resnet101-conv4": "resnet101conv4", "resnet101conv4": "resnet101conv4", "cct-384": "cct384", "cct384": "cct384"}


def _make_cct384(torch: Any):
    """Dependency-free CCT-14/7x2/384 definition used by the upstream checkpoint."""
    nn, F = torch.nn, torch.nn.functional

    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.num_heads = 6; self.scale = 64 ** -0.5
            self.qkv = nn.Linear(384, 1152, bias=False); self.attn_drop = nn.Dropout(0.1)
            self.proj = nn.Linear(384, 384); self.proj_drop = nn.Dropout(0.0)
        def forward(self, x):
            b, n, c = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, 6, 64).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
            x = (self.attn_drop(x) @ v).transpose(1, 2).reshape(b, n, c)
            return self.proj_drop(self.proj(x))

    class Block(nn.Module):
        def __init__(self, rate):
            super().__init__(); self.pre_norm = nn.LayerNorm(384); self.self_attn = Attention()
            self.linear1 = nn.Linear(384, 1152); self.dropout1 = nn.Dropout(0.0)
            self.norm1 = nn.LayerNorm(384); self.linear2 = nn.Linear(1152, 384); self.dropout2 = nn.Dropout(0.0)
            self.drop_path = nn.Identity(); self.activation = F.gelu
        def forward(self, x):
            x = self.norm1(x + self.drop_path(self.self_attn(self.pre_norm(x))))
            return x + self.drop_path(self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x))))))

    class Tokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            def layer(in_channels, out_channels):
                return nn.Sequential(nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
            self.conv_layers = nn.Sequential(layer(3, 64), layer(64, 384)); self.flattener = nn.Flatten(2, 3)
        def forward(self, x): return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__(); self.embedding_dim = 384; self.sequence_length = 576; self.seq_pool = True
            self.attention_pool = nn.Linear(384, 1); self.positional_emb = nn.Parameter(torch.zeros(1, 576, 384))
            self.dropout = nn.Dropout(0.0); self.blocks = nn.ModuleList([Block(i / 13 * 0.1) for i in range(14)]); self.norm = nn.LayerNorm(384)
        def forward(self, x):
            x = self.dropout(x + self.positional_emb)
            for block in self.blocks: x = block(x)
            return self.norm(x)

    class CCT(nn.Module):
        def __init__(self): super().__init__(); self.tokenizer = Tokenizer(); self.classifier = Classifier(); self.aggregation = None
        def forward(self, x): return self.classifier(self.tokenizer(x))
    return CCT()


def _make_model(torch: Any, torchvision: Any, backbone_name: str, num_clusters: int = 64):
    """Build the upstream CNN/CCT backbone with its compatible NetVLAD head."""

    nn = torch.nn
    functional = torch.nn.functional
    feature_dim = 256

    class NetVLADAggregator(nn.Module):
        def __init__(self, dim: int, work_with_tokens: bool = False) -> None:
            super().__init__()
            self.clusters_num = num_clusters
            self.dim = dim
            self.alpha = 0
            self.normalize_input = True
            self.work_with_tokens = work_with_tokens
            self.conv = (nn.Conv1d(dim, num_clusters, kernel_size=1, bias=False) if work_with_tokens else nn.Conv2d(dim, num_clusters, kernel_size=1, bias=False))
            self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

        def forward(self, features):
            if self.work_with_tokens:
                features = features.permute(0, 2, 1)
            batch_size, channels = features.shape[:2]
            features = functional.normalize(features, p=2, dim=1)
            flat_features = features.reshape(batch_size, channels, -1)
            assignments = functional.softmax(
                self.conv(features).reshape(batch_size, self.clusters_num, -1),
                dim=1,
            )

            # Compute one cluster at a time, matching the maintained benchmark.
            # This avoids a much larger B x K x D x H*W temporary tensor.
            vlad = torch.empty(
                (batch_size, self.clusters_num, channels),
                dtype=features.dtype,
                device=features.device,
            )
            for cluster_index in range(self.clusters_num):
                residual = flat_features - self.centroids[cluster_index].view(1, channels, 1)
                weighted = residual * assignments[:, cluster_index : cluster_index + 1]
                vlad[:, cluster_index] = weighted.sum(dim=-1)

            vlad = functional.normalize(vlad, p=2, dim=2)
            return functional.normalize(vlad.reshape(batch_size, -1), p=2, dim=1)

    class NetVLADModel(nn.Module):
        def __init__(self, backbone, dim: int, work_with_tokens: bool = False) -> None:
            super().__init__()
            self.backbone = backbone
            self.aggregation = NetVLADAggregator(dim, work_with_tokens)

        def forward(self, image):
            return self.aggregation(self.backbone(image))

    if backbone_name == "vgg16":
        backbone = nn.Sequential(*list(torchvision.models.vgg16(weights=None).features.children())[:-2]); dim = 512
    elif backbone_name in {"resnet18conv4", "resnet50conv4", "resnet101conv4"}:
        model = getattr(torchvision.models, backbone_name.removesuffix("conv4"))(weights=None)
        backbone = nn.Sequential(*list(model.children())[:-3])
        dim = {"resnet18conv4": 256, "resnet50conv4": 1024, "resnet101conv4": 1024}[backbone_name]
    else:
        backbone = _make_cct384(torch); dim = 384
    return NetVLADModel(backbone, dim, backbone_name == "cct384"), dim


class NetVLAD(VPRAlgorithm):
    """Official ResNet-18 + NetVLAD model pretrained for place recognition."""

    name = "netvlad"

    def __init__(
        self,
        backbone: str = "resnet18conv4",
        trained_on: str = "pitts30k",
        checkpoint_path: str | Path | None = None,
        image_size: tuple[int, int] = (480, 640),
        device: str | None = None,
        batch_size: int = 8,
        pretrained: bool = True,
    ) -> None:
        backbone = _BACKBONE_ALIASES.get(backbone.casefold())
        trained_on = trained_on.casefold()
        if backbone is None:
            raise ValueError(f"backbone must be one of: {', '.join(_CHECKPOINT_IDS)}.")
        if trained_on not in {"pitts30k", "msls"}:
            raise ValueError("trained_on must be either 'pitts30k' or 'msls'.")
        if backbone == "cct384" and image_size != (384, 384):
            raise ValueError("CCT-384 requires image_size=(384, 384).")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain positive (height, width) values.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        try:
            import torch
            import torchvision
        except ImportError as error:
            raise ImportError("NetVLAD requires torch and torchvision.") from error

        self.torch = torch
        self.backbone_name = backbone
        self.trained_on = trained_on
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
        # Match dvgl_benchmark: tensor -> ImageNet normalize -> tensor resize.
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                torchvision.transforms.Resize(tuple(image_size)),
            ]
        )
        self.model, self.descriptor_dim = _make_model(torch, torchvision, backbone)

        if pretrained:
            checkpoint = self._resolve_checkpoint(trained_on, checkpoint_path)
            try:
                self._load_checkpoint(checkpoint)
            except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as error:
                if checkpoint_path is not None:
                    raise ValueError(f"NetVLAD checkpoint is corrupted: {checkpoint}") from error
                print(f"Cached NetVLAD checkpoint is incomplete; downloading it again: {checkpoint}")
                checkpoint.unlink(missing_ok=True)
                self._download_checkpoint(trained_on, checkpoint)
                self._load_checkpoint(checkpoint)
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path cannot be used with pretrained=False.")

        self.model.eval().to(self.device)

    def _resolve_checkpoint(self, trained_on: str, checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is not None:
            path = Path(checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"NetVLAD checkpoint not found: {path}")
            return path

        filename = f"{self.backbone_name}_netvlad_{trained_on}.pth"
        path = Path(self.torch.hub.get_dir()) / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            self._download_checkpoint(trained_on, path)
        return path

    def _download_checkpoint(self, trained_on: str, path: Path) -> None:
        checkpoint_index = 0 if trained_on == "pitts30k" else 1
        checkpoint_id = _CHECKPOINT_IDS[self.backbone_name][checkpoint_index]
        url = f"https://drive.usercontent.google.com/download?id={checkpoint_id}&export=download&confirm=t"
        partial_path = path.with_suffix(path.suffix + ".part")
        partial_path.unlink(missing_ok=True)
        print(f"Downloading official NetVLAD ({trained_on}) checkpoint to {path}")
        try:
            self.torch.hub.download_url_to_file(url, str(partial_path), progress=True)
            self.torch.load(partial_path, map_location="cpu", weights_only=True)
            partial_path.replace(path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = self.torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid NetVLAD checkpoint: {checkpoint_path}")
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = OrderedDict(
                (key.removeprefix("module."), value) for key, value in state_dict.items()
            )
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint is incompatible with {self.backbone_name} NetVLAD with 64 clusters: "
                f"{checkpoint_path}"
            ) from error

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("NetVLAD needs at least one image to encode.")

        # NetVLAD descriptors are large (16,384 floats each), so fill the final
        # array batch by batch instead of retaining every batch and concatenating
        # them, which would briefly duplicate the full descriptor set in memory.
        descriptors = np.empty((len(image_paths), 64 * self.descriptor_dim), dtype=np.float32)
        for start in range(0, len(image_paths), self.batch_size):
            images = []
            for path in image_paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
            batch = self.torch.stack(images).to(self.device)
            with self.torch.inference_mode():
                batch_descriptors = self.model(batch).cpu().numpy()
            descriptors[start : start + len(batch_descriptors)] = batch_descriptors
        return descriptors


register_algorithm(NetVLAD.name, NetVLAD)
