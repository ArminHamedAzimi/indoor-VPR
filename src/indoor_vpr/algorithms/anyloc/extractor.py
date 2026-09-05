from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import torch


Facet = Literal["query", "key", "value", "token"]


class DINOv2PatchExtractor:
    """Extract an intermediate DINOv2 facet for every image patch."""

    def __init__(
        self,
        model_name: str,
        layer: int,
        facet: Facet,
        device: str,
    ) -> None:
        import torch

        self.torch = torch
        self.facet = facet
        self.output: torch.Tensor | None = None
        self.model = cast(
            torch.nn.Module,
            torch.hub.load("facebookresearch/dinov2", model_name),
        ).eval().to(device)

        blocks = cast(torch.nn.ModuleList, getattr(self.model, "blocks"))
        if not -len(blocks) <= layer < len(blocks):
            raise ValueError(f"Layer {layer} is invalid for {model_name} ({len(blocks)} blocks).")
        block = blocks[layer]
        target = block if facet == "token" else cast(Any, block).attn.qkv
        target = cast(torch.nn.Module, target)
        self.hook = target.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: Any) -> None:
        self.output = cast("torch.Tensor", output)

    def __call__(self, image):
        functional = self.torch.nn.functional
        with self.torch.inference_mode():
            self.model(image)
            if self.output is None:
                raise RuntimeError("DINOv2 feature hook did not produce an output.")
            descriptors = self.output[:, 1:, :]
            if self.facet in {"query", "key", "value"}:
                descriptor_size = descriptors.shape[-1] // 3
                facet_index = {"query": 0, "key": 1, "value": 2}[self.facet]
                start = facet_index * descriptor_size
                descriptors = descriptors[..., start : start + descriptor_size]
            descriptors = functional.normalize(descriptors, dim=-1)
        self.output = None
        return descriptors

    def close(self) -> None:
        self.hook.remove()


class CLIPPatchExtractor:
    """Extract local ViT patch tokens from OpenAI CLIP.

    CLIP's pretrained positional embeddings are learned for its native square
    resolution, so images are resized to that resolution by ``AnyLoc`` before
    extraction.  Only ViT variants expose a patch-token grid; ResNet CLIP
    variants therefore are deliberately rejected.
    """

    def __init__(self, model_name: str, device: str) -> None:
        import torch

        self.torch = torch
        self.model = cast(
            torch.nn.Module,
            torch.hub.load("openai/CLIP", "load", model_name, device=device, jit=False)[0],
        ).eval().to(device)
        visual = cast(Any, getattr(self.model, "visual", None))
        if visual is None or not hasattr(visual, "conv1") or not hasattr(visual, "transformer"):
            raise ValueError(
                f"CLIP model '{model_name}' is not a Vision Transformer. "
                "Choose a ViT model such as 'ViT-B/16', 'ViT-B/32', or 'ViT-L/14'."
            )
        self.visual = visual
        self.image_size = int(visual.input_resolution)
        self.patch_size = int(visual.conv1.kernel_size[0])

    def __call__(self, image):
        """Run CLIP's visual ViT and return normalized patch tokens."""
        functional = self.torch.nn.functional
        with self.torch.inference_mode():
            features = self.visual.conv1(image)
            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.permute(0, 2, 1)
            class_token = self.visual.class_embedding.to(features.dtype)
            class_token = class_token + self.torch.zeros(
                features.shape[0], 1, features.shape[-1], dtype=features.dtype, device=features.device
            )
            features = self.torch.cat([class_token, features], dim=1)
            features = features + self.visual.positional_embedding.to(features.dtype)
            features = self.visual.ln_pre(features)
            features = features.permute(1, 0, 2)
            features = self.visual.transformer(features)
            features = features.permute(1, 0, 2)
            features = self.visual.ln_post(features)
            descriptors = functional.normalize(features[:, 1:, :], dim=-1)
        return descriptors
