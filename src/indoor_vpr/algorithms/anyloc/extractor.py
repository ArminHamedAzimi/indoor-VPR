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
