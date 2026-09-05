# AnyLoc adapter

This implementation adapts the official [AnyLoc](https://github.com/AnyLoc/AnyLoc)
DINOv2 + VLAD inference design to the `indoor_vpr.core.VPRAlgorithm` interface.
The upstream project is BSD-3-Clause licensed.

The paper configuration uses `dinov2_vitg14`, layer 31, the value facet, and 32
VLAD clusters. For reproducible inference, download the upstream `cache.zip` and
set `vocabulary_path` to the desired domain's `c_centers.pt`, for example:

```python
{
    "model_name": "dinov2_vitg14",
    "layer": 31,
    "facet": "value",
    "num_clusters": 32,
    "vocabulary_path": "cache/vocabulary/dinov2_vitg14/l31_value_c32/indoor/c_centers.pt",
}
```

When `vocabulary_path=None`, the adapter fits a VLAD vocabulary from local patch
descriptors of the currently selected database. That mode is convenient for new
datasets but is not directly comparable with results using the published AnyLoc
vocabularies.

## CLIP patch descriptors

The adapter can also aggregate local OpenAI CLIP ViT patch tokens with VLAD:

```python
{
    "feature_model": "clip",
    "model_name": "ViT-B/16",
    "num_clusters": 32,
    "vocabulary_path": None,
}
```

The first CLIP run downloads OpenAI's CLIP repository and model weights through
`torch.hub`. Use a separate vocabulary for each CLIP model; published AnyLoc
vocabularies are DINOv2-specific. CLIP uses its native square input resolution,
whereas DINOv2 retains AnyLoc's aspect-ratio-preserving preprocessing.
