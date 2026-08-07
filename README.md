# Indoor VPR

A notebook-first playground for comparing visual place recognition algorithms and
datasets.

## Project layout

```text
indoor-VPR/
├── datasets/                    # Image databases and query sets
├── notebooks/
│   └── vpr_experiments.ipynb    # Main experiment interface
├── src/indoor_vpr/
│   ├── core/                    # Shared framework; no implementations
│   │   ├── algorithm.py         # VPR algorithm contract
│   │   ├── data.py              # Dataset loading interface
│   │   ├── pipeline.py          # Similarity and ranking
│   │   └── registry.py          # Algorithm registration/creation
│   ├── algorithms/              # Concrete implementations only
│   │   ├── anyloc/              # DINOv2 patch features + VLAD
│   │   ├── dinov2.py
│   │   ├── mixvpr.py
│   │   └── rgb_histogram.py
│   └── visualization.py         # Notebook plots
├── match.py                     # Original standalone experiment
└── requirements.txt
```

## Start the notebook

From the repository root:

```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/vpr_experiments.ipynb
```

In the notebook's **Configuration** cell:

- change `DATABASE_DIR` and `QUERY_DIR` to switch datasets;
- change `ALGORITHM` to switch the descriptor method;
- use `MAX_DATABASE_IMAGES` and `MAX_QUERY_IMAGES` for quick experiments;
- edit `ALGORITHM_OPTIONS` to tune an algorithm.

`rgb_histogram` is a fast baseline. `dinov2` and `anyloc` download DINOv2 through
`torch.hub` on first use. `mixvpr` downloads the official GSV-Cities pretrained
checkpoint on first use. All learned methods use CUDA, Apple MPS, or CPU automatically.
AnyLoc can load an official vocabulary or fit one from the selected database;
see its [adapter notes](src/indoor_vpr/algorithms/anyloc/README.md).

MixVPR requires 320 x 320 inputs and offers the official 4096-, 512-, and
128-dimensional checkpoints. The strongest 4096-dimensional model is the default:

```python
create_algorithm("mixvpr", output_dim=4096, batch_size=8)
```

Pass `checkpoint_path="...ckpt"` to use an already downloaded official checkpoint.
These checkpoints were trained on outdoor Google Street View imagery, so evaluate
them on representative indoor data before relying on their rankings.

## Add an algorithm

Implement `VPRAlgorithm` from `indoor_vpr.core` in a new module under
`src/indoor_vpr/algorithms/`, then call `register_algorithm()` at the bottom of
that module and import the implementation in `algorithms/__init__.py`.
