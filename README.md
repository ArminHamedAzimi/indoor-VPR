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
│   │   ├── dinov2.py
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

`rgb_histogram` is a fast baseline. `dinov2` downloads its model through
`torch.hub` on first use and then uses CUDA, Apple MPS, or CPU automatically.

## Add an algorithm

Implement `VPRAlgorithm` from `indoor_vpr.core` in a new module under
`src/indoor_vpr/algorithms/`, then call `register_algorithm()` at the bottom of
that module and import the implementation in `algorithms/__init__.py`.
