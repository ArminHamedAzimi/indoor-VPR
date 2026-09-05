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
│   │   ├── cosplace.py
│   │   ├── dinov2.py
│   │   ├── mixvpr.py
│   │   ├── netvlad.py
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
checkpoint on first use. `cosplace` downloads the official SF-XL pretrained model
on first use. `netvlad` downloads an official pretrained checkpoint from the
maintained deep visual geo-localization benchmark. All learned methods use CUDA,
Apple MPS, or CPU automatically.
AnyLoc can load an official vocabulary or fit one from the selected database;
see its [adapter notes](src/indoor_vpr/algorithms/anyloc/README.md).

MixVPR is a direct inference-only port of AnyLoc's bundled baseline. It uses
320 x 320 inputs and the bundled 4096-dimensional ResNet-50 checkpoint:

```python
create_algorithm("mixvpr", output_dim=4096, batch_size=8)
```

MixVPR uses the official ResNet50 backbone and 4096-dimensional checkpoint.
Choose the image geometry mode independently:

```python
create_algorithm(
    "mixvpr",
    backbone="ResNet50",
    output_dim=4096,
    batch_size=8,
    preprocessing="stretch",  # "stretch", "letterbox", or "two_crops"
)
```

`stretch` is the official 320 x 320 preprocessing. `letterbox` preserves the
whole image aspect ratio with mean-color padding. `two_crops` covers portrait
frames with two overlapping square views and uses the maximum crop-to-crop
cosine similarity.

Pass `checkpoint_path="...ckpt"` to use an already downloaded compatible checkpoint.
These checkpoints were trained on outdoor Google Street View imagery, so evaluate
them on representative indoor data before relying on their rankings.

CosPlace uses the official pretrained SF-XL checkpoints from the upstream release.
The default wrapper uses `ResNet50` with a 2048-dimensional descriptor. Those
weights were trained on outdoor geo-localization data, so they are not a universal
fit for every indoor image set.

NetVLAD uses the ResNet-18 backbone cropped after `layer3` (called
`resnet18conv4` upstream), 64 VLAD clusters, and a 16,384-dimensional descriptor.
The default checkpoint was trained on Pitts30k; an MSLS-trained checkpoint is also
available:

```python
create_algorithm("netvlad", trained_on="pitts30k", batch_size=8)
# Or: create_algorithm("netvlad", trained_on="msls", batch_size=8)
```

This is the pretrained NetVLAD family published by the CosPlace authors' deep
visual geo-localization benchmark and vendored by AnyLoc for its NetVLAD baseline.
Like the MixVPR and CosPlace weights, it was trained on outdoor place-recognition
data, so its indoor performance must be measured on the target dataset.

## ARKit 6-DoF localization with HLoc

The localization workflow uses the two recordings in `datasets/dataset-3`:

1. `notebooks/localization/anyloc.ipynb` extracts pose-synchronized frames,
   runs AnyLoc retrieval, and writes `outputs/localization/anyloc_retrieval.xlsx`.
2. `notebooks/localization/hloc.ipynb` triangulates a metric reference map at
   the recorded ARKit poses and estimates each query pose with HLoc PnP.
3. `notebooks/localization/benchmark_hloc.ipynb` computes translation error,
   rotation error, PnP success rate, and adjustable threshold recalls without
   rerunning either learned model.

The HLoc and benchmark notebooks also render query-reference local feature
correspondences, top-down and 3D camera trajectories, per-frame pose errors,
and PnP inlier counts. Figures are saved under `outputs/localization`.

Install the additional official HLoc stack once:

```bash
./scripts/install_hloc.sh
```

The HLoc adapter runs neural feature/matching stages and PyCOLMAP geometry
stages in separate subprocesses. This avoids the duplicate OpenMP-runtime crash
caused by loading the macOS PyTorch and PyCOLMAP wheels in one process.

ARKit world coordinates are not guaranteed to be shared by separate recording
sessions. The benchmark reports both raw-coordinate error and a first-pose
alignment that assumes the two recordings started at the same physical camera
pose. Use an independently measured cross-session transform for rigorous final
accuracy claims.

## Add an algorithm

Implement `VPRAlgorithm` from `indoor_vpr.core` in a new module under
`src/indoor_vpr/algorithms/`, then call `register_algorithm()` at the bottom of
that module and import the implementation in `algorithms/__init__.py`.
