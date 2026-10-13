# MSTF-SER

Official research code for **MSTF-SER: Motion-Semantic Temporal Fusion Network
for Multi-Person Sports Event Recognition**.

MSTF-SER combines person-centric appearance features, camera-motion-aware
trajectories, sparse player-interaction reasoning, and reliability-gated
temporal fusion. Given an RGB clip and actor tubes, the model predicts event
classes, spatial tubes, temporal boundaries, and tube-quality scores.

> This repository contains source code only. Datasets, pretrained weights,
> predictions, logs, figures, and reported experimental results are not
> distributed here.

## Repository layout

```text
mstf-ser/
|-- src/mstf_ser/          # Model, loss, data, tracking, export, post-processing
|-- scripts/               # Data preparation, training, inference, and evaluation
|-- tests/                 # Unit and smoke tests with synthetic inputs
|-- third_party/MultiSports/ # Official MultiSports evaluator and its notice
|-- docs/                  # Dataset and reproduction instructions
|-- pyproject.toml
`-- requirements.txt
```

## Installation

Python 3.9 or newer is required. Install PyTorch for the CUDA version available
on your machine, then install this package in editable mode:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Data

The code supports MultiSports, JHMDB, and Collective Activity. Obtain each
dataset from its official source and follow [docs/DATASETS.md](docs/DATASETS.md).
No dataset or annotation file is committed to this repository.

## MultiSports workflow

The following commands show the main end-to-end path. Paths are examples and
can be replaced with any local locations.

```bash
# Download and unpack the gated MultiSports release after accepting its terms.
python scripts/download_multisports.py
python scripts/prepare_multisports.py

# Train MSTF-SER.
python scripts/train_multisports.py \
  --annotation data/multisports/data/trainval/multisports_GT.pkl \
  --videos-root data/multisports/videos \
  --output runs/mstf_ser \
  --pretrained-backbone \
  --device cuda:0

# Build tracks from the official person proposals.
python scripts/download_official_person_boxes.py
python scripts/track_official_boxes.py \
  --annotation data/multisports/data/trainval/multisports_GT.pkl \
  --boxes data/multisports/person_boxes/MultiSports_box/val.recall_96.13.pkl \
  --output data/multisports/validation_tracks.pkl

# Infer frame- and video-level detections.
python scripts/infer_multisports.py \
  --annotation data/multisports/data/trainval/multisports_GT.pkl \
  --videos-root data/multisports/videos \
  --tracks data/multisports/validation_tracks.pkl \
  --checkpoint runs/mstf_ser/best.pt \
  --output runs/mstf_ser/predictions \
  --device cuda:0

# Run the official evaluator.
python scripts/evaluate_multisports.py \
  --ground-truth data/multisports/data/trainval/multisports_GT.pkl \
  --predictions runs/mstf_ser/predictions
```

The training objective is intentionally compact:

```text
L = L_classification + 2 L_spatial + L_temporal
```

## Cross-dataset training and baselines

`train_collective.py` fine-tunes the group recognizer on Collective Activity.
`train_multisports.py --classification-only` provides the unified person-tube
classification protocol used for MultiSports and JHMDB. The learning-based
comparison models are implemented in `train_unified_baseline.py`:

- Motion-GRU
- TSN-ShuffleNetV2
- RGB+Motion late fusion

Run `python <script> --help` for all dataset paths and hyperparameters. See
[docs/REPRODUCTION.md](docs/REPRODUCTION.md) for protocol details.

## Optional trackers

The repository includes adapters for ByteTrack and MixSort, but does not vendor
their source trees or weights. Clone those projects separately and pass their
locations through `--bytetrack-root` or `--mixsort-root`. This keeps third-party
code, model weights, and licenses separate from MSTF-SER.

## Testing

Tests use small synthetic clips and do not require research datasets or
checkpoints:

```bash
python -m pytest
```

## License

A project license has not been selected. Add an explicit license before making
the repository public. The official MultiSports evaluator retains its original
license and notice under `third_party/MultiSports/`.

