# DL-WA-CSI

Training and reconstruction code for **“Balancing Sensitivity and Spatial
Fidelity in Deuterium Metabolic Imaging with Weighted-Average CSI and
Prior-Informed Deep Learning Reconstruction.”**

This repository is intentionally limited to the model-training and fixed-model
reconstruction workflow. It does not include the study data, trained model
weights, or the experimental and statistical analysis pipelines used to
produce the manuscript figures and tables.

## Included scope

- Scan-time-matched uniform-average (UA) and weighted-average (WA) acquisition
  operators for preparing network inputs.
- Dynamic water/glucose/lactate FID simulation used by the training pipeline.
- A prior-informed 3D spatial/temporal U-Net with spectral-channel attention,
  multiscale anatomical features, cross-attention, and a temporal Transformer.
- Acquisition-matched DL-UA-CSI and DL-WA-CSI training.
- Fixed-checkpoint reconstruction for acquired CSI data.
- Tests for the training and reconstruction critical paths.

## Installation

The manuscript environment used Python 3.11 and PyTorch 2.0. Newer compatible
PyTorch versions can also be used.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Training data

Training uses co-registered anatomy and spatial priors for three metabolites.
Provide a JSON Lines manifest with paths relative to the manifest file:

```json
{"id":"IXI001-slice042","anatomy":"images/t1.npy","water":"maps/water.npy","glucose":"maps/glucose.npy","lactate":"maps/lactate.npy"}
```

Supported inputs are `.npy`, `.png`, `.tif[f]`, and `.jpg`. Anatomy is retained
at 256×256 by default, and metabolite priors are resampled to 32×32. A legacy
text manifest containing one sample directory per line is also accepted when
each directory contains `anatomy`/`t1`, `water`, `glucose`/`glu`, and
`lactate`/`lac` files.

To generate small synthetic fixtures for a software smoke test:

```bash
python scripts/generate_demo_data.py --output demo-data
```

These fixtures are not study data and must not be used as evidence for the
manuscript's quantitative results.

## Train

Train the DL-WA-CSI branch:

```bash
dlwa-train \
  --train-manifest demo-data/train.jsonl \
  --val-manifest demo-data/val.jsonl \
  --output-dir runs/dl-wa-csi \
  --branch wa \
  --noise-std-min 0.002 \
  --noise-std-max 1.6 \
  --device cuda:0
```

Train the DL-UA-CSI branch with the same architecture and optimization
protocol by changing only the acquisition branch:

```bash
dlwa-train \
  --train-manifest demo-data/train.jsonl \
  --val-manifest demo-data/val.jsonl \
  --output-dir runs/dl-ua-csi \
  --branch ua \
  --noise-std-min 0.002 \
  --noise-std-max 1.6 \
  --device cuda:0
```

The documented training defaults are 30 dynamic frames, 72 FID channels, 150
epochs, Adam with MSE loss, and cosine learning-rate annealing. Inspect all
options with:

```bash
dlwa-train --help
```

The example noise bounds above are software defaults and must be calibrated for
the intended dataset.

## Reconstruct

Provide an NPZ file containing complex image-domain FIDs under `csi`, shaped
`[T,72,32,32]` or `[B,T,72,32,32]`, plus one co-registered anatomical image per
batch item:

```bash
dlwa-infer \
  --checkpoint runs/dl-wa-csi/best.pt \
  --input subject-wa.npz \
  --anatomy subject-t1.npy \
  --output subject-reconstruction.npz \
  --device cuda:0
```

No fitting, retraining, or parameter update occurs during reconstruction. Load
only trusted checkpoints.

## Repository layout

```text
dlwa_csi/
  acquisition.py    # UA/WA acquisition operators
  simulation.py     # dynamic spectral simulation and input formatting
  models.py         # prior-informed reconstruction network
  data.py           # anatomy/metabolite manifests and augmentation
  training.py       # model training
  inference.py      # fixed-checkpoint reconstruction
  checkpointing.py  # model checkpoint loading and saving
  contracts.py      # training/reconstruction metadata validation
scripts/
  generate_demo_data.py
tests/
configs/paper-aligned.json
```

The root `train.py` and `infer.py` modules are convenience entry points.

## Citation

If you use this implementation, cite the accompanying manuscript:

> Chu H, Liu X, Chen G, et al. *Balancing Sensitivity and Spatial Fidelity in
> Deuterium Metabolic Imaging with Weighted-Average CSI and Prior-Informed Deep
> Learning Reconstruction.*

Add the journal citation and DOI after publication.

## Licensing and third-party notices

No project-wide software license has been declared for the original DL-WA-CSI
code. Third-party provenance and license notices are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
