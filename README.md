# DL-WA-CSI

Paper-aligned code for **“Balancing Sensitivity and Spatial Fidelity in
Deuterium Metabolic Imaging with Weighted-Average CSI and Prior-Informed Deep
Learning Reconstruction.”**

DL-WA-CSI couples a scan-time-matched weighted-average (WA) chemical-shift
imaging operator to an anatomy- and spectral-prior-informed reconstruction
network. The central goal is to retain WA's low-spatial-frequency measurement
sensitivity while compensating for its broader spatial response.

> **Reproducibility status.** This repository implements the documented
> acquisition and analytical methods plus an explicit, testable reconstruction,
> training, inference, learned point-probe, and Monte Carlo-analysis scaffold
> aligned to the revised manuscript and Supporting Information. The supplied
> manuscript bundle did **not** contain the study's
> trained checkpoints, IXI subject split/masks, phantom or in-vivo data, exact
> spectral-fitting code, or several simulator constants. Consequently, the
> analytical UA/WA results are exactly reproducible here, while the paper's
> learned-response and biological result tables require those original assets.
> See [Paper alignment and limitations](https://github.com/MVChem/DL-WA-CSI/blob/main/docs/PAPER_ALIGNMENT.md).

## What is implemented

- Exact rounded 32×32 Hann-like repetition schedule: center 263, total 68,106.
- Scan-time-matched uniform (UA) reference and coherent-sum complex-noise
  acquisition physics.
- Dynamic water/glucose/lactate FID synthesis with configurable chemical shifts,
  T₂ decay, 72 spectral samples, and arbitrary dynamic length.
- A pure-PyTorch 3D spatial/temporal U-Net with:
  - early residual frequency-channel attention (FCA),
  - a multiscale high-resolution anatomical encoder,
  - DMI-query/anatomy-key-value cross-attention at every encoder and decoder
    scale,
  - aligned anatomy residuals, and
  - a temporal Transformer bottleneck.
- Acquisition-matched DL-UA-CSI and DL-WA-CSI training with MSE, Adam
  (β₁=0.9, β₂=0.999), 150 epochs, cosine LR 3×10⁻⁴ → 1×10⁻⁶, joint flips, and
  intensity scaling.
- Versioned checkpoints, fixed-model inference, exact analytical point-response
  analysis, and 4,800-scene reliable-recovery protocol scaffolding with
  full-ROI NRMSE, Wilson intervals, and logistic fits.
- Self-contained tMPPCA-style patch PCA, SPIN-SVD-style global low-rank, and
  local CNN-AE reference comparators. The revision omits comparator source and
  hyperparameters, so these are clearly labeled references—not claimed as
  bit-for-bit reproductions of unavailable implementations.
- CPU-fast tests for the physical invariants, network contract, statistics,
  data loading, and checkpoint round trips.

## Install

The manuscript environment was Python 3.11, PyTorch 2.0, and one 24 GB RTX
4090. Newer PyTorch versions are also supported.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

MRSpy is no longer required: the acquisition and compact DMI simulator are
implemented directly so the physical conventions are visible and testable.

Validate the installation:

```bash
pytest -q
dlwa-spatial-response
```

The analytical command should report approximately:

```json
{
  "total_repetitions": 68106,
  "ua_fwhm_voxels": 1.2071401397,
  "wa_fwhm_voxels": 1.9803230108
}
```

These are analytical acquisition PSFs. A learned reconstruction is nonlinear,
so its corresponding curve must be described as a **probe-specific empirical
effective point response**, not a universal system PSF.

A compatible versioned checkpoint trained by this implementation and its
co-registered anatomy can run the full learned probe path with:

```bash
dlwa-spatial-response \
  --checkpoint runs/dl-wa-csi/best.pt \
  --anatomy point-probe-anatomy.npy \
  --device cuda:0 \
  --output point-response.npz
```

## Data manifest

Training uses co-registered anatomy and spatial priors for three metabolites.
Use JSON Lines with paths relative to the manifest:

```json
{"id":"IXI001-slice042","anatomy":"images/t1.npy","water":"maps/water.npy","glucose":"maps/glucose.npy","lactate":"maps/lactate.npy"}
```

Supported inputs are `.npy`, `.png`, `.tif[f]`, and `.jpg`. Anatomy is retained
at 256×256 by default; metabolite priors are resampled to 32×32. NumPy priors
preserve their numeric/relative scale; raster priors are treated as shape masks
and divided by their positive maximum. A legacy text
manifest containing one sample directory per line is accepted when each folder
contains `anatomy`/`t1`, `water`, `glucose`/`glu`, and `lactate`/`lac` files.
Unlike the old prototype, the loader never silently substitutes a zero image
for the anatomical prior.

To exercise the pipeline without study data:

```bash
python scripts/generate_demo_data.py --output demo-data
```

The generated shapes are synthetic software-test fixtures, not study data and
not evidence for the manuscript's quantitative claims.

## Train acquisition-matched branches

DL-WA-CSI:

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

DL-UA-CSI uses the identical architecture and optimization protocol, changing
only the acquisition operator:

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

Documented/schematic values are 30 dynamic frames, 72 FID
channels, four spatial scales, 150 epochs, Adam, MSE, and cosine annealing.
Dynamic length is not hard-coded. Batch size, noise calibration, chemical
shifts/T₂ values, split membership, and augmentation ranges were absent from
the documents; the CLI exposes them rather than presenting guessed values as
paper constants. Inspect every option with `dlwa-train --help`.

The included manifest path provides static metabolite coefficient maps. The
default trainer applies one simulated temporal curve per sample and metabolite,
so all voxels in a given map share that normalized curve and no spatial phase
field is injected. The simulator API also accepts voxel-specific
`[B,T,M,H,W]` maps and a spatial phase term, but the tissue-specific curves,
masks, and phase model used for the study were not supplied. Reproducing that
part of the paper therefore requires integrating those original assets rather
than treating the fallback demo generator as equivalent.

The `0.002–1.6` range above preserves the historical prototype as a software
example; it is **not** calibrated paper noise. Both bounds are required so an
unstated fallback cannot silently become an experimental default. In this
implementation's explicit units, `per_excitation_noise_for_image_sd` converts a desired normalized
image-domain real/imaginary component SD. The SI's reported UA 3.60 mM and WA
0.94 mM values imply per-excitation SDs of approximately 940 and 970 under
center normalization. Their mismatch is another sign that the omitted study
preprocessing/calibration is needed for exact results.

## Inference

Provide an NPZ containing complex image-domain FIDs under `csi`, shaped
`[T,72,32,32]` or `[B,T,72,32,32]`, plus one co-registered anatomy path per
batch item:

> **Checkpoint safety:** Load checkpoints only from a trusted source. PyTorch
> loading requests restricted `weights_only=True` behavior. The compatibility
> fallback for APIs that do not accept that option uses pickle-based loading and
> must never receive an untrusted checkpoint.

```bash
dlwa-infer \
  --checkpoint runs/dl-wa-csi/best.pt \
  --input subject-wa.npz \
  --anatomy subject-t1.npy \
  --output subject-reconstruction.npz \
  --device cuda:0
```

No test-time fitting, retraining, or parameter update occurs.

The shipped training target is a magnitude FID, while `reconstruction` is a
**real-valued magnitude-domain estimate** and is not constrained to be
nonnegative; input phase is not reconstructed. This is an explicit, usable
fallback for the underspecified preprocessing in the revision, but it is not a
complex quantitative spectral reconstruction. In particular, do not pass this
array to `fit_metabolite_maps`: mixtures of FID magnitudes cannot be separated
by a linear magnitude-basis fit. Paper-equivalent concentration maps require
the missing study representation, preprocessing, and spectral-fitting
workflow—or a separately validated complex-preserving or magnitude-domain
estimator—which was not provided.

## Reliable-recovery analysis

Generate new concentration assignments following the stated rules (48 levels ×
100 independent scenes = 4,800 scenes):

```bash
dlwa-reliable-recovery generate --output protocol.npz
```

The SI prose does not uniquely identify the designated target tube. The
artwork appears to indicate zero-based tube index 5, at `(11,20)`, which is the
explicit default and can be changed with `--target-tube-index`.
The manuscript's original random seed, assignment identities, and sampling
probabilities were not provided. This command records a new explicit seed and
uses independent uniform draws from the allowed concentration set as a
documented reproducible instantiation—not as a reconstruction of the paper's
exact random scenes.

After all four fixed pipelines produce concentration maps, place arrays named
`ua_csi`, `wa_csi`, `dl_ua_csi`, and `dl_wa_csi` with shape
`[48,100,32,32]` into an NPZ and run:

```bash
dlwa-reliable-recovery score \
  --protocol protocol.npz \
  --estimates concentration-maps.npz \
  --output recovery-summary.npz
```

Only the designated 5×5 tube ROI is scored. Success requires full-voxel ROI
NRMSE **strictly below 10%**. Reported concentrations are simulation-derived
reliable-recovery summaries, not analytical or experimental detection limits.

## Repository layout

```text
dlwa_csi/
  acquisition.py    # UA/WA repetition maps, coherent acquisition, PSF/FWHM
  simulation.py     # dynamic spectral model and input formatting
  models.py         # prior-informed 3D Transformer U-Net
  data.py           # anatomy/metabolite manifests and augmentation
  training.py       # documented optimizer plus explicit fallback simulator
  inference.py      # fixed-checkpoint reconstruction
  experiments.py    # eight-tube Monte Carlo protocol
  metrics.py        # NRMSE, Wilson intervals, logistic model
  baselines.py      # explicit tMPPCA/SPIN-SVD reference paths and CNN-AE
scripts/
  spatial_response.py
  reliable_recovery.py
  generate_demo_data.py
tests/
configs/paper-aligned.json
docs/PAPER_ALIGNMENT.md
```

The root `train.py`/`infer.py` modules and `model.*` imports are packaged as
path-compatible aliases for the historical repository layout. The model aliases
do **not** preserve the old Diffusers-style constructor or raw-checkpoint format;
use versioned checkpoints created by the current training command.
Exact training resume must keep the same resolved device type (`cpu` or
`cuda`) because PyTorch generator-state formats are not portable between them;
the checkpoint contract rejects a cross-type resume instead of silently
changing the random trajectory.

## Interpretation guardrails

- Peak-normalized PSF/effective-response widths compare localization, not
  absolute sensitivity.
- WA is an acquisition-time repetition redistribution, not post-processing
  apodization.
- The learned point curve is probe-specific because the network is nonlinear.
- Nonlinear DL output is not assigned a conventional linear analytical LOD.
- In-vivo images have no voxelwise metabolite ground truth and support only
  cautious statements about signal stability and spatial delineation—not
  absolute accuracy or proven biological rim–core gradients.

## Citation

If you use this implementation, cite the accompanying manuscript:

> Chu H, Liu X, Chen G, et al. *Balancing Sensitivity and Spatial Fidelity in
> Deuterium Metabolic Imaging with Weighted-Average CSI and Prior-Informed Deep
> Learning Reconstruction.*

Add the journal citation and DOI after publication.

## Licensing and third-party notices

No project-wide software license has been declared for the original DL-WA-CSI
code. Do not infer permission to reuse it from the licenses of historical
third-party components. Their provenance and license notices are recorded in
[THIRD_PARTY_NOTICES.md](https://github.com/MVChem/DL-WA-CSI/blob/main/THIRD_PARTY_NOTICES.md).
