# Paper alignment and reproducibility boundary

This document separates implemented manuscript requirements from numerical
claims that cannot be independently reproduced from the supplied files.

## Authoritative sources audited

- `0811_manscript_v4.docx`
- `0811_SI_v4.docx`
- `0811_Response_letter_v4.docx`

The manuscript points to `https://gitee.com/txz32102/DL-WA-CSI`. The SI and
response letter contain no repository URL. None of the three documents includes
a data archive or trained checkpoint.

## Requirement-to-code map

| Paper requirement | Implementation / evidence |
|---|---|
| 32×32 rounded Hann schedule, center 263, total 68,106 | `dlwa_csi.acquisition.hann_repetition_map`; golden test |
| Scan-time-matched constant UA map | `matched_uniform_repetition_map`; exact total test |
| Coherent repetition sum; signal ×N, noise SD ×√N | `apply_repetition_acquisition` |
| Explicit image-noise calibration under chosen normalization | `per_excitation_noise_for_image_sd` |
| Post-acquisition apodization kept distinct from WA | `separable_hann_window`, `apply_kspace_apodization` |
| Complex 2-D k-space encoding and inverse reconstruction | `apply_repetition_acquisition` |
| Dynamic metabolite maps, chemical shifts, T₂ decay, complex FIDs | `SpectralModel` and `synthesize_dynamic_fids`; API accepts voxel-specific dynamics |
| WA-degraded/clean supervised pairs | `prepare_network_pair` |
| DMI `[B,T,C,H,W]`, arbitrary T, paper C=72 | `PriorInformedUNet3D` contract tests |
| High-resolution co-registered anatomy input | multiscale `_AnatomyEncoder`; 32×32/256×256-compatible test |
| DMI Q, anatomy K/V cross-attention on encoder and decoder | `_EfficientAnatomicalCrossAttention` at every scale |
| Early spectral-channel attention and residual propagation | `_SpectralChannelAttention` |
| 3-D U-Net with time preserved through spatial pyramid | Conv3d blocks and `(1,2,2)` downsampling |
| Dynamic frames modeled as sequence | temporal Transformer at bottleneck |
| MSE, Adam β=(.9,.999), 150 epochs, cosine 3e-4→1e-6 | `dlwa_csi.training` defaults |
| Joint random flips and intensity scaling | `JointAugmentation` |
| Same architecture for DL-UA and DL-WA | `--branch` changes only repetition map |
| Fixed-model inference | `dlwa-infer`; no optimizer/test-time update |
| UA/WA analytical PSF, 2¹⁸ interpolation, half-height width | `sinc_interpolated_profile`, `profile_fwhm`, CLI |
| UA 1.207 and WA 1.980 nominal voxels | golden tests reproduce 1.207140 and 1.980323 |
| Acquisition-only point probe equals analytical route | 72-channel point-probe golden test |
| Fixed-checkpoint learned point-probe workflow | `dlwa-spatial-response --checkpoint ... --anatomy ...` |
| Fixed eight-tube geometry and 0.25–12 mM grid | `generate_recovery_protocol` and tests |
| 100 repeats/level, replacement allowed, 4,800 scenes | protocol generator and invariant tests |
| Full-voxel target-ROI NRMSE <10% | `summarize_recovery_maps`; strict-threshold test |
| Empirical probability and 95% Wilson intervals | `metrics.wilson_interval` |
| Binary logistic fit and P=.95 crossing | `fit_logistic_binary`, `logistic_crossing` |
| Named tMPPCA/SPIN-SVD/CNN-AE paths | transparent references in `baselines.py`; exact external settings unavailable |

## Exact analytical values

The SI defines

```text
W_WA(k,l) = round(1 + (263-1)/4
                        * [1+cos(2πk/32)]
                        * [1+cos(2πl/32)])
```

Ties-to-even rounding is required: half-away-from-zero produces 68,110 rather
than the stated 68,106. Tests therefore make the center and total authoritative.
The 32-point central k-space line is centered in 262,144 points, inverse
transformed, peak-normalized, and measured at the nearest linearly interpolated
half-height crossings. The results are:

| Acquisition | FWHM (nominal voxels) |
|---|---:|
| UA-CSI | 1.2071401397 |
| WA-CSI | 1.9803230108 |

## Assets needed for the remaining paper numbers

The following were neither present in the repository nor supplied with the
revision bundle:

1. Fixed DL-UA-CSI and DL-WA-CSI state dictionaries used in the figures.
2. The exact 72-point training spectral model, phase model, intensity scaling,
   and branch noise calibration.
3. IXI subject IDs, train/validation/test split, FSL BET/FAST commands, tissue
   masks, manual tumor masks, coefficient matrices, and dynamic curves.
4. Simulation, physical phantom, and in-vivo DMI tensors plus predefined ROIs.
5. Exact spectral fitting, tMPPCA, SPIN-SVD, and local CNN-AE configurations.
6. Seeds/repeat identities behind means and standard deviations.

Without these, it would be misleading to claim independent reproduction of:

- the 1.221-voxel learned lactate effective response;
- Table S1/S3/S4 PSNR and SSIM values;
- branch noise SDs 3.60 and 0.94 mM;
- P=.95 crossings 7.25, 4.82, 3.79, and 2.31 mM;
- phantom calibration/LOD values or in-vivo findings.

The code exposes missing simulator choices explicitly and can ingest compatible
data. An original state dictionary would first require a rigorously validated
conversion to this repository's versioned architecture and recorded runtime
contract. It does not encode paper-reported outputs as if they were newly
computed results.

## Implemented fallback boundaries

The manifest-based trainer receives static metabolite coefficient maps and
generates one temporal curve per sample/metabolite. Thus every nonzero voxel in
one metabolite map shares the same normalized time course. Although
`synthesize_dynamic_fids` supports voxel-specific `[B,T,M,H,W]` dynamics and a
spatial phase term, the default training path does not invent the missing
tissue-class/tumor curves or phase model.

Training targets are magnitude-FID channels; the unconstrained network produces
a real-valued magnitude-domain estimate. The inference NPZ therefore does not
contain reconstructed complex FIDs. `fit_metabolite_maps` deliberately rejects
this output: for a
mixture, `abs(sum(a_m b_m))` is not `sum(a_m abs(b_m))`, so linear fitting of
magnitude basis functions is invalid. Exact quantitative metabolite mapping
requires the missing study representation, preprocessing, and spectral fitter,
or a separately validated complex-preserving or magnitude-domain estimator.

The reliable-recovery generator samples the seven context concentrations IID
uniformly from the stated allowed grid. Replacement is documented in the SI,
but its sampling probabilities, random seed, and original scene identities are
not. The generator is therefore a reproducible protocol instantiation and the
scoring command expects externally generated four-pipeline concentration maps;
it is not an end-to-end reproduction of the unavailable Monte Carlo run.

## Forward-model convention

Main-text Eq. (1) places `N_ex(k)` outside a parenthesis containing both signal
and noise, which would make noise SD grow as `N`. The SI explicitly derives the
coherent sum of independent repetitions (S1.5–S1.7): signal grows as `N`, noise
variance as `N`, and noise SD as `sqrt(N)`. The implementation follows that SI
derivation. Relatedly, the main text's “approximately 4 times SNR” wording
corresponds to the SI's approximately 4× *center mean amplitude*; the stated
center measurement-SNR gain is its square root, approximately 2×.

The SI proves that before branch-specific intensity normalization the global
image-domain noise variance at fixed total repetitions is independent of the
repetition distribution. It later reports different normalized UA/WA image
noise SDs without fully specifying that normalization. This implementation
uses an explicit `normalization="center"`: coherent k-space sums are divided by
the branch's largest repetition count. This preserves the low-frequency signal
scale and makes the convention inspectable. Change or calibrate it only with a
recorded configuration.

The simulator parameter `noise_std_per_excitation` is the standard deviation of
each real and imaginary component. If `CN(0, sigma²)` is instead defined as
total complex variance, pass `sigma/sqrt(2)` under this convention.

## Scope of claims

- Call Monte Carlo output **simulation-derived reliable recovery**, not a
  practical, analytical, or experimental detection limit.
- Call a fixed nonlinear network's point result a **probe-specific empirical
  effective point response**, not an object-independent system PSF.
- Do not infer absolute in-vivo reconstruction accuracy without ground truth.
- Do not claim biological heterogeneity or rim–core gradients are proven by
  reconstruction alone.
