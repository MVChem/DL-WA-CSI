"""Torch-native dynamic DMI signal synthesis and network formatting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .acquisition import apply_repetition_acquisition
from .contracts import NETWORK_SCALE_QUANTILE


@dataclass(frozen=True)
class SpectralModel:
    """A compact three-metabolite FID model used by the training simulator."""

    spectral_points: int = 72
    spectral_bandwidth_hz: float = 4065.0
    peak_offsets_hz: tuple[float, ...] = (0.0, -55.0, -209.0)
    t2_seconds: tuple[float, ...] = (0.080, 0.070, 0.060)
    metabolite_names: tuple[str, ...] = ("water", "glucose", "lactate")

    def __post_init__(self) -> None:
        count = len(self.metabolite_names)
        if count == 0 or len(self.peak_offsets_hz) != count or len(self.t2_seconds) != count:
            raise ValueError("metabolite_names, peak_offsets_hz, and t2_seconds must match")
        if self.spectral_points < 2 or self.spectral_bandwidth_hz <= 0:
            raise ValueError("spectral_points and spectral_bandwidth_hz must be positive")
        if any(value <= 0 for value in self.t2_seconds):
            raise ValueError("all T2 values must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SPECTRAL_MODEL = SpectralModel()


def spectral_basis(
    model: SpectralModel = DEFAULT_SPECTRAL_MODEL,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return complex metabolite basis FIDs with shape ``[M, C]``."""

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    sample_time = torch.arange(
        model.spectral_points, dtype=real_dtype, device=device
    ) / model.spectral_bandwidth_hz
    offsets = torch.tensor(model.peak_offsets_hz, dtype=real_dtype, device=device)[:, None]
    t2 = torch.tensor(model.t2_seconds, dtype=real_dtype, device=device)[:, None]
    decay = torch.exp(-sample_time[None, :] / t2)
    phase = torch.exp(-2j * torch.pi * offsets * sample_time[None, :])
    return (decay * phase).to(dtype=dtype)


def synthesize_dynamic_fids(
    metabolite_maps: torch.Tensor,
    concentration_curves: torch.Tensor | None = None,
    *,
    model: SpectralModel = DEFAULT_SPECTRAL_MODEL,
    spatial_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthesize clean complex image-domain FIDs.

    ``metabolite_maps`` may be static ``[B,M,H,W]`` priors paired with curves
    shaped ``[T,M]`` or ``[B,T,M]``. For tissue-/voxel-specific dynamics it may
    instead contain the complete amplitudes as ``[B,T,M,H,W]``, in which case
    ``concentration_curves`` must be omitted. Output is ``[B,T,C,H,W]``.
    """

    if metabolite_maps.ndim not in (4, 5):
        raise ValueError(
            "metabolite_maps must have shape [B,M,H,W] or [B,T,M,H,W]"
        )
    if not metabolite_maps.is_floating_point():
        metabolite_maps = metabolite_maps.float()
    real_dtype = metabolite_maps.dtype
    if metabolite_maps.ndim == 4:
        batch, metabolites, _, _ = metabolite_maps.shape
    else:
        batch, _, metabolites, _, _ = metabolite_maps.shape
    if metabolites != len(model.metabolite_names):
        raise ValueError(
            f"expected {len(model.metabolite_names)} metabolites, got {metabolites}"
        )
    if metabolite_maps.ndim == 5:
        if concentration_curves is not None:
            raise ValueError("omit concentration_curves when dynamic maps are supplied")
        amplitudes = metabolite_maps
    else:
        if concentration_curves is None:
            raise ValueError("static metabolite maps require concentration_curves")
        if concentration_curves.ndim == 2:
            concentration_curves = concentration_curves.unsqueeze(0).expand(batch, -1, -1)
        if concentration_curves.ndim != 3 or concentration_curves.shape[0] != batch:
            raise ValueError("concentration_curves must have shape [T,M] or [B,T,M]")
        if concentration_curves.shape[-1] != metabolites:
            raise ValueError("curve metabolite dimension does not match the maps")
        amplitudes = (
            concentration_curves.to(
                device=metabolite_maps.device, dtype=real_dtype
            )[:, :, :, None, None]
            * metabolite_maps[:, None]
        )

    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    basis = spectral_basis(model, dtype=complex_dtype, device=metabolite_maps.device)
    fids = torch.einsum("btmhw,mc->btchw", amplitudes.to(complex_dtype), basis)
    if spatial_phase is not None:
        if spatial_phase.shape != (batch, *metabolite_maps.shape[-2:]):
            raise ValueError("spatial_phase must have shape [B, H, W]")
        if not spatial_phase.is_floating_point():
            raise TypeError("spatial_phase must use a floating-point dtype")
        phase = torch.exp(-1j * spatial_phase.to(device=fids.device, dtype=real_dtype))
        fids = fids * phase[:, None, None]
    return fids


def fit_metabolite_maps(
    spatiospectral_data: torch.Tensor,
    *,
    model: SpectralModel = DEFAULT_SPECTRAL_MODEL,
    nonnegative: bool = True,
) -> torch.Tensor:
    """Fit water/glucose/lactate amplitudes with a transparent linear model.

    Parameters
    ----------
    spatiospectral_data:
        ``[B,T,C,H,W]`` complex FIDs. Magnitude channels cannot be fitted by
        linearly combining magnitude basis functions because
        ``abs(sum(a*b)) != sum(a*abs(b))``.
    nonnegative:
        Return coefficient magnitudes (complex case) or clamp real estimates
        at zero. The study's exact spectral-fitting procedure was not supplied,
        so this helper is an explicit reference rather than a claimed match.
    """

    if spatiospectral_data.ndim != 5:
        raise ValueError("spatiospectral_data must have shape [B, T, C, H, W]")
    if spatiospectral_data.shape[2] != model.spectral_points:
        raise ValueError(
            f"expected {model.spectral_points} spectral points, "
            f"got {spatiospectral_data.shape[2]}"
        )
    if not torch.is_complex(spatiospectral_data):
        raise TypeError(
            "linear spectral fitting requires complex FIDs; the exact fitting "
            "method for magnitude network outputs was not supplied"
        )
    complex_dtype = (
        torch.complex128
        if spatiospectral_data.dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    basis = spectral_basis(model, dtype=complex_dtype, device=spatiospectral_data.device)
    design = basis.transpose(0, 1)
    values = spatiospectral_data.to(complex_dtype)
    inverse = torch.linalg.pinv(design)
    coefficients = torch.einsum("mc,btchw->btmhw", inverse, values)
    if nonnegative:
        if torch.is_complex(coefficients):
            coefficients = coefficients.abs()
        else:
            coefficients = coefficients.clamp_min(0)
    return coefficients


def random_smooth_curves(
    batch_size: int,
    dynamic_frames: int,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate smooth water/glucose/lactate concentration trajectories."""

    if batch_size < 1 or dynamic_frames < 1:
        raise ValueError("batch_size and dynamic_frames must be positive")
    controls = 5
    random_values = torch.rand(
        (batch_size, 3, controls), dtype=dtype, device=device, generator=generator
    )
    maxima = 30.0 + 30.0 * torch.rand(
        (batch_size, 1, 1), dtype=dtype, device=device, generator=generator
    )
    water = 20.0 + random_values[:, 0:1] * (maxima - 20.0)
    glucose = 5.0 * random_values[:, 1:2]
    lactate = 5.0 * random_values[:, 2:3]
    values = torch.cat((water, glucose, lactate), dim=1)
    if dynamic_frames == 1:
        curves = values[:, :, :1]
    else:
        curves = F.interpolate(values, size=dynamic_frames, mode="linear", align_corners=True)
        # A short reflected moving average removes interpolation corners while
        # retaining independently sampled trajectories.
        if dynamic_frames >= 5:
            padded = F.pad(curves, (2, 2), mode="reflect")
            kernel = torch.tensor([1, 4, 6, 4, 1], dtype=dtype, device=device) / 16.0
            curves = F.conv1d(
                padded.reshape(batch_size * 3, 1, -1), kernel.reshape(1, 1, -1)
            ).reshape(batch_size, 3, dynamic_frames)
    return curves.transpose(1, 2).contiguous()


@dataclass(frozen=True)
class NetworkPair:
    degraded: torch.Tensor
    target: torch.Tensor
    complex_degraded: torch.Tensor
    scale: torch.Tensor


def prepare_network_pair(
    clean_fids: torch.Tensor,
    repetition_map: torch.Tensor,
    *,
    noise_std_per_excitation: float | torch.Tensor,
    generator: torch.Generator | None = None,
    percentile: float = NETWORK_SCALE_QUANTILE,
) -> NetworkPair:
    """Create the acquisition-matched magnitude input/target pair.

    The same per-sample scale is derived from the acquired input and applied to
    both input and target. This avoids target leakage and exactly matches the
    scale available to :func:`format_inference` at deployment.
    """

    if clean_fids.ndim != 5:
        raise ValueError("clean_fids must have shape [B, T, C, H, W]")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must lie in (0, 1]")
    result = apply_repetition_acquisition(
        clean_fids,
        repetition_map,
        noise_std_per_excitation=noise_std_per_excitation,
        normalization="center",
        generator=generator,
    )
    degraded_magnitude = result.image.abs()
    target_magnitude = clean_fids.abs()
    flattened = degraded_magnitude.flatten(1)
    scale = torch.quantile(flattened, percentile, dim=1).clamp_min(1e-8)
    shape = (clean_fids.shape[0],) + (1,) * (clean_fids.ndim - 1)
    scale_view = scale.reshape(shape)
    return NetworkPair(
        degraded=(degraded_magnitude / scale_view).to(target_magnitude.dtype),
        target=(target_magnitude / scale_view).to(target_magnitude.dtype),
        complex_degraded=result.image,
        scale=scale,
    )


def format_inference(
    acquired_fids: torch.Tensor,
    *,
    percentile: float = NETWORK_SCALE_QUANTILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitude-format acquired complex FIDs without target leakage."""

    if acquired_fids.ndim != 5:
        raise ValueError("acquired_fids must have shape [B, T, C, H, W]")
    magnitude = acquired_fids.abs() if torch.is_complex(acquired_fids) else acquired_fids.abs()
    scale = torch.quantile(magnitude.flatten(1), percentile, dim=1).clamp_min(1e-8)
    scale_view = scale.reshape((magnitude.shape[0],) + (1,) * (magnitude.ndim - 1))
    return magnitude / scale_view, scale
