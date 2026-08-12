"""Transparent comparator implementations for matched UA/WA branches.

The revision names tMPPCA, SPIN-SVD, and a locally trained CNN autoencoder but
does not provide their code or hyperparameters. The functions below are
self-contained reference comparators with explicit settings; they must not be
presented as exact reproductions of unavailable third-party implementations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_spatiospectral(data: Tensor) -> None:
    if not isinstance(data, Tensor):
        raise TypeError("data must be a torch.Tensor")
    if data.ndim < 3:
        raise ValueError("data must have signal dimensions followed by [H, W]")
    if data.shape[-2] < 2 or data.shape[-1] < 2:
        raise ValueError("spatial dimensions must be at least 2")
    if not (data.is_floating_point() or torch.is_complex(data)):
        raise TypeError("data must be floating point or complex")


def _mp_rank(singular_values: Tensor, rows: int, columns: int) -> int:
    """Estimate retained rank with a finite-matrix Marchenko-Pastur edge."""

    components = singular_values.numel()
    squared = singular_values.real.square()
    for rank in range(components):
        remaining_rows = rows - rank
        remaining_columns = columns - rank
        if remaining_rows <= 0 or remaining_columns <= 0:
            break
        noise_variance = squared[rank:].sum() / (remaining_rows * remaining_columns)
        upper_edge = noise_variance * (
            remaining_rows**0.5 + remaining_columns**0.5
        ) ** 2
        if squared[rank] <= upper_edge:
            return rank
    return components


def tmppca_reference(
    data: Tensor,
    *,
    patch_size: int = 5,
    stride: int = 2,
    rank: int | None = None,
) -> Tensor:
    """Patchwise tensor MP-PCA reference denoiser.

    All leading signal axes (normally dynamic and FID) form the measurement
    dimension, while each spatial patch supplies voxel observations. Overlap is
    averaged. Pass ``rank`` for a fixed-rank ablation or leave it unset for the
    MP-edge estimate.
    """

    _validate_spatiospectral(data)
    if patch_size < 2 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer of at least 3")
    if stride < 1:
        raise ValueError("stride must be positive")
    if stride > patch_size:
        raise ValueError("stride cannot exceed patch_size because it would leave gaps")
    if rank is not None and rank < 0:
        raise ValueError("rank cannot be negative")

    height, width = data.shape[-2:]
    signal_shape = data.shape[:-2]
    features = int(torch.tensor(signal_shape).prod().item())
    flat = data.reshape(features, height, width)
    radius = patch_size // 2
    padded = F.pad(flat, (radius, radius, radius, radius), mode="reflect")
    output = torch.zeros_like(padded)
    weights = torch.zeros(
        (1, padded.shape[-2], padded.shape[-1]),
        dtype=data.real.dtype,
        device=data.device,
    )
    y_positions = list(range(0, height, stride))
    x_positions = list(range(0, width, stride))
    if y_positions[-1] != height - 1:
        y_positions.append(height - 1)
    if x_positions[-1] != width - 1:
        x_positions.append(width - 1)

    for y in y_positions:
        for x in x_positions:
            patch = padded[:, y : y + patch_size, x : x + patch_size]
            matrix = patch.reshape(features, -1)
            mean = matrix.mean(dim=1, keepdim=True)
            centered = matrix - mean
            u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            retained = _mp_rank(singular_values, *centered.shape) if rank is None else rank
            retained = min(retained, singular_values.numel())
            if retained:
                reconstructed = (u[:, :retained] * singular_values[:retained]) @ vh[:retained]
                reconstructed = reconstructed + mean
            else:
                reconstructed = mean.expand_as(centered)
            output[:, y : y + patch_size, x : x + patch_size] += reconstructed.reshape(
                features, patch_size, patch_size
            )
            weights[:, y : y + patch_size, x : x + patch_size] += 1
    cropped = output[:, radius : radius + height, radius : radius + width]
    cropped_weights = weights[:, radius : radius + height, radius : radius + width]
    return (cropped / cropped_weights.clamp_min(1)).reshape(*signal_shape, height, width)


def spin_svd_reference(data: Tensor, *, rank: int = 8) -> Tensor:
    """Global spatial/spectral low-rank SVD reference comparator."""

    _validate_spatiospectral(data)
    if rank < 1:
        raise ValueError("rank must be positive")
    height, width = data.shape[-2:]
    matrix = data.reshape(-1, height * width)
    mean = matrix.mean(dim=1, keepdim=True)
    centered = matrix - mean
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    retained = min(rank, singular_values.numel())
    reconstructed = (u[:, :retained] * singular_values[:retained]) @ vh[:retained]
    return (reconstructed + mean).reshape_as(data)


class CNNAutoencoder(nn.Module):
    """Local 3-D convolutional autoencoder comparator without anatomy input."""

    def __init__(self, spectral_channels: int = 72, hidden_channels: int = 48) -> None:
        super().__init__()
        if spectral_channels < 1 or hidden_channels < 2:
            raise ValueError("spectral_channels and hidden_channels must be positive")
        self.spectral_channels = spectral_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(spectral_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels * 2,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, spectral_channels, 3, padding=1),
        )

    def forward(self, dmi: Tensor) -> Tensor:
        if dmi.ndim != 5:
            raise ValueError("dmi must have shape [B, T, C, H, W]")
        if dmi.shape[2] != self.spectral_channels:
            raise ValueError(f"expected {self.spectral_channels} spectral channels")
        internal = dmi.permute(0, 2, 1, 3, 4)
        encoded = self.encoder(internal)
        decoded = self.decoder(encoded)
        decoded = F.interpolate(
            decoded,
            size=(dmi.shape[1], dmi.shape[3], dmi.shape[4]),
            mode="trilinear",
            align_corners=False,
        )
        return decoded.permute(0, 2, 1, 3, 4).contiguous()
