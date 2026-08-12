"""Scan-time-matched UA-CSI and WA-CSI acquisition operators.

The implementation follows Supporting Information equations S1.4--S1.15.
Repetitions are coherently summed, so a repetition count ``N(k)`` multiplies
the noise-free k-space signal while the standard deviation of independent
per-excitation noise grows as ``sqrt(N(k))``.

All repetition maps use *centered* k-space ordering.  The transform helpers
therefore apply ``ifftshift`` before an FFT and ``fftshift`` afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

MatrixSize = int | Sequence[int]


def _matrix_size_2d(matrix_size: MatrixSize) -> tuple[int, int]:
    if isinstance(matrix_size, int):
        matrix_size = (matrix_size, matrix_size)
    if len(matrix_size) != 2:
        raise ValueError("matrix_size must be an int or a two-element sequence")
    nx, ny = (int(value) for value in matrix_size)
    if nx < 2 or ny < 2:
        raise ValueError("both matrix dimensions must be at least 2")
    return nx, ny


def hann_repetition_map(
    matrix_size: MatrixSize = (32, 32),
    center_repetitions: int = 263,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the paper's rounded, separable Hann-like WA schedule.

    For the paper configuration (32 x 32, ``center_repetitions=263``), the
    returned map has a center value of 263, an outer value of 1, and a total
    of 68,106 repetitions.
    """

    nx, ny = _matrix_size_2d(matrix_size)
    if center_repetitions < 1:
        raise ValueError("center_repetitions must be positive")

    work_dtype = torch.float64
    kx = torch.arange(nx, dtype=work_dtype, device=device) - nx // 2
    ky = torch.arange(ny, dtype=work_dtype, device=device) - ny // 2
    x, y = torch.meshgrid(kx, ky, indexing="ij")
    two_pi = 2.0 * torch.pi
    schedule = 1.0 + (center_repetitions - 1.0) / 4.0 * (
        1.0 + torch.cos(two_pi * x / nx)
    ) * (1.0 + torch.cos(two_pi * y / ny))
    return torch.round(schedule).to(dtype=dtype)


def separable_hann_window(
    matrix_size: MatrixSize = (32, 32),
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the centered ideal 2-D Hann window used for apodization controls."""

    nx, ny = _matrix_size_2d(matrix_size)
    kx = torch.arange(nx, dtype=torch.float64, device=device) - nx // 2
    ky = torch.arange(ny, dtype=torch.float64, device=device) - ny // 2
    x, y = torch.meshgrid(kx, ky, indexing="ij")
    window = 0.25 * (1.0 + torch.cos(2.0 * torch.pi * x / nx)) * (
        1.0 + torch.cos(2.0 * torch.pi * y / ny)
    )
    return window.to(dtype=dtype)


def matched_uniform_repetition_map(
    wa_map: torch.Tensor | None = None,
    *,
    matrix_size: MatrixSize = (32, 32),
    center_repetitions: int = 263,
    integer: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a UA schedule with exactly the WA schedule's scan-time budget.

    The analytical map is spatially constant and can therefore contain a
    fractional value (68,106 / 1,024 for the paper configuration).  Set
    ``integer=True`` to obtain the nearest-balanced physical schedule while
    preserving the exact integer total; the extra repetitions are assigned
    from the center outwards.
    """

    if wa_map is None:
        wa_map = hann_repetition_map(
            matrix_size,
            center_repetitions,
            dtype=torch.float64,
            device=device,
        )
    if wa_map.ndim != 2:
        raise ValueError("wa_map must be two-dimensional")
    if torch.any(wa_map < 0):
        raise ValueError("repetition counts cannot be negative")

    nx, ny = wa_map.shape
    total = wa_map.to(torch.float64).sum()
    if not integer:
        return torch.full(
            (nx, ny),
            total / (nx * ny),
            dtype=dtype,
            device=wa_map.device,
        )

    total_int = int(round(float(total.item())))
    base, remainder = divmod(total_int, nx * ny)
    result = torch.full((nx, ny), base, dtype=torch.int64, device=wa_map.device)
    if remainder:
        x = torch.arange(nx, device=wa_map.device) - nx // 2
        y = torch.arange(ny, device=wa_map.device) - ny // 2
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        order = torch.argsort((xx.square() + yy.square()).flatten(), stable=True)
        result.flatten()[order[:remainder]] += 1
    return result.to(dtype=dtype)


@dataclass(frozen=True)
class AcquisitionResult:
    """Outputs of a coherent repetition-weighted acquisition."""

    image: torch.Tensor
    kspace: torch.Tensor
    repetition_map: torch.Tensor
    normalization: torch.Tensor


def apply_kspace_apodization(
    image_fids: torch.Tensor,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply post-acquisition windowing as a distinct smoothing baseline.

    This multiplies already acquired data in k-space. It does not change the
    measurement SNR of any retained k-space sample and must not be described as
    weighted-average acquisition.
    """

    if image_fids.ndim < 2:
        raise ValueError("image_fids must have at least two spatial dimensions")
    if not torch.is_complex(image_fids):
        image_fids = torch.complex(image_fids, torch.zeros_like(image_fids))
    if window is None:
        window = separable_hann_window(
            image_fids.shape[-2:], dtype=image_fids.real.dtype, device=image_fids.device
        )
    if window.ndim != 2 or tuple(window.shape) != tuple(image_fids.shape[-2:]):
        raise ValueError("window must match the final two image dimensions")
    centered_kspace = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image_fids, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    centered_kspace = centered_kspace * window.to(
        device=image_fids.device, dtype=image_fids.real.dtype
    )
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(centered_kspace, dim=(-2, -1)), dim=(-2, -1)
        ),
        dim=(-2, -1),
    )


def _complex_noise_like(
    tensor: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    real_dtype = tensor.real.dtype
    kwargs = {"dtype": real_dtype, "device": tensor.device, "generator": generator}
    real = torch.randn(tensor.shape, **kwargs)
    imag = torch.randn(tensor.shape, **kwargs)
    return torch.complex(real, imag)


def apply_repetition_acquisition(
    image_fids: torch.Tensor,
    repetition_map: torch.Tensor,
    *,
    noise_std_per_excitation: float | torch.Tensor = 0.0,
    normalization: Literal["center", "none"] | float | torch.Tensor = "center",
    generator: torch.Generator | None = None,
) -> AcquisitionResult:
    """Apply the physical coherent-sum acquisition to complex image FIDs.

    Parameters
    ----------
    image_fids:
        Complex tensor with spatial dimensions in the final two positions;
        common shapes are ``[B, T, C, H, W]`` and ``[C, H, W]``.
    repetition_map:
        Centered ``[H, W]`` acquisition repetition counts or analytical
        weights.
    noise_std_per_excitation:
        Standard deviation of *each real and imaginary component* of a single
        excitation's independent complex Gaussian noise.
    normalization:
        ``"center"`` divides the coherent sum by the largest repetition count.
        This is an explicit implementation convention because the revision did
        not supply its exact fixed input scaling. ``"none"`` leaves the
        coherent sum unchanged.

    Returns
    -------
    AcquisitionResult
        Both centered k-space and reconstructed complex image-domain data.
    """

    if image_fids.ndim < 2:
        raise ValueError("image_fids must have at least two spatial dimensions")
    if not torch.is_complex(image_fids):
        image_fids = torch.complex(image_fids, torch.zeros_like(image_fids))
    if repetition_map.ndim != 2:
        raise ValueError("repetition_map must have shape [H, W]")
    if tuple(image_fids.shape[-2:]) != tuple(repetition_map.shape):
        raise ValueError(
            "spatial shape mismatch: image_fids ends in "
            f"{tuple(image_fids.shape[-2:])}, map is {tuple(repetition_map.shape)}"
        )
    if torch.any(repetition_map < 0):
        raise ValueError("repetition counts cannot be negative")

    real_dtype = image_fids.real.dtype
    repetitions = repetition_map.to(device=image_fids.device, dtype=real_dtype)
    expand_shape = (1,) * (image_fids.ndim - 2) + repetitions.shape
    repetitions = repetitions.reshape(expand_shape)

    kspace_clean = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image_fids, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    kspace = repetitions * kspace_clean

    noise_level = torch.as_tensor(
        noise_std_per_excitation, dtype=real_dtype, device=image_fids.device
    )
    if torch.any(noise_level < 0):
        raise ValueError("noise_std_per_excitation cannot be negative")
    if bool(torch.any(noise_level != 0)):
        kspace = kspace + _complex_noise_like(kspace, generator=generator) * (
            noise_level * torch.sqrt(repetitions)
        )

    if isinstance(normalization, str):
        if normalization == "center":
            norm = repetitions.max()
        elif normalization == "none":
            norm = torch.ones((), dtype=real_dtype, device=image_fids.device)
        else:
            raise ValueError("normalization must be 'center', 'none', or a number")
    else:
        norm = torch.as_tensor(normalization, dtype=real_dtype, device=image_fids.device)
    if bool(torch.any(norm <= 0)):
        raise ValueError("normalization must be positive")
    kspace = kspace / norm

    reconstructed = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    return AcquisitionResult(
        image=reconstructed,
        kspace=kspace,
        repetition_map=repetition_map,
        normalization=norm,
    )


def per_excitation_noise_for_image_sd(
    target_component_sd: float,
    repetition_map: torch.Tensor,
    *,
    normalization: Literal["center", "none"] | float = "center",
) -> float:
    """Calibrate per-excitation component SD to an image component SD.

    This follows SI Eq. S1.9 and the transform convention in
    :func:`apply_repetition_acquisition`. Historical prototype ``noise_level``
    values are not assumed to share these explicit physical units.
    """

    if target_component_sd < 0:
        raise ValueError("target_component_sd cannot be negative")
    if repetition_map.ndim != 2 or torch.any(repetition_map < 0):
        raise ValueError("repetition_map must be a nonnegative 2-D tensor")
    if isinstance(normalization, str):
        if normalization == "center":
            norm = float(repetition_map.max().item())
        elif normalization == "none":
            norm = 1.0
        else:
            raise ValueError("normalization must be 'center', 'none', or positive")
    else:
        norm = float(normalization)
    if norm <= 0:
        raise ValueError("normalization must be positive")
    spatial_samples = repetition_map.numel()
    total_repetitions = float(repetition_map.to(torch.float64).sum().item())
    if total_repetitions <= 0:
        raise ValueError("repetition_map must have a positive total")
    return target_component_sd * spatial_samples * norm / total_repetitions**0.5


def sinc_interpolated_profile(
    repetition_map: torch.Tensor,
    *,
    interpolation_points: int = 2**18,
    axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the paper's peak-normalized analytical central PSF line."""

    if repetition_map.ndim != 2:
        raise ValueError("repetition_map must be two-dimensional")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    n = repetition_map.shape[axis]
    if repetition_map.shape[0] != repetition_map.shape[1]:
        raise ValueError("the published PSF workflow assumes a square matrix")
    if interpolation_points < n or interpolation_points % 2:
        raise ValueError("interpolation_points must be an even integer >= matrix size")

    weights = repetition_map.to(torch.float64).sum(dim=1 - axis)
    padded = torch.zeros(interpolation_points, dtype=torch.float64, device=weights.device)
    start = interpolation_points // 2 - n // 2
    padded[start : start + n] = weights
    profile = torch.abs(
        torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(padded)))
    )
    profile = profile / profile.max()
    coordinates = (
        torch.arange(interpolation_points, dtype=torch.float64, device=weights.device)
        - interpolation_points // 2
    ) * n / interpolation_points
    return coordinates, profile


def channel_l2_point_profile(
    channel_images: torch.Tensor,
    *,
    interpolation_points: int = 2**18,
    row: int | None = None,
    center_peak: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinc-interpolate channel rows separately, then combine their L2 norm.

    This implements SI Section 3 panel B step (5). Input is ``[C,H,W]`` and may
    be real or complex. The returned coordinates are in nominal voxels.
    """

    if channel_images.ndim != 3:
        raise ValueError("channel_images must have shape [C,H,W]")
    _, height, width = channel_images.shape
    if height != width:
        raise ValueError("the published point-response workflow assumes square images")
    if interpolation_points < width or interpolation_points % 2:
        raise ValueError("interpolation_points must be an even integer >= image width")
    row = height // 2 if row is None else row
    if not 0 <= row < height:
        raise ValueError("row is outside the image")
    rows = channel_images[:, row, :]
    if not torch.is_complex(rows):
        rows = torch.complex(rows, torch.zeros_like(rows))
    row_kspace = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(rows, dim=-1), dim=-1), dim=-1
    )
    padded = torch.zeros(
        (rows.shape[0], interpolation_points), dtype=rows.dtype, device=rows.device
    )
    start = interpolation_points // 2 - width // 2
    padded[:, start : start + width] = row_kspace
    interpolated = torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(padded, dim=-1), dim=-1), dim=-1
    )
    profile = torch.sqrt(torch.sum(torch.abs(interpolated) ** 2, dim=0))
    profile = profile / profile.max().clamp_min(torch.finfo(profile.dtype).eps)
    if center_peak:
        shift = interpolation_points // 2 - int(torch.argmax(profile).item())
        profile = torch.roll(profile, shifts=shift)
    coordinates = (
        torch.arange(
            interpolation_points, dtype=profile.dtype, device=channel_images.device
        )
        - interpolation_points // 2
    ) * width / interpolation_points
    return coordinates, profile


def profile_fwhm(
    coordinates: torch.Tensor,
    profile: torch.Tensor,
    *,
    level: float = 0.5,
) -> float:
    """Measure FWHM using the nearest crossings and linear interpolation."""

    if coordinates.ndim != 1 or profile.ndim != 1 or coordinates.shape != profile.shape:
        raise ValueError("coordinates and profile must be equal-length 1-D tensors")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    peak = int(torch.argmax(profile).item())
    left_below = torch.nonzero(profile[:peak] < level).flatten()
    right_below = torch.nonzero(profile[peak:] < level).flatten()
    if left_below.numel() == 0 or right_below.numel() == 0:
        raise ValueError("profile does not cross the requested level on both sides")
    left_low = int(left_below[-1].item())
    right_low = peak + int(right_below[0].item())

    def crossing(i0: int, i1: int) -> torch.Tensor:
        x0, x1 = coordinates[i0], coordinates[i1]
        y0, y1 = profile[i0], profile[i1]
        return x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    left = crossing(left_low, left_low + 1)
    right = crossing(right_low - 1, right_low)
    return float((right - left).item())
