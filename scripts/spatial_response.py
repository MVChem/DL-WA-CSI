#!/usr/bin/env python3
"""Reproduce the analytical UA/WA portion of manuscript Figure 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dlwa_csi.acquisition import (
    apply_repetition_acquisition,
    channel_l2_point_profile,
    hann_repetition_map,
    matched_uniform_repetition_map,
    profile_fwhm,
    sinc_interpolated_profile,
)
from dlwa_csi.checkpointing import load_model
from dlwa_csi.contracts import (
    ACQUISITION_NORMALIZATION,
    PAPER_ANATOMY_SIZE,
    PAPER_MATRIX_SIZE,
    WA_SCHEDULE,
    checkpoint_runtime_contract,
)
from dlwa_csi.inference import _normalized_anatomy
from dlwa_csi.simulation import SpectralModel, format_inference, spectral_basis
from dlwa_csi.training import _resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .npz destination for repetition maps and dense profiles",
    )
    parser.add_argument(
        "--interpolation-points",
        type=int,
        default=2**18,
        help="Dense zero-padded IFFT size (paper: 262144)",
    )
    parser.add_argument("--checkpoint", type=Path, help="Fixed DL-WA checkpoint")
    parser.add_argument(
        "--anatomy",
        type=Path,
        help="Co-registered anatomy for the learned point probe",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--anatomy-size",
        type=int,
        help="Must match checkpoint metadata; defaults to its recorded size",
    )
    return parser


def _validate_learned_checkpoint(
    model: torch.nn.Module, payload: dict[str, object]
) -> tuple[SpectralModel, int, dict[str, object]]:
    acquisition, preprocessing = checkpoint_runtime_contract(payload, expected_branch="wa")
    if tuple(acquisition["matrix_size"]) != PAPER_MATRIX_SIZE:
        raise ValueError("the published point probe requires a 32x32 acquisition")
    if acquisition["normalization"] != ACQUISITION_NORMALIZATION:
        raise ValueError("the published point probe requires center normalization")
    if acquisition["schedule"] != WA_SCHEDULE:
        raise ValueError("the checkpoint does not record the published WA schedule")
    if acquisition.get("wa_center_repetitions") != 263:
        raise ValueError("the published point probe requires WA center weight 263")
    if acquisition.get("wa_total_repetitions") != 68_106:
        raise ValueError("the published point probe requires WA total 68106")
    if tuple(preprocessing["anatomy_size"]) != PAPER_ANATOMY_SIZE:
        raise ValueError("the published point probe requires 256x256 anatomy")
    if getattr(model, "in_channels", None) != 72:
        raise ValueError("the published point probe requires a 72-channel checkpoint")
    spectral_config = payload.get("spectral_model")
    if not isinstance(spectral_config, dict) or not spectral_config:
        raise ValueError("checkpoint must record a nonempty spectral_model")
    spectral_model = SpectralModel(**spectral_config)
    if spectral_model.spectral_points != 72:
        raise ValueError("checkpoint spectral model must contain 72 FID points")
    lactate_indices = [
        index
        for index, name in enumerate(spectral_model.metabolite_names)
        if name.casefold() == "lactate"
    ]
    if len(lactate_indices) != 1:
        raise ValueError("checkpoint spectral model must identify exactly one lactate basis")
    return spectral_model, lactate_indices[0], preprocessing


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interpolation_points < 32:
        raise ValueError("--interpolation-points must be at least 32")
    if args.anatomy_size is not None and args.anatomy_size < 1:
        raise ValueError("--anatomy-size must be positive")
    wa = hann_repetition_map()
    ua = matched_uniform_repetition_map(wa)
    coordinates, ua_profile = sinc_interpolated_profile(
        ua, interpolation_points=args.interpolation_points
    )
    _, wa_profile = sinc_interpolated_profile(
        wa, interpolation_points=args.interpolation_points
    )
    result = {
        "matrix_size": 32,
        "wa_center_repetitions": int(wa.max().item()),
        "total_repetitions": int(wa.sum().item()),
        "ua_weight": float(ua[0, 0].item()),
        "ua_fwhm_voxels": profile_fwhm(coordinates, ua_profile),
        "wa_fwhm_voxels": profile_fwhm(coordinates, wa_profile),
        "interpretation": "analytical acquisition PSFs; peak-normalized width, not sensitivity",
    }
    dl_profile = None
    if (args.checkpoint is None) != (args.anatomy is None):
        raise ValueError("--checkpoint and --anatomy must be supplied together")
    if args.checkpoint is not None:
        device = _resolve_device(args.device)
        model, payload = load_model(args.checkpoint, device=device)
        spectral_model, lactate_index, preprocessing = _validate_learned_checkpoint(
            model, payload
        )
        anatomy_size = int(preprocessing["anatomy_size"][0])
        if args.anatomy_size is not None and args.anatomy_size != anatomy_size:
            raise ValueError("--anatomy-size must match checkpoint preprocessing metadata")
        point = torch.zeros((1, 1, 72, 32, 32), dtype=torch.complex64, device=device)
        point[0, 0, :, 16, 16] = spectral_basis(
            spectral_model, device=device
        )[lactate_index]
        wa_input = apply_repetition_acquisition(
            point,
            wa.to(device=device, dtype=torch.float32),
            noise_std_per_excitation=0.0,
            normalization=ACQUISITION_NORMALIZATION,
        ).image
        network_input, scale = format_inference(
            wa_input, percentile=float(preprocessing["input_scale_quantile"])
        )
        anatomy = _normalized_anatomy(args.anatomy, anatomy_size).to(device)
        model.eval()
        with torch.inference_mode():
            prediction = model(network_input, anatomy)[0, 0] * scale[0]
        dl_coordinates, dl_profile = channel_l2_point_profile(
            prediction,
            interpolation_points=args.interpolation_points,
            center_peak=False,
        )
        peak_offset = float(dl_coordinates[torch.argmax(dl_profile)].item())
        learned_width = profile_fwhm(dl_coordinates, dl_profile)
        result.update(
            {
                "dl_wa_effective_fwhm_voxels": learned_width,
                "dl_wa_to_ua_width_ratio": learned_width / result["ua_fwhm_voxels"],
                "dl_wa_peak_offset_voxels": peak_offset,
                "learned_interpretation": (
                    "probe-specific empirical effective point response of this fixed "
                    "checkpoint/preprocessing/anatomy; not a system PSF"
                ),
            }
        )
    print(json.dumps(result, indent=2))
    if args.output is not None:
        if args.output.suffix.lower() != ".npz":
            raise ValueError("--output must end in .npz")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            coordinates=coordinates.cpu().numpy(),
            ua_profile=ua_profile.cpu().numpy(),
            wa_profile=wa_profile.cpu().numpy(),
            **(
                {"dl_wa_profile": dl_profile.cpu().numpy()}
                if dl_profile is not None
                else {}
            ),
            ua_repetition_map=ua.cpu().numpy(),
            wa_repetition_map=wa.cpu().numpy(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
