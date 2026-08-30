"""Run a fixed DL-UA-CSI or DL-WA-CSI checkpoint on acquired complex FIDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpointing import load_model
from .contracts import checkpoint_runtime_contract
from .data import load_scalar_image
from .simulation import format_inference
from .training import _resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ containing complex `csi` with [T,C,H,W] or [B,T,C,H,W]",
    )
    parser.add_argument(
        "--anatomy",
        type=Path,
        nargs="+",
        required=True,
        help="One co-registered anatomy path per CSI batch item",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--anatomy-size",
        type=int,
        help="Must match checkpoint metadata; defaults to its recorded size",
    )
    return parser


def _normalized_anatomy(path: Path, size: int) -> torch.Tensor:
    anatomy = load_scalar_image(path)
    anatomy = F.interpolate(
        anatomy[None, None], size=(size, size), mode="bilinear", align_corners=False
    )
    anatomy = anatomy - anatomy.amin(dim=(-2, -1), keepdim=True)
    return anatomy / anatomy.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.anatomy_size is not None and args.anatomy_size < 1:
        raise ValueError("--anatomy-size must be positive")
    device = _resolve_device(args.device)
    model, payload = load_model(args.checkpoint, device=device)
    acquisition, preprocessing = checkpoint_runtime_contract(payload)
    recorded_anatomy_size = tuple(preprocessing["anatomy_size"])
    if recorded_anatomy_size[0] != recorded_anatomy_size[1]:
        raise ValueError("inference currently requires square checkpoint anatomy_size")
    anatomy_size = args.anatomy_size or recorded_anatomy_size[0]
    if (anatomy_size, anatomy_size) != recorded_anatomy_size:
        raise ValueError("--anatomy-size must match checkpoint preprocessing metadata")
    model.eval()
    with np.load(args.input, allow_pickle=False) as archive:
        if "csi" not in archive:
            raise ValueError("input NPZ must contain a `csi` array")
        csi_array = archive["csi"]
    if not np.iscomplexobj(csi_array):
        raise ValueError("`csi` must be complex image-domain FID data")
    csi = torch.from_numpy(csi_array)
    if csi.ndim == 4:
        csi = csi.unsqueeze(0)
    if csi.ndim != 5:
        raise ValueError("`csi` must have shape [T,C,H,W] or [B,T,C,H,W]")
    if tuple(csi.shape[-2:]) != tuple(acquisition["matrix_size"]):
        raise ValueError("CSI spatial size must match checkpoint acquisition metadata")
    csi = csi.to(device=device, dtype=torch.complex64)
    if len(args.anatomy) != csi.shape[0]:
        raise ValueError(
            f"received {len(args.anatomy)} anatomy images for CSI batch {csi.shape[0]}"
        )
    anatomy = torch.cat(
        [_normalized_anatomy(path, anatomy_size) for path in args.anatomy]
    ).to(device)
    network_input, scale = format_inference(
        csi, percentile=float(preprocessing["input_scale_quantile"])
    )
    with torch.inference_mode():
        normalized = model(network_input, anatomy)
    scale_view = scale.reshape((scale.shape[0],) + (1,) * (normalized.ndim - 1))
    reconstruction = normalized * scale_view
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        reconstruction=reconstruction.cpu().numpy(),
        magnitude_domain_estimate=reconstruction.cpu().numpy(),
        normalized_reconstruction=normalized.cpu().numpy(),
        input_magnitude=csi.abs().cpu().numpy(),
        scale=scale.cpu().numpy(),
        branch=np.asarray(acquisition["branch"]),
        acquisition_normalization=np.asarray(acquisition["normalization"]),
        input_representation=np.asarray(preprocessing["network_input_representation"]),
        output_representation=np.asarray(preprocessing["network_output_representation"]),
        input_scale_quantile=np.asarray(preprocessing["input_scale_quantile"]),
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
