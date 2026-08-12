"""Versioned checkpoint helpers shared by training and inference."""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .models import PriorInformedUNet3D

CHECKPOINT_FORMAT_VERSION = 1


def atomic_torch_save(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write a checkpoint atomically so interrupted jobs keep the prior file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a DL-WA-CSI checkpoint."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if "weights_only" in inspect.signature(torch.load).parameters:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    else:  # PyTorch 2.0 did not yet expose weights_only.
        warnings.warn(
            "this PyTorch version uses pickle-based checkpoint loading; load only "
            "trusted DL-WA-CSI checkpoints",
            RuntimeWarning,
            stacklevel=2,
        )
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    required = {"format_version", "model_config", "model_state"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing: {', '.join(sorted(missing))}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {payload['format_version']}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def load_model(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[PriorInformedUNet3D, dict[str, Any]]:
    """Construct a model from checkpoint config and restore its weights."""

    payload = load_checkpoint(path, map_location=device)
    model = PriorInformedUNet3D.from_config(payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, payload
