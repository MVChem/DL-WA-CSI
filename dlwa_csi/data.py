"""Datasets for anatomy-guided, on-the-fly DMI simulation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

METABOLITE_KEYS = ("water", "glucose", "lactate")
_LEGACY_NAMES = {
    "water": ("water",),
    "glucose": ("glucose", "glu"),
    "lactate": ("lactate", "lac"),
    "anatomy": ("anatomy", "t1", "t1w", "reference"),
}
_IMAGE_EXTENSIONS = (".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _resolve_path(path: str | Path, base: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def load_scalar_image(path: str | Path) -> torch.Tensor:
    """Load a 2-D scalar image from NumPy or a common image format."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("F"), dtype=np.float32).copy()
    tensor = torch.as_tensor(array, dtype=torch.float32).squeeze()
    if tensor.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D scalar image, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path} contains non-finite values")
    return tensor


def _minmax(image: torch.Tensor) -> torch.Tensor:
    image = image - image.min()
    maximum = image.max()
    if maximum > 0:
        image = image / maximum
    return image


def _metabolite_prior(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """Load a nonnegative spatial coefficient map without per-map min-maxing.

    NumPy arrays preserve their numeric scale and therefore relative tissue and
    metabolite coefficients. Raster images have no physical numeric metadata,
    so they are divided by their positive maximum and treated as shape masks.
    Constant positive masks remain one instead of collapsing to zero.
    """

    image = _resize(load_scalar_image(path), size)
    if torch.any(image < 0):
        raise ValueError(f"metabolite prior cannot contain negative values: {path}")
    if path.suffix.lower() != ".npy":
        image = image / image.max().clamp_min(1e-8)
    return image


def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        image[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0]


@dataclass(frozen=True)
class JointAugmentation:
    """Joint spatial flips plus metabolite intensity scaling."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    intensity_scale: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self) -> None:
        probabilities = (
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("flip probabilities must lie in [0, 1]")
        if self.intensity_scale[0] <= 0 or self.intensity_scale[1] < self.intensity_scale[0]:
            raise ValueError("intensity_scale must be positive and ordered")

    def __call__(
        self, anatomy: torch.Tensor, metabolites: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < self.horizontal_flip_probability:
            anatomy = anatomy.flip(-1)
            metabolites = metabolites.flip(-1)
        if torch.rand(()) < self.vertical_flip_probability:
            anatomy = anatomy.flip(-2)
            metabolites = metabolites.flip(-2)
        low, high = self.intensity_scale
        scale = low + (high - low) * torch.rand(())
        return anatomy.contiguous(), (metabolites * scale).contiguous()


def _find_legacy_file(folder: Path, kind: str) -> Path:
    for stem in _LEGACY_NAMES[kind]:
        for extension in _IMAGE_EXTENSIONS:
            candidate = folder / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    names = ", ".join(
        f"{stem}{extension}" for stem in _LEGACY_NAMES[kind] for extension in _IMAGE_EXTENSIONS
    )
    raise FileNotFoundError(f"missing {kind} in {folder}; tried {names}")


def _legacy_records(lines: Iterable[str], manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(lines, start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        folder = _resolve_path(value, manifest_path.parent)
        if not folder.is_dir():
            raise FileNotFoundError(f"line {number}: folder does not exist: {folder}")
        records.append(
            {
                "id": folder.name,
                "anatomy": _find_legacy_file(folder, "anatomy"),
                **{key: _find_legacy_file(folder, key) for key in METABOLITE_KEYS},
            }
        )
    return records


def _jsonl_records(lines: Iterable[str], manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = {"anatomy", *METABOLITE_KEYS}
    for number, raw in enumerate(lines, start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            record = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {manifest_path}:{number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"record {number} must be a JSON object")
        missing = required.difference(record)
        if missing:
            raise ValueError(f"record {number} is missing: {', '.join(sorted(missing))}")
        normalized = dict(record)
        for key in required:
            normalized[key] = _resolve_path(record[key], manifest_path.parent)
        normalized.setdefault("id", f"sample-{number:05d}")
        records.append(normalized)
    return records


class AnatomicalDMIDataset(Dataset[dict[str, Any]]):
    """Load co-registered anatomy plus three spatial metabolite priors.

    JSONL is the preferred format, one object per line::

        {"id":"IXI001-42", "anatomy":"images/t1.npy",
         "water":"maps/water.npy", "glucose":"maps/glucose.npy",
         "lactate":"maps/lactate.npy"}

    Paths are resolved relative to the manifest. For compatibility, a ``.txt``
    manifest may list one sample directory per line. Unlike the historical
    loader, each directory must contain a real anatomy image; silently replacing
    the paper's anatomical prior with zeros is intentionally prohibited.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        dmi_size: tuple[int, int] = (32, 32),
        anatomy_size: tuple[int, int] = (256, 256),
        augment: JointAugmentation | None = None,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        lines = self.manifest.read_text(encoding="utf-8").splitlines()
        if self.manifest.suffix.lower() in (".jsonl", ".json"):
            self.records = _jsonl_records(lines, self.manifest)
        else:
            self.records = _legacy_records(lines, self.manifest)
        if not self.records:
            raise ValueError(f"manifest has no samples: {self.manifest}")
        self.dmi_size = tuple(int(value) for value in dmi_size)
        self.anatomy_size = tuple(int(value) for value in anatomy_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        anatomy = _minmax(_resize(load_scalar_image(record["anatomy"]), self.anatomy_size))
        metabolite_images = []
        for key in METABOLITE_KEYS:
            image = _metabolite_prior(Path(record[key]), self.dmi_size)
            metabolite_images.append(image)
        metabolites = torch.stack(metabolite_images)
        if self.augment is not None:
            # Flip at native output resolutions by sampling the decisions once.
            hflip = torch.rand(()) < self.augment.horizontal_flip_probability
            vflip = torch.rand(()) < self.augment.vertical_flip_probability
            if hflip:
                anatomy, metabolites = anatomy.flip(-1), metabolites.flip(-1)
            if vflip:
                anatomy, metabolites = anatomy.flip(-2), metabolites.flip(-2)
            low, high = self.augment.intensity_scale
            metabolites = metabolites * (low + (high - low) * torch.rand(()))
        return {
            "id": str(record["id"]),
            "anatomy": anatomy.contiguous(),
            "metabolites": metabolites.contiguous(),
        }
