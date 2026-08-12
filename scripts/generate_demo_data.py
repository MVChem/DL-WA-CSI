#!/usr/bin/env python3
"""Generate small synthetic anatomy/metabolite fixtures for a pipeline smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _sample(seed: int, size: int = 96) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    brain = ((x / 0.82) ** 2 + (y / 0.95) ** 2 < 1).astype(np.float32)
    anatomy = brain * (
        0.55
        + 0.25 * np.cos(8 * x) * np.cos(7 * y)
        + 0.15 * np.exp(-((x - 0.25) ** 2 + (y + 0.12) ** 2) / 0.025)
    )
    anatomy += rng.normal(0, 0.015, anatomy.shape).astype(np.float32)
    anatomy = np.clip(anatomy, 0, None)
    water = brain * (0.7 + 0.3 * anatomy / max(float(anatomy.max()), 1e-8))
    glucose = brain * np.exp(-((x + 0.18) ** 2 + (y - 0.08) ** 2) / 0.32)
    lactate = brain * np.exp(-((x - 0.28) ** 2 + (y + 0.10) ** 2) / 0.055)
    return {
        "anatomy": anatomy.astype(np.float32),
        "water": water.astype(np.float32),
        "glucose": glucose.astype(np.float32),
        "lactate": lactate.astype(np.float32),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.train_samples < 1 or args.val_samples < 1:
        raise ValueError("train and validation sample counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, count, offset in (
        ("train", args.train_samples, 0),
        ("val", args.val_samples, args.train_samples),
    ):
        records = []
        split_dir = args.output / split
        split_dir.mkdir(exist_ok=True)
        for index in range(count):
            sample_id = f"demo-{split}-{index:03d}"
            arrays = _sample(args.seed + offset + index)
            record = {"id": sample_id}
            for name, array in arrays.items():
                destination = split_dir / f"{sample_id}-{name}.npy"
                np.save(destination, array, allow_pickle=False)
                record[name] = str(destination.relative_to(args.output))
            records.append(record)
        manifest = args.output / f"{split}.jsonl"
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    print(f"wrote demo manifests beneath {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
