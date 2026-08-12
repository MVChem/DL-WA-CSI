#!/usr/bin/env python3
"""Generate or score the paper's Monte Carlo reliable-recovery protocol.

This script deliberately calls the result *simulation-derived reliable
recovery*, not an analytical or experimental detection limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dlwa_csi.experiments import (
    PAPER_METHODS,
    RecoveryProtocol,
    generate_recovery_protocol,
    summarize_recovery_maps,
)
from dlwa_csi.metrics import fit_logistic_binary, logistic_crossing


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write randomized tube assignments")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--repeats", type=int, default=100)
    generate.add_argument("--target-tube-index", type=int, default=5)
    generate.add_argument("--seed", type=int, default=20260811)

    score = subparsers.add_parser("score", help="score reconstructed concentration maps")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument(
        "--estimates",
        type=Path,
        required=True,
        help="NPZ with one [48,repeats,32,32] array per paper method",
    )
    score.add_argument("--output", type=Path, help="Optional summary NPZ")
    return parser


def _save_protocol(path: Path, protocol: RecoveryProtocol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        levels=protocol.levels,
        assignments=protocol.assignments,
        target_tube_index=np.asarray(protocol.target_tube_index),
        masks=protocol.masks,
        seed=np.asarray(protocol.seed),
        context_sampling=np.asarray(protocol.context_sampling),
    )


def _load_protocol(path: Path) -> RecoveryProtocol:
    with np.load(path, allow_pickle=False) as values:
        return RecoveryProtocol(
            levels=values["levels"],
            assignments=values["assignments"],
            target_tube_index=int(values["target_tube_index"]),
            masks=values["masks"].astype(bool),
            seed=int(values["seed"]),
            context_sampling=(
                str(values["context_sampling"])
                if "context_sampling" in values
                else "unknown_legacy_protocol"
            ),
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        protocol = generate_recovery_protocol(
            repeats=args.repeats,
            target_tube_index=args.target_tube_index,
            seed=args.seed,
        )
        _save_protocol(args.output, protocol)
        print(
            json.dumps(
                {
                    "levels": int(protocol.levels.size),
                    "repeats_per_level": protocol.repeats,
                    "scenes": int(protocol.levels.size * protocol.repeats),
                    "target_tube_index": protocol.target_tube_index,
                    "context_sampling": protocol.context_sampling,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    protocol = _load_protocol(args.protocol)
    output_values: dict[str, np.ndarray] = {"levels": protocol.levels}
    report: dict[str, object] = {
        "criterion": "target-tube full-voxel ROI NRMSE < 10%",
        "context_sampling": protocol.context_sampling,
        "interpretation": "simulation-derived reliable recovery; not analytical/experimental LOD",
        "methods": {},
    }
    with np.load(args.estimates, allow_pickle=False) as estimates:
        missing = [method for method in PAPER_METHODS if method not in estimates]
        if missing:
            raise ValueError(f"estimate archive is missing: {', '.join(missing)}")
        for method in PAPER_METHODS:
            summary = summarize_recovery_maps(estimates[method], protocol)
            expanded_levels = np.repeat(summary.levels, protocol.repeats)
            alpha, beta = fit_logistic_binary(expanded_levels, summary.reliable.reshape(-1))
            crossing = logistic_crossing(alpha, beta, probability=0.95)
            report["methods"][method] = {
                "alpha": alpha,
                "beta": beta,
                "p95_crossing_mM": crossing,
            }
            output_values[f"{method}_probability"] = summary.probability
            output_values[f"{method}_wilson_low"] = summary.wilson_low
            output_values[f"{method}_wilson_high"] = summary.wilson_high
            output_values[f"{method}_nrmse_percent"] = summary.nrmse_percent
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
