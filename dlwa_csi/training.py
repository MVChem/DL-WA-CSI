"""Training entry point for acquisition-matched DL-UA-CSI and DL-WA-CSI."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .acquisition import hann_repetition_map, matched_uniform_repetition_map
from .checkpointing import CHECKPOINT_FORMAT_VERSION, atomic_torch_save, load_checkpoint
from .contracts import training_runtime_contract
from .data import AnatomicalDMIDataset, JointAugmentation
from .models import PriorInformedUNet3D
from .simulation import (
    SpectralModel,
    prepare_network_pair,
    random_smooth_curves,
    synthesize_dynamic_fids,
)

PAPER_EPOCHS = 150
PAPER_INITIAL_LR = 3e-4
PAPER_MIN_LR = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    train_manifest: str
    val_manifest: str
    output_dir: str
    branch: str
    epochs: int
    batch_size: int
    dynamic_frames: int
    spectral_points: int
    learning_rate: float
    minimum_learning_rate: float
    noise_std_min: float
    noise_std_max: float
    spectral_bandwidth_hz: float
    peak_offsets_hz: tuple[float, ...]
    t2_seconds: tuple[float, ...]
    horizontal_flip_probability: float
    vertical_flip_probability: float
    intensity_scale_min: float
    intensity_scale_max: float
    channels: tuple[int, ...]
    attention_heads: tuple[int, ...]
    temporal_heads: int
    temporal_layers: int
    seed: int
    device: str
    num_workers: int
    amp: bool
    save_every: int
    resume: str | None


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("all comma-separated values must be positive")
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one number is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", default="runs/dl-wa-csi")
    parser.add_argument(
        "--branch",
        choices=("wa", "ua"),
        default="wa",
        help="Acquisition used to generate inputs; architecture is identical",
    )
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-frames", type=int, default=30)
    parser.add_argument("--spectral-points", type=int, default=72)
    parser.add_argument("--learning-rate", type=float, default=PAPER_INITIAL_LR)
    parser.add_argument("--minimum-learning-rate", type=float, default=PAPER_MIN_LR)
    parser.add_argument(
        "--noise-std-min",
        type=float,
        required=True,
        help=(
            "Required per-excitation real/imaginary k-space noise SD lower bound; "
            "the revision omits its calibration"
        ),
    )
    parser.add_argument(
        "--noise-std-max",
        type=float,
        required=True,
        help="Required because the revision omits training-noise calibration",
    )
    parser.add_argument("--spectral-bandwidth-hz", type=float, default=4065.0)
    parser.add_argument(
        "--peak-offsets-hz",
        type=_csv_floats,
        default=(0.0, -55.0, -209.0),
        help="Water, glucose, and lactate frequency offsets",
    )
    parser.add_argument(
        "--t2-seconds",
        type=_csv_floats,
        default=(0.080, 0.070, 0.060),
        help="Water, glucose, and lactate T2 values",
    )
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.5)
    parser.add_argument("--intensity-scale-min", type=float, default=0.9)
    parser.add_argument("--intensity-scale-max", type=float, default=1.1)
    parser.add_argument("--channels", type=_csv_ints, default=(72, 72, 72, 72))
    parser.add_argument("--attention-heads", type=_csv_ints, default=(4, 4, 4, 4))
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", help="Resume optimizer/model/scheduler from checkpoint")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = ("epochs", "batch_size", "dynamic_frames", "spectral_points", "save_every")
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.learning_rate <= 0 or args.minimum_learning_rate < 0:
        raise ValueError("learning rates must be non-negative and initial LR positive")
    if args.minimum_learning_rate > args.learning_rate:
        raise ValueError("minimum learning rate cannot exceed initial learning rate")
    if args.noise_std_min < 0 or args.noise_std_max < args.noise_std_min:
        raise ValueError("noise range must be non-negative and ordered")
    if args.spectral_bandwidth_hz <= 0:
        raise ValueError("--spectral-bandwidth-hz must be positive")
    if len(args.peak_offsets_hz) != 3 or len(args.t2_seconds) != 3:
        raise ValueError("frequency offsets and T2 values must each contain three entries")
    if any(value <= 0 for value in args.t2_seconds):
        raise ValueError("all T2 values must be positive")
    flip_probabilities = (
        args.horizontal_flip_probability,
        args.vertical_flip_probability,
    )
    if any(not 0 <= value <= 1 for value in flip_probabilities):
        raise ValueError("flip probabilities must lie in [0, 1]")
    if args.intensity_scale_min <= 0 or args.intensity_scale_max < args.intensity_scale_min:
        raise ValueError("intensity scale range must be positive and ordered")
    if len(args.channels) != len(args.attention_heads):
        raise ValueError("--channels and --attention-heads must have equal lengths")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _make_loader(
    dataset: AnatomicalDMIDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def _noise_level(config: TrainingConfig, generator: torch.Generator, device: torch.device) -> float:
    value = torch.rand((), generator=generator, device=device).item()
    return config.noise_std_min + value * (config.noise_std_max - config.noise_std_min)


def _batch_pair(
    batch: dict[str, Any],
    *,
    repetition_map: torch.Tensor,
    spectral_model: SpectralModel,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    metabolites = batch["metabolites"].to(device, non_blocking=True)
    anatomy = batch["anatomy"].to(device, non_blocking=True)
    curves = random_smooth_curves(
        metabolites.shape[0],
        config.dynamic_frames,
        device=device,
        generator=generator,
        dtype=metabolites.dtype,
    )
    clean = synthesize_dynamic_fids(metabolites, curves, model=spectral_model)
    pair = prepare_network_pair(
        clean,
        repetition_map,
        noise_std_per_excitation=_noise_level(config, generator, device),
        generator=generator,
    )
    return pair.degraded, pair.target, anatomy


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _evaluate(
    model: PriorInformedUNet3D,
    loader: DataLoader,
    *,
    repetition_map: torch.Tensor,
    spectral_model: SpectralModel,
    config: TrainingConfig,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    samples = 0
    generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="validate", leave=False):
            degraded, target, anatomy = _batch_pair(
                batch,
                repetition_map=repetition_map,
                spectral_model=spectral_model,
                config=config,
                generator=generator,
                device=device,
            )
            with _autocast(device, amp):
                prediction = model(degraded, anatomy)
                loss = nn.functional.mse_loss(prediction, target)
            count = degraded.shape[0]
            total_loss += float(loss.item()) * count
            samples += count
    return total_loss / max(samples, 1)


def _checkpoint_payload(
    model: PriorInformedUNet3D,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    *,
    config: TrainingConfig,
    spectral_model: SpectralModel,
    epoch: int,
    best_val_loss: float,
    simulation_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_config": model.get_config(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "simulation_generator_state": simulation_generator.get_state(),
        "rng_device_type": simulation_generator.device.type,
        "loader_generator_state": loader_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "training_config": asdict(config),
        "spectral_model": spectral_model.to_dict(),
    }
    payload.update(training_runtime_contract(config.branch))
    return payload


def _cpu_rng_state(state: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize serialized generator state for PyTorch's set_state APIs."""

    if not isinstance(state, torch.Tensor):
        raise TypeError(f"checkpoint {name} must be a tensor")
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng_states(
    payload: dict[str, Any],
    *,
    simulation_generator: torch.Generator,
    loader_generator: torch.Generator,
    device: torch.device,
) -> None:
    """Restore CPU/CUDA RNGs after a checkpoint loaded on any target device."""

    checkpoint_device_type = payload.get("rng_device_type")
    if checkpoint_device_type not in {"cpu", "cuda"}:
        raise ValueError("checkpoint is missing a valid rng_device_type")
    if checkpoint_device_type != device.type:
        raise ValueError(
            "exact resume cannot change RNG device type from "
            f"{checkpoint_device_type!r} to {device.type!r}"
        )
    if "simulation_generator_state" in payload:
        simulation_generator.set_state(
            _cpu_rng_state(
                payload["simulation_generator_state"], "simulation_generator_state"
            )
        )
    if "loader_generator_state" in payload:
        loader_generator.set_state(
            _cpu_rng_state(payload["loader_generator_state"], "loader_generator_state")
        )
    if "torch_rng_state" in payload:
        torch.set_rng_state(_cpu_rng_state(payload["torch_rng_state"], "torch_rng_state"))
    cuda_states = payload.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_states:
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("checkpoint cuda_rng_state_all must be a sequence")
        torch.cuda.set_rng_state_all(
            [
                _cpu_rng_state(state, f"cuda_rng_state_all[{index}]")
                for index, state in enumerate(cuda_states)
            ]
        )


def run_training(config: TrainingConfig) -> Path:
    """Train one acquisition branch and return its best checkpoint path."""

    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    amp = bool(config.amp and device.type == "cuda")
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentation = JointAugmentation(
        horizontal_flip_probability=config.horizontal_flip_probability,
        vertical_flip_probability=config.vertical_flip_probability,
        intensity_scale=(config.intensity_scale_min, config.intensity_scale_max)
    )
    train_dataset = AnatomicalDMIDataset(config.train_manifest, augment=augmentation)
    val_dataset = AnatomicalDMIDataset(config.val_manifest)
    loader_generator = torch.Generator().manual_seed(config.seed)
    train_loader = _make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=loader_generator,
        device=device,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed + 1),
        device=device,
    )

    model = PriorInformedUNet3D(
        in_channels=config.spectral_points,
        channels=config.channels,
        attention_heads=config.attention_heads,
        temporal_heads=config.temporal_heads,
        temporal_layers=config.temporal_layers,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.minimum_learning_rate,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 0
    best_val_loss = math.inf
    train_generator = torch.Generator(device=device).manual_seed(config.seed + 2)

    if config.resume:
        payload = load_checkpoint(config.resume, map_location=device)
        if payload["model_config"] != model.get_config():
            raise ValueError("resume checkpoint model config does not match CLI config")
        previous_config = payload.get("training_config")
        if not isinstance(previous_config, dict):
            raise ValueError("resume checkpoint is missing its training config")
        continuation_keys = (
            "train_manifest",
            "val_manifest",
            "output_dir",
            "branch",
            "epochs",
            "batch_size",
            "dynamic_frames",
            "spectral_points",
            "noise_std_min",
            "noise_std_max",
            "learning_rate",
            "minimum_learning_rate",
            "spectral_bandwidth_hz",
            "peak_offsets_hz",
            "t2_seconds",
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "intensity_scale_min",
            "intensity_scale_max",
            "seed",
            "num_workers",
            "amp",
        )
        changed = [
            key
            for key in continuation_keys
            if previous_config.get(key) != getattr(config, key)
        ]
        if changed:
            raise ValueError(
                "resume configuration changed trajectory-defining fields: "
                + ", ".join(changed)
            )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload.get("scaler_state", {}))
        _restore_rng_states(
            payload,
            simulation_generator=train_generator,
            loader_generator=loader_generator,
            device=device,
        )
        start_epoch = int(payload.get("epoch", 0))
        best_val_loss = float(payload.get("best_val_loss", math.inf))

    # Write provenance only after a resume checkpoint has passed all trajectory
    # compatibility checks. A rejected resume must not alter a valid run record.
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    spectral_model = SpectralModel(
        spectral_points=config.spectral_points,
        spectral_bandwidth_hz=config.spectral_bandwidth_hz,
        peak_offsets_hz=config.peak_offsets_hz,
        t2_seconds=config.t2_seconds,
    )
    wa_map = hann_repetition_map(dtype=torch.float32, device=device)
    repetition_map = (
        wa_map
        if config.branch == "wa"
        else matched_uniform_repetition_map(wa_map, dtype=torch.float32)
    )
    log_path = output_dir / "metrics.jsonl"
    best_path = output_dir / "best.pt"

    for epoch in range(start_epoch, config.epochs):
        started = time.time()
        model.train()
        running_loss = 0.0
        samples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for batch in progress:
            degraded, target, anatomy = _batch_pair(
                batch,
                repetition_map=repetition_map,
                spectral_model=spectral_model,
                config=config,
                generator=train_generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                prediction = model(degraded, anatomy)
                loss = nn.functional.mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = degraded.shape[0]
            running_loss += float(loss.item()) * count
            samples += count
            progress.set_postfix(loss=running_loss / samples)
        train_loss = running_loss / max(samples, 1)
        val_loss = _evaluate(
            model,
            val_loader,
            repetition_map=repetition_map,
            spectral_model=spectral_model,
            config=config,
            device=device,
            amp=amp,
        )
        scheduler.step()
        improved = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            config=config,
            spectral_model=spectral_model,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            simulation_generator=train_generator,
            loader_generator=loader_generator,
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, best_path)
        if (epoch + 1) % config.save_every == 0:
            atomic_torch_save(payload, output_dir / f"epoch-{epoch + 1:04d}.pt")
        record = {
            "epoch": epoch + 1,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
    return best_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    config = TrainingConfig(**vars(args))
    best_path = run_training(config)
    print(f"best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
