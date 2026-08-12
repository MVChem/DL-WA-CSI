"""Reproducible protocol helpers for the paper's validation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import roi_nrmse_percent, tube_masks, wilson_interval

PAPER_METHODS = ("ua_csi", "wa_csi", "dl_ua_csi", "dl_wa_csi")


def concentration_grid(
    minimum: float = 0.25,
    maximum: float = 12.0,
    step: float = 0.25,
) -> np.ndarray:
    """Return the inclusive 0.25--12.00 mM paper grid without float drift."""

    if minimum <= 0 or maximum < minimum or step <= 0:
        raise ValueError("require 0 < minimum <= maximum and step > 0")
    count_float = (maximum - minimum) / step
    count = int(round(count_float))
    if not np.isclose(count_float, count):
        raise ValueError("range must be exactly divisible by step")
    return minimum + step * np.arange(count + 1, dtype=np.float64)


@dataclass(frozen=True)
class RecoveryProtocol:
    """All randomized concentration assignments for the Monte Carlo study."""

    levels: np.ndarray
    assignments: np.ndarray
    target_tube_index: int
    masks: np.ndarray
    seed: int
    context_sampling: str = "iid_uniform_with_replacement"

    @property
    def repeats(self) -> int:
        return int(self.assignments.shape[1])

    def ground_truth_maps(self, level_index: int) -> np.ndarray:
        """Materialize ``[repeats, 32, 32]`` lactate concentration maps."""

        if not 0 <= level_index < self.levels.size:
            raise IndexError(level_index)
        return np.einsum(
            "rt,thw->rhw", self.assignments[level_index], self.masks.astype(np.float64)
        )


def generate_recovery_protocol(
    *,
    repeats: int = 100,
    target_tube_index: int = 5,
    seed: int = 20260811,
) -> RecoveryProtocol:
    """Generate a reproducible 48-level, eight-tube protocol instantiation.

    The SI artwork appears to indicate tube index 5 (corner ``(11, 20)``), but
    the prose does not uniquely identify it. It is therefore explicit and
    configurable here. Other tube concentrations are independently sampled
    uniformly with replacement from the target level through 12 mM. Uniform
    probabilities are an explicit implementation choice because the SI does
    not state the original sampling distribution or seed.
    """

    levels = concentration_grid()
    masks = tube_masks()
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0 <= target_tube_index < masks.shape[0]:
        raise ValueError("target_tube_index is outside the eight-tube layout")
    rng = np.random.default_rng(seed)
    assignments = np.empty((levels.size, repeats, masks.shape[0]), dtype=np.float64)
    for level_index, target in enumerate(levels):
        permitted = levels[level_index:]
        assignments[level_index] = rng.choice(
            permitted, size=(repeats, masks.shape[0]), replace=True
        )
        assignments[level_index, :, target_tube_index] = target
    return RecoveryProtocol(levels, assignments, target_tube_index, masks, seed)


@dataclass(frozen=True)
class RecoverySummary:
    levels: np.ndarray
    probability: np.ndarray
    wilson_low: np.ndarray
    wilson_high: np.ndarray
    nrmse_percent: np.ndarray
    reliable: np.ndarray


def summarize_recovery_maps(
    estimates: np.ndarray,
    protocol: RecoveryProtocol,
    *,
    threshold_percent: float = 10.0,
) -> RecoverySummary:
    """Score maps using only the designated tube's full-voxel ROI NRMSE."""

    estimates = np.asarray(estimates, dtype=np.float64)
    expected = (protocol.levels.size, protocol.repeats, 32, 32)
    if estimates.shape != expected:
        raise ValueError(f"estimates must have shape {expected}, got {estimates.shape}")
    if threshold_percent <= 0:
        raise ValueError("threshold_percent must be positive")
    target_mask = protocol.masks[protocol.target_tube_index]
    nrmse = np.empty(expected[:2], dtype=np.float64)
    for level_index, target in enumerate(protocol.levels):
        for repeat in range(protocol.repeats):
            nrmse[level_index, repeat] = roi_nrmse_percent(
                estimates[level_index, repeat],
                target,
                target_mask,
                normalization=float(target),
            )
    # The paper's criterion is strictly less than 10%, not <= 10%.
    reliable = nrmse < threshold_percent
    probability = reliable.mean(axis=1)
    intervals = np.asarray(
        [wilson_interval(int(row.sum()), protocol.repeats) for row in reliable]
    )
    return RecoverySummary(
        levels=protocol.levels.copy(),
        probability=probability,
        wilson_low=intervals[:, 0],
        wilson_high=intervals[:, 1],
        nrmse_percent=nrmse,
        reliable=reliable,
    )
