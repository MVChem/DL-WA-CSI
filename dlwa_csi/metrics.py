"""Metrics and statistics used by the paper's validation workflows."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def roi_nrmse_percent(
    estimate: np.ndarray,
    reference: np.ndarray | float,
    mask: np.ndarray,
    *,
    normalization: float,
) -> float:
    """Return ROI NRMSE in percent using the paper's full-voxel criterion."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if estimate.shape != mask.shape:
        raise ValueError("estimate and mask must have the same shape")
    if reference.ndim and reference.shape != estimate.shape:
        raise ValueError("array reference must have the same shape as estimate")
    if not mask.any():
        raise ValueError("mask cannot be empty")
    if normalization <= 0:
        raise ValueError("normalization must be positive")
    squared_error = np.square(estimate - reference)
    return 100.0 * float(np.sqrt(np.mean(squared_error[mask]))) / normalization


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson score interval (95% by default)."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return center - half_width, center + half_width


def logistic_probability(concentration: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    exponent = np.clip(-(alpha + beta * np.asarray(concentration)), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(exponent))


def fit_logistic_binary(
    concentrations: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    """Fit ``P(c)=1/(1+exp(-(alpha+beta*c)))`` to binary outcomes."""

    concentrations = np.asarray(concentrations, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if concentrations.shape != outcomes.shape or concentrations.ndim != 1:
        raise ValueError("concentrations and outcomes must be equal-length 1-D arrays")
    if not np.all(np.isin(outcomes, (0.0, 1.0))):
        raise ValueError("outcomes must be binary")
    if np.unique(outcomes).size < 2:
        raise ValueError("both successful and unsuccessful outcomes are needed")
    def negative_log_likelihood(parameters: np.ndarray) -> float:
        logits = parameters[0] + parameters[1] * concentrations
        return float(np.sum(np.logaddexp(0.0, logits) - outcomes * logits))

    result = minimize(
        negative_log_likelihood,
        x0=np.asarray((-5.0, 1.0)),
        method="L-BFGS-B",
        bounds=((-100.0, 100.0), (-100.0, 100.0)),
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def logistic_crossing(alpha: float, beta: float, probability: float = 0.95) -> float:
    """Return the concentration where a fitted logistic reaches probability."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    if beta == 0:
        raise ValueError("beta cannot be zero")
    return (math.log(probability / (1.0 - probability)) - alpha) / beta


TUBE_CORNERS: tuple[tuple[int, int], ...] = (
    (4, 9),
    (11, 9),
    (18, 9),
    (25, 9),
    (4, 20),
    (11, 20),
    (18, 20),
    (25, 20),
)


def tube_masks(
    matrix_size: int = 32,
    tube_size: int = 5,
    corners: tuple[tuple[int, int], ...] = TUBE_CORNERS,
) -> np.ndarray:
    """Return the eight fixed paper ROIs as ``[8, H, W]`` boolean masks."""

    masks = np.zeros((len(corners), matrix_size, matrix_size), dtype=bool)
    for index, (x, y) in enumerate(corners):
        if x < 0 or y < 0 or x + tube_size > matrix_size or y + tube_size > matrix_size:
            raise ValueError(f"tube at {(x, y)} falls outside the matrix")
        # The paper specifies (x, y); arrays are indexed [y, x].
        masks[index, y : y + tube_size, x : x + tube_size] = True
    if np.any(masks.sum(axis=0) > 1):
        raise ValueError("tube masks overlap")
    return masks
