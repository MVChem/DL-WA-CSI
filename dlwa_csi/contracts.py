"""Runtime contracts recorded in checkpoints and enforced at inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PAPER_MATRIX_SIZE = (32, 32)
PAPER_ANATOMY_SIZE = (256, 256)
NETWORK_SCALE_QUANTILE = 0.995
NETWORK_INPUT_REPRESENTATION = "magnitude_fid"
NETWORK_OUTPUT_REPRESENTATION = "real_valued_magnitude_fid_estimate"
ACQUISITION_NORMALIZATION = "center"
WA_SCHEDULE = "rounded_hann_a263_ties_to_even"
UA_SCHEDULE = "scan_time_matched_uniform_from_wa"


def training_runtime_contract(branch: str) -> dict[str, dict[str, Any]]:
    """Return the explicit acquisition/preprocessing metadata used by training."""

    if branch not in {"ua", "wa"}:
        raise ValueError("branch must be 'ua' or 'wa'")
    return {
        "acquisition": {
            "branch": branch,
            "matrix_size": list(PAPER_MATRIX_SIZE),
            "normalization": ACQUISITION_NORMALIZATION,
            "schedule": WA_SCHEDULE if branch == "wa" else UA_SCHEDULE,
            "wa_center_repetitions": 263,
            "wa_total_repetitions": 68_106,
            "ua_analytical_weight": 68_106 / 1_024,
        },
        "preprocessing": {
            "network_input_representation": NETWORK_INPUT_REPRESENTATION,
            "network_output_representation": NETWORK_OUTPUT_REPRESENTATION,
            "input_scale_quantile": NETWORK_SCALE_QUANTILE,
            "anatomy_normalization": "bilinear_resize_then_per_image_minmax",
            "anatomy_size": list(PAPER_ANATOMY_SIZE),
            "temporal_dynamics": "one_global_curve_per_sample_and_metabolite",
            "spatial_phase": "none_in_default_trainer",
        },
    }


def _positive_pair(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"checkpoint {name} must contain two positive integers")
    return int(value[0]), int(value[1])


def checkpoint_runtime_contract(
    payload: Mapping[str, Any],
    *,
    expected_branch: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and return checkpoint acquisition/preprocessing metadata."""

    acquisition = payload.get("acquisition")
    preprocessing = payload.get("preprocessing")
    if not isinstance(acquisition, dict) or not isinstance(preprocessing, dict):
        raise ValueError(
            "checkpoint is missing explicit acquisition/preprocessing metadata; "
            "convert it before inference"
        )
    branch = acquisition.get("branch")
    if branch not in {"ua", "wa"}:
        raise ValueError("checkpoint acquisition.branch must be 'ua' or 'wa'")
    if expected_branch is not None and branch != expected_branch:
        raise ValueError(f"checkpoint acquisition branch must be {expected_branch!r}")
    training_config = payload.get("training_config")
    if isinstance(training_config, dict) and training_config.get("branch") != branch:
        raise ValueError("checkpoint training and acquisition branch metadata disagree")
    _positive_pair(acquisition.get("matrix_size"), "acquisition.matrix_size")
    if acquisition.get("normalization") not in {"center", "none"}:
        raise ValueError("checkpoint acquisition normalization must be 'center' or 'none'")
    if not isinstance(acquisition.get("schedule"), str):
        raise ValueError("checkpoint acquisition schedule is missing")

    _positive_pair(preprocessing.get("anatomy_size"), "preprocessing.anatomy_size")
    if (
        preprocessing.get("anatomy_normalization")
        != "bilinear_resize_then_per_image_minmax"
    ):
        raise ValueError("unsupported checkpoint anatomy normalization")
    if preprocessing.get("network_input_representation") != NETWORK_INPUT_REPRESENTATION:
        raise ValueError("unsupported checkpoint network input representation")
    if preprocessing.get("network_output_representation") != NETWORK_OUTPUT_REPRESENTATION:
        raise ValueError("unsupported checkpoint network output representation")
    quantile = preprocessing.get("input_scale_quantile")
    valid_quantile = (
        isinstance(quantile, (int, float))
        and not isinstance(quantile, bool)
        and 0 < quantile <= 1
    )
    if not valid_quantile:
        raise ValueError("checkpoint input scale quantile must lie in (0, 1]")
    return acquisition, preprocessing
