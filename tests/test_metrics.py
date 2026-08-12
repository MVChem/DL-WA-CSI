import numpy as np
import pytest

from dlwa_csi.metrics import (
    fit_logistic_binary,
    logistic_crossing,
    roi_nrmse_percent,
    tube_masks,
    wilson_interval,
)


def test_tube_geometry_matches_protocol():
    masks = tube_masks()
    assert masks.shape == (8, 32, 32)
    assert np.all(masks.sum(axis=(1, 2)) == 25)
    assert masks.sum() == 200
    assert not np.any(masks.sum(axis=0) > 1)
    expected_corners = (
        (4, 9),
        (11, 9),
        (18, 9),
        (25, 9),
        (4, 20),
        (11, 20),
        (18, 20),
        (25, 20),
    )
    for mask, (x, y) in zip(masks, expected_corners, strict=True):
        ys, xs = np.nonzero(mask)
        assert (xs.min(), ys.min()) == (x, y)
        assert (xs.max(), ys.max()) == (x + 4, y + 4)


def test_roi_nrmse_is_full_voxel_error_and_strict_threshold_can_be_applied():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    estimate = np.full((4, 4), 10.0)
    estimate[1, 1] = 12.0
    assert roi_nrmse_percent(estimate, 10.0, mask, normalization=10.0) == pytest.approx(10.0)
    assert not (roi_nrmse_percent(estimate, 10.0, mask, normalization=10.0) < 10.0)


def test_wilson_interval_known_case():
    low, high = wilson_interval(95, 100)
    assert low == pytest.approx(0.88825, abs=1e-5)
    assert high == pytest.approx(0.97846, abs=1e-5)


def test_logistic_crossing():
    alpha, beta = -4.0, 2.0
    c = logistic_crossing(alpha, beta, 0.95)
    assert c == pytest.approx((np.log(19.0) + 4.0) / 2.0)


def test_binary_logistic_fit_is_maximum_likelihood():
    concentrations = np.repeat(np.arange(1.0, 6.0), 20)
    rng = np.random.default_rng(12)
    probabilities = 1.0 / (1.0 + np.exp(-(-4.0 + concentrations)))
    outcomes = rng.binomial(1, probabilities).astype(float)
    alpha, beta = fit_logistic_binary(concentrations, outcomes)
    assert alpha < 0
    assert beta > 0
    fitted = 1.0 / (1.0 + np.exp(-(alpha + beta * concentrations)))
    score = np.asarray(
        [np.sum(fitted - outcomes), np.sum((fitted - outcomes) * concentrations)]
    )
    assert np.linalg.norm(score) < 1e-3
