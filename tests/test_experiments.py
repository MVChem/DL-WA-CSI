import numpy as np
import pytest

from dlwa_csi.experiments import (
    concentration_grid,
    generate_recovery_protocol,
    summarize_recovery_maps,
)


def test_paper_protocol_scene_count_and_assignment_rules():
    protocol = generate_recovery_protocol(seed=10)
    assert concentration_grid().shape == (48,)
    assert protocol.assignments.shape == (48, 100, 8)
    assert protocol.levels[0] == pytest.approx(0.25)
    assert protocol.levels[-1] == pytest.approx(12.0)
    for index, level in enumerate(protocol.levels):
        assert np.all(protocol.assignments[index, :, protocol.target_tube_index] == level)
        assert np.all(protocol.assignments[index] >= level)
        assert np.all(protocol.assignments[index] <= 12.0)
        scaled = protocol.assignments[index] / 0.25
        assert np.allclose(scaled, np.round(scaled))


def test_scoring_uses_designated_tube_only_and_strict_criterion():
    protocol = generate_recovery_protocol(repeats=2, seed=1)
    estimates = np.empty((48, 2, 32, 32), dtype=np.float64)
    for level_index, level in enumerate(protocol.levels):
        truth = protocol.ground_truth_maps(level_index)
        estimates[level_index] = truth
        target = protocol.masks[protocol.target_tube_index]
        # Exactly 10% uniform error fails because the threshold is strict.
        estimates[level_index, 1, target] = level * 1.10
        # Arbitrarily corrupt non-target tubes; they must not affect the score.
        estimates[level_index, 0, ~target] = 999.0
    summary = summarize_recovery_maps(estimates, protocol)
    assert np.all(summary.reliable[:, 0])
    assert not np.any(summary.reliable[:, 1])
    assert np.all(summary.probability == 0.5)
