import pytest
import torch

from dlwa_csi.acquisition import hann_repetition_map
from dlwa_csi.simulation import (
    SpectralModel,
    fit_metabolite_maps,
    prepare_network_pair,
    random_smooth_curves,
    spectral_basis,
    synthesize_dynamic_fids,
)


def test_spectral_basis_and_dynamic_fid_shapes():
    model = SpectralModel(spectral_points=16)
    basis = spectral_basis(model)
    assert basis.shape == (3, 16)
    assert torch.allclose(basis[:, 0], torch.ones(3, dtype=torch.complex64))

    maps = torch.ones((2, 3, 8, 8))
    curves = random_smooth_curves(2, 5, generator=torch.Generator().manual_seed(3))
    fids = synthesize_dynamic_fids(maps, curves, model=model)
    assert fids.shape == (2, 5, 16, 8, 8)
    assert torch.is_complex(fids)


def test_network_pair_uses_inference_available_input_scale_for_both_sides():
    model = SpectralModel(spectral_points=8)
    maps = torch.rand((1, 3, 8, 8), generator=torch.Generator().manual_seed(2))
    curves = torch.ones((1, 2, 3))
    clean = synthesize_dynamic_fids(maps, curves, model=model)
    pair = prepare_network_pair(
        clean,
        hann_repetition_map(8, center_repetitions=17, dtype=torch.float32),
        noise_std_per_excitation=0.0,
    )
    assert pair.degraded.shape == clean.shape
    assert pair.target.shape == clean.shape
    assert pair.scale.shape == (1,)
    assert torch.isfinite(pair.degraded).all()
    expected_scale = torch.quantile(pair.complex_degraded.abs().flatten(1), 0.995, dim=1)
    assert torch.allclose(pair.scale, expected_scale)


def test_complex_spectral_fit_recovers_known_amplitudes_and_supports_phase():
    model = SpectralModel(spectral_points=16)
    maps = torch.rand((1, 3, 4, 4), generator=torch.Generator().manual_seed(8))
    curves = torch.rand((1, 2, 3), generator=torch.Generator().manual_seed(9))
    phase = torch.full((1, 4, 4), 0.4)
    fids = synthesize_dynamic_fids(maps, curves, model=model, spatial_phase=phase)
    fitted = fit_metabolite_maps(fids, model=model)
    expected = curves[:, :, :, None, None] * maps[:, None]
    assert torch.allclose(fitted, expected, atol=2e-5, rtol=2e-5)


def test_voxel_specific_dynamic_maps_are_supported():
    model = SpectralModel(spectral_points=8)
    dynamic_maps = torch.zeros(1, 2, 3, 4, 4)
    dynamic_maps[:, 0, 1, :2, :2] = 2.0
    dynamic_maps[:, 1, 2, 2:, 2:] = 3.0
    fids = synthesize_dynamic_fids(dynamic_maps, model=model)
    fitted = fit_metabolite_maps(fids, model=model)
    assert torch.allclose(fitted, dynamic_maps, atol=2e-5, rtol=2e-5)


def test_magnitude_channels_are_rejected_by_linear_spectral_fit():
    real_channels = torch.ones(1, 2, 16, 4, 4)
    with pytest.raises(TypeError, match="complex FIDs"):
        fit_metabolite_maps(real_channels, model=SpectralModel(spectral_points=16))


def test_integer_static_maps_do_not_truncate_fractional_curves():
    model = SpectralModel(spectral_points=8)
    maps = torch.ones(1, 3, 2, 2, dtype=torch.int64)
    curves = torch.full((1, 1, 3), 0.5)
    fitted = fit_metabolite_maps(
        synthesize_dynamic_fids(maps, curves, model=model), model=model
    )
    assert torch.allclose(fitted, torch.full((1, 1, 3, 2, 2), 0.5), atol=2e-5)


def test_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        synthesize_dynamic_fids(torch.ones(3, 8, 8), torch.ones(2, 3))
