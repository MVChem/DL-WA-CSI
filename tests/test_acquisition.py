import numpy as np
import pytest
import torch

from dlwa_csi.acquisition import (
    apply_kspace_apodization,
    apply_repetition_acquisition,
    channel_l2_point_profile,
    hann_repetition_map,
    matched_uniform_repetition_map,
    per_excitation_noise_for_image_sd,
    profile_fwhm,
    separable_hann_window,
    sinc_interpolated_profile,
)
from dlwa_csi.simulation import spectral_basis


def test_paper_repetition_map_constants():
    wa = hann_repetition_map()
    assert wa.shape == (32, 32)
    assert wa[16, 16].item() == 263
    assert wa[0, 0].item() == 1
    assert wa.sum().item() == 68_106

    ua = matched_uniform_repetition_map(wa)
    assert torch.all(ua == ua[0, 0])
    assert ua.sum().item() == pytest.approx(68_106)
    assert ua[0, 0].item() == pytest.approx(68_106 / 1_024)


def test_paper_analytical_fwhm_golden_values():
    wa = hann_repetition_map()
    ua = matched_uniform_repetition_map(wa)
    x_ua, p_ua = sinc_interpolated_profile(ua)
    x_wa, p_wa = sinc_interpolated_profile(wa)
    assert profile_fwhm(x_ua, p_ua) == pytest.approx(1.20714014, abs=1e-7)
    assert profile_fwhm(x_wa, p_wa) == pytest.approx(1.98032301, abs=1e-7)


@pytest.mark.parametrize("mode", ["ua", "wa"])
def test_point_probe_acquisition_matches_analytical_profile(mode):
    wa = hann_repetition_map(dtype=torch.float32)
    weights = matched_uniform_repetition_map(wa, dtype=torch.float32) if mode == "ua" else wa
    point = torch.zeros((72, 32, 32), dtype=torch.complex64)
    lactate_fid = spectral_basis()[2]
    point[:, 16, 16] = lactate_fid
    acquired = apply_repetition_acquisition(point, weights).image

    # Interpolate each channel separately, then combine by L2 exactly as in SI.
    _, probe = channel_l2_point_profile(acquired, center_peak=False)
    _, analytical = sinc_interpolated_profile(weights)
    assert torch.max(torch.abs(probe.double() - analytical)).item() < 2e-6


def test_noiseless_uniform_acquisition_is_identity_after_center_normalization():
    generator = torch.Generator().manual_seed(4)
    image = torch.complex(
        torch.randn((2, 3, 8, 8), generator=generator),
        torch.randn((2, 3, 8, 8), generator=generator),
    )
    repetitions = torch.full((8, 8), 7.0)
    result = apply_repetition_acquisition(image, repetitions)
    assert torch.allclose(result.image, image, atol=2e-6, rtol=2e-6)


def test_balanced_integer_ua_preserves_total():
    wa = hann_repetition_map()
    ua = matched_uniform_repetition_map(wa, integer=True)
    assert ua.sum().item() == wa.sum().item()
    assert set(np.unique(ua.numpy())) == {66.0, 67.0}


def test_post_acquisition_apodization_is_distinct_from_repetition_redistribution():
    point = torch.zeros((1, 32, 32), dtype=torch.complex64)
    point[:, 16, 16] = 1
    window = separable_hann_window(dtype=torch.float32)
    apodized = apply_kspace_apodization(point, window)
    wa = apply_repetition_acquisition(
        point, hann_repetition_map(dtype=torch.float32), normalization="center"
    ).image
    # Rounding plus the required outer repetition of one makes WA similar but
    # not identical to multiplying already-acquired data by an ideal Hann window.
    assert not torch.equal(apodized, wa)
    assert apodized.abs().max().item() == pytest.approx(wa.abs().max().item(), rel=0.03)


def test_unnormalized_global_noise_variance_depends_on_total_not_distribution():
    generator = torch.Generator().manual_seed(123)
    wa = hann_repetition_map(8, center_repetitions=17, dtype=torch.float32)
    ua = matched_uniform_repetition_map(wa, dtype=torch.float32)
    zeros = torch.zeros((3000, 1, 8, 8), dtype=torch.complex64)
    ua_noise = apply_repetition_acquisition(
        zeros,
        ua,
        noise_std_per_excitation=1.0,
        normalization="none",
        generator=generator,
    ).image.real
    wa_noise = apply_repetition_acquisition(
        zeros,
        wa,
        noise_std_per_excitation=1.0,
        normalization="none",
        generator=generator,
    ).image.real
    expected = float(wa.sum()) / (8 * 8) ** 2
    assert ua_noise.var().item() == pytest.approx(expected, rel=0.03)
    assert wa_noise.var().item() == pytest.approx(expected, rel=0.03)
    normalized_variance_ratio = (wa.max() / ua.max()).square().item()
    observed_ratio = (ua_noise / ua.max()).var().item() / (
        wa_noise / wa.max()
    ).var().item()
    assert observed_ratio == pytest.approx(normalized_variance_ratio, rel=0.05)


def test_image_noise_calibration_matches_reported_branch_scale_formula():
    wa = hann_repetition_map()
    ua = matched_uniform_repetition_map(wa)
    ua_sigma = per_excitation_noise_for_image_sd(3.60, ua)
    wa_sigma = per_excitation_noise_for_image_sd(0.94, wa)
    assert ua_sigma == pytest.approx(939.6, rel=0.002)
    assert wa_sigma == pytest.approx(970.2, rel=0.002)
