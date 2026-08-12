import pytest
import torch

from dlwa_csi.baselines import CNNAutoencoder, spin_svd_reference, tmppca_reference


def test_reference_denoisers_preserve_shape_and_finite_values():
    generator = torch.Generator().manual_seed(7)
    data = torch.randn((2, 3, 7, 7), generator=generator)
    tmppca = tmppca_reference(data, patch_size=3, stride=3)
    spin = spin_svd_reference(data, rank=2)
    assert tmppca.shape == data.shape
    assert spin.shape == data.shape
    assert torch.isfinite(tmppca).all()
    assert torch.isfinite(spin).all()


def test_reference_denoisers_support_complex_data():
    data = torch.complex(torch.randn(2, 3, 5, 5), torch.randn(2, 3, 5, 5))
    assert torch.is_complex(tmppca_reference(data, patch_size=3, stride=3))
    assert torch.is_complex(spin_svd_reference(data, rank=2))


def test_cnn_autoencoder_contract_and_gradient():
    model = CNNAutoencoder(spectral_channels=4, hidden_channels=4)
    inputs = torch.randn(1, 3, 4, 7, 9, requires_grad=True)
    output = model(inputs)
    assert output.shape == inputs.shape
    output.square().mean().backward()
    assert inputs.grad is not None


def test_baseline_validation():
    with pytest.raises(ValueError, match="rank"):
        spin_svd_reference(torch.randn(2, 3, 4), rank=0)
    with pytest.raises(ValueError, match="patch_size"):
        tmppca_reference(torch.randn(2, 4, 4), patch_size=2)
    with pytest.raises(ValueError, match="stride"):
        tmppca_reference(torch.randn(2, 5, 5), patch_size=3, stride=4)


def test_tmppca_patch_coverage_does_not_zero_pixels():
    constant = torch.ones(2, 5, 5)
    denoised = tmppca_reference(constant, patch_size=3, stride=3, rank=1)
    assert torch.allclose(denoised, constant)
