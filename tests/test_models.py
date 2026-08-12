import json

import pytest
import torch

from dlwa_csi.models import PriorInformedUNet3D


def _tiny_model(**overrides):
    config = {
        "in_channels": 3,
        "channels": (4, 8),
        "blocks_per_stage": 1,
        "attention_heads": (1, 2),
        "attention_pool_size": 2,
        "query_chunk_size": 16,
        "temporal_heads": 2,
        "temporal_layers": 1,
        "temporal_mlp_ratio": 2.0,
        "spectral_reduction": 2,
        "norm_groups": 2,
    }
    config.update(overrides)
    return PriorInformedUNet3D(**config)


@pytest.mark.parametrize("frames", [1, 5])
@pytest.mark.parametrize("anatomy_has_channel", [False, True])
def test_output_shape_for_arbitrary_dynamic_length(frames, anatomy_has_channel):
    torch.manual_seed(1)
    model = _tiny_model().eval()
    dmi = torch.randn(1, frames, 3, 7, 9)
    anatomy = torch.randn(1, 1, 7, 9)
    if not anatomy_has_channel:
        anatomy = anatomy[:, 0]

    with torch.no_grad():
        output = model(dmi, anatomy)

    assert output.shape == dmi.shape


def test_anatomical_prior_changes_the_reconstruction():
    torch.manual_seed(2)
    model = _tiny_model().eval()
    dmi = torch.randn(1, 3, 3, 8, 8)
    anatomy_a = torch.zeros(1, 8, 8)
    anatomy_b = torch.randn(1, 8, 8)

    with torch.no_grad():
        output_a = model(dmi, anatomy_a)
        output_b = model(dmi, anatomy_b)

    assert not torch.allclose(output_a, output_b)
    assert (output_a - output_b).abs().mean() > 1e-5


def test_high_resolution_anatomy_is_encoded_at_its_native_scale():
    """The paper pairs a 32x32 DMI grid with a 256x256 anatomical prior."""
    torch.manual_seed(21)
    model = _tiny_model().eval()
    dmi = torch.randn(1, 2, 3, 8, 8)
    anatomy = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        output = model(dmi, anatomy)
    assert output.shape == dmi.shape


def test_gradients_reach_dmi_anatomy_and_model_parameters():
    torch.manual_seed(3)
    model = _tiny_model()
    dmi = torch.randn(1, 2, 3, 6, 6, requires_grad=True)
    anatomy = torch.randn(1, 1, 6, 6, requires_grad=True)

    loss = model(dmi, anatomy).square().mean()
    loss.backward()

    assert dmi.grad is not None and torch.isfinite(dmi.grad).all()
    assert anatomy.grad is not None and torch.isfinite(anatomy.grad).all()
    assert dmi.grad.abs().sum() > 0
    assert anatomy.grad.abs().sum() > 0
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert sum(gradient.abs().sum() for gradient in parameter_gradients) > 0


@pytest.mark.parametrize(
    ("dmi", "anatomy", "message"),
    [
        (torch.randn(1, 3, 6, 6), torch.randn(1, 6, 6), r"\[B, T, C, H, W\]"),
        (torch.randn(1, 2, 4, 6, 6), torch.randn(1, 6, 6), "spectral channels"),
        (torch.randn(1, 2, 3, 6, 6), torch.randn(2, 6, 6), "batch sizes"),
        (torch.randn(1, 2, 3, 6, 6), torch.randn(1, 1, 1), "anatomy spatial dimensions"),
        (torch.randn(1, 2, 3, 6, 6), torch.randn(1, 2, 6, 6), "one channel"),
    ],
)
def test_input_shape_validation(dmi, anatomy, message):
    with pytest.raises(ValueError, match=message):
        _tiny_model()(dmi, anatomy)


def test_dtype_and_constructor_validation():
    model = _tiny_model()
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(1, 2, 3, 6, 6, dtype=torch.int64), torch.ones(1, 6, 6))
    with pytest.raises(ValueError, match="divisible"):
        _tiny_model(channels=(5, 8), attention_heads=(2, 2))
    with pytest.raises(ValueError, match="temporal_heads"):
        _tiny_model(temporal_heads=3)


def test_config_can_be_saved_and_reloaded(tmp_path):
    model = _tiny_model(residual_output=False)

    config_path = model.save_config(tmp_path)
    loaded_json = json.loads(config_path.read_text(encoding="utf-8"))
    restored = PriorInformedUNet3D.from_config(tmp_path)

    assert config_path.name == PriorInformedUNet3D.config_name
    assert loaded_json == model.get_config()
    assert restored.get_config() == model.get_config()
