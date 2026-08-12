import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dlwa_csi.checkpointing import CHECKPOINT_FORMAT_VERSION, atomic_torch_save
from dlwa_csi.contracts import training_runtime_contract
from dlwa_csi.models import PriorInformedUNet3D
from dlwa_csi.simulation import SpectralModel
from scripts.spatial_response import _validate_learned_checkpoint, main


def test_learned_probe_checkpoint_must_be_wa_and_72_channels():
    model = SimpleNamespace(in_channels=72)
    payload = {
        "training_config": {"branch": "wa"},
        "spectral_model": SpectralModel().to_dict(),
        **training_runtime_contract("wa"),
    }
    spectral_model, lactate_index, _ = _validate_learned_checkpoint(model, payload)
    assert spectral_model.spectral_points == 72
    assert lactate_index == 2

    payload.update(training_runtime_contract("ua"))
    payload["training_config"] = {"branch": "ua"}
    with pytest.raises(ValueError, match="branch must be 'wa'"):
        _validate_learned_checkpoint(model, payload)

    payload.update(training_runtime_contract("wa"))
    payload["training_config"] = {"branch": "wa"}
    with pytest.raises(ValueError, match="72-channel"):
        _validate_learned_checkpoint(SimpleNamespace(in_channels=71), payload)

    payload.pop("spectral_model")
    with pytest.raises(ValueError, match="nonempty spectral_model"):
        _validate_learned_checkpoint(model, payload)


def test_learned_probe_resolves_lactate_by_name():
    spectral = SpectralModel(
        metabolite_names=("lactate", "water", "glucose"),
        peak_offsets_hz=(-209.0, 0.0, -55.0),
        t2_seconds=(0.060, 0.080, 0.070),
    )
    payload = {
        "training_config": {"branch": "wa"},
        "spectral_model": spectral.to_dict(),
        **training_runtime_contract("wa"),
    }
    _, lactate_index, _ = _validate_learned_checkpoint(
        SimpleNamespace(in_channels=72), payload
    )
    assert lactate_index == 0


def test_spatial_response_argument_validation():
    with pytest.raises(ValueError, match="anatomy-size"):
        main(["--anatomy-size", "0"])
    with pytest.raises(ValueError, match="at least 32"):
        main(["--interpolation-points", "16"])


def test_fixed_checkpoint_learned_probe_path(tmp_path, capsys):
    model = PriorInformedUNet3D(
        in_channels=72,
        channels=(2,),
        blocks_per_stage=1,
        attention_heads=(1,),
        attention_pool_size=2,
        query_chunk_size=128,
        temporal_heads=1,
        temporal_layers=1,
        norm_groups=1,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = atomic_torch_save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": model.get_config(),
            "model_state": model.state_dict(),
            "training_config": {"branch": "wa"},
            "spectral_model": SpectralModel().to_dict(),
            **training_runtime_contract("wa"),
        },
        tmp_path / "wa.pt",
    )
    anatomy = tmp_path / "anatomy.npy"
    np.save(anatomy, np.arange(64, dtype=np.float32).reshape(8, 8))
    output = tmp_path / "response.npz"

    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--anatomy",
                str(anatomy),
                "--interpolation-points",
                "256",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dl_wa_effective_fwhm_voxels"] == pytest.approx(
        report["wa_fwhm_voxels"], rel=1e-4
    )
    assert report["dl_wa_peak_offset_voxels"] == pytest.approx(0.0, abs=1e-8)
    with np.load(output) as archive:
        assert archive["dl_wa_profile"].shape == (256,)
