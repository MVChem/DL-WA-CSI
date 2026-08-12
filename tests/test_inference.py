import numpy as np
import pytest
import torch

from dlwa_csi.checkpointing import CHECKPOINT_FORMAT_VERSION, atomic_torch_save
from dlwa_csi.contracts import training_runtime_contract
from dlwa_csi.inference import _normalized_anatomy, main
from dlwa_csi.models import PriorInformedUNet3D
from dlwa_csi.simulation import SpectralModel, fit_metabolite_maps


def _inference_fixture(tmp_path):
    model = PriorInformedUNet3D(
        in_channels=3,
        channels=(2,),
        blocks_per_stage=1,
        attention_heads=(1,),
        attention_pool_size=2,
        query_chunk_size=32,
        temporal_heads=1,
        temporal_layers=1,
        norm_groups=1,
    )
    contract = training_runtime_contract("wa")
    contract["acquisition"]["matrix_size"] = [8, 8]
    contract["preprocessing"]["anatomy_size"] = [8, 8]
    checkpoint = atomic_torch_save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": model.get_config(),
            "model_state": model.state_dict(),
            "training_config": {"branch": "wa"},
            "spectral_model": SpectralModel(spectral_points=3).to_dict(),
            **contract,
        },
        tmp_path / "model.pt",
    )
    generator = np.random.default_rng(8)
    csi = (
        generator.normal(size=(2, 1, 3, 8, 8))
        + 1j * generator.normal(size=(2, 1, 3, 8, 8))
    ).astype(np.complex64)
    input_path = tmp_path / "input.npz"
    np.savez_compressed(input_path, csi=csi)
    anatomy_paths = []
    for index in range(2):
        path = tmp_path / f"anatomy-{index}.npy"
        np.save(path, generator.random((8, 8), dtype=np.float32))
        anatomy_paths.append(path)
    return checkpoint, input_path, anatomy_paths


def test_inference_records_real_magnitude_domain_representation(tmp_path):
    checkpoint, input_path, anatomy_paths = _inference_fixture(tmp_path)
    output = tmp_path / "output.npz"
    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(input_path),
                "--anatomy",
                *(str(path) for path in anatomy_paths),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    with np.load(output) as archive:
        estimate = archive["magnitude_domain_estimate"]
        assert estimate.shape == (2, 1, 3, 8, 8)
        assert not np.iscomplexobj(estimate)
        assert archive["output_representation"].item() == (
            "real_valued_magnitude_fid_estimate"
        )
        assert archive["input_scale_quantile"].item() == pytest.approx(0.995)
    with pytest.raises(TypeError, match="complex FIDs"):
        fit_metabolite_maps(
            torch.from_numpy(estimate), model=SpectralModel(spectral_points=3)
        )


def test_inference_requires_matching_anatomy_batch(tmp_path):
    checkpoint, input_path, anatomy_paths = _inference_fixture(tmp_path)
    with pytest.raises(ValueError, match="anatomy images"):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(input_path),
                "--anatomy",
                str(anatomy_paths[0]),
                "--output",
                str(tmp_path / "output.npz"),
                "--device",
                "cpu",
            ]
        )


def test_inference_anatomy_normalization_matches_training_resize_order(tmp_path):
    anatomy = np.zeros((16, 16), dtype=np.float32)
    anatomy[7, 7] = 8.0
    path = tmp_path / "sparse-anatomy.npy"
    np.save(path, anatomy)

    normalized = _normalized_anatomy(path, 8)
    assert normalized.shape == (1, 1, 8, 8)
    assert normalized.min().item() == pytest.approx(0.0)
    assert normalized.max().item() == pytest.approx(1.0)
