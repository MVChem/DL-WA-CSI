import pytest
import torch

from dlwa_csi.checkpointing import CHECKPOINT_FORMAT_VERSION, atomic_torch_save, load_model
from dlwa_csi.models import PriorInformedUNet3D


def test_checkpoint_round_trip(tmp_path):
    model = PriorInformedUNet3D(
        in_channels=3,
        channels=(4, 8),
        blocks_per_stage=1,
        attention_heads=(1, 2),
        attention_pool_size=2,
        query_chunk_size=16,
        temporal_heads=2,
        temporal_layers=1,
    )
    path = atomic_torch_save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": model.get_config(),
            "model_state": model.state_dict(),
        },
        tmp_path / "model.pt",
    )
    restored, payload = load_model(path)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert restored.get_config() == model.get_config()
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(first, second)


def test_checkpoint_validation(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="missing"):
        load_model(path)
