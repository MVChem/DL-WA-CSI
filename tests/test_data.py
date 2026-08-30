import json

import numpy as np
import pytest
import torch

from dlwa_csi.data import AnatomicalDMIDataset, JointAugmentation


def _write_arrays(tmp_path):
    paths = {}
    grid = np.arange(64, dtype=np.float32).reshape(8, 8)
    for index, name in enumerate(("anatomy", "water", "glucose", "lactate"), start=1):
        path = tmp_path / f"{name}.npy"
        np.save(path, grid * index)
        paths[name] = path.name
    return paths


def test_jsonl_dataset_loads_real_anatomy_and_metabolites(tmp_path):
    paths = _write_arrays(tmp_path)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps({"id": "case-1", **paths}) + "\n", encoding="utf-8")
    dataset = AnatomicalDMIDataset(manifest, dmi_size=(4, 4), anatomy_size=(16, 16))
    sample = dataset[0]
    assert sample["id"] == "case-1"
    assert sample["anatomy"].shape == (16, 16)
    assert sample["metabolites"].shape == (3, 4, 4)
    assert sample["anatomy"].max().item() == pytest.approx(1.0)


def test_manifest_rejects_missing_anatomy(tmp_path):
    paths = _write_arrays(tmp_path)
    del paths["anatomy"]
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(paths) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="anatomy"):
        AnatomicalDMIDataset(manifest)


def test_legacy_folder_requires_anatomy(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    array = np.ones((8, 8), dtype=np.float32)
    for name in ("water", "glu", "lac"):
        np.save(sample_dir / f"{name}.npy", array)
    manifest = tmp_path / "samples.txt"
    manifest.write_text(str(sample_dir) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="anatomy"):
        AnatomicalDMIDataset(manifest)


def test_joint_augmentation_validation():
    with pytest.raises(ValueError):
        JointAugmentation(horizontal_flip_probability=2.0)


def test_numpy_metabolite_scale_and_constant_maps_are_preserved(tmp_path):
    anatomy = np.arange(64, dtype=np.float32).reshape(8, 8)
    np.save(tmp_path / "anatomy.npy", anatomy)
    for name, value in (("water", 1.0), ("glucose", 2.0), ("lactate", 3.0)):
        np.save(tmp_path / f"{name}.npy", np.full((8, 8), value, dtype=np.float32))
    record = {
        "anatomy": "anatomy.npy",
        "water": "water.npy",
        "glucose": "glucose.npy",
        "lactate": "lactate.npy",
    }
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    maps = AnatomicalDMIDataset(manifest, dmi_size=(8, 8))[0]["metabolites"]
    assert torch.all(maps[0] == 1.0)
    assert torch.all(maps[1] == 2.0)
    assert torch.all(maps[2] == 3.0)
