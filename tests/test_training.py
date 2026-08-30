import pytest
import torch

from dlwa_csi.training import _restore_rng_states, build_parser


def test_training_requires_both_explicit_noise_bounds():
    required = ["--train-manifest", "train.jsonl", "--val-manifest", "val.jsonl"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(required + ["--noise-std-max", "1"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(required + ["--noise-std-min", "0"])
    parsed = build_parser().parse_args(
        required + ["--noise-std-min", "0", "--noise-std-max", "1"]
    )
    assert parsed.noise_std_min == 0
    assert parsed.noise_std_max == 1


def test_resume_rng_states_are_normalized_to_contiguous_cpu_tensors(monkeypatch):
    simulation_generator = torch.Generator().manual_seed(3)
    loader_generator = torch.Generator().manual_seed(4)
    observed: dict[str, object] = {}

    class RecordingGenerator:
        def __init__(self, name):
            self.name = name

        def set_state(self, state):
            assert state.device.type == "cpu"
            assert state.dtype == torch.uint8
            assert state.is_contiguous()
            observed[self.name] = state.clone()

    monkeypatch.setattr(
        torch,
        "set_rng_state",
        lambda state: observed.update(torch=state.clone()),
    )
    _restore_rng_states(
        {
            "rng_device_type": "cpu",
            "simulation_generator_state": simulation_generator.get_state()[::2],
            "loader_generator_state": loader_generator.get_state()[::2],
            "torch_rng_state": torch.get_rng_state()[::2],
            "cuda_rng_state_all": [],
        },
        simulation_generator=RecordingGenerator("simulation"),
        loader_generator=RecordingGenerator("loader"),
        device=torch.device("cpu"),
    )
    assert set(observed) == {"simulation", "loader", "torch"}
    assert observed["torch"].device.type == "cpu"
    assert observed["torch"].is_contiguous()


def test_resume_rejects_cpu_cuda_rng_device_type_changes():
    with pytest.raises(ValueError, match="cannot change RNG device type"):
        _restore_rng_states(
            {"rng_device_type": "cuda"},
            simulation_generator=torch.Generator(),
            loader_generator=torch.Generator(),
            device=torch.device("cpu"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_resume_accepts_rng_states_loaded_on_cuda():
    device = torch.device("cuda:0")
    simulation_generator = torch.Generator(device=device).manual_seed(11)
    loader_generator = torch.Generator().manual_seed(12)
    payload = {
        "rng_device_type": "cuda",
        "simulation_generator_state": simulation_generator.get_state().to(device),
        "loader_generator_state": loader_generator.get_state().to(device),
        "torch_rng_state": torch.get_rng_state().to(device),
        "cuda_rng_state_all": [state.to(device) for state in torch.cuda.get_rng_state_all()],
    }
    _restore_rng_states(
        payload,
        simulation_generator=simulation_generator,
        loader_generator=loader_generator,
        device=device,
    )
    assert torch.rand((), generator=simulation_generator, device=device).isfinite()
