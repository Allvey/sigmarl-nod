import pytest
import torch
from tensordict import TensorDict

from scripts.check_runtime_environment import (
    assert_finite_tensors,
    check_road_traffic_rollout,
    check_tanh_normal,
    main,
    runtime_versions,
)


def test_assert_finite_tensors_accepts_nested_finite_values():
    tensordict = TensorDict(
        {
            "agents": TensorDict(
                {"observation": torch.tensor([[0.0, 1.0]])},
                batch_size=[1],
            )
        },
        batch_size=[1],
    )

    assert_finite_tensors(tensordict, context="reset")


def test_assert_finite_tensors_identifies_non_finite_nested_key():
    tensordict = TensorDict(
        {
            "agents": TensorDict(
                {"observation": torch.tensor([[0.0, float("nan")]])},
                batch_size=[1],
            )
        },
        batch_size=[1],
    )

    with pytest.raises(
        RuntimeError,
        match=r"rollout.*agents.*observation",
    ):
        assert_finite_tensors(tensordict, context="rollout")


def test_check_tanh_normal_returns_finite_sample_and_log_prob():
    result = check_tanh_normal()

    assert set(result) == {"sample", "log_prob"}
    assert torch.isfinite(result["sample"]).all()
    assert torch.isfinite(result["log_prob"]).all()


def test_runtime_versions_rejects_version_drift(tmp_path):
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(
        "\n".join(
            (
                "Torch==0.0.0",
                "torchrl==0.2.1",
                "tensordict==0.2.1",
                "vmas==1.4.1",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=r"torch.*expected=0\.0\.0.*actual=2\.1\.0",
    ):
        runtime_versions(requirements_path=requirements_path)


def test_runtime_versions_rejects_missing_required_pin(tmp_path):
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(
        "\n".join(
            (
                "torch==2.1.0",
                "torchrl==0.2.1",
                "tensordict==0.2.1",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"vmas.*exact.*pin"):
        runtime_versions(requirements_path=requirements_path)


def test_road_traffic_rollout_completes_two_real_steps():
    assert check_road_traffic_rollout(steps=2) == 2


def test_main_returns_one_and_reports_invalid_rollout_steps(capsys):
    return_code = main(["--steps", "1"])

    captured = capsys.readouterr()
    assert return_code == 1
    assert "[FAIL]" in captured.err
    assert "rollout steps must be between 2 and 10" in captured.err


def test_main_converts_import_error_to_clear_failure(monkeypatch, capsys):
    def fail_runtime_versions():
        raise ImportError("torch import is broken")

    monkeypatch.setattr(
        "scripts.check_runtime_environment.runtime_versions",
        fail_runtime_versions,
    )

    return_code = main(["--steps", "2"])

    captured = capsys.readouterr()
    assert return_code == 1
    assert "[FAIL]" in captured.err
    assert "torch import is broken" in captured.err
