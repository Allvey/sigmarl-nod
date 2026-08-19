from pathlib import Path

import main_training_baseline
from utilities.baseline_config import REPO_ROOT, load_baseline_config


def test_main_rejects_non_repo_root_before_training(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(
        main_training_baseline,
        "mappo_cavs",
        lambda **kwargs: called.append(kwargs),
    )

    result = main_training_baseline.main(
        ["--baseline", "base_mappo", "--smoke"]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert called == []
    assert "[FAIL]" in captured.err
    assert "repository root" in captured.err


def test_main_forces_wandb_disabled(tmp_path, monkeypatch):
    resolved = load_baseline_config(
        "base_mappo", smoke=True, output_root=tmp_path, run_id="main-test"
    )
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(
        main_training_baseline,
        "load_baseline_config",
        lambda *args, **kwargs: resolved,
    )
    monkeypatch.setattr(main_training_baseline, "mappo_cavs", lambda **kwargs: None)
    monkeypatch.setattr(
        main_training_baseline,
        "materialize_metrics",
        lambda output_dir: Path(output_dir) / "metrics.json",
    )
    monkeypatch.setattr(
        main_training_baseline,
        "validate_baseline_artifacts",
        lambda *args, **kwargs: {"iterations": 2},
    )

    result = main_training_baseline.main(
        ["--baseline", "base_mappo", "--smoke"]
    )

    assert result == 0
    assert main_training_baseline.os.environ["WANDB_MODE"] == "disabled"
