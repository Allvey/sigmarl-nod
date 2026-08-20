import json
from pathlib import Path

import main_testing_opinion
import main_training_opinion


def test_training_validate_only_reports_config_without_starting_training(capsys):
    result = main_training_opinion.main(["--validate-only"])

    captured = capsys.readouterr()
    assert result == 0
    assert "[PASS]" in captured.out
    assert "rho_c=0.5" in captured.out
    assert not hasattr(main_training_opinion, "mappo_cavs")


def test_testing_validate_only_reports_config_without_starting_rollout(capsys):
    result = main_testing_opinion.main(["--validate-only"])

    captured = capsys.readouterr()
    assert result == 0
    assert "[PASS]" in captured.out
    assert "rho_c=0.5" in captured.out
    assert not hasattr(main_testing_opinion, "mappo_cavs")


def test_training_entrypoint_runs_validated_trainer(monkeypatch, tmp_path, capsys):
    class FakeEnv:
        def close(self):
            pass

    class FakeTrainer:
        env = FakeEnv()

        def fit(self):
            return tmp_path / "final_opinion.pt"

    monkeypatch.setattr(
        "utilities.opinion.trainer.build_opinion_trainer",
        lambda loaded, smoke, output_dir: FakeTrainer(),
    )
    result = main_training_opinion.main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "[PASS]" in captured.out
    assert "final_opinion.pt" in captured.out


def test_testing_entrypoint_requires_checkpoint(capsys):
    result = main_testing_opinion.main([])

    captured = capsys.readouterr()
    assert result == 1
    assert "[FAIL]" in captured.err
    assert "--checkpoint" in captured.err


def test_testing_entrypoint_runs_checkpoint_evaluation(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "opinion.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        "utilities.opinion.evaluation.evaluate_opinion_checkpoint",
        lambda *args, **kwargs: {"reward_mean": 1.0},
    )

    result = main_testing_opinion.main(["--checkpoint", str(checkpoint), "--smoke"])

    captured = capsys.readouterr()
    assert result == 0
    assert "[PASS]" in captured.out
    assert "reward_mean" in captured.out


def test_entrypoint_rejects_invalid_config_before_any_runtime_construction(
    tmp_path, capsys
):
    raw = json.loads(Path("config_opinion.json").read_text(encoding="utf-8"))
    raw["is_using_opponent_modeling"] = True
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(raw), encoding="utf-8")

    result = main_training_opinion.main(
        ["--config", str(invalid_path), "--validate-only"]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "[FAIL]" in captured.err
    assert "is_using_opponent_modeling" in captured.err
