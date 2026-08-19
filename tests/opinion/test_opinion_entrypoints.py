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


def test_training_entrypoint_returns_distinct_not_implemented_status(capsys):
    result = main_training_opinion.main([])

    captured = capsys.readouterr()
    assert result == 2
    assert "[NOT IMPLEMENTED]" in captured.err
    assert "M2" in captured.err
    assert "training" in captured.err.lower()


def test_testing_entrypoint_returns_distinct_not_implemented_status(capsys):
    result = main_testing_opinion.main([])

    captured = capsys.readouterr()
    assert result == 2
    assert "[NOT IMPLEMENTED]" in captured.err
    assert "M2" in captured.err
    assert "testing" in captured.err.lower()


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
