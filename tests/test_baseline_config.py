import json
from pathlib import Path

import pytest

from utilities.baseline_config import (
    BASELINE_NAMES,
    load_baseline_config,
    validate_baseline_artifacts,
    validate_baseline_config,
    write_resolved_config,
)
from utilities.helper_training import Parameters


METHOD_KEYS = {
    "is_append_current_pos_to_short_refs_for_topology",
    "is_using_opponent_modeling",
    "is_using_prioritized_marl",
    "n_topology_nearing_agents_observed",
    "prioritization_method",
    "use_topology_neighbor_selection",
    "topology_loss_weight",
    "where_to_save",
}


def test_committed_baselines_are_comparable_except_method_and_output_fields():
    base = load_baseline_config("base_mappo")
    tsc = load_baseline_config("tsc")

    assert BASELINE_NAMES == ("base_mappo", "tsc")
    assert {k: v for k, v in base.items() if k not in METHOD_KEYS} == {
        k: v for k, v in tsc.items() if k not in METHOD_KEYS
    }
    assert base["seed"] == tsc["seed"] == 7


def test_base_mappo_disables_every_non_mappo_method_path():
    base = load_baseline_config("base_mappo")

    assert base["is_using_opponent_modeling"] is False
    assert base["is_using_prioritized_marl"] is False
    assert base["use_topology_neighbor_selection"] is False
    assert base["topology_loss_weight"] == 0.0
    assert base["prioritization_method"] == "none"
    assert base["is_append_current_pos_to_short_refs_for_topology"] is False
    assert base["n_nearing_agents_observed"] == 2
    assert base["n_topology_nearing_agents_observed"] == 2
    validate_baseline_config("base_mappo", base)


def test_tsc_baseline_preserves_current_project_method_switches():
    tsc = load_baseline_config("tsc")

    assert tsc["is_using_opponent_modeling"] is True
    assert tsc["is_using_prioritized_marl"] is False
    assert tsc["prioritization_method"] == "soft_label"
    assert tsc["use_topology_neighbor_selection"] is True
    assert tsc["topology_loss_weight"] == 0.5
    assert tsc["is_append_current_pos_to_short_refs_for_topology"] is True
    assert tsc["n_nearing_agents_observed"] == 2
    assert tsc["n_topology_nearing_agents_observed"] == 3
    validate_baseline_config("tsc", tsc)


@pytest.mark.parametrize(
    ("name", "field", "invalid"),
    (
        ("base_mappo", "is_append_current_pos_to_short_refs_for_topology", True),
        ("base_mappo", "n_topology_nearing_agents_observed", 3),
        ("tsc", "is_append_current_pos_to_short_refs_for_topology", False),
        ("tsc", "n_topology_nearing_agents_observed", 2),
    ),
)
def test_validator_rejects_topology_input_contract_drift(name, field, invalid):
    config = load_baseline_config(name, run_id="topology-input-contract")
    config[field] = invalid

    with pytest.raises(ValueError, match=field):
        validate_baseline_config(name, config)


def test_original_training_config_still_loads_as_tsc_path():
    parameters = Parameters.from_json("config.json")

    assert parameters.is_using_opponent_modeling is True
    assert parameters.is_using_prioritized_marl is False
    assert parameters.prioritization_method == "soft_label"
    assert parameters.use_topology_neighbor_selection is True
    assert parameters.topology_loss_weight == 0.5


def test_smoke_overrides_resolved_copy_without_mutating_committed_config():
    committed_path = Path("configs/baselines/base_mappo.json")
    before = committed_path.read_text(encoding="utf-8")
    resolved = load_baseline_config("base_mappo", smoke=True)

    assert resolved["n_iters"] == 2
    assert resolved["max_steps"] == 8
    assert resolved["frames_per_batch"] == 16
    assert resolved["num_epochs"] == 1
    assert resolved["minibatch_size"] == 8
    assert resolved["device"] == "cpu"
    assert resolved["is_load_model"] is False
    assert resolved["is_load_final_model"] is False
    assert resolved["is_continue_train"] is False
    assert resolved["is_save_intermediate_model"] is True
    assert resolved["is_testing_mode"] is False
    assert committed_path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_resolved_output_path_is_canonical_and_snapshot_is_written(tmp_path, name):
    config = load_baseline_config(name, output_root=tmp_path, run_id="test-run")

    assert config["where_to_save"].endswith("/")
    assert Path(config["where_to_save"]) == tmp_path / name / "runs" / "test-run"
    snapshot = write_resolved_config(config)
    assert snapshot == Path(config["where_to_save"]) / "resolved_config.json"
    assert json.loads(snapshot.read_text(encoding="utf-8")) == config


def test_validator_rejects_residual_tsc_switch_in_base_config():
    base = load_baseline_config("base_mappo")
    base["topology_loss_weight"] = 0.1

    with pytest.raises(ValueError, match="topology_loss_weight"):
        validate_baseline_config("base_mappo", base)


def test_resolved_runs_receive_unique_output_directories(tmp_path):
    first = load_baseline_config("base_mappo", smoke=True, output_root=tmp_path)
    second = load_baseline_config("base_mappo", smoke=True, output_root=tmp_path)

    assert first["where_to_save"] != second["where_to_save"]
    assert Path(first["where_to_save"]).parent.name == "runs"
    assert Path(second["where_to_save"]).parent.name == "runs"


@pytest.mark.parametrize("run_id", (".", "..", ".hidden", "-option", "../escape"))
def test_resolver_rejects_unsafe_run_id_path_segments(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        load_baseline_config("base_mappo", output_root=tmp_path, run_id=run_id)


def test_resolver_accepts_safe_single_segment_run_id(tmp_path):
    config = load_baseline_config(
        "base_mappo", output_root=tmp_path, run_id="Smoke_01.valid-id"
    )

    assert Path(config["where_to_save"]).name == "Smoke_01.valid-id"


def test_snapshot_writer_rejects_reusing_existing_run_leaf(tmp_path):
    first = load_baseline_config(
        "base_mappo", output_root=tmp_path, run_id="duplicate-run"
    )
    second = load_baseline_config(
        "base_mappo", output_root=tmp_path, run_id="duplicate-run"
    )
    write_resolved_config(first)

    with pytest.raises(RuntimeError, match=r"run directory already exists.*duplicate-run"):
        write_resolved_config(second)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("seed", "7"),
        ("n_iters", 2.0),
        ("frames_per_batch", True),
        ("is_load_model", 0),
        ("is_save_intermediate_model", 1),
    ),
)
def test_validator_rejects_non_exact_config_types(field, invalid):
    config = load_baseline_config("base_mappo", run_id="type-test")
    config[field] = invalid

    with pytest.raises(ValueError, match=field):
        validate_baseline_config("base_mappo", config)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("device", "cuda", "CPU"),
        ("is_load_model", True, "from scratch"),
        ("is_load_final_model", True, "from scratch"),
        ("is_continue_train", True, "from scratch"),
        ("is_save_intermediate_model", False, "intermediate"),
    ),
)
def test_validator_rejects_execution_contract_drift(field, invalid, message):
    config = load_baseline_config("base_mappo", run_id="contract-test")
    config[field] = invalid

    with pytest.raises(ValueError, match=message):
        validate_baseline_config("base_mappo", config)


def test_validator_rejects_unknown_parameter_key():
    config = load_baseline_config("base_mappo", run_id="unknown-key")
    config["minibach_size"] = 512

    with pytest.raises(ValueError, match=r"unknown.*minibach_size"):
        validate_baseline_config("base_mappo", config)


def test_validator_rejects_string_boolean_for_any_committed_switch():
    config = load_baseline_config("base_mappo", run_id="bool-schema")
    config["is_prb"] = "false"

    with pytest.raises(ValueError, match=r"is_prb.*bool"):
        validate_baseline_config("base_mappo", config)


@pytest.mark.parametrize("invalid", ("0.0002", float("nan")))
def test_validator_rejects_invalid_learning_rate_numeric(invalid):
    config = load_baseline_config("base_mappo", run_id="numeric-schema")
    config["lr"] = invalid

    with pytest.raises(ValueError, match=r"lr.*numeric|lr.*finite"):
        validate_baseline_config("base_mappo", config)


@pytest.mark.parametrize(
    "missing_field",
    (
        "device",
        "is_load_model",
        "is_load_final_model",
        "is_continue_train",
        "is_save_intermediate_model",
        "is_testing_mode",
    ),
)
def test_validator_reports_missing_execution_contract_fields(missing_field):
    config = load_baseline_config("base_mappo", run_id="missing-schema")
    del config[missing_field]

    with pytest.raises(ValueError, match=rf"missing fields.*{missing_field}"):
        validate_baseline_config("base_mappo", config)


@pytest.mark.parametrize(
    "invalid",
    (
        [1.0, 0.0],
        [0.5, 0.5, float("nan")],
        [1.1, -0.1, 0.0],
        [0.5, 0.25, 0.0],
        [True, 0.0, 0.0],
    ),
)
def test_validator_rejects_invalid_scenario_probability_vector(invalid):
    config = load_baseline_config("base_mappo", run_id="probability-schema")
    config["cpm_scenario_probabilities"] = invalid

    with pytest.raises(ValueError, match="cpm_scenario_probabilities"):
        validate_baseline_config("base_mappo", config)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("scenario_name", "other_scenario"),
        ("scenario_type", "intersection_1"),
        ("n_agents", 5),
    ),
)
def test_validator_rejects_frozen_baseline_identity_drift(field, invalid):
    config = load_baseline_config("base_mappo", run_id="identity-schema")
    config[field] = invalid

    with pytest.raises(ValueError, match=field):
        validate_baseline_config("base_mappo", config)


def _write_artifacts(config, *, tsc=False, metric_length=None):
    output_dir = Path(config["where_to_save"])
    write_resolved_config(config)
    for filename in ("final_policy.pth", "final_critic.pth"):
        (output_dir / filename).write_text("placeholder", encoding="utf-8")
    if tsc:
        for filename in ("final_topology.pth", "final_action_predictor.pth"):
            (output_dir / filename).write_text("placeholder", encoding="utf-8")
    length = config["n_iters"] if metric_length is None else metric_length
    metrics = {
        "episode_reward_mean_list": [-1.0] * length,
        "collision_agents_rate_list": [0.0] * length,
        "collision_lanelets_rate_list": [0.0] * length,
        "collision_total_rate_list": [0.0] * length,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return output_dir


def test_artifact_validator_enforces_base_and_tsc_checkpoint_contract(tmp_path):
    base = load_baseline_config(
        "base_mappo", smoke=True, output_root=tmp_path, run_id="base-run"
    )
    base_dir = _write_artifacts(base)

    summary = validate_baseline_artifacts("base_mappo", base_dir, base)
    assert summary["iterations"] == 2

    (base_dir / "final_topology.pth").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not produce"):
        validate_baseline_artifacts("base_mappo", base_dir, base)

    tsc = load_baseline_config(
        "tsc", smoke=True, output_root=tmp_path, run_id="tsc-run"
    )
    tsc_dir = _write_artifacts(tsc, tsc=True)
    summary = validate_baseline_artifacts("tsc", tsc_dir, tsc)
    assert summary["iterations"] == 2


def test_tsc_artifact_validator_rejects_priority_checkpoint(tmp_path):
    tsc = load_baseline_config(
        "tsc", smoke=True, output_root=tmp_path, run_id="priority-run"
    )
    output_dir = _write_artifacts(tsc, tsc=True)
    (output_dir / "final_priority_policy.pth").write_text(
        "placeholder", encoding="utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match=r"tsc.*must not produce.*final_priority_policy\.pth",
    ):
        validate_baseline_artifacts("tsc", output_dir, tsc)


def test_parent_checkpoint_cannot_satisfy_current_tsc_run(tmp_path):
    tsc = load_baseline_config(
        "tsc", smoke=True, output_root=tmp_path, run_id="isolated-run"
    )
    output_dir = _write_artifacts(tsc, tsc=False)
    stable_root = output_dir.parents[1]
    (stable_root / "final_topology.pth").write_text("old", encoding="utf-8")
    (stable_root / "final_action_predictor.pth").write_text(
        "old", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="final_topology.pth"):
        validate_baseline_artifacts("tsc", output_dir, tsc)


def test_artifact_validator_requires_exact_iteration_count(tmp_path):
    base = load_baseline_config(
        "base_mappo", output_root=tmp_path, run_id="full-run"
    )
    output_dir = _write_artifacts(base, metric_length=2)

    with pytest.raises(RuntimeError, match=r"exactly 250.*got 2"):
        validate_baseline_artifacts("base_mappo", output_dir, base)


def test_artifact_validator_rejects_snapshot_mismatch(tmp_path):
    base = load_baseline_config(
        "base_mappo", smoke=True, output_root=tmp_path, run_id="snapshot-run"
    )
    output_dir = _write_artifacts(base)
    mismatched = dict(base)
    mismatched["seed"] = 8
    (output_dir / "resolved_config.json").write_text(
        json.dumps(mismatched), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="resolved_config.json.*does not match"):
        validate_baseline_artifacts("base_mappo", output_dir, base)


def test_artifact_validator_rejects_boolean_reward_metric(tmp_path):
    base = load_baseline_config(
        "base_mappo", smoke=True, output_root=tmp_path, run_id="bool-metric"
    )
    output_dir = _write_artifacts(base)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["episode_reward_mean_list"][0] = True
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"episode_reward_mean_list.*numeric"):
        validate_baseline_artifacts("base_mappo", output_dir, base)


@pytest.mark.parametrize(
    "metric_key",
    (
        "collision_agents_rate_list",
        "collision_lanelets_rate_list",
        "collision_total_rate_list",
    ),
)
def test_artifact_validator_rejects_out_of_range_collision_rate(
    tmp_path, metric_key
):
    base = load_baseline_config(
        "base_mappo",
        smoke=True,
        output_root=tmp_path,
        run_id="collision-metric",
    )
    output_dir = _write_artifacts(base)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics[metric_key][0] = 1.1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(RuntimeError, match=rf"{metric_key}.*\[0, 1\]"):
        validate_baseline_artifacts("base_mappo", output_dir, base)
