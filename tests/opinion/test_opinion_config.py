import json
import math
from pathlib import Path

import pytest

from utilities.helper_training import Parameters
from utilities.opinion.config import (
    OPINION_CONFIG_FIELDS,
    OPINION_STAGES,
    OpinionConfig,
    load_opinion_experiment_config,
)


CONFIG_PATH = Path("config_opinion.json")


def _raw_experiment():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_experiment(tmp_path, data):
    path = tmp_path / "opinion.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_recommended_opinion_config_loads_and_computes_critical_urgency():
    loaded = load_opinion_experiment_config(CONFIG_PATH)

    assert loaded.parameters.use_opinion_marl is True
    assert loaded.opinion.stage == "base"
    assert loaded.opinion.rho_c == pytest.approx(0.5)
    assert loaded.opinion.to_dict()["rho_c"] == pytest.approx(0.5)
    assert set(loaded.parameters.opinion_config) == OPINION_CONFIG_FIELDS
    resolved = loaded.to_dict()
    assert resolved["opinion_config"]["rho_c"] == pytest.approx(0.5)
    assert resolved["use_opinion_marl"] is True


def test_opinion_schema_matches_the_committed_configuration_exactly():
    raw = _raw_experiment()["opinion_config"]

    assert set(raw) == OPINION_CONFIG_FIELDS
    assert OPINION_STAGES == ("base", "evidence", "joint")


@pytest.mark.parametrize("stage", OPINION_STAGES)
def test_all_three_training_stages_are_valid(stage):
    raw = _raw_experiment()["opinion_config"]
    raw["stage"] = stage

    config = OpinionConfig.from_dict(raw)

    assert config.stage == stage


@pytest.mark.parametrize("stage", ("", "train", "tsc", 1))
def test_invalid_stage_is_rejected(stage):
    raw = _raw_experiment()["opinion_config"]
    raw["stage"] = stage

    with pytest.raises(ValueError, match="stage"):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("n_candidates", 0),
        ("chunk_length", -1),
        ("chunks_per_minibatch", 1.5),
        ("evidence_hidden_dim", True),
        ("evidence_num_layers", "2"),
        ("n_substeps", 0),
    ),
)
def test_positive_integer_contract(field, invalid):
    raw = _raw_experiment()["opinion_config"]
    raw[field] = invalid

    with pytest.raises(ValueError, match=field):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("include_z_in_critic", 0),
        ("log_pair_diagnostics", "true"),
    ),
)
def test_boolean_fields_require_real_booleans(field, invalid):
    raw = _raw_experiment()["opinion_config"]
    raw[field] = invalid

    with pytest.raises(ValueError, match=field):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("b_max", 0.0),
        ("b_temperature", float("nan")),
        ("kappa", "1.0"),
        ("nu", True),
        ("alpha", float("inf")),
        ("eta", 0.0),
        ("z0", -1.0),
        ("z_clip", 0.0),
        ("lr_actor", 0.0),
        ("lr_evidence", -1.0),
        ("lr_critic", float("nan")),
        ("ttc_horizon", 0.0),
        ("urgency_time_scale", 0.0),
        ("urgency_distance_temperature", 0.0),
    ),
)
def test_strict_positive_finite_numeric_contract(field, invalid):
    raw = _raw_experiment()["opinion_config"]
    raw[field] = invalid

    with pytest.raises(ValueError, match=field):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("residual_scale_start", -0.1),
        ("residual_scale_target", -0.1),
        ("neutral_loss_weight", -0.1),
        ("magnitude_loss_weight", -0.1),
        ("safe_distance", -0.1),
    ),
)
def test_non_negative_numeric_contract(field, invalid):
    raw = _raw_experiment()["opinion_config"]
    raw[field] = invalid

    with pytest.raises(ValueError, match=field):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize("invalid", (-0.1, 1.1, float("nan"), "0.5"))
def test_residual_warmup_fraction_is_a_finite_unit_interval_value(invalid):
    raw = _raw_experiment()["opinion_config"]
    raw["residual_warmup_fraction"] = invalid

    with pytest.raises(ValueError, match="residual_warmup_fraction"):
        OpinionConfig.from_dict(raw)


def test_dynamics_must_have_a_reachable_supercritical_regime():
    raw = _raw_experiment()["opinion_config"]
    raw["nu"] = 0.4
    raw["alpha"] = 2.0
    raw["kappa"] = 1.0

    with pytest.raises(ValueError, match=r"nu.*alpha.*kappa"):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize("field", ("kappa", "nu", "alpha"))
def test_rho_c_is_always_derived_not_accepted_as_input(field):
    raw = _raw_experiment()["opinion_config"]
    raw[field] = raw[field] * 1.1
    config = OpinionConfig.from_dict(raw)

    assert config.rho_c == pytest.approx(
        config.kappa / (config.nu * config.alpha)
    )
    assert math.isfinite(config.rho_c)


def test_missing_and_unknown_opinion_fields_are_rejected():
    raw = _raw_experiment()["opinion_config"]
    del raw["z0"]
    raw["legacy_topology_weight"] = 0.5

    with pytest.raises(ValueError, match=r"missing.*z0.*unknown.*legacy"):
        OpinionConfig.from_dict(raw)


def test_derived_rho_c_cannot_be_supplied_by_json():
    raw = _raw_experiment()["opinion_config"]
    raw["rho_c"] = 0.5

    with pytest.raises(ValueError, match=r"unknown.*rho_c"):
        OpinionConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("use_opinion_marl", False),
        ("is_using_opponent_modeling", True),
        ("is_using_prioritized_marl", True),
        ("use_topology_neighbor_selection", True),
        ("topology_loss_weight", 0.5),
        ("prioritization_method", "soft_label"),
        ("is_append_current_pos_to_short_refs_for_topology", True),
        ("n_topology_nearing_agents_observed", 3),
    ),
)
def test_opinion_experiment_rejects_tsc_or_disabled_opinion_switches(
    tmp_path, field, invalid
):
    raw = _raw_experiment()
    raw[field] = invalid
    path = _write_experiment(tmp_path, raw)

    with pytest.raises(ValueError, match=field):
        load_opinion_experiment_config(path)


def test_opinion_experiment_rejects_unstable_discrete_dynamics(tmp_path):
    raw = _raw_experiment()
    raw["dt"] = 2.0
    raw["opinion_config"]["eta"] = 1.0
    raw["opinion_config"]["kappa"] = 1.0
    path = _write_experiment(tmp_path, raw)

    with pytest.raises(ValueError, match=r"dt.*eta.*kappa.*< 2"):
        load_opinion_experiment_config(path)


def test_candidate_count_must_fit_the_four_agent_experiment(tmp_path):
    raw = _raw_experiment()
    raw["opinion_config"]["n_candidates"] = 4
    path = _write_experiment(tmp_path, raw)

    with pytest.raises(ValueError, match=r"n_candidates.*n_agents"):
        load_opinion_experiment_config(path)


def test_root_config_rejects_missing_and_unknown_fields(tmp_path):
    raw = _raw_experiment()
    del raw["seed"]
    raw["topology_teacher"] = True
    path = _write_experiment(tmp_path, raw)

    with pytest.raises(ValueError, match=r"missing.*seed.*unknown.*topology_teacher"):
        load_opinion_experiment_config(path)


def test_legacy_parameters_keep_opinion_disabled_by_default():
    parameters = Parameters.from_json("config.json")

    assert parameters.use_opinion_marl is False
    assert parameters.opinion_config is None
