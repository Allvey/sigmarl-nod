"""Configuration and artifact checks for reproducible training baselines."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import re
import secrets
from typing import Any, Dict, Mapping, Optional

from utilities.helper_training import Parameters


BASELINE_NAMES = ("base_mappo", "tsc")
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "baselines"
SMOKE_OVERRIDES = {
    "n_iters": 2,
    "max_steps": 8,
    "frames_per_batch": 16,
    "num_epochs": 1,
    "minibatch_size": 8,
    "device": "cpu",
    "is_load_model": False,
    "is_load_final_model": False,
    "is_continue_train": False,
    "is_save_intermediate_model": True,
    "is_testing_mode": False,
}
METRIC_KEYS = (
    "episode_reward_mean_list",
    "collision_agents_rate_list",
    "collision_lanelets_rate_list",
    "collision_total_rate_list",
)
FORBIDDEN_BASE_ARTIFACT_PARTS = ("topology", "action_predictor", "priority")
POSITIVE_INTEGER_FIELDS = (
    "seed",
    "n_agents",
    "n_iters",
    "frames_per_batch",
    "num_epochs",
    "minibatch_size",
    "max_steps",
    "n_steps_stored",
    "n_points_short_term",
    "n_nearing_agents_observed",
    "n_topology_nearing_agents_observed",
)
BOOLEAN_FIELDS = (
    "is_add_noise",
    "is_append_current_pos_to_short_refs_for_topology",
    "is_apply_mask",
    "is_challenging_initial_state_buffer",
    "is_ego_view",
    "is_load_model",
    "is_load_final_model",
    "is_continue_train",
    "is_observe_distance_to_agents",
    "is_observe_distance_to_boundaries",
    "is_observe_distance_to_center_line",
    "is_observe_ref_path_other_agents",
    "is_observe_vertices",
    "is_partial_observation",
    "is_prb",
    "is_save_eval_results",
    "is_save_intermediate_model",
    "is_testing_mode",
    "is_use_mtv_distance",
    "is_using_opponent_modeling",
    "is_using_prioritized_marl",
    "is_visualize_short_term_path",
    "use_topology_neighbor_selection",
)
STRING_FIELDS = (
    "scenario_name",
    "device",
    "scenario_type",
    "model_name",
    "where_to_save",
    "prioritization_method",
)
NUMERIC_FIELDS = (
    "dt",
    "lr",
    "lr_action_predictor",
    "lr_min",
    "max_grad_norm",
    "clip_epsilon",
    "gamma",
    "lmbda",
    "entropy_eps",
    "topology_loss_weight",
    "episode_reward_mean_current",
    "episode_reward_intermediate",
    "topology_selection_threshold",
)
BASELINE_CONFIG_FIELDS = frozenset(
    POSITIVE_INTEGER_FIELDS
    + BOOLEAN_FIELDS
    + STRING_FIELDS
    + NUMERIC_FIELDS
    + ("cpm_scenario_probabilities",)
)
FROZEN_COMMON_VALUES = {
    "scenario_name": "road_traffic",
    "scenario_type": "CPM_mixed",
    "n_agents": 4,
    "device": "cpu",
    "seed": 7,
    "n_nearing_agents_observed": 2,
}


def _canonical_output_dir(path: Path) -> str:
    return str(path.resolve()) + "/"


def _new_run_id(smoke: bool) -> str:
    mode = "smoke" if smoke else "full"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{mode}-{timestamp}-{secrets.token_hex(4)}"


def validate_baseline_config(name: str, config: Mapping[str, Any]) -> None:
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")

    missing = BASELINE_CONFIG_FIELDS.difference(config)
    if missing:
        raise ValueError(f"{name} config is missing fields: {sorted(missing)}")

    valid_parameter_keys = set(inspect.signature(Parameters.__init__).parameters) - {
        "self"
    }
    unknown = set(config).difference(BASELINE_CONFIG_FIELDS)
    if unknown:
        invalid_parameters = unknown.difference(valid_parameter_keys)
        detail = "parameter" if invalid_parameters else "baseline schema"
        raise ValueError(f"{name} config has unknown {detail} keys: {sorted(unknown)}")

    for field in POSITIVE_INTEGER_FIELDS:
        if type(config[field]) is not int:
            raise ValueError(f"{name}.{field} must have type int")
        if config[field] <= 0:
            raise ValueError(f"{name}.{field} must be positive")
    for field in BOOLEAN_FIELDS:
        if type(config[field]) is not bool:
            raise ValueError(f"{name}.{field} must have type bool")
    for field in STRING_FIELDS:
        if type(config[field]) is not str:
            raise ValueError(f"{name}.{field} must have type str")
        if field != "model_name" and not config[field]:
            raise ValueError(f"{name}.{field} must not be empty")
    for field in NUMERIC_FIELDS:
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{field} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{name}.{field} must be finite")

    for field in ("dt", "lr", "lr_action_predictor", "lr_min", "max_grad_norm"):
        if config[field] <= 0:
            raise ValueError(f"{name}.{field} must be positive")
    if config["lr_min"] > config["lr"]:
        raise ValueError(f"{name}.lr_min must not exceed lr")
    if not 0 < config["clip_epsilon"] <= 1:
        raise ValueError(f"{name}.clip_epsilon must be in (0, 1]")
    for field in ("gamma", "lmbda", "topology_selection_threshold"):
        if not 0 <= config[field] <= 1:
            raise ValueError(f"{name}.{field} must be in [0, 1]")
    for field in ("entropy_eps", "topology_loss_weight"):
        if config[field] < 0:
            raise ValueError(f"{name}.{field} must be non-negative")

    probabilities = config["cpm_scenario_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != 3:
        raise ValueError(
            f"{name}.cpm_scenario_probabilities must be a list of length 3"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in probabilities
    ):
        raise ValueError(
            f"{name}.cpm_scenario_probabilities must contain finite values in [0, 1]"
        )
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name}.cpm_scenario_probabilities must sum to 1")

    for field, expected_value in FROZEN_COMMON_VALUES.items():
        if config[field] != expected_value:
            expected_label = "CPU" if field == "device" else repr(expected_value)
            raise ValueError(
                f"{name}.{field} must remain {expected_label}, got {config[field]!r}"
            )
    if config["frames_per_batch"] % config["max_steps"] != 0:
        raise ValueError("frames_per_batch must be divisible by max_steps")
    if config["frames_per_batch"] % config["minibatch_size"] != 0:
        raise ValueError("frames_per_batch must be divisible by minibatch_size")
    if any(
        config[field]
        for field in ("is_load_model", "is_load_final_model", "is_continue_train")
    ):
        raise ValueError(f"{name} baseline must train from scratch")
    if not config["is_save_intermediate_model"]:
        raise ValueError(f"{name} baseline must save intermediate artifacts")
    if config["is_testing_mode"]:
        raise ValueError(f"{name} baseline training must disable testing mode")

    if name == "base_mappo":
        expected = {
            "is_append_current_pos_to_short_refs_for_topology": False,
            "is_using_opponent_modeling": False,
            "is_using_prioritized_marl": False,
            "prioritization_method": "none",
            "use_topology_neighbor_selection": False,
            "topology_loss_weight": 0.0,
            "n_topology_nearing_agents_observed": 2,
        }
    else:
        expected = {
            "is_append_current_pos_to_short_refs_for_topology": True,
            "is_using_opponent_modeling": True,
            "is_using_prioritized_marl": False,
            "prioritization_method": "soft_label",
            "use_topology_neighbor_selection": True,
            "topology_loss_weight": 0.5,
            "n_topology_nearing_agents_observed": 3,
        }
    for field, expected_value in expected.items():
        if config[field] != expected_value:
            raise ValueError(
                f"{name}.{field} must be {expected_value!r}, got {config[field]!r}"
            )


def load_baseline_config(
    name: str,
    *,
    smoke: bool = False,
    output_root: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
    path = CONFIG_ROOT / f"{name}.json"
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    resolved = copy.deepcopy(config)
    if smoke:
        resolved.update(SMOKE_OVERRIDES)
    if run_id is None:
        run_id = _new_run_id(smoke)
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id
    ):
        raise ValueError(
            "run_id must be one safe path segment beginning with a letter or digit"
        )
    if output_root is None:
        baseline_root = REPO_ROOT / "outputs" / "baselines" / name
    else:
        baseline_root = Path(output_root) / name
    runs_root = baseline_root / "runs"
    output_dir = runs_root / run_id
    if output_dir.resolve().parent != runs_root.resolve():
        raise ValueError("run_id resolved outside the baseline runs directory")
    resolved["where_to_save"] = _canonical_output_dir(output_dir)
    validate_baseline_config(name, resolved)
    return resolved


def write_resolved_config(config: Mapping[str, Any]) -> Path:
    output_dir = Path(str(config["where_to_save"]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(f"run directory already exists: {output_dir}") from error
    path = output_dir / "resolved_config.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(config), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def materialize_metrics(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    candidates = sorted(
        output_dir.glob("*_data.json"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise RuntimeError(f"no training data JSON found under {output_dir}")
    with candidates[-1].open("r", encoding="utf-8") as stream:
        training_data = json.load(stream)
    metrics = {key: training_data.get(key) for key in METRIC_KEYS}
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def validate_baseline_artifacts(
    name: str,
    output_dir: Path,
    resolved_config: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    expected_output_dir = Path(str(resolved_config["where_to_save"]))
    if output_dir.resolve() != expected_output_dir.resolve():
        raise RuntimeError(
            f"artifact directory {output_dir} does not match resolved where_to_save"
        )
    for filename in (
        "final_policy.pth",
        "final_critic.pth",
        "resolved_config.json",
        "metrics.json",
    ):
        if not (output_dir / filename).is_file():
            raise RuntimeError(f"missing baseline artifact: {output_dir / filename}")

    with (output_dir / "resolved_config.json").open("r", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    if snapshot != dict(resolved_config):
        raise RuntimeError("resolved_config.json does not match this run's configuration")

    checkpoint_names = [path.name.lower() for path in output_dir.glob("*.pth")]
    if name == "base_mappo":
        unexpected = [
            filename
            for filename in checkpoint_names
            if any(part in filename for part in FORBIDDEN_BASE_ARTIFACT_PARTS)
        ]
        if unexpected:
            raise RuntimeError(
                f"base_mappo must not produce TSC/priority checkpoints: {unexpected}"
            )
    elif name == "tsc":
        for filename in ("final_topology.pth", "final_action_predictor.pth"):
            if filename not in checkpoint_names:
                raise RuntimeError(f"tsc is missing checkpoint: {filename}")
        unexpected = [
            filename for filename in checkpoint_names if "priority" in filename
        ]
        if unexpected:
            raise RuntimeError(
                f"tsc with priority=false must not produce priority checkpoints: {unexpected}"
            )
    else:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")

    with (output_dir / "metrics.json").open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    expected_iterations = resolved_config["n_iters"]
    lengths = []
    for key in METRIC_KEYS:
        values = metrics.get(key)
        if not isinstance(values, list) or len(values) != expected_iterations:
            actual_length = len(values) if isinstance(values, list) else "non-list"
            raise RuntimeError(
                f"{key} must contain exactly {expected_iterations} iterations; "
                f"got {actual_length}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise RuntimeError(f"{key} must contain finite numeric values")
        if key != "episode_reward_mean_list" and any(
            not 0 <= value <= 1 for value in values
        ):
            raise RuntimeError(f"{key} must contain rates in [0, 1]")
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise RuntimeError(f"metric lengths differ: {dict(zip(METRIC_KEYS, lengths))}")
    return {
        "iterations": lengths[0],
        "reward": metrics["episode_reward_mean_list"],
        "collision": metrics["collision_total_rate_list"],
    }
