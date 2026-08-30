"""Paired, multi-environment-seed A5/A6 checkpoint comparison."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torchrl.envs.utils import step_mdp

from utilities.avocado.config import (
    AVOCADOConfigError,
    _exact_keys,
    _integer,
    _number,
    _object,
    _string,
)
from utilities.avocado.road_config import A3RoadExperimentConfig, RoadCaseConfig
from utilities.avocado_marl.a5_benchmark import run_a5_rollout
from utilities.avocado_marl.a5_config import A5ExperimentConfig
from utilities.avocado_marl.a6_config import A6ExperimentConfig
from utilities.avocado_marl.a6_trainer import (
    A6OneStepTrainer,
    _reset_mask,
    resolve_latest_a6_checkpoint,
)
from utilities.avocado_marl.bridge import A4BridgeTrace
from utilities.avocado_marl.config import A4ExperimentConfig
from utilities.constants import SCENARIOS
from utilities.experiment_artifacts import atomic_write_json


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


# A positive paired delta is desirable for "higher" metrics and undesirable
# for "lower" metrics. Metrics absent here are diagnostic rather than gates.
METRIC_DIRECTIONS = {
    "mean_reward_per_agent_step": "higher",
    "agent_collision_events_per_1000_steps": "lower",
    "lane_collision_events_per_1000_steps": "lower",
    "wrong_entry_events_per_1000_steps": "lower",
    "route_completion_events_per_1000_steps": "higher",
    "mean_reference_distance_meters": "lower",
    "p95_reference_distance_meters": "lower",
    "mean_measured_speed_mps": "higher",
    "shield_intervention_rate": "lower",
    "p95_absolute_speed_change_mps": "lower",
    "p95_absolute_steering_change_degrees": "lower",
    "conflict_p95_absolute_steering_change_degrees": "lower",
    "conflict_steering_reversal_rate": "lower",
    "conflict_stopped_action_rate": "lower",
}


@dataclass(frozen=True)
class A6ComparisonEvaluationConfig:
    scenarios: Tuple[str, ...]
    environment_seeds: Tuple[int, ...]
    parallel_environments: int
    max_steps: int
    stop_speed_threshold_mps: float
    steering_reversal_threshold_degrees: float
    ttc_bins_seconds: Tuple[float, ...]

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> "A6ComparisonEvaluationConfig":
        raw = _object(raw, "evaluation")
        _exact_keys(raw, set(cls.__dataclass_fields__), "evaluation")
        scenarios_raw = raw["scenarios"]
        seeds_raw = raw["environment_seeds"]
        bins_raw = raw["ttc_bins_seconds"]
        if not isinstance(scenarios_raw, list) or not scenarios_raw:
            raise AVOCADOConfigError("evaluation.scenarios must be a non-empty list.")
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise AVOCADOConfigError(
                "evaluation.environment_seeds must be a non-empty list."
            )
        if not isinstance(bins_raw, list) or len(bins_raw) < 2:
            raise AVOCADOConfigError(
                "evaluation.ttc_bins_seconds must contain at least two edges."
            )
        result = cls(
            scenarios=tuple(
                _string(value, "evaluation.scenarios[]") for value in scenarios_raw
            ),
            environment_seeds=tuple(
                _integer(value, "evaluation.environment_seeds[]", minimum=0)
                for value in seeds_raw
            ),
            parallel_environments=_integer(
                raw["parallel_environments"],
                "evaluation.parallel_environments",
            ),
            max_steps=_integer(raw["max_steps"], "evaluation.max_steps"),
            stop_speed_threshold_mps=_number(
                raw["stop_speed_threshold_mps"],
                "evaluation.stop_speed_threshold_mps",
            ),
            steering_reversal_threshold_degrees=_number(
                raw["steering_reversal_threshold_degrees"],
                "evaluation.steering_reversal_threshold_degrees",
            ),
            ttc_bins_seconds=tuple(
                _number(value, "evaluation.ttc_bins_seconds[]")
                for value in bins_raw
            ),
        )
        unknown = sorted(set(result.scenarios) - set(SCENARIOS))
        if unknown:
            raise AVOCADOConfigError(f"Unknown comparison scenarios: {unknown}.")
        if len(set(result.scenarios)) != len(result.scenarios):
            raise AVOCADOConfigError("evaluation.scenarios must not contain duplicates.")
        if len(set(result.environment_seeds)) != len(result.environment_seeds):
            raise AVOCADOConfigError(
                "evaluation.environment_seeds must not contain duplicates."
            )
        if result.stop_speed_threshold_mps < 0.0:
            raise AVOCADOConfigError(
                "evaluation.stop_speed_threshold_mps must be non-negative."
            )
        if result.steering_reversal_threshold_degrees < 0.0:
            raise AVOCADOConfigError(
                "evaluation.steering_reversal_threshold_degrees must be non-negative."
            )
        if result.ttc_bins_seconds[0] != 0.0 or any(
            right <= left
            for left, right in zip(
                result.ttc_bins_seconds, result.ttc_bins_seconds[1:]
            )
        ):
            raise AVOCADOConfigError(
                "evaluation.ttc_bins_seconds must start at 0 and increase strictly."
            )
        return result


@dataclass(frozen=True)
class A6ComparisonConfig:
    a6_config: Path
    output_root: str
    evaluation: A6ComparisonEvaluationConfig

    @classmethod
    def from_json(cls, path: Path) -> "A6ComparisonConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "a6_config",
            "output_root",
            "evaluation",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != 1:
            raise AVOCADOConfigError("A6 comparison schema_version must be 1.")
        if raw["method"] != "avocado_marl" or raw["stage"] != "a6_comparison":
            raise AVOCADOConfigError(
                "Expected method='avocado_marl' and stage='a6_comparison'."
            )
        a6_config = Path(_string(raw["a6_config"], "a6_config"))
        if not a6_config.is_file():
            raise AVOCADOConfigError(f"A6 config does not exist: {a6_config}")
        return cls(
            a6_config=a6_config,
            output_root=_string(raw["output_root"], "output_root"),
            evaluation=A6ComparisonEvaluationConfig.from_dict(raw["evaluation"]),
        )


@dataclass(frozen=True)
class _RolloutArtifacts:
    metrics: Dict[str, float]
    trace: A4BridgeTrace
    correction_features: Tensor
    checkpoint_iteration: int
    base_actor_source_hash: str


def _quantile(values: Tensor, probability: float) -> float:
    return float(torch.quantile(values.float(), probability)) if values.numel() else 0.0


def _absolute_summary(values: Tensor, prefix: str) -> Dict[str, float]:
    values = values.abs().float()
    return {
        f"mean_absolute_{prefix}": float(values.mean()) if values.numel() else 0.0,
        f"p95_absolute_{prefix}": _quantile(values, 0.95),
        f"maximum_absolute_{prefix}": float(values.max()) if values.numel() else 0.0,
    }


def _trace_metrics(
    trace: A4BridgeTrace,
    correction_features: Tensor,
    *,
    maximum_correction: float,
    velocity_obstacle_horizon: float,
    stop_speed_threshold_mps: float,
    steering_reversal_threshold_degrees: float,
    ttc_bins_seconds: Sequence[float],
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    actions = trace.executed_action.float()
    pair_mask = trace.pair_mask.bool()
    correction_pair_mask = correction_features[..., -1].bool()
    reset_mask = trace.reset_mask.bool()
    conflict = trace.conflict_mask.bool()
    correction = trace.estimate_correction[correction_pair_mask]
    heuristic = trace.heuristic_estimate[pair_mask]
    fused = trace.fused_estimate[pair_mask]
    opinion = trace.opinion[pair_mask]
    fusion_difference = (trace.fused_estimate - trace.heuristic_estimate)[pair_mask]

    metrics: Dict[str, float] = {
        "perceived_pair_samples": float(pair_mask.sum()),
        "correction_pair_samples": float(correction_pair_mask.sum()),
        "conflict_action_samples": float(conflict.sum()),
        "correction_saturation_rate": (
            float((correction.abs() >= 0.99 * maximum_correction).float().mean())
            if correction.numel()
            else 0.0
        ),
        "fusion_saturation_rate": (
            float((fused.abs() >= 0.99).float().mean()) if fused.numel() else 0.0
        ),
        "invalid_pair_maximum_absolute_correction": (
            float(trace.estimate_correction[~correction_pair_mask].abs().max())
            if bool((~correction_pair_mask).any())
            else 0.0
        ),
    }
    metrics.update(_absolute_summary(correction, "correction"))
    metrics.update(_absolute_summary(heuristic, "heuristic_estimate"))
    metrics.update(_absolute_summary(fused, "fused_estimate"))
    metrics.update(_absolute_summary(opinion, "opinion"))
    metrics.update(_absolute_summary(fusion_difference, "fusion_difference"))

    if actions.shape[0] > 1:
        valid_change = ~reset_mask[:-1]
        conflict_window = (conflict[:-1] | conflict[1:]) & valid_change
        speed_change = (actions[1:, ..., 0] - actions[:-1, ..., 0]).abs()
        steering_change = torch.rad2deg(
            (actions[1:, ..., 1] - actions[:-1, ..., 1]).abs()
        )
        valid_speed_change = speed_change[valid_change]
        valid_steering_change = steering_change[valid_change]
        conflict_steering_change = steering_change[conflict_window]
        threshold = math.radians(steering_reversal_threshold_degrees)
        reversal = (
            (actions[1:, ..., 1] * actions[:-1, ..., 1] < 0)
            & (actions[1:, ..., 1].abs() > threshold)
            & (actions[:-1, ..., 1].abs() > threshold)
            & conflict_window
        )
        metrics.update(
            {
                "mean_absolute_speed_change_mps": (
                    float(valid_speed_change.mean())
                    if valid_speed_change.numel()
                    else 0.0
                ),
                "p95_absolute_speed_change_mps": _quantile(
                    valid_speed_change, 0.95
                ),
                "mean_absolute_steering_change_degrees": (
                    float(valid_steering_change.mean())
                    if valid_steering_change.numel()
                    else 0.0
                ),
                "p95_absolute_steering_change_degrees": _quantile(
                    valid_steering_change, 0.95
                ),
                "conflict_mean_absolute_steering_change_degrees": (
                    float(conflict_steering_change.mean())
                    if conflict_steering_change.numel()
                    else 0.0
                ),
                "conflict_p95_absolute_steering_change_degrees": _quantile(
                    conflict_steering_change, 0.95
                ),
                "conflict_steering_reversal_rate": (
                    float(reversal.sum()) / int(conflict_window.sum())
                    if bool(conflict_window.any())
                    else 0.0
                ),
            }
        )
        previous_pair_reset = reset_mask[:-1, :, :, None] | reset_mask[
            :-1, :, None, :
        ]
        valid_pair_change = (
            correction_pair_mask[:-1]
            & correction_pair_mask[1:]
            & ~previous_pair_reset
        )
        correction_reversal = (
            trace.estimate_correction[:-1] * trace.estimate_correction[1:] < 0
        ) & valid_pair_change
        metrics["correction_sign_switch_rate"] = (
            float(correction_reversal.sum()) / int(valid_pair_change.sum())
            if bool(valid_pair_change.any())
            else 0.0
        )
    else:
        metrics.update(
            {
                "mean_absolute_speed_change_mps": 0.0,
                "p95_absolute_speed_change_mps": 0.0,
                "mean_absolute_steering_change_degrees": 0.0,
                "p95_absolute_steering_change_degrees": 0.0,
                "conflict_mean_absolute_steering_change_degrees": 0.0,
                "conflict_p95_absolute_steering_change_degrees": 0.0,
                "conflict_steering_reversal_rate": 0.0,
                "correction_sign_switch_rate": 0.0,
            }
        )
    conflict_actions = actions[..., 0].abs()[conflict]
    metrics["conflict_stopped_action_rate"] = (
        float((conflict_actions <= stop_speed_threshold_mps).float().mean())
        if conflict_actions.numel()
        else 0.0
    )

    ttc_seconds = (
        correction_features[..., 8].float() * velocity_obstacle_horizon
    )
    ttc_bins: Dict[str, Dict[str, float]] = {}
    intervals = tuple(zip(ttc_bins_seconds, ttc_bins_seconds[1:]))
    for index, (lower, upper) in enumerate(intervals):
        upper_selected = (
            ttc_seconds <= upper
            if index == len(intervals) - 1
            else ttc_seconds < upper
        )
        selected = correction_pair_mask & (ttc_seconds >= lower) & upper_selected
        values = trace.estimate_correction[selected]
        closing = "]" if index == len(intervals) - 1 else ")"
        ttc_bins[f"[{lower:g},{upper:g}{closing}"] = {
            "pair_samples": int(selected.sum()),
            "mean_absolute_correction": (
                float(values.abs().mean()) if values.numel() else 0.0
            ),
            "p95_absolute_correction": _quantile(values.abs(), 0.95),
        }
    return metrics, ttc_bins


def _a5_rollout(
    config: A6ExperimentConfig,
    *,
    scenario_type: str,
    seed: int,
    parallel_environments: int,
    max_steps: int,
) -> _RolloutArtifacts:
    a5_config = A5ExperimentConfig.from_json(config.a5_config)
    n_agents = int(SCENARIOS[scenario_type]["n_agents"])
    case = RoadCaseConfig(
        name=f"{scenario_type}_{n_agents}",
        scenario_type=scenario_type,
        n_agents=n_agents,
    )
    result = run_a5_rollout(
        a5_config,
        case,
        run_directory=config.base_policy.run_directory,
        checkpoint=config.base_policy.policy_checkpoint,
        episodes_override=parallel_environments,
        max_steps_override=max_steps,
        seed_override=seed,
    )
    base = result.metrics
    metrics = {
        "mean_reward_per_agent_step": base.mean_reward_per_agent_step,
        "agent_collision_events_per_1000_steps": base.agent_collision_events_per_1000_steps,
        "lane_collision_events_per_1000_steps": base.lane_collision_events_per_1000_steps,
        "wrong_entry_events_per_1000_steps": base.wrong_entry_events_per_1000_steps,
        "route_completion_events_per_1000_steps": base.route_completion_events_per_1000_steps,
        "mean_reference_distance_meters": base.mean_reference_distance_meters,
        "p95_reference_distance_meters": base.p95_reference_distance_meters,
        "mean_measured_speed_mps": base.mean_measured_speed_mps,
        "shield_intervention_rate": base.bridge.shield_intervention_rate,
        "action_intervention_rate": base.bridge.action_intervention_rate,
        "mean_nominal_executed_speed_difference_mps": base.bridge.mean_absolute_speed_change_mps,
        "mean_nominal_executed_steering_difference_degrees": base.bridge.mean_absolute_steering_change_degrees,
    }
    return _RolloutArtifacts(
        metrics=metrics,
        trace=result.trace,
        correction_features=result.correction_features,
        checkpoint_iteration=0,
        base_actor_source_hash="",
    )


@torch.no_grad()
def _a6_rollout(
    config: A6ExperimentConfig,
    checkpoint: Path,
    *,
    scenario_type: str,
    seed: int,
    parallel_environments: int,
    max_steps: int,
) -> _RolloutArtifacts:
    trainer = A6OneStepTrainer(
        config,
        environment_steps_override=max_steps,
        parallel_environments_override=parallel_environments,
        scenario_type_override=scenario_type,
        seed_override=seed,
        load_critic=False,
    )
    try:
        checkpoint_iteration = trainer.load_checkpoint(checkpoint)
        trainer.rollout.reset_all()
        tensordict = trainer.env.reset()
        reward_sum = 0.0
        agent_collisions = 0
        lane_collisions = 0
        wrong_entries = 0
        route_completions = 0
        reference_values = []
        measured_speed_sum = 0.0
        action_samples = 0
        correction_features = []
        for step in range(max_steps):
            rollout_step = trainer.rollout.step(tensordict, deterministic=True)
            correction_features.append(rollout_step.features.cpu())
            step_tensordict = trainer.env.step(rollout_step.tensordict)
            reward_sum += float(
                step_tensordict.get(("next", "agents", "reward")).sum()
            )
            scenario = trainer.scenario
            reset_mask = _reset_mask(scenario)
            agent_collisions += int(scenario.a3_last_agent_collisions.sum())
            lane_collisions += int(scenario.a3_last_lane_collisions.sum())
            wrong_entries += int(scenario.a3_last_wrong_entries.sum())
            route_completions += int(scenario.a3_last_route_completions.sum())
            reference_values.append(
                scenario.a3_last_reference_distances.detach().cpu().reshape(-1)
            )
            measured = torch.stack(
                [agent.state.vel for agent in scenario.world.agents], dim=1
            )
            measured_speed_sum += float(
                torch.linalg.vector_norm(measured, dim=-1).sum()
            )
            action_samples += parallel_environments * scenario.n_agents
            trainer.rollout.reset_agents(reset_mask)
            if step + 1 < max_steps:
                tensordict = step_mdp(
                    step_tensordict,
                    keep_other=True,
                    exclude_action=False,
                    exclude_reward=True,
                    reward_keys=trainer.env.reward_keys,
                    action_keys=trainer.env.action_keys,
                    done_keys=trainer.env.done_keys,
                )
        bridge = trainer.rollout.execution_bridge
        bridge_metrics = bridge.metrics()
        all_reference = torch.cat(reference_values)
        event_scale = 1000.0 / max(action_samples, 1)
        metrics = {
            "mean_reward_per_agent_step": reward_sum / max(action_samples, 1),
            "agent_collision_events_per_1000_steps": agent_collisions * event_scale,
            "lane_collision_events_per_1000_steps": lane_collisions * event_scale,
            "wrong_entry_events_per_1000_steps": wrong_entries * event_scale,
            "route_completion_events_per_1000_steps": route_completions * event_scale,
            "mean_reference_distance_meters": float(all_reference.mean()),
            "p95_reference_distance_meters": _quantile(all_reference, 0.95),
            "mean_measured_speed_mps": measured_speed_sum / max(action_samples, 1),
            "shield_intervention_rate": bridge_metrics.shield_intervention_rate,
            "action_intervention_rate": bridge_metrics.action_intervention_rate,
            "mean_nominal_executed_speed_difference_mps": bridge_metrics.mean_absolute_speed_change_mps,
            "mean_nominal_executed_steering_difference_degrees": bridge_metrics.mean_absolute_steering_change_degrees,
        }
        return _RolloutArtifacts(
            metrics=metrics,
            trace=bridge.trace(),
            correction_features=torch.stack(correction_features),
            checkpoint_iteration=checkpoint_iteration,
            base_actor_source_hash=trainer.base_actor_source_hash,
        )
    finally:
        trainer.close()


def _confidence_summary(values: Sequence[float]) -> Dict[str, float]:
    count = len(values)
    average = mean(values)
    if count < 2:
        return {
            "count": count,
            "mean": average,
            "standard_deviation": 0.0,
            "ci95_low": average,
            "ci95_high": average,
        }
    deviation = stdev(values)
    critical = _T_CRITICAL_95.get(count - 1, 1.96)
    margin = critical * deviation / math.sqrt(count)
    return {
        "count": count,
        "mean": average,
        "standard_deviation": deviation,
        "ci95_low": average - margin,
        "ci95_high": average + margin,
    }


def _paired_comparisons(
    records: Sequence[Mapping[str, Any]], scenarios: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for scenario in scenarios:
        by_stage_seed = {
            (record["stage"], record["seed"]): record["metrics"]
            for record in records
            if record["scenario_type"] == scenario
        }
        seeds = sorted(
            seed
            for stage, seed in by_stage_seed
            if stage == "a5" and ("a6", seed) in by_stage_seed
        )
        metric_names = sorted(
            set.intersection(
                *(set(by_stage_seed[(stage, seed)]) for seed in seeds for stage in ("a5", "a6"))
            )
        )
        comparisons = {}
        for metric in metric_names:
            deltas = [
                float(by_stage_seed[("a6", seed)][metric])
                - float(by_stage_seed[("a5", seed)][metric])
                for seed in seeds
            ]
            summary = _confidence_summary(deltas)
            direction = METRIC_DIRECTIONS.get(metric, "diagnostic")
            if len(deltas) < 2 and direction != "diagnostic":
                conclusion = "insufficient_seeds"
            elif direction == "higher":
                conclusion = (
                    "favors_a6"
                    if summary["ci95_low"] > 0.0
                    else "favors_a5"
                    if summary["ci95_high"] < 0.0
                    else "inconclusive"
                )
            elif direction == "lower":
                conclusion = (
                    "favors_a6"
                    if summary["ci95_high"] < 0.0
                    else "favors_a5"
                    if summary["ci95_low"] > 0.0
                    else "inconclusive"
                )
            else:
                conclusion = "diagnostic"
            comparisons[metric] = {
                "direction": direction,
                "delta_definition": "A6 - A5",
                **summary,
                "conclusion": conclusion,
            }
        result[scenario] = {"paired_seeds": seeds, "metrics": comparisons}
    return result


def _stage_aggregates(
    records: Sequence[Mapping[str, Any]], scenarios: Sequence[str]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    result = {}
    for scenario in scenarios:
        result[scenario] = {}
        for stage in ("a5", "a6"):
            selected = [
                record["metrics"]
                for record in records
                if record["scenario_type"] == scenario and record["stage"] == stage
            ]
            names = sorted(set.intersection(*(set(item) for item in selected)))
            result[scenario][stage] = {
                name: _confidence_summary([float(item[name]) for item in selected])
                for name in names
            }
    return result


def _paired_ttc_comparisons(
    records: Sequence[Mapping[str, Any]], scenarios: Sequence[str]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    result = {}
    for scenario in scenarios:
        selected = [
            record
            for record in records
            if record["scenario_type"] == scenario
        ]
        by_stage_seed = {
            (record["stage"], record["seed"]): record["ttc_bins"]
            for record in selected
        }
        seeds = sorted(
            seed
            for stage, seed in by_stage_seed
            if stage == "a5" and ("a6", seed) in by_stage_seed
        )
        bins = sorted(
            set.intersection(
                *(
                    set(by_stage_seed[(stage, seed)])
                    for seed in seeds
                    for stage in ("a5", "a6")
                )
            )
        )
        result[scenario] = {}
        for bin_name in bins:
            result[scenario][bin_name] = {}
            for metric in (
                "pair_samples",
                "mean_absolute_correction",
                "p95_absolute_correction",
            ):
                a5_values = [
                    float(by_stage_seed[("a5", seed)][bin_name][metric])
                    for seed in seeds
                ]
                a6_values = [
                    float(by_stage_seed[("a6", seed)][bin_name][metric])
                    for seed in seeds
                ]
                deltas = [a6 - a5 for a5, a6 in zip(a5_values, a6_values)]
                result[scenario][bin_name][metric] = {
                    "a5": _confidence_summary(a5_values),
                    "a6": _confidence_summary(a6_values),
                    "paired_delta_a6_minus_a5": _confidence_summary(deltas),
                }
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# A5/A6 paired comparison",
        "",
        "Delta is A6 - A5. A conclusion requires the paired 95% confidence interval to exclude zero.",
        "",
    ]
    headline = tuple(METRIC_DIRECTIONS)
    for scenario, comparison in summary["paired_comparisons"].items():
        stage_aggregates = summary["stage_aggregates"][scenario]
        lines.extend(
            (
                f"## {scenario}",
                "",
                "| Metric | Direction | A5 mean | A6 mean | Paired delta | 95% CI | Conclusion |",
                "|---|---:|---:|---:|---:|---:|---|",
            )
        )
        for name in headline:
            item = comparison["metrics"][name]
            lines.append(
                f"| {name} | {item['direction']} | "
                f"{stage_aggregates['a5'][name]['mean']:.6g} | "
                f"{stage_aggregates['a6'][name]['mean']:.6g} | "
                f"{item['mean']:.6g} | "
                f"[{item['ci95_low']:.6g}, {item['ci95_high']:.6g}] | "
                f"{item['conclusion']} |"
            )
        lines.extend(
            (
                "",
                "### TTC-binned correction diagnostics",
                "",
                "| TTC bin (s) | A5 mean | A6 mean | Paired delta | 95% CI |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for bin_name, bin_metrics in summary["paired_ttc_comparisons"][
            scenario
        ].items():
            item = bin_metrics["mean_absolute_correction"]
            delta = item["paired_delta_a6_minus_a5"]
            lines.append(
                f"| {bin_name} | {item['a5']['mean']:.6g} | "
                f"{item['a6']['mean']:.6g} | {delta['mean']:.6g} | "
                f"[{delta['ci95_low']:.6g}, {delta['ci95_high']:.6g}] |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_a5_a6_comparison(
    config_path: Path,
    *,
    checkpoint: Optional[Path] = None,
    output_directory: Optional[Path] = None,
    scenarios_override: Optional[Sequence[str]] = None,
    seeds_override: Optional[Sequence[int]] = None,
    parallel_environments_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> Path:
    comparison = A6ComparisonConfig.from_json(config_path)
    config = A6ExperimentConfig.from_json(comparison.a6_config)
    a5_config = A5ExperimentConfig.from_json(config.a5_config)
    a4_config = A4ExperimentConfig.from_json(a5_config.a4_config)
    a3_config = A3RoadExperimentConfig.from_json(a4_config.a3_config)
    evaluation = comparison.evaluation
    scenarios = tuple(scenarios_override or evaluation.scenarios)
    seeds = tuple(seeds_override or evaluation.environment_seeds)
    parallel_environments = (
        evaluation.parallel_environments
        if parallel_environments_override is None
        else int(parallel_environments_override)
    )
    max_steps = (
        evaluation.max_steps if max_steps_override is None else int(max_steps_override)
    )
    if not scenarios or any(value not in SCENARIOS for value in scenarios):
        raise ValueError("Comparison scenarios must be non-empty and valid.")
    if not seeds or any(type(value) is not int or value < 0 for value in seeds):
        raise ValueError("Comparison seeds must be non-negative integers.")
    if parallel_environments <= 0 or max_steps <= 0:
        raise ValueError("Comparison environments and max_steps must be positive.")
    selected_checkpoint = Path(
        checkpoint or resolve_latest_a6_checkpoint(config.output_root)
    ).expanduser().resolve()
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"A6 checkpoint does not exist: {selected_checkpoint}")
    if output_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path(comparison.output_root) / timestamp
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    records = []
    for scenario_type in scenarios:
        for seed in seeds:
            for stage, runner in (("a5", _a5_rollout), ("a6", _a6_rollout)):
                print(
                    f"[A6 comparison] stage={stage} scenario={scenario_type} "
                    f"seed={seed} envs={parallel_environments} steps={max_steps}",
                    flush=True,
                )
                if stage == "a5":
                    artifacts = runner(
                        config,
                        scenario_type=scenario_type,
                        seed=seed,
                        parallel_environments=parallel_environments,
                        max_steps=max_steps,
                    )
                else:
                    artifacts = runner(
                        config,
                        selected_checkpoint,
                        scenario_type=scenario_type,
                        seed=seed,
                        parallel_environments=parallel_environments,
                        max_steps=max_steps,
                    )
                trace_metrics, ttc_bins = _trace_metrics(
                    artifacts.trace,
                    artifacts.correction_features,
                    maximum_correction=config.y_correction.maximum_correction,
                    velocity_obstacle_horizon=(
                        a3_config.parameters.velocity_obstacle_horizon
                    ),
                    stop_speed_threshold_mps=evaluation.stop_speed_threshold_mps,
                    steering_reversal_threshold_degrees=(
                        evaluation.steering_reversal_threshold_degrees
                    ),
                    ttc_bins_seconds=evaluation.ttc_bins_seconds,
                )
                metrics = {**artifacts.metrics, **trace_metrics}
                record = {
                    "stage": stage,
                    "scenario_type": scenario_type,
                    "seed": seed,
                    "parallel_environments": parallel_environments,
                    "max_steps": max_steps,
                    "checkpoint_iteration": artifacts.checkpoint_iteration,
                    "base_actor_source_hash": artifacts.base_actor_source_hash,
                    "metrics": metrics,
                    "ttc_bins": ttc_bins,
                }
                records.append(record)
                atomic_write_json(
                    output_directory / f"{stage}_{scenario_type}_seed{seed}.json",
                    record,
                )
    summary = {
        "schema_version": 1,
        "method": "avocado_marl",
        "stage": "a6_comparison",
        "a6_config": str(comparison.a6_config.resolve()),
        "a6_checkpoint": str(selected_checkpoint),
        "comparison_contract": {
            "baseline": "A5 frozen zero YCorrectionNet",
            "candidate": "A6 learned bounded YCorrectionNet",
            "paired_seed": True,
            "delta_definition": "A6 - A5",
            "confidence_interval": "two-sided paired Student t, 95%",
            "parallel_environments": parallel_environments,
            "max_steps": max_steps,
            "scenarios": list(scenarios),
            "environment_seeds": list(seeds),
            "base_policy_checkpoint": str(
                config.base_policy.policy_checkpoint.resolve()
            ),
            "base_policy_checkpoint_sha256": _file_sha256(
                config.base_policy.policy_checkpoint
            ),
        },
        "records": records,
        "stage_aggregates": _stage_aggregates(records, scenarios),
        "paired_comparisons": _paired_comparisons(records, scenarios),
        "paired_ttc_comparisons": _paired_ttc_comparisons(records, scenarios),
    }
    atomic_write_json(output_directory / "summary.json", summary)
    (output_directory / "report.md").write_text(
        _markdown_report(summary), encoding="utf-8"
    )
    return output_directory
