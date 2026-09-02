"""Paired long-horizon validation for PSB-MARL P3.2, P3.3, and P5."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from utilities.experiment_artifacts import (
    atomic_write_json,
    resolve_latest_testable_run,
    resolve_policy_checkpoint,
    write_artifact_manifest,
)
from utilities.psb_marl.checkpoint import sha256_file
from utilities.psb_marl.config import PSBConfigError
from utilities.psb_marl.p3_dual import continuous_safety_costs


def _load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _verify_p32_run(experiment, run: Path, checkpoint_path: Optional[Path]):
    assert experiment.parent_run is not None
    assert experiment.primal_dual is not None
    is_p5 = experiment.stage == "p5_joint_psb_marl"
    is_p33 = experiment.stage == "p3_paired_differential_primal_dual_ppo"
    is_paired = is_p33 or is_p5
    label = "P5" if is_p5 else ("P3.3" if is_p33 else "P3.2")
    manifest = _load_json(run / "deployment_manifest.json", f"{label} manifest")
    resolved = _load_json(run / "psb_config_resolved.json", f"{label} runtime")
    status = _load_json(run / "training_status.json", f"{label} status")
    expected_runtime = (
        experiment.p5_runtime_config()
        if is_p5
        else (
            experiment.p33_runtime_config()
            if is_p33
            else experiment.p32_runtime_config()
        )
    )
    runtime = resolved.get("runtime_config")
    if runtime != expected_runtime:
        raise PSBConfigError(f"{label} runtime does not match the locked config.")
    if (
        manifest.get("method") != "psb_marl"
        or manifest.get("stage") != experiment.stage
        or manifest.get("selected")
        != (
            "base_fallback_pending_p5_validation"
            if is_p5
            else "base_fallback_pending_p3_3_validation"
            if is_p33
            else "base_fallback_pending_p3_2_validation"
        )
        or manifest.get("actor_learning_enabled") is not True
        or manifest.get("dual_learning_enabled") is not True
        or (
            is_paired
            and manifest.get("paired_differential_learning_enabled") is not True
        )
        or (
            is_paired
            and manifest.get("paired_episode_boundaries_synchronized")
            is not True
        )
        or (
            is_paired
            and manifest.get("paired_episode_boundary_mode")
            != "union_truncate_and_common_seed_reset"
        )
        or status.get("status") != "completed"
    ):
        raise PSBConfigError(f"Selected run is not a completed {label} run.")

    candidate = run / "candidate_policy.pth"
    checkpoint = resolve_policy_checkpoint(
        run, candidate if checkpoint_path is None else checkpoint_path
    )
    if checkpoint.name != "candidate_policy.pth":
        raise PSBConfigError(f"{label} testing accepts only candidate_policy.pth.")
    required = {
        "candidate_critic.pth",
        "source_differential_critic.pth",
        "p3_dual_state.pt",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
        "final_policy.pth",
        "final_critic.pth",
    }
    if is_paired:
        required.add("candidate_differential_critic.pth")
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise PSBConfigError(f"{label} run is missing artifacts: {missing}.")

    candidate_hash = sha256_file(candidate)
    candidate_critic_hash = sha256_file(run / "candidate_critic.pth")
    base_policy_hash = sha256_file(experiment.base.policy_checkpoint)
    base_critic_hash = sha256_file(experiment.base.critic_checkpoint)
    source_differential_hash = sha256_file(
        experiment.parent_run
        / (
            "candidate_differential_critic.pth"
            if is_p5
            else "candidate_critic.pth"
        )
    )
    checks = {
        "candidate_policy_hash_matches_manifest": (
            manifest.get("candidate_policy_sha256") == candidate_hash
        ),
        "candidate_critic_hash_matches_manifest": (
            manifest.get("candidate_critic_sha256") == candidate_critic_hash
        ),
        "source_differential_critic_preserved": (
            sha256_file(run / "source_differential_critic.pth")
            == source_differential_hash
            and manifest.get("source_differential_critic_sha256")
            == source_differential_hash
        ),
        "base_fallback_policy_preserved": (
            sha256_file(run / "base_fallback_policy.pth") == base_policy_hash
        ),
        "base_fallback_critic_preserved": (
            sha256_file(run / "base_fallback_critic.pth") == base_critic_hash
        ),
        "deployed_policy_remains_base": (
            sha256_file(run / "final_policy.pth") == base_policy_hash
        ),
        "deployed_critic_remains_base": (
            sha256_file(run / "final_critic.pth") == base_critic_hash
        ),
        "evaluated_checkpoint_is_candidate": (
            sha256_file(checkpoint) == candidate_hash
        ),
    }
    if is_paired:
        online_hash = sha256_file(run / "candidate_differential_critic.pth")
        online_payload = torch.load(
            run / "candidate_differential_critic.pth", map_location="cpu"
        )
        checks.update(
            {
                "candidate_differential_critic_hash_matches_manifest": (
                    manifest.get("candidate_differential_critic_sha256")
                    == online_hash
                ),
                "paired_differential_learning_enabled": (
                    manifest.get("paired_differential_learning_enabled") is True
                ),
                "paired_episode_boundaries_synchronized": (
                    manifest.get("paired_episode_boundaries_synchronized")
                    is True
                ),
                "paired_episode_boundary_mode_matches": (
                    manifest.get("paired_episode_boundary_mode")
                    == "union_truncate_and_common_seed_reset"
                ),
                "candidate_differential_critic_stage_matches": (
                    isinstance(online_payload, dict)
                    and online_payload.get("stage") == experiment.stage
                ),
                "candidate_differential_critic_runtime_matches": (
                    isinstance(online_payload, dict)
                    and online_payload.get("runtime_config") == runtime
                ),
            }
        )
    if is_p5:
        checks.update(
            {
                "candidate_backbone_trainable": (
                    manifest.get("candidate_backbone_trainable") is True
                ),
                "absolute_critic_learning_enabled": (
                    manifest.get("absolute_critic_learning_enabled") is True
                ),
                "source_base_policy_preserved": (
                    sha256_file(run / "base_fallback_policy.pth")
                    == base_policy_hash
                ),
            }
        )
    dual_state = torch.load(run / "p3_dual_state.pt", map_location="cpu")
    if not isinstance(dual_state, dict):
        raise PSBConfigError(f"{label} dual state must be a dictionary.")
    config = experiment.primal_dual
    expected_dual = {
        "vehicle_budget": config.vehicle_budget,
        "lane_budget": config.lane_budget,
        "vehicle_learning_rate": config.vehicle_learning_rate,
        "lane_learning_rate": config.lane_learning_rate,
        "maximum_multiplier": config.maximum_multiplier,
    }
    dual_checks = {
        f"dual_{name}_matches_config": float(dual_state.get(name, math.nan))
        == float(value)
        for name, value in expected_dual.items()
    }
    for name in ("vehicle_multiplier", "lane_multiplier"):
        value = float(dual_state.get(name, math.nan))
        dual_checks[f"{name}_finite_and_projected"] = (
            math.isfinite(value) and 0.0 <= value <= config.maximum_multiplier
        )
    dual_checks["dual_constraint_normalization_matches_config"] = bool(
        dual_state.get("normalize_constraints", False)
    ) == bool(config.normalize_constraints)
    dual_checks["dual_active_constraints_match_config"] = tuple(
        dual_state.get("active_constraints", ("vehicle", "lane"))
    ) == tuple(config.active_constraints)
    checks.update(dual_checks)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{label} artifact integrity failed: {failed}")
    return checkpoint, runtime, checks, dual_state


def _next_info(rollout, name: str) -> torch.Tensor:
    try:
        return rollout.get(("next", "agents", "info", name))
    except KeyError:
        return rollout.get(("agents", "info", name))


def _safety_summary(rollout, *, lane_safety_margin: float) -> Dict[str, float]:
    costs = continuous_safety_costs(
        urgency=_next_info(rollout, "urgency"),
        confidence=_next_info(rollout, "confidence"),
        pair_mask=_next_info(rollout, "pair_mask").bool(),
        distance_left=_next_info(rollout, "distance_left_b"),
        distance_right=_next_info(rollout, "distance_right_b"),
        vehicle_collision=_next_info(rollout, "is_collision_with_agents"),
        lane_collision=_next_info(rollout, "is_collision_with_lanelets"),
        lane_safety_margin=lane_safety_margin,
    )
    return {
        "vehicle_cost_mean": float(costs.vehicle.float().mean().item()),
        "lane_cost_mean": float(costs.lane.float().mean().item()),
        "vehicle_cost_max": float(costs.vehicle.float().max().item()),
        "lane_cost_max": float(costs.lane.float().max().item()),
    }


def _mean_confidence_upper(values: Sequence[float], z_value: float):
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = float(tensor.mean().item())
    radius = (
        0.0
        if tensor.numel() == 1
        else float(z_value * tensor.std(unbiased=True).item() / tensor.numel() ** 0.5)
    )
    return mean, radius, mean + radius


def p32_safety_gate(
    candidate_rollouts: Sequence[Dict[str, object]],
    base_rollouts: Sequence[Dict[str, object]],
    *,
    vehicle_budget: float,
    lane_budget: float,
    confidence_z: float,
    active_constraints: Sequence[str] = ("vehicle", "lane"),
    budget_applicable: bool = True,
) -> Dict[str, object]:
    """Check absolute CMDP budgets and report paired Base-relative changes."""

    if not candidate_rollouts or len(candidate_rollouts) != len(base_rollouts):
        raise ValueError("P3.2 safety gate requires paired rollout summaries.")
    result: Dict[str, object] = {}
    active = set(active_constraints)
    passed = True
    for channel, budget in (("vehicle", vehicle_budget), ("lane", lane_budget)):
        candidate = [float(item[f"{channel}_cost_mean"]) for item in candidate_rollouts]
        base = [float(item[f"{channel}_cost_mean"]) for item in base_rollouts]
        difference = [left - right for left, right in zip(candidate, base)]
        mean, radius, upper = _mean_confidence_upper(candidate, confidence_z)
        delta_mean, delta_radius, delta_upper = _mean_confidence_upper(
            difference, confidence_z
        )
        dualized = channel in active
        channel_passed = upper <= float(budget)
        if dualized and budget_applicable:
            passed = passed and channel_passed
        result[channel] = {
            "dualized": dualized,
            "candidate_mean": mean,
            "candidate_confidence_radius": radius,
            "candidate_upper_bound": upper,
            "budget": float(budget),
            "budget_passed": bool(channel_passed),
            "difference_candidate_minus_base_mean": delta_mean,
            "difference_confidence_radius": delta_radius,
            "difference_upper_bound": delta_upper,
        }
    return {
        "passed": bool(passed),
        "status": (
            "not_applicable_cross_scenario"
            if not budget_applicable
            else (
                "passed_absolute_safety_budgets"
                if passed
                else "failed_absolute_safety_budgets"
            )
        ),
        "budget_applicable": bool(budget_applicable),
        "confidence_z": float(confidence_z),
        **result,
    }


def p33_efficacy_gate(
    performance_gate: Dict[str, object],
    safety_gate: Dict[str, object],
) -> Dict[str, object]:
    """Require reward superiority or risk superiority at noninferior reward."""

    reward_superiority = float(performance_gate["reward_lower_bound"]) > 0.0
    vehicle_risk_superiority = (
        float(safety_gate["vehicle"]["difference_upper_bound"]) < 0.0
        and performance_gate.get("reward_passed") is True
    )
    return {
        "passed": bool(reward_superiority or vehicle_risk_superiority),
        "criterion": "reward_superiority_or_safe_risk_superiority",
        "reward_superiority": bool(reward_superiority),
        "vehicle_risk_superiority_with_reward_noninferiority": bool(
            vehicle_risk_superiority
        ),
    }


def test_p32(
    experiment,
    *,
    run_directory: Optional[Path],
    checkpoint_path: Optional[Path],
    scenario_type: Optional[str],
    max_steps: int,
    episodes: int,
    seeds: Optional[Sequence[int]],
    render: bool,
    save_simulation_video: bool,
    compare_base: bool,
    promote_if_noninferior: bool,
    psb_action_projection: Optional[str],
    report_label: Optional[str],
) -> Dict[str, object]:
    """Run a quarantined P3.2/P3.3/P5 Candidate against Base with CRN."""

    label = (
        "P5"
        if experiment.stage == "p5_joint_psb_marl"
        else (
            "P3.3"
            if experiment.stage
            == "p3_paired_differential_primal_dual_ppo"
            else "P3.2"
        )
    )

    if not compare_base:
        raise ValueError(f"{label} requires --compare-base.")
    if promote_if_noninferior:
        raise ValueError(
            f"{label} validation does not promote candidates automatically."
        )
    if psb_action_projection is not None:
        raise ValueError(f"{label} action projection is locked by P2.1-U.")
    if report_label is not None and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", report_label
    ) is None:
        raise ValueError(f"{label} report label is unsafe.")
    selected_run = (
        resolve_latest_testable_run(experiment.output_root)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    checkpoint, runtime, artifact_checks, dual_state = _verify_p32_run(
        experiment, selected_run, checkpoint_path
    )
    selected_seeds = tuple(seeds) if seeds is not None else (experiment.seed,)
    if not selected_seeds or any(type(seed) is not int or seed < 0 for seed in selected_seeds):
        raise ValueError(f"{label} seeds must be non-negative integers.")
    if episodes <= 0 or max_steps <= 1:
        raise ValueError(f"{label} episodes and max_steps are invalid.")
    if render and (episodes != 1 or len(selected_seeds) != 1):
        raise ValueError(f"{label} rendering requires one episode and one seed.")
    if save_simulation_video:
        raise ValueError(
            f"{label} paired validation does not support video capture."
        )

    from main_testing import test_base
    from utilities.psb_marl.evaluator import (
        _p2_noninferiority_gate,
        _paired_performance,
        _rollout_summary,
    )

    assert experiment.conflict_graph is not None
    assert experiment.primal_dual is not None
    candidate_rollouts = []
    base_rollouts = []
    comparisons = []
    for seed in selected_seeds:
        candidate_td = test_base(
            experiment.output_root,
            selected_run,
            checkpoint,
            save_simulation_video=False,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seed=seed,
            render=render,
            opinion_pair_info_config=experiment.conflict_graph.to_dict(),
            psb_runtime_config=runtime,
        )
        base_td = test_base(
            str(experiment.base.run_directory),
            experiment.base.run_directory,
            experiment.base.policy_checkpoint,
            save_simulation_video=False,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seed=seed,
            render=False,
            opinion_pair_info_config=experiment.conflict_graph.to_dict(),
        )
        candidate = _rollout_summary(
            candidate_td, seed, p2_runtime_config=runtime
        )
        base = _rollout_summary(base_td, seed)
        candidate.update(
            _safety_summary(
                candidate_td,
                lane_safety_margin=experiment.primal_dual.lane_safety_margin,
            )
        )
        base.update(
            _safety_summary(
                base_td,
                lane_safety_margin=experiment.primal_dual.lane_safety_margin,
            )
        )
        comparison = _paired_performance(candidate, base)
        comparison.update(
            {
                "vehicle_cost_difference_candidate_minus_base": (
                    candidate["vehicle_cost_mean"] - base["vehicle_cost_mean"]
                ),
                "lane_cost_difference_candidate_minus_base": (
                    candidate["lane_cost_mean"] - base["lane_cost_mean"]
                ),
            }
        )
        candidate_rollouts.append(candidate)
        base_rollouts.append(base)
        comparisons.append(comparison)

    promotion = dict(runtime["promotion"])
    performance_gate = _p2_noninferiority_gate(
        comparisons,
        promotion,
        candidate_rollouts=candidate_rollouts,
        proximal=dict(runtime["proximal"]),
        branch_adapter=dict(runtime["branch_adapter"]),
        action_projection=str(runtime["branch_adapter"]["action_projection"]),
    )
    safety_gate = p32_safety_gate(
        candidate_rollouts,
        base_rollouts,
        vehicle_budget=experiment.primal_dual.vehicle_budget,
        lane_budget=experiment.primal_dual.lane_budget,
        confidence_z=float(promotion["confidence_z"]),
        active_constraints=experiment.primal_dual.active_constraints,
        budget_applicable=(
            (experiment.base_run_config["scenario_type"] if scenario_type is None else scenario_type)
            == experiment.base_run_config["scenario_type"]
        ),
    )
    noninferiority_and_budget_passed = bool(
        performance_gate.get("passed") is True
        and safety_gate.get("passed") is True
    )
    efficacy_gate = None
    if label in {"P3.3", "P5"}:
        efficacy_gate = p33_efficacy_gate(performance_gate, safety_gate)
    passed = bool(
        noninferiority_and_budget_passed
        and (efficacy_gate is None or efficacy_gate["passed"] is True)
    )
    report_prefix = (
        "p5" if label == "P5" else ("p3_3" if label == "P3.3" else "p3_2")
    )
    report_name = f"{report_prefix}_manual_validation.json"
    if report_label is not None:
        report_name = (
            f"{report_prefix}_manual_validation_{report_label}.json"
        )
    report: Dict[str, object] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": experiment.stage,
        "run_directory": str(selected_run),
        "checkpoint": str(checkpoint),
        "passed": passed,
        "status": "passed" if passed else "failed",
        "deployment": "base_fallback",
        "promotion_result": "not_requested_candidate_quarantined",
        "evaluation_protocol": {
            "scenario_type": scenario_type,
            "max_steps": int(max_steps),
            "episodes": int(episodes),
            "seeds": [int(seed) for seed in selected_seeds],
            "common_random_numbers": True,
        },
        "artifact_checks": artifact_checks,
        "dual_state": dual_state,
        "candidate_rollouts": candidate_rollouts,
        "base_rollouts": base_rollouts,
        "paired_comparisons": comparisons,
        "performance_noninferiority_gate": performance_gate,
        "safety_budget_gate": safety_gate,
    }
    if efficacy_gate is not None:
        report["efficacy_gate"] = efficacy_gate
    atomic_write_json(selected_run / report_name, report)
    atomic_write_json(
        selected_run / "comparison_to_base.json",
        {
            "schema_version": 1,
            "status": report["status"],
            "deployment": "base_fallback",
            "candidate_checkpoint": "candidate_policy.pth",
            "performance_noninferiority_gate": performance_gate,
            "safety_budget_gate": safety_gate,
            **(
                {"efficacy_gate": efficacy_gate}
                if efficacy_gate is not None
                else {}
            ),
            "report": report_name,
        },
    )
    write_artifact_manifest(selected_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
