"""Manual evaluation entrypoint for PSB-MARL stages."""

from __future__ import annotations

import json
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
from utilities.psb_marl.checkpoint import copy_checkpoint_exact, sha256_file
from utilities.psb_marl.config import PSBConfigError, load_psb_experiment


def _load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _verify_p0_run(
    run_directory: Path,
    checkpoint_path: Optional[Path],
    source_policy: Path,
    source_critic: Path,
) -> tuple[Path, Dict[str, object]]:
    checkpoint = resolve_policy_checkpoint(run_directory, checkpoint_path)
    if checkpoint.name != "final_policy.pth":
        raise PSBConfigError("P0 testing accepts only final_policy.pth.")
    manifest = _load_json(
        run_directory / "deployment_manifest.json", "P0 deployment manifest"
    )
    equivalence = _load_json(
        run_directory / "p0_equivalence.json", "P0 equivalence proof"
    )
    if (
        manifest.get("method") != "psb_marl"
        or manifest.get("stage") != "p0_base_passthrough"
        or manifest.get("selected") != "base_passthrough"
    ):
        raise PSBConfigError("Selected run is not a valid PSB P0 passthrough run.")
    packaged_critic = run_directory / "final_critic.pth"
    source_policy_hash = sha256_file(source_policy)
    source_critic_hash = sha256_file(source_critic)
    packaged_policy_hash = sha256_file(checkpoint)
    packaged_critic_hash = sha256_file(packaged_critic)
    checks = {
        "policy_bytes_identical": packaged_policy_hash == source_policy_hash,
        "critic_bytes_identical": packaged_critic_hash == source_critic_hash,
        "manifest_policy_hash_matches": (
            manifest.get("policy_sha256") == packaged_policy_hash
        ),
        "manifest_critic_hash_matches": (
            manifest.get("critic_sha256") == packaged_critic_hash
        ),
        "equivalence_policy_hash_matches": (
            equivalence.get("policy_sha256") == packaged_policy_hash
        ),
        "equivalence_critic_hash_matches": (
            equivalence.get("critic_sha256") == packaged_critic_hash
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P0 checkpoint equivalence failed: {failed}")
    return checkpoint, {
        **checks,
        "source_policy_sha256": source_policy_hash,
        "source_critic_sha256": source_critic_hash,
        "packaged_policy_sha256": packaged_policy_hash,
        "packaged_critic_sha256": packaged_critic_hash,
    }


def _verify_p1_run(
    run_directory: Path,
    checkpoint_path: Optional[Path],
    source_policy: Path,
    source_critic: Path,
    expected_runtime: Dict[str, object],
) -> tuple[Path, Dict[str, object], Dict[str, object]]:
    checkpoint = resolve_policy_checkpoint(run_directory, checkpoint_path)
    if checkpoint.name != "final_policy.pth":
        raise PSBConfigError("P1 testing accepts only final_policy.pth.")
    manifest = _load_json(
        run_directory / "deployment_manifest.json", "P1 deployment manifest"
    )
    equivalence = _load_json(
        run_directory / "p1_equivalence.json", "P1 equivalence proof"
    )
    certification = _load_json(
        run_directory / "p1_certification.json", "P1 certification"
    )
    if (
        manifest.get("method") != "psb_marl"
        or manifest.get("stage") != "p1_zero_control_equivalence"
        or manifest.get("selected") != "p1_zero_control_sidecar"
    ):
        raise PSBConfigError("Selected run is not a valid PSB P1 run.")
    if certification.get("passed") is not True:
        raise PSBConfigError("Selected P1 run did not pass proximal certification.")

    layer_path = run_directory / "final_psb_layer.pth"
    if not layer_path.is_file():
        raise FileNotFoundError(f"P1 layer checkpoint does not exist: {layer_path}")
    layer_payload = torch.load(layer_path, map_location="cpu")
    layer_hash = sha256_file(layer_path)
    runtime_config = layer_payload.get("runtime_config")
    if runtime_config != expected_runtime:
        raise PSBConfigError("P1 layer runtime does not match the current config.")
    if layer_payload.get("trainable_parameters") != 0:
        raise PSBConfigError("P1 layer checkpoint must have zero trainable parameters.")

    packaged_critic = run_directory / "final_critic.pth"
    source_policy_hash = sha256_file(source_policy)
    source_critic_hash = sha256_file(source_critic)
    packaged_policy_hash = sha256_file(checkpoint)
    packaged_critic_hash = sha256_file(packaged_critic)
    checks = {
        "policy_bytes_identical_to_base": packaged_policy_hash == source_policy_hash,
        "critic_bytes_identical_to_base": packaged_critic_hash == source_critic_hash,
        "manifest_policy_hash_matches": (
            manifest.get("policy_sha256") == packaged_policy_hash
        ),
        "manifest_critic_hash_matches": (
            manifest.get("critic_sha256") == packaged_critic_hash
        ),
        "equivalence_policy_hash_matches": (
            equivalence.get("policy_sha256") == packaged_policy_hash
        ),
        "equivalence_critic_hash_matches": (
            equivalence.get("critic_sha256") == packaged_critic_hash
        ),
        "manifest_layer_hash_matches": manifest.get("layer_sha256") == layer_hash,
        "equivalence_layer_hash_matches": (
            equivalence.get("layer_sha256") == layer_hash
        ),
        "zero_control": equivalence.get("b_max") == 0.0,
        "zero_actor_context": equivalence.get("actor_context_gain") == 0.0,
        "untouched_action_path": (
            equivalence.get("action_path") == "untouched_base_actor"
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P1 structural equivalence failed: {failed}")
    return checkpoint, {
        **checks,
        "source_policy_sha256": source_policy_hash,
        "source_critic_sha256": source_critic_hash,
        "packaged_policy_sha256": packaged_policy_hash,
        "packaged_critic_sha256": packaged_critic_hash,
        "packaged_layer_sha256": layer_hash,
    }, runtime_config


def _verify_p2_run(
    run_directory: Path,
    checkpoint_path: Optional[Path],
    source_policy: Path,
    source_critic: Path,
    expected_runtime: Dict[str, object],
) -> tuple[Path, Dict[str, object], Dict[str, object]]:
    manifest = _load_json(
        run_directory / "deployment_manifest.json", "P2 deployment manifest"
    )
    resolved = _load_json(
        run_directory / "psb_config_resolved.json", "P2 resolved config"
    )
    if (
        manifest.get("method") != "psb_marl"
        or manifest.get("stage") != "p2_frozen_base_bifurcation"
    ):
        raise PSBConfigError("Selected run is not a valid PSB P2 run.")
    runtime_config = resolved.get("runtime_config")
    if runtime_config != expected_runtime:
        raise PSBConfigError("P2 runtime does not match the current config.")

    candidate = run_directory / "candidate_policy.pth"
    if checkpoint_path is None:
        checkpoint = resolve_policy_checkpoint(run_directory, candidate)
    else:
        checkpoint = resolve_policy_checkpoint(run_directory, checkpoint_path)
    selected = str(manifest.get("selected"))
    if checkpoint.name == "final_policy.pth":
        if selected != "candidate_promoted":
            raise PSBConfigError(
                "P2 final_policy.pth is still the Base fallback; evaluate "
                "candidate_policy.pth before promotion."
            )
    elif checkpoint.name != "candidate_policy.pth":
        raise PSBConfigError(
            "P2 evaluation accepts candidate_policy.pth, or final_policy.pth "
            "after promotion."
        )

    candidate_critic = run_directory / "candidate_critic.pth"
    fallback_policy = run_directory / "base_fallback_policy.pth"
    fallback_critic = run_directory / "base_fallback_critic.pth"
    final_policy = run_directory / "final_policy.pth"
    final_critic = run_directory / "final_critic.pth"
    for path in (
        candidate,
        candidate_critic,
        fallback_policy,
        fallback_critic,
        final_policy,
        final_critic,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"P2 artifact does not exist: {path}")

    source_policy_hash = sha256_file(source_policy)
    source_critic_hash = sha256_file(source_critic)
    candidate_hash = sha256_file(candidate)
    candidate_critic_hash = sha256_file(candidate_critic)
    fallback_policy_hash = sha256_file(fallback_policy)
    fallback_critic_hash = sha256_file(fallback_critic)
    expected_final_policy_hash = (
        candidate_hash if selected == "candidate_promoted" else fallback_policy_hash
    )
    expected_final_critic_hash = (
        candidate_critic_hash
        if selected == "candidate_promoted"
        else fallback_critic_hash
    )
    checks = {
        "base_fallback_policy_matches_source": (
            fallback_policy_hash == source_policy_hash
        ),
        "base_fallback_critic_matches_source": (
            fallback_critic_hash == source_critic_hash
        ),
        "candidate_policy_hash_matches_manifest": (
            manifest.get("candidate_policy_sha256") == candidate_hash
        ),
        "candidate_critic_hash_matches_manifest": (
            manifest.get("candidate_critic_sha256") == candidate_critic_hash
        ),
        "deployed_policy_matches_selection": (
            sha256_file(final_policy) == expected_final_policy_hash
        ),
        "deployed_critic_matches_selection": (
            sha256_file(final_critic) == expected_final_critic_hash
        ),
        "evaluated_checkpoint_is_candidate": (
            sha256_file(checkpoint) == candidate_hash
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P2 artifact integrity failed: {failed}")
    return checkpoint, {
        **checks,
        "deployment_selection": selected,
        "source_policy_sha256": source_policy_hash,
        "source_critic_sha256": source_critic_hash,
        "candidate_policy_sha256": candidate_hash,
        "candidate_critic_sha256": candidate_critic_hash,
    }, runtime_config


def _rollout_summary(
    out_td,
    seed: int,
    *,
    p2_runtime_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    action = out_td.get(("agents", "action"))
    reward = out_td.get(("next", "agents", "reward"))
    try:
        collision_with_agents = out_td.get(
            ("next", "agents", "info", "is_collision_with_agents")
        ).bool()
        collision_with_lanelets = out_td.get(
            ("next", "agents", "info", "is_collision_with_lanelets")
        ).bool()
    except KeyError:
        collision_shape = reward.shape[:-1]
        collision_with_agents = torch.zeros(
            collision_shape, dtype=torch.bool, device=reward.device
        )
        collision_with_lanelets = torch.zeros_like(collision_with_agents)
    collision = collision_with_agents | collision_with_lanelets
    summary = {
        "seed": int(seed),
        "environment_count": int(action.shape[0]),
        "time_steps": int(action.shape[1]),
        "action_samples": int(action.numel() // action.shape[-1]),
        "nonfinite_action_count": int(
            (~torch.isfinite(action).all(dim=-1)).sum().item()
        ),
        "nonfinite_reward_count": int((~torch.isfinite(reward)).sum().item()),
        "mean_reward_per_agent_step": float(reward.float().mean().item()),
        "collision_with_agents_rate": float(
            collision_with_agents.float().mean().item()
        ),
        "collision_with_lanelets_rate": float(
            collision_with_lanelets.float().mean().item()
        ),
        "total_collision_rate": float(collision.float().mean().item()),
    }
    try:
        z_next = out_td.get(("agents", "psb", "z_next_dense"))
        residual = out_td.get(("agents", "psb", "root_residual"))
        denominator = out_td.get(("agents", "psb", "root_denominator"))
        control = out_td.get(("agents", "psb", "b"))
    except KeyError:
        return summary
    summary.update(
        {
            "max_abs_z": float(z_next.abs().max().item()),
            "max_antisymmetry_error": float(
                (z_next + z_next.transpose(-1, -2)).abs().max().item()
            ),
            "max_root_residual": float(residual.abs().max().item()),
            "min_root_denominator": float(denominator.min().item()),
            "max_abs_b": float(control.abs().max().item()),
            "nonfinite_z_count": int((~torch.isfinite(z_next)).sum().item()),
        }
    )
    if p2_runtime_config is not None:
        from utilities.psb_marl.p2_diagnostics import p2_state_diagnostics

        proximal = p2_runtime_config["proximal"]
        branch_adapter = p2_runtime_config["branch_adapter"]
        summary.update(
            p2_state_diagnostics(
                out_td,
                rho_c=float(proximal["rho_c"]),
                z_scale=float(branch_adapter["z_scale"]),
            )
        )
        delta_loc = out_td.get(("agents", "psb", "delta_loc"))
        branch_activity = out_td.get(
            ("agents", "psb", "branch_activity")
        )
        summary["rollout_delta_loc_abs_mean"] = float(
            delta_loc.abs().mean().item()
        )
        summary["rollout_branch_activity_mean"] = float(
            branch_activity.mean().item()
        )
        summary["rollout_branch_activity_max"] = float(
            branch_activity.max().item()
        )
        if branch_adapter.get("conditioning_mode") in {
            "sector_q_gate",
            "supported_sector_q_gate",
        }:
            sector_bound = (
                float(branch_adapter["max_delta_loc"]) * branch_activity
            )
            summary["rollout_sector_bound_max_violation"] = float(
                (delta_loc.abs() - sector_bound)
                .clamp_min(0.0)
                .max()
                .item()
            )
        if delta_loc.shape[-1] == 2:
            speed_delta = delta_loc[..., 0].abs()
            steering_delta = delta_loc[..., 1].abs()
            summary.update(
                {
                    "rollout_delta_speed_abs_mean": float(
                        speed_delta.mean().item()
                    ),
                    "rollout_delta_speed_abs_p95": float(
                        torch.quantile(speed_delta, 0.95).item()
                    ),
                    "rollout_delta_speed_abs_max": float(
                        speed_delta.max().item()
                    ),
                    "rollout_delta_steering_abs_mean": float(
                        steering_delta.mean().item()
                    ),
                    "rollout_delta_steering_abs_p95": float(
                        torch.quantile(steering_delta, 0.95).item()
                    ),
                    "rollout_delta_steering_abs_max": float(
                        steering_delta.max().item()
                    ),
                }
            )
        delta_log_scale = out_td.get(
            ("agents", "psb", "delta_log_scale")
        )
        summary["rollout_delta_log_scale_abs_mean"] = float(
            delta_log_scale.abs().mean().item()
        )
        summary["rollout_delta_log_scale_abs_max"] = float(
            delta_log_scale.abs().max().item()
        )
        summary["rollout_scale_matches_base_exactly"] = bool(
            torch.equal(
                out_td.get(("agents", "scale")),
                out_td.get(("agents", "psb", "base_scale")),
            )
        )
    return summary


def _paired_equivalence(psb_td, base_td, seed: int) -> Dict[str, object]:
    psb_action = psb_td.get(("agents", "action"))
    base_action = base_td.get(("agents", "action"))
    psb_reward = psb_td.get(("next", "agents", "reward"))
    base_reward = base_td.get(("next", "agents", "reward"))
    if psb_action.shape != base_action.shape or psb_reward.shape != base_reward.shape:
        return {
            "seed": int(seed),
            "shape_match": False,
            "actions_exactly_equal": False,
            "rewards_exactly_equal": False,
            "max_abs_action_difference": float("inf"),
            "max_abs_reward_difference": float("inf"),
        }
    return {
        "seed": int(seed),
        "shape_match": True,
        "actions_exactly_equal": bool(torch.equal(psb_action, base_action)),
        "rewards_exactly_equal": bool(torch.equal(psb_reward, base_reward)),
        "max_abs_action_difference": float(
            (psb_action - base_action).abs().max().item()
        ),
        "max_abs_reward_difference": float(
            (psb_reward - base_reward).abs().max().item()
        ),
    }


def _paired_performance(
    candidate_summary: Dict[str, object],
    base_summary: Dict[str, object],
) -> Dict[str, object]:
    return {
        "seed": int(candidate_summary["seed"]),
        "candidate_mean_reward": float(
            candidate_summary["mean_reward_per_agent_step"]
        ),
        "base_mean_reward": float(base_summary["mean_reward_per_agent_step"]),
        "reward_difference_candidate_minus_base": float(
            candidate_summary["mean_reward_per_agent_step"]
            - base_summary["mean_reward_per_agent_step"]
        ),
        "candidate_total_collision_rate": float(
            candidate_summary["total_collision_rate"]
        ),
        "base_total_collision_rate": float(
            base_summary["total_collision_rate"]
        ),
        "collision_difference_candidate_minus_base": float(
            candidate_summary["total_collision_rate"]
            - base_summary["total_collision_rate"]
        ),
        "candidate_vehicle_collision_rate": float(
            candidate_summary["collision_with_agents_rate"]
        ),
        "base_vehicle_collision_rate": float(
            base_summary["collision_with_agents_rate"]
        ),
        "vehicle_collision_difference_candidate_minus_base": float(
            candidate_summary["collision_with_agents_rate"]
            - base_summary["collision_with_agents_rate"]
        ),
        "candidate_lane_collision_rate": float(
            candidate_summary["collision_with_lanelets_rate"]
        ),
        "base_lane_collision_rate": float(
            base_summary["collision_with_lanelets_rate"]
        ),
        "lane_collision_difference_candidate_minus_base": float(
            candidate_summary["collision_with_lanelets_rate"]
            - base_summary["collision_with_lanelets_rate"]
        ),
    }


def _confidence_bound(values: Sequence[float], z_value: float) -> tuple[float, float]:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = float(tensor.mean().item())
    standard_error = (
        0.0
        if tensor.numel() == 1
        else float(tensor.std(unbiased=True).item() / tensor.numel() ** 0.5)
    )
    return mean, float(z_value * standard_error)


def _p2_noninferiority_gate(
    comparisons: Sequence[Dict[str, object]],
    promotion: Dict[str, object],
    *,
    candidate_rollouts: Optional[Sequence[Dict[str, object]]] = None,
    proximal: Optional[Dict[str, object]] = None,
    branch_adapter: Optional[Dict[str, object]] = None,
    action_projection: str = "full",
) -> Dict[str, object]:
    minimum = int(promotion["minimum_paired_seeds"])
    if len(comparisons) < minimum:
        return {
            "passed": False,
            "status": "insufficient_paired_seeds",
            "paired_seed_count": len(comparisons),
            "minimum_paired_seeds": minimum,
        }
    reward_values = [
        float(item["reward_difference_candidate_minus_base"])
        for item in comparisons
    ]
    collision_values = [
        float(item["collision_difference_candidate_minus_base"])
        for item in comparisons
    ]
    z_value = float(promotion["confidence_z"])
    reward_mean, reward_radius = _confidence_bound(reward_values, z_value)
    collision_mean, collision_radius = _confidence_bound(
        collision_values, z_value
    )
    reward_lower_bound = reward_mean - reward_radius
    collision_upper_bound = collision_mean + collision_radius
    reward_margin = float(promotion["reward_margin"])
    collision_margin = float(promotion["collision_margin"])
    reward_passed = reward_lower_bound >= -reward_margin
    collision_passed = collision_upper_bound <= collision_margin
    lane_collision_margin = promotion.get("lane_collision_margin")
    lane_collision_statistics = {}
    lane_collision_passed = True
    if lane_collision_margin is not None:
        try:
            lane_collision_values = [
                float(item["lane_collision_difference_candidate_minus_base"])
                for item in comparisons
            ]
        except KeyError as error:
            raise ValueError(
                "The lane collision gate requires paired lane collision rates."
            ) from error
        lane_mean, lane_radius = _confidence_bound(
            lane_collision_values, z_value
        )
        lane_upper_bound = lane_mean + lane_radius
        lane_collision_passed = lane_upper_bound <= float(
            lane_collision_margin
        )
        lane_collision_statistics = {
            "lane_collision_difference_mean": lane_mean,
            "lane_collision_confidence_radius": lane_radius,
            "lane_collision_upper_bound": lane_upper_bound,
            "lane_collision_margin": float(lane_collision_margin),
            "lane_collision_passed": bool(lane_collision_passed),
        }
    structural_checks = {}
    if candidate_rollouts is not None and proximal is not None:
        residual_tolerance = float(proximal["residual_tolerance"])
        b_max = float(proximal["b_max"])
        structural_checks = {
            "finite_actions": all(
                int(item["nonfinite_action_count"]) == 0
                for item in candidate_rollouts
            ),
            "finite_rewards": all(
                int(item["nonfinite_reward_count"]) == 0
                for item in candidate_rollouts
            ),
            "finite_bifurcation_state": all(
                int(item.get("nonfinite_z_count", 1)) == 0
                for item in candidate_rollouts
            ),
            "antisymmetric_state": all(
                float(item.get("max_antisymmetry_error", float("inf")))
                <= 1e-6
                for item in candidate_rollouts
            ),
            "proximal_residual": all(
                float(item.get("max_root_residual", float("inf")))
                <= residual_tolerance
                for item in candidate_rollouts
            ),
            "positive_root_denominator": all(
                float(item.get("min_root_denominator", float("-inf"))) > 0.0
                for item in candidate_rollouts
            ),
            "bounded_control": all(
                float(item.get("max_abs_b", float("inf"))) <= b_max + 1e-6
                for item in candidate_rollouts
            ),
        }
        if (
            branch_adapter is not None
            and float(branch_adapter["max_delta_log_scale"]) == 0.0
        ):
            structural_checks["base_scale_exactly_preserved"] = all(
                item.get("rollout_scale_matches_base_exactly") is True
                and float(
                    item.get(
                        "rollout_delta_log_scale_abs_max", float("inf")
                    )
                )
                == 0.0
                for item in candidate_rollouts
            )
        if action_projection == "longitudinal_only":
            structural_checks["base_steering_mean_exactly_preserved"] = all(
                float(
                    item.get(
                        "rollout_delta_steering_abs_max", float("inf")
                    )
                )
                == 0.0
                for item in candidate_rollouts
            )
        if (
            branch_adapter is not None
            and branch_adapter.get("conditioning_mode")
            in {"sector_q_gate", "supported_sector_q_gate"}
        ):
            structural_checks["sector_bound_satisfied"] = all(
                float(
                    item.get(
                        "rollout_sector_bound_max_violation",
                        float("inf"),
                    )
                )
                <= 1e-7
                for item in candidate_rollouts
            )
    structural_passed = all(structural_checks.values())
    passed = (
        reward_passed
        and collision_passed
        and lane_collision_passed
        and structural_passed
    )
    return {
        "passed": bool(passed),
        "status": (
            "passed_paired_confidence_bounds"
            if passed
            else "failed_paired_confidence_bounds"
        ),
        "paired_seed_count": len(comparisons),
        "minimum_paired_seeds": minimum,
        "confidence_z": z_value,
        "reward_difference_mean": reward_mean,
        "reward_confidence_radius": reward_radius,
        "reward_lower_bound": reward_lower_bound,
        "reward_margin": reward_margin,
        "reward_passed": bool(reward_passed),
        "collision_difference_mean": collision_mean,
        "collision_confidence_radius": collision_radius,
        "collision_upper_bound": collision_upper_bound,
        "collision_margin": collision_margin,
        "collision_passed": bool(collision_passed),
        **lane_collision_statistics,
        "structural_checks": structural_checks,
        "structural_passed": bool(structural_passed),
    }


def test_psb(
    config_path: Path,
    *,
    run_directory: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    scenario_type: Optional[str] = None,
    max_steps: int = 1200,
    episodes: int = 1,
    seeds: Optional[Sequence[int]] = None,
    render: bool = True,
    save_simulation_video: bool = False,
    compare_base: bool = False,
    promote_if_noninferior: bool = False,
    psb_action_projection: Optional[str] = None,
    report_label: Optional[str] = None,
) -> Dict[str, object]:
    """Verify the selected PSB stage, then run its manual road protocol."""

    experiment = load_psb_experiment(config_path)
    if experiment.stage in {
        "p3_primal_dual_ppo",
        "p3_paired_differential_primal_dual_ppo",
    }:
        from utilities.psb_marl.p3_dual_evaluation import test_p32

        return test_p32(
            experiment,
            run_directory=run_directory,
            checkpoint_path=checkpoint_path,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seeds=seeds,
            render=render,
            save_simulation_video=save_simulation_video,
            compare_base=compare_base,
            promote_if_noninferior=promote_if_noninferior,
            psb_action_projection=psb_action_projection,
            report_label=report_label,
        )
    if experiment.stage == "p3_differential_critic":
        from utilities.psb_marl.p3_critic_training import test_p31

        return test_p31(
            experiment,
            run_directory=run_directory,
            checkpoint_path=checkpoint_path,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seeds=seeds,
            render=render,
            save_simulation_video=save_simulation_video,
            compare_base=compare_base,
            promote_if_noninferior=promote_if_noninferior,
            psb_action_projection=psb_action_projection,
            report_label=report_label,
        )
    if experiment.stage == "p3_paired_rollout_equivalence":
        from utilities.psb_marl.p3_pairing import test_p3_pairing

        return test_p3_pairing(
            experiment,
            run_directory=run_directory,
            checkpoint_path=checkpoint_path,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seeds=seeds,
            render=render,
            save_simulation_video=save_simulation_video,
            compare_base=compare_base,
            promote_if_noninferior=promote_if_noninferior,
            psb_action_projection=psb_action_projection,
            report_label=report_label,
        )
    if report_label is not None and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", report_label
    ) is None:
        raise ValueError(
            "PSB report label must contain 1-64 letters, digits, underscores, "
            "or hyphens and must start with a letter or digit."
        )
    configured_action_projection = (
        experiment.branch_adapter.action_projection
        if experiment.stage == "p2_frozen_base_bifurcation"
        and experiment.branch_adapter is not None
        else "full"
    )
    action_projection = (
        configured_action_projection
        if psb_action_projection is None
        else psb_action_projection
    )
    inference_only_counterfactual = (
        psb_action_projection is not None
        and action_projection != configured_action_projection
    )
    if action_projection not in {"full", "longitudinal_only"}:
        raise ValueError("Unsupported PSB action projection.")
    if (
        action_projection != "full"
        and experiment.stage != "p2_frozen_base_bifurcation"
    ):
        raise ValueError("PSB action projection is supported only for P2.")
    if inference_only_counterfactual and promote_if_noninferior:
        raise ValueError(
            "Inference-only action projections cannot be promoted. Train a "
            "dedicated projected policy before promotion."
        )
    selected_run = (
        resolve_latest_testable_run(experiment.output_root)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    runtime_config = None
    if experiment.stage == "p0_base_passthrough":
        checkpoint, equivalence = _verify_p0_run(
            selected_run,
            checkpoint_path,
            experiment.base.policy_checkpoint,
            experiment.base.critic_checkpoint,
        )
    elif experiment.stage == "p1_zero_control_equivalence":
        checkpoint, equivalence, runtime_config = _verify_p1_run(
            selected_run,
            checkpoint_path,
            experiment.base.policy_checkpoint,
            experiment.base.critic_checkpoint,
            experiment.p1_runtime_config(),
        )
    else:
        checkpoint, equivalence, runtime_config = _verify_p2_run(
            selected_run,
            checkpoint_path,
            experiment.base.policy_checkpoint,
            experiment.base.critic_checkpoint,
            experiment.p2_runtime_config(),
        )
    selected_seeds = tuple(seeds) if seeds is not None else (experiment.seed,)
    if not selected_seeds:
        raise ValueError("At least one testing seed is required.")
    if any(type(seed) is not int or seed < 0 for seed in selected_seeds):
        raise ValueError("Testing seeds must be non-negative integers.")
    if episodes <= 0 or max_steps <= 1:
        raise ValueError("episodes must be positive and max_steps must exceed 1.")
    if render and (episodes != 1 or len(selected_seeds) != 1):
        raise ValueError("Rendering requires one episode and one seed.")
    if save_simulation_video and (episodes != 1 or len(selected_seeds) != 1):
        raise ValueError("Video capture requires one episode and one seed.")

    # Local import avoids a circular dependency while main_testing.py dispatches
    # PSB configurations to this module.
    from main_testing import test_base

    rollouts = []
    base_rollouts = []
    paired_comparisons = []
    for seed in selected_seeds:
        out_td = test_base(
            experiment.output_root,
            selected_run,
            checkpoint,
            save_simulation_video=save_simulation_video,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seed=seed,
            render=render,
            opinion_pair_info_config=(
                experiment.conflict_graph.to_dict()
                if experiment.conflict_graph is not None
                else None
            ),
            psb_runtime_config=runtime_config,
            psb_action_projection=psb_action_projection,
        )
        candidate_summary = _rollout_summary(
            out_td,
            seed,
            p2_runtime_config=(
                runtime_config
                if experiment.stage == "p2_frozen_base_bifurcation"
                else None
            ),
        )
        rollouts.append(candidate_summary)
        if compare_base and experiment.stage == "p1_zero_control_equivalence":
            base_td = test_base(
                experiment.output_root,
                selected_run,
                checkpoint,
                save_simulation_video=False,
                scenario_type=scenario_type,
                max_steps=max_steps,
                episodes=episodes,
                seed=seed,
                render=False,
            )
            paired_comparisons.append(_paired_equivalence(out_td, base_td, seed))
        elif compare_base and experiment.stage == "p2_frozen_base_bifurcation":
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
            )
            base_summary = _rollout_summary(base_td, seed)
            base_rollouts.append(base_summary)
            paired_comparisons.append(
                _paired_performance(candidate_summary, base_summary)
            )

    paired_passed = bool(paired_comparisons) and all(
        item["shape_match"]
        and item["actions_exactly_equal"]
        and item["rewards_exactly_equal"]
        for item in paired_comparisons
    ) if experiment.stage == "p1_zero_control_equivalence" else False
    noninferiority_gate: Dict[str, object] = {"status": "not_applicable"}
    if experiment.stage == "p0_base_passthrough":
        noninferiority_result = "proven_by_identical_policy_checkpoint"
        comparison_mode = (
            "byte_exact_base_equivalence" if compare_base else "p0_integrity"
        )
        promotion_result = (
            "base_passthrough_already_selected"
            if promote_if_noninferior
            else "not_requested"
        )
        report_name = "p0_manual_validation.json"
    elif experiment.stage == "p1_zero_control_equivalence":
        noninferiority_result = (
            "proven_by_exact_paired_actions"
            if compare_base and paired_passed
            else (
                "failed_exact_paired_actions"
                if compare_base
                else "proven_structurally_by_zero_actor_context"
            )
        )
        comparison_mode = (
            "paired_same_seed_exact_action_equivalence"
            if compare_base
            else "p1_structural_integrity"
        )
        promotion_result = (
            "p1_zero_control_sidecar_selected"
            if promote_if_noninferior and (not compare_base or paired_passed)
            else (
                "rejected"
                if promote_if_noninferior
                else "not_requested"
            )
        )
        report_name = "p1_manual_validation.json"
    else:
        comparison_mode = (
            "paired_same_seed_statistical_noninferiority"
            if compare_base
            else "p2_candidate_integrity"
        )
        noninferiority_gate = (
            _p2_noninferiority_gate(
                paired_comparisons,
                dict(experiment.promotion.to_dict()),
                candidate_rollouts=rollouts,
                proximal=dict(runtime_config["proximal"]),
                branch_adapter=dict(runtime_config["branch_adapter"]),
                action_projection=action_projection,
            )
            if compare_base
            else {"passed": False, "status": "not_evaluated"}
        )
        noninferiority_result = str(noninferiority_gate["status"])
        if inference_only_counterfactual:
            promotion_result = "not_applicable_inference_counterfactual"
            comparison_mode = (
                "paired_same_seed_longitudinal_projection_counterfactual"
                if compare_base and action_projection == "longitudinal_only"
                else "p2_action_projection_counterfactual"
            )
            report_name = "p2_counterfactual_{}_validation.json".format(
                action_projection
            )
        elif promote_if_noninferior:
            manifest = _load_json(
                selected_run / "deployment_manifest.json",
                "P2 deployment manifest",
            )
            if compare_base and noninferiority_gate.get("passed") is True:
                deployed_policy_hash = copy_checkpoint_exact(
                    selected_run / "candidate_policy.pth",
                    selected_run / "final_policy.pth",
                )
                deployed_critic_hash = copy_checkpoint_exact(
                    selected_run / "candidate_critic.pth",
                    selected_run / "final_critic.pth",
                )
                manifest["selected"] = "candidate_promoted"
                promotion_result = "candidate_promoted"
            else:
                deployed_policy_hash = copy_checkpoint_exact(
                    selected_run / "base_fallback_policy.pth",
                    selected_run / "final_policy.pth",
                )
                deployed_critic_hash = copy_checkpoint_exact(
                    selected_run / "base_fallback_critic.pth",
                    selected_run / "final_critic.pth",
                )
                manifest["selected"] = "base_fallback_rejected_candidate"
                promotion_result = "rejected_base_fallback_retained"
            manifest["policy_sha256"] = deployed_policy_hash
            manifest["critic_sha256"] = deployed_critic_hash
            manifest["last_promotion_result"] = promotion_result
            atomic_write_json(
                selected_run / "deployment_manifest.json", manifest
            )
        else:
            promotion_result = "not_requested"
        if not inference_only_counterfactual:
            report_name = "p2_manual_validation.json"
    if report_label is not None:
        report_name = f"{Path(report_name).stem}_{report_label}.json"

    report: Dict[str, object] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": experiment.stage,
        "run_directory": str(selected_run),
        "checkpoint": str(checkpoint),
        "report_label": report_label,
        "training_seed": (
            None
            if runtime_config is None
            else runtime_config.get("training_seed", experiment.seed)
        ),
        "evaluation_protocol": {
            "scenario_type": (
                experiment.base_run_config["scenario_type"]
                if scenario_type is None
                else scenario_type
            ),
            "max_steps": int(max_steps),
            "episodes": int(episodes),
            "seeds": [int(seed) for seed in selected_seeds],
            "compare_base": bool(compare_base),
        },
        "configured_action_projection": configured_action_projection,
        "action_projection": action_projection,
        "inference_only_counterfactual": inference_only_counterfactual,
        "comparison_mode": comparison_mode,
        "noninferiority_result": noninferiority_result,
        "promotion_result": promotion_result,
        "equivalence": equivalence,
        "rollouts": rollouts,
        "base_rollouts": base_rollouts,
        "paired_comparisons": paired_comparisons,
        "noninferiority_gate": noninferiority_gate,
    }
    atomic_write_json(selected_run / report_name, report)
    if experiment.stage == "p2_frozen_base_bifurcation":
        comparison_name = (
            "comparison_to_base_{}.json".format(action_projection)
            if inference_only_counterfactual
            else "comparison_to_base.json"
        )
        if report_label is not None:
            comparison_name = (
                f"{Path(comparison_name).stem}_{report_label}.json"
            )
        atomic_write_json(
            selected_run / comparison_name,
            {
                "schema_version": 1,
                "reference": "recorded_base_source",
                "status": noninferiority_result,
                "deployment": (
                    "candidate"
                    if promotion_result == "candidate_promoted"
                    else "base_fallback"
                ),
                "promotion_result": promotion_result,
                "noninferiority_gate": noninferiority_gate,
                "report": report_name,
            },
        )
    write_artifact_manifest(selected_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if (
        compare_base
        and experiment.stage == "p1_zero_control_equivalence"
        and not paired_passed
    ):
        raise RuntimeError(
            "P1 failed exact paired Base equivalence; see p1_manual_validation.json."
        )
    if (
        promote_if_noninferior
        and experiment.stage == "p2_frozen_base_bifurcation"
        and promotion_result != "candidate_promoted"
    ):
        raise RuntimeError(
            "P2 candidate was not promoted; Base fallback remains deployed. "
            "See p2_manual_validation.json."
        )
    return report
