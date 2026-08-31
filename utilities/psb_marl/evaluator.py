"""Manual evaluation entrypoint for PSB-MARL stages."""

from __future__ import annotations

import json
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


def _rollout_summary(out_td, seed: int) -> Dict[str, object]:
    action = out_td.get(("agents", "action"))
    reward = out_td.get(("next", "agents", "reward"))
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
) -> Dict[str, object]:
    """Verify the selected PSB stage, then run its manual road protocol."""

    experiment = load_psb_experiment(config_path)
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
    else:
        checkpoint, equivalence, runtime_config = _verify_p1_run(
            selected_run,
            checkpoint_path,
            experiment.base.policy_checkpoint,
            experiment.base.critic_checkpoint,
            experiment.p1_runtime_config(),
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
        )
        rollouts.append(_rollout_summary(out_td, seed))
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

    paired_passed = bool(paired_comparisons) and all(
        item["shape_match"]
        and item["actions_exactly_equal"]
        and item["rewards_exactly_equal"]
        for item in paired_comparisons
    )
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
    else:
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

    report: Dict[str, object] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": experiment.stage,
        "run_directory": str(selected_run),
        "checkpoint": str(checkpoint),
        "comparison_mode": comparison_mode,
        "noninferiority_result": noninferiority_result,
        "promotion_result": promotion_result,
        "equivalence": equivalence,
        "rollouts": rollouts,
        "paired_comparisons": paired_comparisons,
    }
    atomic_write_json(selected_run / report_name, report)
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
    return report
