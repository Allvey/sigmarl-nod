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


def _rollout_summary(out_td, seed: int) -> Dict[str, object]:
    action = out_td.get(("agents", "action"))
    reward = out_td.get(("next", "agents", "reward"))
    return {
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
    """Verify P0 identity, then execute the packaged Base policy manually."""

    experiment = load_psb_experiment(config_path)
    selected_run = (
        resolve_latest_testable_run(experiment.output_root)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    checkpoint, equivalence = _verify_p0_run(
        selected_run,
        checkpoint_path,
        experiment.base.policy_checkpoint,
        experiment.base.critic_checkpoint,
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
        )
        rollouts.append(_rollout_summary(out_td, seed))

    report: Dict[str, object] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": experiment.stage,
        "run_directory": str(selected_run),
        "checkpoint": str(checkpoint),
        "comparison_mode": (
            "byte_exact_base_equivalence" if compare_base else "p0_integrity"
        ),
        "noninferiority_result": "proven_by_identical_policy_checkpoint",
        "promotion_result": (
            "base_passthrough_already_selected"
            if promote_if_noninferior
            else "not_requested"
        ),
        "equivalence": equivalence,
        "rollouts": rollouts,
    }
    atomic_write_json(selected_run / "p0_manual_validation.json", report)
    write_artifact_manifest(selected_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
