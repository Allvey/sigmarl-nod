"""PSB-MARL training entrypoint.

P0 performs no optimization.  It validates and packages the selected
Base-MAPPO policy/critic pair without changing a single model byte.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from utilities.experiment_artifacts import (
    atomic_write_json,
    create_run_directory,
    initialize_run,
    mark_latest_completed_run,
    write_artifact_manifest,
    write_training_status,
)
from utilities.psb_marl.checkpoint import (
    copy_checkpoint_exact,
    save_p0_training_checkpoint,
    save_p1_layer_checkpoint,
    save_p1_training_checkpoint,
    sha256_file,
)
from utilities.psb_marl.certification import certify_p1_layer
from utilities.psb_marl.config import PSBConfigError, load_psb_experiment
from utilities.psb_marl.proximal import ProximalSaturatingBifurcation


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def train_psb(
    config_path: Path,
    *,
    resume_checkpoint: Optional[Path] = None,
    iterations_override: Optional[int] = None,
) -> Path:
    """Run the configured PSB stage and return its isolated run directory."""

    experiment = load_psb_experiment(config_path)
    if experiment.stage == "p3_paired_differential_primal_dual_ppo":
        return _train_p33(
            experiment,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if experiment.stage == "p3_primal_dual_ppo":
        return _train_p32(
            experiment,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if experiment.stage == "p3_differential_critic":
        from utilities.psb_marl.p3_critic_training import train_p31

        return train_p31(
            experiment,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if experiment.stage == "p3_paired_rollout_equivalence":
        from utilities.psb_marl.p3_pairing import package_p3_pairing

        return package_p3_pairing(
            experiment,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if experiment.stage == "p2_frozen_base_bifurcation":
        return _train_p2(
            experiment,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if experiment.stage == "p1_zero_control_equivalence":
        return _package_p1(
            experiment,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if resume_checkpoint is not None:
        raise PSBConfigError("P0 is immutable and does not support resume.")
    if iterations_override is not None:
        raise PSBConfigError("P0 performs no optimization; --iterations is invalid.")

    output_root = str(Path(experiment.output_root).expanduser().resolve())
    run_directory = create_run_directory(
        output_root=output_root,
        method="psb-p0",
        seed=experiment.seed,
    )
    resolved_base = dict(experiment.base_run_config)
    resolved_base.update(
        {
            "where_to_save": str(run_directory) + os.sep,
            "output_root": output_root,
            "run_id": run_directory.name,
            "artifact_logging_enabled": True,
        }
    )
    initialize_run(
        run_directory=run_directory,
        source_config=experiment.source_config,
        resolved_config=resolved_base,
        method="psb_marl",
        stage=experiment.stage,
    )
    try:
        policy_destination = run_directory / "final_policy.pth"
        critic_destination = run_directory / "final_critic.pth"
        policy_hash = copy_checkpoint_exact(
            experiment.base.policy_checkpoint, policy_destination
        )
        critic_hash = copy_checkpoint_exact(
            experiment.base.critic_checkpoint, critic_destination
        )
        config_fingerprint = _fingerprint(experiment.source_config)
        save_p0_training_checkpoint(
            run_directory / "final_checkpoint.pt",
            policy_checkpoint=policy_destination,
            critic_checkpoint=critic_destination,
            policy_sha256=policy_hash,
            critic_sha256=critic_hash,
            source_base_run=experiment.base.run_directory,
            config_fingerprint=config_fingerprint,
        )
        equivalence = {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": experiment.stage,
            "equivalence_kind": "byte_exact_checkpoint_passthrough",
            "source_base_run": str(experiment.base.run_directory),
            "source_policy_checkpoint": str(experiment.base.policy_checkpoint),
            "source_critic_checkpoint": str(experiment.base.critic_checkpoint),
            "packaged_policy_checkpoint": str(policy_destination.resolve()),
            "packaged_critic_checkpoint": str(critic_destination.resolve()),
            "policy_sha256": policy_hash,
            "critic_sha256": critic_hash,
            "policy_bytes_identical": True,
            "critic_bytes_identical": True,
            "action_path": "base_mappo_only",
            "trainable_psb_parameters": 0,
        }
        atomic_write_json(run_directory / "p0_equivalence.json", equivalence)
        atomic_write_json(
            run_directory / "psb_config_resolved.json",
            {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_base_policy": str(experiment.base.policy_checkpoint),
                "resolved_base_critic": str(experiment.base.critic_checkpoint),
                "config_fingerprint": config_fingerprint,
            },
        )
        atomic_write_json(
            run_directory / "deployment_manifest.json",
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "selected": "base_passthrough",
                "policy_checkpoint": "final_policy.pth",
                "critic_checkpoint": "final_critic.pth",
                "policy_sha256": policy_hash,
                "critic_sha256": critic_hash,
            },
        )
        write_training_status(
            run_directory,
            status="completed",
            iteration=0,
        )
        write_artifact_manifest(run_directory)
        mark_latest_completed_run(output_root, run_directory)
    except BaseException as error:
        write_training_status(
            run_directory,
            status="failed",
            iteration=0,
            error=f"{type(error).__name__}: {error}",
        )
        write_artifact_manifest(run_directory)
        raise

    return run_directory


def _train_p32(
    experiment,
    *,
    resume_checkpoint: Optional[Path],
    iterations_override: Optional[int],
) -> Path:
    """Continue the certified P2 Actor with projected primal-dual PPO."""

    if resume_checkpoint is not None:
        raise PSBConfigError("P3.2 resume is not enabled in the first version.")
    assert experiment.parent_run is not None
    assert experiment.conflict_graph is not None
    assert experiment.primal_dual is not None
    from main_training import train_base

    iterations = (
        experiment.primal_dual.iterations
        if iterations_override is None
        else iterations_override
    )
    if type(iterations) is not int or iterations <= 0:
        raise PSBConfigError("P3.2 --iterations must be a positive integer.")
    parameters = experiment.base_parameters
    parameters.seed = experiment.effective_training_seed
    parameters.where_to_save = str(
        Path(experiment.output_root).expanduser().resolve()
    )
    parameters.n_iters = iterations
    parameters.total_frames = parameters.frames_per_batch * iterations
    runtime = experiment.p32_runtime_config()
    runtime["training"] = {
        **runtime["training"],
        "iterations": iterations,
    }
    run = train_base(
        parameters=parameters,
        source_config=experiment.source_config,
        run_label="psb-p3-dual",
        supplementary_snapshots={
            "psb_config_resolved.json": {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_parent_run": str(experiment.parent_run),
                "runtime_config": runtime,
                "config_fingerprint": _fingerprint(experiment.source_config),
            }
        },
        comparison_payload={
            "schema_version": 1,
            "status": "pending_manual_paired_validation",
            "deployment": "base_fallback",
            "candidate_checkpoint": "candidate_policy.pth",
            "dual_learning_enabled": True,
        },
        opinion_pair_info_config=experiment.conflict_graph.to_dict(),
        psb_runtime_config=runtime,
        artifact_method="psb_marl",
        artifact_stage=experiment.stage,
        resume_checkpoint=None,
    )
    differential_hash = copy_checkpoint_exact(
        experiment.parent_run / "candidate_critic.pth",
        run / "source_differential_critic.pth",
    )
    required = (
        "candidate_policy.pth",
        "candidate_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
        "final_policy.pth",
        "final_critic.pth",
        "p3_dual_state.pt",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise RuntimeError(f"P3.2 training is missing artifacts: {missing}.")
    atomic_write_json(
        run / "deployment_manifest.json",
        {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": experiment.stage,
            "selected": "base_fallback_pending_p3_2_validation",
            "actor_learning_enabled": True,
            "dual_learning_enabled": True,
            "candidate_policy": "candidate_policy.pth",
            "candidate_critic": "candidate_critic.pth",
            "dual_state": "p3_dual_state.pt",
            "source_differential_critic": "source_differential_critic.pth",
            "source_differential_critic_sha256": differential_hash,
            "candidate_policy_sha256": sha256_file(
                run / "candidate_policy.pth"
            ),
            "candidate_critic_sha256": sha256_file(
                run / "candidate_critic.pth"
            ),
            "base_policy_sha256": sha256_file(
                run / "base_fallback_policy.pth"
            ),
            "base_critic_sha256": sha256_file(
                run / "base_fallback_critic.pth"
            ),
        },
    )
    write_artifact_manifest(run)
    return run


def _train_p33(
    experiment,
    *,
    resume_checkpoint: Optional[Path],
    iterations_override: Optional[int],
) -> Path:
    """Train the P2 Actor with paired differential primal-dual PPO."""

    if resume_checkpoint is not None:
        raise PSBConfigError("P3.3 resume is not enabled in the first version.")
    assert experiment.parent_run is not None
    assert experiment.conflict_graph is not None
    assert experiment.primal_dual is not None
    assert experiment.paired_differential is not None
    from main_training import train_base

    iterations = (
        experiment.primal_dual.iterations
        if iterations_override is None
        else iterations_override
    )
    if type(iterations) is not int or iterations <= 0:
        raise PSBConfigError("P3.3 --iterations must be a positive integer.")
    parameters = experiment.base_parameters
    parameters.seed = experiment.effective_training_seed
    parameters.where_to_save = str(
        Path(experiment.output_root).expanduser().resolve()
    )
    parameters.n_iters = iterations
    parameters.total_frames = parameters.frames_per_batch * iterations
    runtime = experiment.p33_runtime_config()
    runtime["training"] = {
        **runtime["training"],
        "iterations": iterations,
    }
    run = train_base(
        parameters=parameters,
        source_config=experiment.source_config,
        run_label="psb-p3-diff-dual",
        supplementary_snapshots={
            "psb_config_resolved.json": {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_parent_run": str(experiment.parent_run),
                "runtime_config": runtime,
                "config_fingerprint": _fingerprint(experiment.source_config),
            }
        },
        comparison_payload={
            "schema_version": 1,
            "status": "pending_manual_paired_superiority_validation",
            "deployment": "base_fallback",
            "candidate_checkpoint": "candidate_policy.pth",
            "dual_learning_enabled": True,
            "paired_differential_learning_enabled": True,
            "paired_episode_boundary_mode": (
                "union_truncate_and_common_seed_reset"
            ),
        },
        opinion_pair_info_config=experiment.conflict_graph.to_dict(),
        psb_runtime_config=runtime,
        artifact_method="psb_marl",
        artifact_stage=experiment.stage,
        resume_checkpoint=None,
    )
    differential_hash = copy_checkpoint_exact(
        experiment.parent_run / "candidate_critic.pth",
        run / "source_differential_critic.pth",
    )
    required = (
        "candidate_policy.pth",
        "candidate_critic.pth",
        "candidate_differential_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
        "final_policy.pth",
        "final_critic.pth",
        "p3_dual_state.pt",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise RuntimeError(f"P3.3 training is missing artifacts: {missing}.")
    atomic_write_json(
        run / "deployment_manifest.json",
        {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": experiment.stage,
            "selected": "base_fallback_pending_p3_3_validation",
            "actor_learning_enabled": True,
            "dual_learning_enabled": True,
            "paired_differential_learning_enabled": True,
            "paired_episode_boundaries_synchronized": True,
            "paired_episode_boundary_mode": (
                "union_truncate_and_common_seed_reset"
            ),
            "candidate_policy": "candidate_policy.pth",
            "candidate_scalar_critic": "candidate_critic.pth",
            "candidate_differential_critic": (
                "candidate_differential_critic.pth"
            ),
            "dual_state": "p3_dual_state.pt",
            "source_differential_critic": "source_differential_critic.pth",
            "source_differential_critic_sha256": differential_hash,
            "candidate_policy_sha256": sha256_file(
                run / "candidate_policy.pth"
            ),
            "candidate_critic_sha256": sha256_file(
                run / "candidate_critic.pth"
            ),
            "candidate_differential_critic_sha256": sha256_file(
                run / "candidate_differential_critic.pth"
            ),
            "base_policy_sha256": sha256_file(
                run / "base_fallback_policy.pth"
            ),
            "base_critic_sha256": sha256_file(
                run / "base_fallback_critic.pth"
            ),
        },
    )
    write_artifact_manifest(run)
    return run


def _train_p2(
    experiment,
    *,
    resume_checkpoint: Optional[Path],
    iterations_override: Optional[int],
) -> Path:
    """Train a quarantined P2 candidate while retaining exact Base fallback."""

    assert experiment.parent_run is not None
    assert experiment.conflict_graph is not None
    assert experiment.training is not None
    from main_training import train_base

    parameters = experiment.base_parameters
    # P2 robustness studies vary only the stochastic PSB training process while
    # retaining the exact same frozen Base checkpoint.
    parameters.seed = experiment.effective_training_seed
    parameters.where_to_save = str(
        Path(experiment.output_root).expanduser().resolve()
    )
    iterations = (
        experiment.training.iterations
        if iterations_override is None
        else iterations_override
    )
    if type(iterations) is not int or iterations <= 0:
        raise PSBConfigError("P2 --iterations must be a positive integer.")
    parameters.n_iters = iterations
    parameters.total_frames = parameters.frames_per_batch * iterations
    runtime_config = experiment.p2_runtime_config()
    comparison_payload = {
        "schema_version": 1,
        "reference": "recorded_base_source",
        "status": "pending_manual_paired_validation",
        "automated_performance_validation": False,
        "deployment": "base_fallback",
        "candidate_checkpoint": "candidate_policy.pth",
        "promotion": runtime_config["promotion"],
    }
    run_directory = train_base(
        parameters=parameters,
        source_config=experiment.source_config,
        run_label="psb-p2",
        supplementary_snapshots={
            "psb_config_resolved.json": {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_parent_run": str(experiment.parent_run),
                "runtime_config": runtime_config,
                "config_fingerprint": _fingerprint(experiment.source_config),
            }
        },
        comparison_payload=comparison_payload,
        opinion_pair_info_config=experiment.conflict_graph.to_dict(),
        psb_runtime_config=runtime_config,
        artifact_method="psb_marl",
        artifact_stage=experiment.stage,
        resume_checkpoint=resume_checkpoint,
    )

    base_policy = run_directory / "base_fallback_policy.pth"
    base_critic = run_directory / "base_fallback_critic.pth"
    candidate_policy = run_directory / "candidate_policy.pth"
    candidate_critic = run_directory / "candidate_critic.pth"
    for path in (base_policy, base_critic, candidate_policy, candidate_critic):
        if not path.is_file():
            raise RuntimeError(f"P2 training did not create required artifact: {path}")
    base_policy_hash = sha256_file(base_policy)
    base_critic_hash = sha256_file(base_critic)
    if base_policy_hash != sha256_file(experiment.base.policy_checkpoint):
        raise RuntimeError("P2 Base fallback policy is not byte-identical to Base.")
    if base_critic_hash != sha256_file(experiment.base.critic_checkpoint):
        raise RuntimeError("P2 Base fallback critic is not byte-identical to Base.")
    atomic_write_json(
        run_directory / "deployment_manifest.json",
        {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": experiment.stage,
            "selected": "base_fallback_pending_validation",
            "policy_checkpoint": "final_policy.pth",
            "critic_checkpoint": "final_critic.pth",
            "base_fallback_policy": "base_fallback_policy.pth",
            "base_fallback_critic": "base_fallback_critic.pth",
            "candidate_policy": "candidate_policy.pth",
            "candidate_critic": "candidate_critic.pth",
            "base_policy_sha256": base_policy_hash,
            "base_critic_sha256": base_critic_hash,
            "candidate_policy_sha256": sha256_file(candidate_policy),
            "candidate_critic_sha256": sha256_file(candidate_critic),
            "promotion_gate": runtime_config["promotion"],
        },
    )
    write_artifact_manifest(run_directory)
    mark_latest_completed_run(experiment.output_root, run_directory)
    return run_directory


def _package_p1(
    experiment,
    *,
    resume_checkpoint: Optional[Path],
    iterations_override: Optional[int],
) -> Path:
    if resume_checkpoint is not None:
        raise PSBConfigError("P1 is immutable and does not support resume.")
    if iterations_override is not None:
        raise PSBConfigError(
            "P1 performs no PPO optimization; --iterations is invalid."
        )
    assert experiment.parent_run is not None
    assert experiment.conflict_graph is not None
    assert experiment.proximal is not None

    output_root = str(Path(experiment.output_root).expanduser().resolve())
    run_directory = create_run_directory(
        output_root=output_root,
        method="psb-p1",
        seed=experiment.seed,
    )
    resolved_base = dict(experiment.base_run_config)
    resolved_base.update(
        {
            "where_to_save": str(run_directory) + os.sep,
            "output_root": output_root,
            "run_id": run_directory.name,
            "artifact_logging_enabled": True,
        }
    )
    initialize_run(
        run_directory=run_directory,
        source_config=experiment.source_config,
        resolved_config=resolved_base,
        method="psb_marl",
        stage=experiment.stage,
    )
    try:
        policy_destination = run_directory / "final_policy.pth"
        critic_destination = run_directory / "final_critic.pth"
        policy_hash = copy_checkpoint_exact(
            experiment.parent_run / "final_policy.pth", policy_destination
        )
        critic_hash = copy_checkpoint_exact(
            experiment.parent_run / "final_critic.pth", critic_destination
        )
        runtime_config = experiment.p1_runtime_config()
        layer = ProximalSaturatingBifurcation.from_runtime_config(
            runtime_config["proximal"]
        )
        certification = certify_p1_layer(layer)
        config_fingerprint = _fingerprint(experiment.source_config)
        save_p1_layer_checkpoint(
            run_directory / "final_psb_layer.pth", runtime_config
        )
        layer_hash = sha256_file(run_directory / "final_psb_layer.pth")
        save_p1_training_checkpoint(
            run_directory / "final_checkpoint.pt",
            policy_checkpoint=policy_destination,
            critic_checkpoint=critic_destination,
            policy_sha256=policy_hash,
            critic_sha256=critic_hash,
            parent_run=experiment.parent_run,
            runtime_config=runtime_config,
            config_fingerprint=config_fingerprint,
        )
        atomic_write_json(run_directory / "p1_certification.json", certification)
        atomic_write_json(
            run_directory / "p1_equivalence.json",
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "equivalence_kind": "zero_control_action_path_isolation",
                "parent_run": str(experiment.parent_run),
                "source_base_run": str(experiment.base.run_directory),
                "policy_sha256": policy_hash,
                "critic_sha256": critic_hash,
                "layer_sha256": layer_hash,
                "policy_bytes_identical_to_base": True,
                "critic_bytes_identical_to_base": True,
                "control_mode": "zero",
                "b_max": 0.0,
                "actor_context_gain": 0.0,
                "action_path": "untouched_base_actor",
                "trainable_psb_parameters": 0,
            },
        )
        atomic_write_json(
            run_directory / "psb_config_resolved.json",
            {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_parent_run": str(experiment.parent_run),
                "runtime_config": runtime_config,
                "config_fingerprint": config_fingerprint,
            },
        )
        atomic_write_json(
            run_directory / "deployment_manifest.json",
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "selected": "p1_zero_control_sidecar",
                "policy_checkpoint": "final_policy.pth",
                "critic_checkpoint": "final_critic.pth",
                "layer_checkpoint": "final_psb_layer.pth",
                "policy_sha256": policy_hash,
                "critic_sha256": critic_hash,
                "layer_sha256": layer_hash,
                "actor_context_gain": 0.0,
            },
        )
        write_training_status(
            run_directory,
            status="completed",
            iteration=0,
        )
        write_artifact_manifest(run_directory)
        mark_latest_completed_run(output_root, run_directory)
    except BaseException as error:
        write_training_status(
            run_directory,
            status="failed",
            iteration=0,
            error=f"{type(error).__name__}: {error}",
        )
        write_artifact_manifest(run_directory)
        raise
    return run_directory
