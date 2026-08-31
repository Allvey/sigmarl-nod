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
