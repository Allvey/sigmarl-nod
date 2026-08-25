"""Opinion-MARL training entry point through the configurable M9 trainer."""

import argparse
import json
from pathlib import Path

from main_training import train_base
from utilities.experiment_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    resolve_evidence_critic_pair,
    resolve_latest_evidence_run,
    resolve_latest_run,
    resolve_latest_testable_run,
    resolve_policy_critic_pair,
)
from utilities.opinion.config import (
    load_opinion_experiment,
    require_m9_supported_mode,
)


DEFAULT_CONFIG_FILE = Path("config_opinion.json")


def _resolve_m6_base_actor_source(
    evidence_run_directory: Path,
    recorded_base_actor: Path,
    current_base_actor: Path,
) -> tuple[Path, str]:
    """Reuse the exact frozen Actor that produced the selected M5 EvidenceNet.

    New runs save ``source_base_actor.pth`` before their first iteration.  M5
    runs produced before that safeguard still contain the same frozen state as
    ``final_base_actor.pth`` after a clean completion, so keep that artifact as
    an explicit backwards-compatible fallback.
    """

    candidates = (
        (evidence_run_directory / "source_base_actor.pth", "m5_preserved_source"),
        (evidence_run_directory / "final_base_actor.pth", "m5_final_base_actor"),
    )
    for candidate, checkpoint_kind in candidates:
        if candidate.is_file():
            return candidate.resolve(), checkpoint_kind

    if (
        recorded_base_actor == current_base_actor.resolve()
        and recorded_base_actor.is_file()
    ):
        return recorded_base_actor, "m5_recorded_source"

    raise FileNotFoundError(
        "The selected M5 EvidenceNet has no reusable frozen Base Actor. "
        "Expected source_base_actor.pth or final_base_actor.pth in "
        f"{evidence_run_directory}; recorded source={recorded_base_actor}. "
        "Restart M5 with the current artifact-saving logic before M6."
    )


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    resume_checkpoint: Path = None,
) -> Path:
    experiment = load_opinion_experiment(config_file)
    require_m9_supported_mode(experiment)
    conflict_config = experiment.config.opinion.conflict_graph
    emits_pair_info = conflict_config.emit_pair_info
    bridge_config = experiment.config.opinion.policy_bridge
    stateful_config = experiment.config.opinion.stateful
    sequence_config = experiment.config.opinion.sequence_ppo
    trainer_config = experiment.config.opinion.trainer
    opinion_policy_config = None
    resolved_opinion_config = experiment.resolved_opinion_config()
    opinion_values = experiment.source_config["opinion"]
    initialize_from_scratch = bool(
        trainer_config.enabled and trainer_config.initialization == "none"
    )
    if initialize_from_scratch:
        use_base_ppo_update = not experiment.config.opinion.residual.apply_to_action
        opinion_policy_config = {
            "mode": bridge_config.mode,
            "freeze_base_actor": bridge_config.freeze_base_actor,
            "evidence": opinion_values["evidence"],
            "dynamics": opinion_values["dynamics"],
            "residual": opinion_values["residual"],
            "freeze_evidence": stateful_config.freeze_evidence,
            "zero_threshold": stateful_config.zero_threshold,
            "evidence_learning_rate_scale": opinion_values["sequence_ppo"][
                "evidence_learning_rate_scale"
            ],
            "use_base_ppo_update": use_base_ppo_update,
            "sequence_buffer_enabled": not use_base_ppo_update,
            "sequence_evidence_training": not use_base_ppo_update,
            "chunk_length": sequence_config.chunk_length,
            "neutral_loss_coefficient": (
                sequence_config.neutral_loss_coefficient
            ),
            "magnitude_loss_coefficient": (
                sequence_config.magnitude_loss_coefficient
            ),
            "initialize_evidence_random": True,
            "initialize_from_scratch": True,
            "trainer": opinion_values["trainer"],
            "resume_checkpoint": (
                None
                if resume_checkpoint is None
                else str(Path(resume_checkpoint).expanduser().resolve())
            ),
        }
        resolved_opinion_config["initialization_contract"] = (
            "random_actor_evidence_critic"
        )
    if bridge_config.enabled and not initialize_from_scratch:
        try:
            base_run_directory = resolve_latest_run(
                bridge_config.base_output_root
            )
            base_run_status = "completed"
        except FileNotFoundError:
            base_run_directory = resolve_latest_testable_run(
                bridge_config.base_output_root
            )
            base_run_status = "incomplete"
            base_training_status_path = (
                base_run_directory / "training_status.json"
            )
            if base_training_status_path.is_file():
                with base_training_status_path.open(
                    "r", encoding="utf-8"
                ) as file:
                    recorded_status = json.load(file).get("status")
                if recorded_status in {"running", "failed", "completed"}:
                    base_run_status = recorded_status
            print(
                f"[WARNING] No completed Base run is available (selected Base "
                f"status: {base_run_status}). The Opinion stage will be "
                "initialized from a matched intermediate Base Actor/Critic pair. "
                "This is suitable for pipeline development, but formal performance "
                "experiments should use a fully trained Base run."
            )
        base_resolved_config_path = base_run_directory / "config_resolved.json"
        if not base_resolved_config_path.is_file():
            raise FileNotFoundError(
                "M5 requires the Base run config snapshot: "
                f"{base_resolved_config_path}"
            )
        with base_resolved_config_path.open("r", encoding="utf-8") as file:
            base_run_config = json.load(file)
        for field_name, expected_value in experiment.base_source_config.items():
            if field_name == "where_to_save":
                continue
            actual_value = base_run_config.get(field_name)
            if actual_value != expected_value:
                raise ValueError(
                    "The selected Base run does not match the referenced Base "
                    f"config at '{field_name}': expected {expected_value!r}, "
                    f"found {actual_value!r}."
                )
        base_actor_checkpoint, base_critic_checkpoint = (
            resolve_policy_critic_pair(base_run_directory)
        )
        base_checkpoint_kind = (
            "final"
            if base_actor_checkpoint.name == "final_policy.pth"
            else "intermediate_best_reward"
        )

        # Use the validated JSON-shaped values here: EvidenceConfig.from_dict
        # intentionally requires hidden_sizes to remain a JSON list.
        opinion_policy_config = {
            "mode": bridge_config.mode,
            "freeze_base_actor": bridge_config.freeze_base_actor,
            "base_run_directory": str(base_run_directory),
            "base_actor_checkpoint": str(base_actor_checkpoint),
            "base_critic_checkpoint": str(base_critic_checkpoint),
            "evidence": opinion_values["evidence"],
            "residual": opinion_values["residual"],
            "evidence_learning_rate_scale": opinion_values["sequence_ppo"][
                "evidence_learning_rate_scale"
            ],
        }
        if (
            stateful_config.enabled
            and trainer_config.enabled
            and stateful_config.evidence_output_root is None
        ):
            opinion_policy_config.update(
                {
                    "dynamics": opinion_values["dynamics"],
                    "freeze_evidence": stateful_config.freeze_evidence,
                    "zero_threshold": stateful_config.zero_threshold,
                }
            )
        elif stateful_config.enabled:
            evidence_run_directory = resolve_latest_evidence_run(
                stateful_config.evidence_output_root
            )
            evidence_checkpoint, evidence_critic_checkpoint = (
                resolve_evidence_critic_pair(evidence_run_directory)
            )
            evidence_source_snapshot = (
                evidence_run_directory / "opinion_config_resolved.json"
            )
            if not evidence_source_snapshot.is_file():
                raise FileNotFoundError(
                    "M6 requires the M5 opinion config snapshot: "
                    f"{evidence_source_snapshot}"
                )
            with evidence_source_snapshot.open("r", encoding="utf-8") as file:
                evidence_source_config = json.load(file)
            source_opinion = evidence_source_config.get("opinion", {})
            source_bridge = source_opinion.get("policy_bridge", {})
            source_stateful = source_opinion.get("stateful")
            # M5 runs completed before the M6 schema extension either omit
            # this field or persist it as null. Both unambiguously mean the
            # stateless Direct-Evidence stage; malformed non-object values are
            # still rejected.
            if source_stateful is None:
                source_stateful_enabled = False
            elif isinstance(source_stateful, dict):
                source_stateful_enabled = source_stateful.get("enabled")
            else:
                source_stateful_enabled = None
            if (
                evidence_source_config.get("stage") != "evidence"
                or source_bridge.get("mode") != "direct_evidence"
                or source_stateful_enabled is not False
            ):
                raise ValueError(
                    "M6 evidence_output_root must select an M5 Direct-Evidence run."
                )
            if source_opinion.get("evidence") != opinion_values["evidence"]:
                raise ValueError(
                    "M6 EvidenceNet configuration must exactly match its M5 source."
                )
            recorded_base_actor = Path(
                str(evidence_source_config["resolved_base_actor_checkpoint"])
            ).expanduser().resolve()
            base_actor_checkpoint, base_checkpoint_kind = (
                _resolve_m6_base_actor_source(
                    evidence_run_directory=evidence_run_directory,
                    recorded_base_actor=recorded_base_actor,
                    current_base_actor=base_actor_checkpoint,
                )
            )
            opinion_policy_config["base_actor_checkpoint"] = str(
                base_actor_checkpoint
            )
            base_critic_checkpoint = evidence_critic_checkpoint
            opinion_policy_config["base_critic_checkpoint"] = str(
                evidence_critic_checkpoint
            )
            opinion_policy_config.update(
                {
                    "evidence_checkpoint": str(evidence_checkpoint),
                    "dynamics": opinion_values["dynamics"],
                    "freeze_evidence": stateful_config.freeze_evidence,
                    "zero_threshold": stateful_config.zero_threshold,
                }
            )
            resolved_opinion_config["resolved_evidence_run_directory"] = str(
                evidence_run_directory
            )
            resolved_opinion_config["resolved_evidence_checkpoint"] = str(
                evidence_checkpoint
            )
            resolved_opinion_config["resolved_evidence_critic_checkpoint"] = str(
                evidence_critic_checkpoint
            )
            if sequence_config.enabled and not trainer_config.enabled:
                try:
                    sequence_source_run = resolve_latest_run(
                        sequence_config.source_output_root
                    )
                    sequence_source_status = "completed"
                except FileNotFoundError:
                    sequence_source_run = resolve_latest_testable_run(
                        sequence_config.source_output_root
                    )
                    sequence_source_status = "incomplete"
                    source_stage = "M7" if sequence_config.train_evidence else "M6"
                    target_stage = "M8" if sequence_config.train_evidence else "M7"
                    print(
                        f"[WARNING] {target_stage} is initialized from an "
                        f"incomplete {source_stage} run. This is suitable for "
                        "pipeline development only."
                    )
                sequence_policy_checkpoint, sequence_critic_checkpoint = (
                    resolve_policy_critic_pair(sequence_source_run)
                )
                sequence_opinion_snapshot = (
                    sequence_source_run / "opinion_config_resolved.json"
                )
                if not sequence_opinion_snapshot.is_file():
                    raise FileNotFoundError(
                        "Sequence training requires its source opinion config "
                        f"snapshot: {sequence_opinion_snapshot}"
                    )
                with sequence_opinion_snapshot.open("r", encoding="utf-8") as file:
                    sequence_source_config = json.load(file)
                source_opinion = sequence_source_config.get("opinion", {})
                source_bridge = source_opinion.get("policy_bridge", {})
                source_stateful = source_opinion.get("stateful", {})
                source_sequence = source_opinion.get("sequence_ppo", {})
                common_source_valid = (
                    source_bridge.get("mode") == "stateful_opinion"
                    and source_stateful.get("enabled") is True
                )
                if sequence_config.train_evidence:
                    source_valid = (
                        common_source_valid
                        and sequence_source_config.get("stage") == "sequence"
                        and source_stateful.get("freeze_evidence") is True
                        and source_sequence.get("enabled") is True
                        and source_sequence.get("train_evidence", False) is False
                    )
                    source_description = "an M7 frozen-Evidence Sequence-Buffer run"
                else:
                    source_valid = (
                        common_source_valid
                        and sequence_source_config.get("stage") == "evidence"
                        and source_sequence.get("enabled", False) is False
                    )
                    source_description = (
                        "an M6 Stateful run with Sequence Buffer disabled"
                    )
                if not source_valid:
                    raise ValueError(
                        "sequence_ppo.source_output_root must select "
                        f"{source_description}."
                    )
                for section_name in ("evidence", "dynamics", "residual"):
                    if source_opinion.get(section_name) != opinion_values[section_name]:
                        raise ValueError(
                            "Sequence configuration does not match its source at "
                            f"opinion.{section_name}."
                        )
                base_critic_checkpoint = sequence_critic_checkpoint
                opinion_policy_config.update(
                    {
                        "base_critic_checkpoint": str(sequence_critic_checkpoint),
                        "initial_policy_checkpoint": str(sequence_policy_checkpoint),
                        "sequence_buffer_enabled": True,
                        "sequence_evidence_training": (
                            sequence_config.train_evidence
                        ),
                        "chunk_length": sequence_config.chunk_length,
                        "neutral_loss_coefficient": (
                            sequence_config.neutral_loss_coefficient
                        ),
                        "magnitude_loss_coefficient": (
                            sequence_config.magnitude_loss_coefficient
                        ),
                    }
                )
                resolved_opinion_config.update(
                    {
                        "resolved_sequence_source_run_directory": str(
                            sequence_source_run
                        ),
                        "resolved_sequence_source_status": sequence_source_status,
                        "resolved_sequence_policy_checkpoint": str(
                            sequence_policy_checkpoint
                        ),
                        "resolved_sequence_critic_checkpoint": str(
                            sequence_critic_checkpoint
                        ),
                    }
                )
        if trainer_config.enabled:
            opinion_policy_config.update(
                {
                    "dynamics": opinion_values["dynamics"],
                    "freeze_evidence": stateful_config.freeze_evidence,
                    "zero_threshold": stateful_config.zero_threshold,
                    "sequence_buffer_enabled": True,
                    "sequence_evidence_training": True,
                    "chunk_length": sequence_config.chunk_length,
                    "neutral_loss_coefficient": (
                        sequence_config.neutral_loss_coefficient
                    ),
                    "magnitude_loss_coefficient": (
                        sequence_config.magnitude_loss_coefficient
                    ),
                    "initialize_evidence_random": (
                        trainer_config.initialization == "base"
                    ),
                    "trainer": opinion_values["trainer"],
                    "resume_checkpoint": (
                        None
                        if resume_checkpoint is None
                        else str(Path(resume_checkpoint).expanduser().resolve())
                    ),
                }
            )
            if trainer_config.initialization == "opinion":
                try:
                    trainer_source_run = resolve_latest_run(
                        trainer_config.source_output_root
                    )
                    trainer_source_status = "completed"
                except FileNotFoundError:
                    trainer_source_run = resolve_latest_testable_run(
                        trainer_config.source_output_root
                    )
                    trainer_source_status = "incomplete"
                    print(
                        "[WARNING] M9 is initialized from an incomplete Opinion "
                        "run. This is suitable for pipeline development only."
                    )
                trainer_policy, trainer_critic = resolve_policy_critic_pair(
                    trainer_source_run
                )
                trainer_snapshot = (
                    trainer_source_run / "opinion_config_resolved.json"
                )
                if not trainer_snapshot.is_file():
                    raise FileNotFoundError(
                        "M9 Opinion initialization requires: "
                        f"{trainer_snapshot}"
                    )
                with trainer_snapshot.open("r", encoding="utf-8") as file:
                    trainer_source_config = json.load(file)
                source_opinion = trainer_source_config.get("opinion", {})
                if (
                    trainer_source_config.get("stage")
                    not in {"sequence_ppo", "joint"}
                    or source_opinion.get("policy_bridge", {}).get("mode")
                    != "stateful_opinion"
                    or source_opinion.get("sequence_ppo", {}).get("train_evidence")
                    is not True
                ):
                    raise ValueError(
                        "M9 Opinion initialization must select an M8/M9 "
                        "stateful Sequence-PPO run."
                    )
                for section_name in ("evidence", "dynamics", "residual"):
                    if source_opinion.get(section_name) != opinion_values[section_name]:
                        raise ValueError(
                            "M9 configuration does not match its Opinion source "
                            f"at opinion.{section_name}."
                        )
                base_critic_checkpoint = trainer_critic
                opinion_policy_config.update(
                    {
                        "initial_policy_checkpoint": str(trainer_policy),
                        "base_critic_checkpoint": str(trainer_critic),
                    }
                )
                resolved_opinion_config.update(
                    {
                        "resolved_trainer_source_run_directory": str(
                            trainer_source_run
                        ),
                        "resolved_trainer_source_status": trainer_source_status,
                        "resolved_trainer_policy_checkpoint": str(trainer_policy),
                        "resolved_trainer_critic_checkpoint": str(trainer_critic),
                    }
                )
            else:
                # Base initialization keeps the already validated Base
                # Actor/Critic and starts EvidenceNet from its near-neutral
                # random initialization.
                opinion_policy_config.pop("evidence_checkpoint", None)
                opinion_policy_config.pop("initial_policy_checkpoint", None)
                opinion_policy_config["base_actor_checkpoint"] = str(
                    base_actor_checkpoint
                )
                opinion_policy_config["base_critic_checkpoint"] = str(
                    base_critic_checkpoint
                )
        resolved_opinion_config["resolved_base_run_directory"] = str(
            base_run_directory
        )
        resolved_opinion_config["resolved_base_run_status"] = base_run_status
        resolved_opinion_config["resolved_base_checkpoint_kind"] = (
            base_checkpoint_kind
        )
        resolved_opinion_config["resolved_base_actor_checkpoint"] = str(
            base_actor_checkpoint
        )
        resolved_opinion_config["resolved_base_critic_checkpoint"] = str(
            base_critic_checkpoint
        )

    if trainer_config.enabled:
        scratch_suffix = "-from-scratch" if initialize_from_scratch else ""
        run_label = (
            f"m9-{trainer_config.mode.replace('_', '-')}{scratch_suffix}"
        )
        artifact_stage = (
            f"trainer_{trainer_config.mode}_from_scratch"
            if initialize_from_scratch
            else f"trainer_{trainer_config.mode}"
        )
        expected_behavior = (
            "m9_joint_from_scratch"
            if initialize_from_scratch
            else f"m9_{trainer_config.mode}"
        )
        comparison_note = (
            "M9 jointly trains randomly initialized Base Actor, EvidenceNet, "
            "and Central Critic for one SigmaRL-aligned budget without loading "
            "earlier-stage weights."
            if initialize_from_scratch
            else "M9 uses the unified configurable trainer. Base Actor is either "
            "frozen, jointly optimized, or activated after Evidence warmup; "
            "EvidenceNet and Critic retain separate parameter groups."
        )
    elif sequence_config.enabled and sequence_config.train_evidence:
        run_label = "m8-sequence-ppo"
        artifact_stage = "evidence_sequence_ppo"
        expected_behavior = "differentiable_opinion_sequence_ppo"
        comparison_note = (
            "M8 freezes the Base Actor, unrolls Opinion Dynamics through each "
            "truncated chunk, and trains EvidenceNet from PPO advantage while "
            "the Central Critic is optimized independently."
        )
    elif sequence_config.enabled:
        run_label = "m7-sequence-buffer"
        artifact_stage = "sequence_buffer"
        expected_behavior = "sequence_buffer_noop_policy"
        comparison_note = (
            "M7 preserves complete time chunks and z_init while keeping the "
            "M6 Actor/Evidence frozen; only the Central Critic is trained."
        )
    elif stateful_config.enabled:
        run_label = "m6-stateful-opinion"
        artifact_stage = "stateful_rollout"
        expected_behavior = "stateful_opinion_rollout"
        comparison_note = (
            "M6 freezes Base Actor and EvidenceNet, evolves z_dense once per "
            "physical step, and trains only the unchanged Central Critic."
        )
    elif bridge_config.enabled:
        run_label = "m5-direct-evidence"
        artifact_stage = "evidence_direct"
        expected_behavior = "direct_evidence_candidate"
        comparison_note = (
            "M5 freezes the Base Actor and trains a stateless EvidenceNet that "
            "applies a bounded speed-location residual."
        )
    else:
        run_label = "m4-pair-info" if emits_pair_info else "opinion-off-base"
        artifact_stage = experiment.config.stage
        expected_behavior = "base_equivalent"
        comparison_note = (
            "M4 emits physical pair tensors through environment info, but "
            "policy, reward, action, and optimizer remain Base-equivalent."
            if emits_pair_info
            else "Opinion is disabled and reuses the R1 Base path."
        )
    return train_base(
        parameters=experiment.parameters,
        source_config=experiment.source_config,
        run_label=run_label,
        supplementary_snapshots={
            "base_config_source.json": experiment.base_source_config,
            "opinion_config_resolved.json": resolved_opinion_config,
        },
        comparison_payload={
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "reference": "R1 Base-MAPPO with the same seed and budget",
            "status": "pending_user_validation",
            "expected_behavior": expected_behavior,
            "automated_performance_validation": False,
            "note": comparison_note,
        },
        opinion_pair_info_config=(
            conflict_config.to_dict() if emits_pair_info else None
        ),
        opinion_policy_config=opinion_policy_config,
        artifact_method="opinion_marl",
        artifact_stage=artifact_stage,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Opinion-MARL through the configurable M9 trainer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Opinion experiment configuration (default: config_opinion.json).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an M9 run from latest_checkpoint.pt or a periodic checkpoint.",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.resume)
