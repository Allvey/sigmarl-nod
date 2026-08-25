"""Opinion-MARL training entry point through the M6 Stateful stage."""

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
    require_m6_supported_mode,
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


def main(config_file: Path = DEFAULT_CONFIG_FILE) -> Path:
    experiment = load_opinion_experiment(config_file)
    require_m6_supported_mode(experiment)
    conflict_config = experiment.config.opinion.conflict_graph
    emits_pair_info = conflict_config.emit_pair_info
    bridge_config = experiment.config.opinion.policy_bridge
    stateful_config = experiment.config.opinion.stateful
    opinion_policy_config = None
    resolved_opinion_config = experiment.resolved_opinion_config()
    if bridge_config.enabled:
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
        opinion_values = experiment.source_config["opinion"]
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
        if stateful_config.enabled:
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

    run_label = (
        "m6-stateful-opinion"
        if stateful_config.enabled
        else (
            "m5-direct-evidence"
            if bridge_config.enabled
            else ("m4-pair-info" if emits_pair_info else "opinion-off-base")
        )
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
            "expected_behavior": (
                "stateful_opinion_rollout"
                if stateful_config.enabled
                else (
                    "direct_evidence_candidate"
                    if bridge_config.enabled
                    else "base_equivalent"
                )
            ),
            "automated_performance_validation": False,
            "note": (
                "M6 freezes Base Actor and EvidenceNet, evolves z_dense once per "
                "physical step, and trains only the unchanged Central Critic."
                if stateful_config.enabled
                else (
                    "M5 freezes the Base Actor and trains a stateless EvidenceNet "
                    "that applies a bounded speed-location residual."
                    if bridge_config.enabled
                    else (
                        "M4 emits physical pair tensors through environment info, "
                        "but policy, reward, action, and optimizer remain Base-equivalent."
                        if emits_pair_info
                        else "Opinion is disabled and reuses the R1 Base path."
                    )
                )
            ),
        },
        opinion_pair_info_config=(
            conflict_config.to_dict() if emits_pair_info else None
        ),
        opinion_policy_config=opinion_policy_config,
        artifact_method="opinion_marl",
        artifact_stage=(
            "stateful_rollout"
            if stateful_config.enabled
            else (
                "evidence_direct"
                if bridge_config.enabled
                else experiment.config.stage
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the staged Opinion-MARL method through M6."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Opinion experiment configuration (default: config_opinion.json).",
    )
    arguments = parser.parse_args()
    main(arguments.config)
