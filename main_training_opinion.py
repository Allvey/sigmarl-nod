"""Opinion-MARL training entry point through the M4 information-only stage."""

import argparse
from pathlib import Path

from main_training import train_base
from utilities.experiment_artifacts import ARTIFACT_SCHEMA_VERSION
from utilities.opinion.config import (
    load_opinion_experiment,
    require_base_noop_mode,
)


DEFAULT_CONFIG_FILE = Path("config_opinion.json")


def main(config_file: Path = DEFAULT_CONFIG_FILE) -> Path:
    experiment = load_opinion_experiment(config_file)
    require_base_noop_mode(experiment)
    conflict_config = experiment.config.opinion.conflict_graph
    emits_pair_info = conflict_config.emit_pair_info
    return train_base(
        parameters=experiment.parameters,
        source_config=experiment.source_config,
        run_label="m4-pair-info" if emits_pair_info else "opinion-off-base",
        supplementary_snapshots={
            "base_config_source.json": experiment.base_source_config,
            "opinion_config_resolved.json": experiment.resolved_opinion_config(),
        },
        comparison_payload={
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "reference": "R1 Base-MAPPO with the same seed and budget",
            "status": "pending_user_validation",
            "expected_behavior": "base_equivalent",
            "automated_performance_validation": False,
            "note": (
                "M4 emits physical pair tensors through environment info, but "
                "the policy, reward, action, and optimizer remain Base-equivalent."
                if emits_pair_info
                else "Opinion is disabled and reuses the R1 Base path."
            ),
        },
        opinion_pair_info_config=(
            conflict_config.to_dict() if emits_pair_info else None
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the staged Opinion-MARL method through M4."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Opinion experiment configuration (default: config_opinion.json).",
    )
    arguments = parser.parse_args()
    main(arguments.config)
