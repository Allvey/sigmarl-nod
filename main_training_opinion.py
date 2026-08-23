"""Opinion-MARL training entry point.

M2 provides the typed configuration and preserves an executable no-op/Base
path. Opinion computation is deliberately unavailable until its mathematical
modules are implemented in later milestones.
"""

import argparse
from pathlib import Path

from main_training import train_base
from utilities.experiment_artifacts import ARTIFACT_SCHEMA_VERSION
from utilities.opinion.config import (
    load_opinion_experiment,
    require_m2_base_mode,
)


DEFAULT_CONFIG_FILE = Path("config_opinion.json")


def main(config_file: Path = DEFAULT_CONFIG_FILE) -> Path:
    experiment = load_opinion_experiment(config_file)
    require_m2_base_mode(experiment)
    return train_base(
        parameters=experiment.parameters,
        source_config=experiment.source_config,
        run_label="opinion-off-base",
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
            "note": "M2 has Opinion disabled and reuses the R1 Base path.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the staged Opinion-MARL method (M2 Base/no-op mode)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Opinion experiment configuration (default: config_opinion.json).",
    )
    arguments = parser.parse_args()
    main(arguments.config)
