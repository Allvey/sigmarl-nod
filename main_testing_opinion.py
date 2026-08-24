"""Opinion-MARL testing entry point through the M5 Direct-Evidence stage."""

import argparse
from pathlib import Path
from typing import Optional

from main_testing import test_base
from utilities.opinion.config import (
    load_opinion_experiment,
    require_m5_supported_mode,
)


DEFAULT_CONFIG_FILE = Path("config_opinion.json")


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    run_directory: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> None:
    experiment = load_opinion_experiment(config_file)
    require_m5_supported_mode(experiment)
    conflict_config = experiment.config.opinion.conflict_graph
    bridge_config = experiment.config.opinion.policy_bridge
    opinion_policy_config = None
    if bridge_config.enabled:
        # Preserve JSON list types expected by the strict runtime parsers.
        opinion_values = experiment.source_config["opinion"]
        opinion_policy_config = {
            "mode": bridge_config.mode,
            "freeze_base_actor": bridge_config.freeze_base_actor,
            "evidence": opinion_values["evidence"],
            "residual": opinion_values["residual"],
            "evidence_learning_rate_scale": opinion_values["sequence_ppo"][
                "evidence_learning_rate_scale"
            ],
        }
    test_base(
        experiment.config.output_root,
        run_directory,
        checkpoint_path,
        opinion_pair_info_config=(
            conflict_config.to_dict() if conflict_config.emit_pair_info else None
        ),
        opinion_policy_config=opinion_policy_config,
        opinion_visualization_config=(
            {
                "agent_id": bridge_config.visualize_agent_id,
                "prediction_horizon_seconds": (
                    conflict_config.prediction_horizon_seconds
                ),
                "sensing_distance_meters": conflict_config.sensing_distance_meters,
            }
            if conflict_config.emit_pair_info
            else None
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test an Opinion-MARL run through the M5 Direct-Evidence stage."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Opinion experiment configuration (default: config_opinion.json).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Test this exact run instead of auto-resolving a run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Load this exact final_policy.pth or reward<value>_policy.pth. "
            "Its parent directory is used as --run-dir when --run-dir is omitted."
        ),
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.run_dir, arguments.checkpoint)
