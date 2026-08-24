"""Opinion-MARL testing entry point through the M4 information-only stage."""

import argparse
from pathlib import Path
from typing import Optional

from main_testing import test_base
from utilities.opinion.config import (
    load_opinion_experiment,
    require_base_noop_mode,
)


DEFAULT_CONFIG_FILE = Path("config_opinion.json")


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    run_directory: Optional[Path] = None,
) -> None:
    experiment = load_opinion_experiment(config_file)
    require_base_noop_mode(experiment)
    conflict_config = experiment.config.opinion.conflict_graph
    test_base(
        experiment.config.output_root,
        run_directory,
        opinion_pair_info_config=(
            conflict_config.to_dict() if conflict_config.emit_pair_info else None
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test an Opinion-MARL run through the M4 information stage."
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
        help="Test this exact run instead of the latest completed run.",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.run_dir)
