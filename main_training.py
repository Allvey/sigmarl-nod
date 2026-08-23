"""Standard Base-MAPPO training entry point.

R1 keeps the original SigmaRL 1.2.0 optimizer path and adds only isolated run
directories plus reproducible artifacts around it.
"""

import json
import os
import argparse
from pathlib import Path

from utilities.experiment_artifacts import (
    create_run_directory,
    initialize_run,
    mark_latest_completed_run,
    write_artifact_manifest,
    write_training_status,
)
from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


DEFAULT_CONFIG_FILE = Path("config.json")


def main(config_file: Path = DEFAULT_CONFIG_FILE) -> None:
    with config_file.open("r", encoding="utf-8") as file:
        source_config = json.load(file)

    parameters = Parameters.from_dict(source_config)
    output_root = str(Path(parameters.where_to_save).expanduser().resolve())
    run_directory = create_run_directory(
        output_root=output_root,
        method="base",
        seed=parameters.seed,
    )

    parameters.output_root = output_root
    parameters.run_id = run_directory.name
    parameters.artifact_logging_enabled = True
    parameters.where_to_save = str(run_directory) + os.sep

    initialize_run(
        run_directory=run_directory,
        source_config=source_config,
        resolved_config=dict(parameters.to_dict()),
    )

    try:
        mappo_cavs(parameters=parameters)
        write_training_status(
            run_directory,
            status="completed",
            iteration=parameters.n_iters,
        )
        write_artifact_manifest(run_directory)
        mark_latest_completed_run(output_root, run_directory)
    except BaseException as error:
        write_training_status(
            run_directory,
            status="failed",
            iteration=None,
            error=f"{type(error).__name__}: {error}",
        )
        write_artifact_manifest(run_directory)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SigmaRL Base-MAPPO.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Training configuration (default: config.json).",
    )
    arguments = parser.parse_args()
    main(arguments.config)
