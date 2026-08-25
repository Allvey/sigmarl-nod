"""Standard Base-MAPPO training entry point.

R1 keeps the original SigmaRL 1.2.0 optimizer path and adds only isolated run
directories plus reproducible artifacts around it.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from utilities.experiment_artifacts import (
    atomic_write_json,
    create_run_directory,
    initialize_run,
    mark_latest_completed_run,
    write_artifact_manifest,
    write_training_status,
)
from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


DEFAULT_CONFIG_FILE = Path("config.json")


def train_base(
    parameters: Parameters,
    source_config: Mapping[str, Any],
    run_label: str = "base",
    supplementary_snapshots: Optional[Mapping[str, Mapping[str, Any]]] = None,
    comparison_payload: Optional[Mapping[str, Any]] = None,
    opinion_pair_info_config: Optional[Mapping[str, object]] = None,
    opinion_policy_config: Optional[Mapping[str, object]] = None,
    artifact_method: str = "base_mappo",
    artifact_stage: str = "base",
    resume_checkpoint: Optional[Path] = None,
) -> Path:
    """Run Base-MAPPO, optionally with the staged M4/M5 Opinion side paths."""

    output_root = str(Path(parameters.where_to_save).expanduser().resolve())
    if resume_checkpoint is None:
        run_directory = create_run_directory(
            output_root=output_root,
            method=run_label,
            seed=parameters.seed,
        )
    else:
        resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        run_directory = resume_checkpoint.parent
        expected_runs_root = Path(output_root).expanduser().resolve() / "runs"
        if run_directory.parent != expected_runs_root:
            raise ValueError(
                "Resume checkpoint must belong directly to a run under the "
                f"configured output root: {expected_runs_root}"
            )

    parameters.output_root = output_root
    parameters.run_id = run_directory.name
    parameters.artifact_logging_enabled = True
    parameters.where_to_save = str(run_directory) + os.sep

    if resume_checkpoint is None:
        initialize_run(
            run_directory=run_directory,
            source_config=dict(source_config),
            resolved_config=dict(parameters.to_dict()),
            method=artifact_method,
            stage=artifact_stage,
        )
    else:
        write_training_status(
            run_directory,
            status="running",
            iteration=None,
        )
    try:
        if resume_checkpoint is None:
            for filename, payload in (supplementary_snapshots or {}).items():
                if Path(filename).name != filename or not filename.endswith(".json"):
                    raise ValueError(
                        "Supplementary snapshot names must be plain .json filenames."
                    )
                atomic_write_json(run_directory / filename, dict(payload))

        mappo_cavs(
            parameters=parameters,
            opinion_pair_info_config=opinion_pair_info_config,
            opinion_policy_config=opinion_policy_config,
            artifact_method=artifact_method,
            artifact_stage=artifact_stage,
            training_resume_checkpoint=resume_checkpoint,
        )
        write_training_status(
            run_directory,
            status="completed",
            iteration=parameters.n_iters,
        )
        if comparison_payload is not None:
            atomic_write_json(
                run_directory / "comparison_to_base.json",
                dict(comparison_payload),
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

    return run_directory


def main(config_file: Path = DEFAULT_CONFIG_FILE) -> Path:
    with config_file.open("r", encoding="utf-8") as file:
        source_config: Dict[str, Any] = json.load(file)

    parameters = Parameters.from_dict(source_config)
    return train_base(parameters=parameters, source_config=source_config)


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
