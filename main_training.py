"""Standard Base-MAPPO training entry point.

R1 keeps the original SigmaRL 1.2.0 optimizer path and adds only isolated run
directories plus reproducible artifacts around it.
"""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from utilities.experiment_artifacts import (
    atomic_write_json,
    create_run_directory,
    initialize_run,
    mark_latest_completed_run,
    write_artifact_manifest,
    write_training_status,
)

if TYPE_CHECKING:
    from utilities.helper_training import Parameters


DEFAULT_CONFIG_FILE = Path("config.json")


def train_base(
    parameters: Parameters,
    source_config: Mapping[str, Any],
    run_label: str = "base",
    supplementary_snapshots: Optional[Mapping[str, Mapping[str, Any]]] = None,
    comparison_payload: Optional[Mapping[str, Any]] = None,
    opinion_pair_info_config: Optional[Mapping[str, object]] = None,
    opinion_policy_config: Optional[Mapping[str, object]] = None,
    psb_runtime_config: Optional[Mapping[str, object]] = None,
    artifact_method: str = "base_mappo",
    artifact_stage: str = "base",
    resume_checkpoint: Optional[Path] = None,
) -> Path:
    """Run Base-MAPPO, optionally with the staged M4/M5 Opinion side paths."""

    # Keep configuration dispatch and PSB P0 packaging importable without the
    # optional VMAS runtime. Base training still loads the identical function
    # immediately before it is needed.
    from utilities.mappo_cavs import mappo_cavs

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
            psb_runtime_config=psb_runtime_config,
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


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    resume_checkpoint: Optional[Path] = None,
    iterations_override: Optional[int] = None,
) -> Path:
    with config_file.open("r", encoding="utf-8") as file:
        source_config: Dict[str, Any] = json.load(file)

    method = source_config.get("method")
    if method == "psb_marl":
        # Keep the standard Base entry independent from PSB imports unless a
        # PSB configuration explicitly selects the new path.
        from utilities.psb_marl.trainer import train_psb

        return train_psb(
            config_file,
            resume_checkpoint=resume_checkpoint,
            iterations_override=iterations_override,
        )
    if method == "opinion_marl":
        if iterations_override is not None:
            raise ValueError(
                "--iterations is not supported by the Opinion-MARL dispatcher."
            )
        from main_training_opinion import main as train_opinion

        return train_opinion(config_file, resume_checkpoint)
    if method not in (None, "base_mappo"):
        raise ValueError(f"Unsupported training method: {method!r}")

    if method == "base_mappo":
        source_config = {
            key: value for key, value in source_config.items() if key != "method"
        }
    from utilities.helper_training import Parameters

    parameters = Parameters.from_dict(source_config)
    if iterations_override is not None:
        if type(iterations_override) is not int or iterations_override <= 0:
            raise ValueError("--iterations must be a positive integer.")
        parameters.n_iters = iterations_override
        parameters.total_frames = parameters.frames_per_batch * iterations_override
    return train_base(
        parameters=parameters,
        source_config=source_config,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SigmaRL Base, Opinion-MARL, or PSB-MARL."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Base, Opinion-MARL, or PSB-MARL configuration.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional method-specific training checkpoint to resume.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional short-run iteration override when the stage supports it.",
    )
    arguments = parser.parse_args()
    run_directory = main(
        arguments.config,
        arguments.resume,
        arguments.iterations,
    )
    print(f"[INFO] Run directory: {run_directory}")
