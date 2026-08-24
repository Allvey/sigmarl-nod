# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import argparse
from pathlib import Path
from typing import Mapping, Optional

from vmas.simulator.utils import save_video

from utilities.experiment_artifacts import resolve_latest_run
from utilities.helper_training import Parameters, SaveData
from utilities.constants import SCENARIOS
from utilities.mappo_cavs import mappo_cavs


DEFAULT_CONFIG_FILE = Path("config.json")


def _load_run_parameters(run_directory: Path) -> Parameters:
    resolved_config_path = run_directory / "config_resolved.json"
    if resolved_config_path.is_file():
        with resolved_config_path.open("r", encoding="utf-8") as file:
            return Parameters.from_dict(json.load(file))

    # Compatibility with an output produced before R1.
    legacy_json_files = sorted(run_directory.glob("reward*_data.json"))
    if not legacy_json_files:
        raise FileNotFoundError(
            f"No config_resolved.json or legacy reward data found in {run_directory}."
        )
    with legacy_json_files[-1].open("r", encoding="utf-8") as file:
        return SaveData.from_dict(json.load(file)).parameters


def test_base(
    output_root: str,
    run_directory: Optional[Path] = None,
    opinion_pair_info_config: Optional[Mapping[str, object]] = None,
) -> None:
    if run_directory is None:
        run_directory = resolve_latest_run(output_root)
    else:
        run_directory = run_directory.expanduser().resolve()
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_directory}")
    parameters = _load_run_parameters(run_directory)

    parameters.where_to_save = str(run_directory) + os.sep
    parameters.artifact_logging_enabled = False
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = True
    parameters.is_save_eval_results = False
    parameters.is_load_model = True
    parameters.is_load_final_model = True
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.max_steps = 1200  # 1200 steps correspond to one minute at dt=0.05.
    parameters.num_vmas_envs = 1

    # Evaluate on the saved training scenario by default. Cross-scenario
    # evaluation will use an explicit M10 evaluation configuration.
    parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
    parameters.is_save_simulation_video = False
    parameters.is_visualize_short_term_path = False
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = True

    env, policy, priority_module, parameters = mappo_cavs(
        parameters=parameters,
        opinion_pair_info_config=opinion_pair_info_config,
    )
    rollout_result = env.rollout(
        max_steps=parameters.max_steps - 1,
        policy=policy,
        priority_module=priority_module,
        callback=lambda env, _: env.render(
            mode="rgb_array", visualize_when_rgb=True
        ),
        auto_cast_to_device=True,
        break_when_any_done=False,
        is_save_simulation_video=parameters.is_save_simulation_video,
    )
    if parameters.is_save_simulation_video:
        out_td, frame_list = rollout_result
        save_video(str(run_directory / "video"), frame_list, fps=1 / parameters.dt)
    else:
        out_td = rollout_result


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    run_directory: Optional[Path] = None,
) -> None:
    source_parameters = Parameters.from_json(str(config_file))
    test_base(source_parameters.where_to_save, run_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a completed SigmaRL Base run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Configuration whose where_to_save selects the run (default: config.json).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Test this exact run directory instead of the latest completed run.",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.run_dir)
