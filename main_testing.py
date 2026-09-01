# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence

from utilities.experiment_artifacts import (
    resolve_latest_testable_run,
    resolve_policy_checkpoint,
)
from utilities.constants import SCENARIOS


DEFAULT_CONFIG_FILE = Path("config.json")


def _load_run_parameters(run_directory: Path):
    from utilities.helper_training import Parameters, SaveData

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
    checkpoint_path: Optional[Path] = None,
    opinion_pair_info_config: Optional[Mapping[str, object]] = None,
    opinion_policy_config: Optional[Mapping[str, object]] = None,
    opinion_visualization_config: Optional[Mapping[str, object]] = None,
    psb_runtime_config: Optional[Mapping[str, object]] = None,
    psb_action_projection: Optional[str] = None,
    save_simulation_video: bool = False,
    scenario_type: Optional[str] = None,
    max_steps: int = 1200,
    episodes: int = 1,
    seed: Optional[int] = None,
    render: bool = True,
):
    # Import simulator-heavy modules only for an actual rollout. This keeps
    # config dispatch, --help, and P0 filesystem checks lightweight.
    from utilities.mappo_cavs import mappo_cavs

    if type(max_steps) is not int or max_steps <= 1:
        raise ValueError("max_steps must be an integer greater than 1.")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer.")
    if render and episodes != 1:
        raise ValueError("Live rendering requires exactly one environment.")
    if checkpoint_path is not None and run_directory is None:
        run_directory = checkpoint_path.expanduser().resolve().parent
    elif run_directory is None:
        run_directory = resolve_latest_testable_run(output_root)
    else:
        run_directory = run_directory.expanduser().resolve()
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_directory}")
    checkpoint_path = resolve_policy_checkpoint(run_directory, checkpoint_path)
    print(f"[INFO] Testing run: {run_directory}")
    print(f"[INFO] Testing policy checkpoint: {checkpoint_path}")
    parameters = _load_run_parameters(run_directory)
    if scenario_type is not None:
        if scenario_type not in SCENARIOS:
            raise ValueError(f"Unknown testing scenario_type: {scenario_type}")
        parameters.scenario_type = scenario_type

    parameters.where_to_save = str(run_directory) + os.sep
    parameters.artifact_logging_enabled = False
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = bool(render)
    parameters.is_save_eval_results = False
    parameters.is_load_model = True
    parameters.is_load_final_model = True
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.max_steps = max_steps
    parameters.num_vmas_envs = episodes
    if seed is not None:
        parameters.seed = seed

    # Evaluate on the saved training scenario by default. Cross-scenario
    # evaluation will use an explicit M10 evaluation configuration.
    parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
    parameters.is_save_simulation_video = bool(save_simulation_video)
    parameters.is_visualize_short_term_path = False
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = True

    env, policy, priority_module, parameters = mappo_cavs(
        parameters=parameters,
        opinion_pair_info_config=opinion_pair_info_config,
        opinion_policy_config=opinion_policy_config,
        psb_runtime_config=psb_runtime_config,
        psb_action_projection=psb_action_projection,
        policy_checkpoint_path=checkpoint_path,
    )

    def render_callback(render_env, tensordict):
        if opinion_visualization_config is not None:
            from utilities.opinion.visualization import (
                update_opinion_visualization,
            )

            update_opinion_visualization(
                render_env,
                tensordict,
                opinion_visualization_config,
            )
        return render_env.render(mode="rgb_array", visualize_when_rgb=True)

    callback = (
        render_callback
        if render or parameters.is_save_simulation_video
        else None
    )
    # Policy construction can initialize a different number of modules for
    # Base and PSB.  Re-seed immediately before rollout so paired evaluation
    # uses common environment and action-sampling randomness rather than RNG
    # offsets caused by network construction.
    import random

    import numpy as np
    import torch

    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parameters.seed)
    try:
        rollout_result = env.rollout(
            max_steps=parameters.max_steps - 1,
            policy=policy,
            priority_module=priority_module,
            callback=callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=parameters.is_save_simulation_video,
        )
        if parameters.is_save_simulation_video:
            from vmas.simulator.utils import save_video

            out_td, frame_list = rollout_result
            video_path = run_directory / "video"
            save_video(str(video_path), frame_list, fps=1 / parameters.dt)
            print(f"[INFO] Saved simulation video: {video_path}.mp4")
        else:
            out_td = rollout_result
        return out_td
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    run_directory: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    *,
    scenario_type: Optional[str] = None,
    max_steps: int = 1200,
    episodes: int = 1,
    seeds: Optional[Sequence[int]] = None,
    render: bool = True,
    save_simulation_video: bool = False,
    compare_base: bool = False,
    promote_if_noninferior: bool = False,
    psb_action_projection: Optional[str] = None,
    psb_report_label: Optional[str] = None,
):
    with config_file.open("r", encoding="utf-8") as stream:
        source_config = json.load(stream)
    method = source_config.get("method")
    if method == "psb_marl":
        from utilities.psb_marl.evaluator import test_psb

        return test_psb(
            config_file,
            run_directory=run_directory,
            checkpoint_path=checkpoint_path,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seeds=seeds,
            render=render,
            save_simulation_video=save_simulation_video,
            compare_base=compare_base,
            promote_if_noninferior=promote_if_noninferior,
            psb_action_projection=psb_action_projection,
            report_label=psb_report_label,
        )
    if method == "opinion_marl":
        if (
            scenario_type is not None
            or max_steps != 1200
            or episodes != 1
            or seeds is not None
            or not render
            or compare_base
            or promote_if_noninferior
            or psb_action_projection is not None
            or psb_report_label is not None
        ):
            raise ValueError(
                "Advanced unified testing options are not yet supported by the "
                "legacy Opinion-MARL entrypoint."
            )
        from main_testing_opinion import main as test_opinion

        return test_opinion(
            config_file,
            run_directory,
            checkpoint_path,
            save_simulation_video,
        )
    if method not in (None, "base_mappo"):
        raise ValueError(f"Unsupported testing method: {method!r}")
    if method == "base_mappo":
        source_config = {
            key: value for key, value in source_config.items() if key != "method"
        }
    if compare_base or promote_if_noninferior or psb_action_projection is not None:
        raise ValueError(
            "PSB comparison, promotion, and action projection options apply "
            "only to PSB runs."
        )
    from utilities.helper_training import Parameters

    source_parameters = Parameters.from_dict(source_config)
    selected_seeds = tuple(seeds) if seeds is not None else (source_parameters.seed,)
    if render and (episodes != 1 or len(selected_seeds) != 1):
        raise ValueError("Rendering requires one episode and one seed.")
    if save_simulation_video and (episodes != 1 or len(selected_seeds) != 1):
        raise ValueError("Video capture requires one episode and one seed.")
    outputs = []
    for selected_seed in selected_seeds:
        outputs.append(
            test_base(
                source_parameters.where_to_save,
                run_directory,
                checkpoint_path,
                save_simulation_video=save_simulation_video,
                scenario_type=scenario_type,
                max_steps=max_steps,
                episodes=episodes,
                seed=selected_seed,
                render=render,
            )
        )
    return outputs[0] if len(outputs) == 1 else outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a Base, Opinion-MARL, or PSB-MARL policy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Base, Opinion-MARL, or PSB-MARL configuration.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Test this exact run directory instead of auto-resolving a run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Load this exact final_policy.pth, candidate_policy.pth, or "
            "reward<value>_policy.pth. "
            "Its parent directory is used as --run-dir when --run-dir is omitted."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default=None,
        help="Override the saved training scenario.",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more non-negative environment seeds.",
    )
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", dest="render", action="store_true")
    render_group.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="For PSB, verify or evaluate against its recorded Base source.",
    )
    parser.add_argument(
        "--promote-if-noninferior",
        action="store_true",
        help="For supported PSB stages, apply the configured promotion gate.",
    )
    parser.add_argument(
        "--psb-action-projection",
        choices=("full", "longitudinal_only"),
        default=None,
        help=(
            "Inference-only P2 counterfactual. longitudinal_only preserves "
            "the Base steering mean and applies only the learned speed correction."
        ),
    )
    parser.add_argument(
        "--psb-report-label",
        default=None,
        help=(
            "Optional safe suffix for an independent PSB evaluation report; "
            "it does not change policy or evaluation semantics."
        ),
    )
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.run_dir,
        arguments.checkpoint,
        scenario_type=arguments.scenario,
        max_steps=arguments.max_steps,
        episodes=arguments.episodes,
        seeds=arguments.seeds,
        render=arguments.render,
        save_simulation_video=arguments.save_video,
        compare_base=arguments.compare_base,
        promote_if_noninferior=arguments.promote_if_noninferior,
        psb_action_projection=arguments.psb_action_projection,
        psb_report_label=arguments.psb_report_label,
    )
