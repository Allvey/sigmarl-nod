"""Run A3 AVOCADO-KB directly in SigmaRL's road-traffic environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from utilities.avocado.road_benchmark import (
    run_live_road_case,
    run_road_benchmark,
)
from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.constants import SCENARIOS


DEFAULT_CONFIG_FILE = Path("configs/avocado/a3_road_environment.json")

# Select the scenario used by main_testing_avocado_kb.py in --render mode.
TEST_SCENARIO_TYPE = (
    # "intersection_2"
    # "roundabout_1"
    # "CPM_entire"
    # "CPM_mixed"
    "on_ramp_1"
    # roundabout_1, intersection_1/2/3, CPM_mixed
)


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    *,
    output_directory: Optional[Path] = None,
    save_plots: bool = True,
    save_videos: bool = False,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    fail_on_gate: bool = True,
    render_live: bool = False,
    render_case: Optional[str] = None,
    render_scenario_type: Optional[str] = None,
    render_planner: str = "avocado_kb",
) -> None:
    config = A3RoadExperimentConfig.from_json(config_file)
    if render_live:
        if episodes_override not in (None, 1):
            raise ValueError("Live rendering supports exactly one environment.")
        if render_case is not None and render_scenario_type is not None:
            raise ValueError(
                "Select either render_case or render_scenario_type, not both."
            )
        selected_scenario = render_scenario_type
        if selected_scenario is None and render_case is None:
            selected_scenario = TEST_SCENARIO_TYPE
        selected_environment = render_case or selected_scenario
        print(
            "[INFO] Starting native VMAS real-time rendering "
            f"({render_planner}, {selected_environment})."
        )
        print("[INFO] Close the viewer or press Ctrl+C to stop early.")
        metrics = run_live_road_case(
            config,
            case_name=render_case,
            scenario_type=selected_scenario,
            planner=render_planner,
            max_steps_override=max_steps_override,
        )
        print("\nA3 real-time rollout complete")
        print(f"case: {metrics.case}")
        print(f"planner: {metrics.planner}")
        print(f"steps: {metrics.executed_steps}")
        print(
            "agent collisions / 1000 steps: "
            f"{metrics.agent_collision_events_per_1000_steps:.3f}"
        )
        print(
            "route completions / 1000 steps: "
            f"{metrics.route_completion_events_per_1000_steps:.3f}"
        )
        print(
            "reference error mean / p95: "
            f"{metrics.mean_reference_distance_meters:.3f} / "
            f"{metrics.p95_reference_distance_meters:.3f} m"
        )
        print(
            "TTC shield intervention rate: "
            f"{100.0 * metrics.shield_intervention_rate:.2f}%"
        )
        return
    result = run_road_benchmark(
        config,
        output_directory=output_directory,
        save_plots=save_plots,
        save_videos=save_videos,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
    )
    print("\nA3 AVOCADO-KB road-environment benchmark")
    print(
        "case                    planner        agent_col lane_col route_done "
        "ref_p95 vel_err steer_sat shield%"
    )
    for item in result.metrics:
        print(
            f"{item.case:<23} {item.planner:<14} "
            f"{item.agent_collision_events_per_1000_steps:>9.3f} "
            f"{item.lane_collision_events_per_1000_steps:>8.3f} "
            f"{item.route_completion_events_per_1000_steps:>10.3f} "
            f"{item.p95_reference_distance_meters:>8.3f} "
            f"{item.mean_tracking_error_mps:>9.3f} "
            f"{item.steering_saturation_rate:>9.3f} "
            f"{100.0 * item.shield_intervention_rate:>7.2f}"
        )
    print(f"\nArtifacts: {result.output_directory}")
    if result.validation.passed:
        print("Validation gate: PASSED")
        for check in result.validation.checks:
            print(f"  [PASS] {check}")
    else:
        print("Validation gate: FAILED")
        for failure in result.validation.failures:
            print(f"  [FAIL] {failure}")
        if fail_on_gate:
            raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic AVOCADO-KB in the same road_traffic "
            "scenario and kinematic bicycle dynamics used by MARL."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--video",
        action="store_true",
        help=(
            "Save an AVOCADO-KB animation per road case (MP4 with FFmpeg, "
            "otherwise GIF)."
        ),
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Open the native VMAS viewer and execute one environment at "
            "dt=0.05 s, like main_testing.py."
        ),
    )
    environment_group = parser.add_mutually_exclusive_group()
    environment_group.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default=None,
        help=(
            "Scenario type used by --render. If omitted, use the "
            "TEST_SCENARIO_TYPE constant."
        ),
    )
    environment_group.add_argument(
        "--case",
        type=str,
        default=None,
        help="Use an exact case from the A3 JSON instead of TEST_SCENARIO_TYPE.",
    )
    parser.add_argument(
        "--planner",
        choices=("path_following", "orca_kb", "avocado_kb"),
        default="avocado_kb",
        help="Controller used by --render (default: avocado_kb).",
    )
    parser.add_argument("--allow-gate-failure", action="store_true")
    arguments = parser.parse_args()
    if arguments.render and arguments.video:
        parser.error("--render and --video are separate modes; choose one.")
    if arguments.render and arguments.episodes not in (None, 1):
        parser.error("--render requires --episodes 1 (or omit --episodes).")
    try:
        main(
            arguments.config,
            output_directory=arguments.output_dir,
            save_plots=not arguments.no_plots,
            save_videos=arguments.video,
            episodes_override=arguments.episodes,
            max_steps_override=arguments.max_steps,
            fail_on_gate=not arguments.allow_gate_failure,
            render_live=arguments.render,
            render_case=arguments.case,
            render_scenario_type=arguments.scenario,
            render_planner=arguments.planner,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Real-time visualization stopped by user.")
