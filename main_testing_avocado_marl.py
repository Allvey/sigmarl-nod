"""Run A4 Base-MAPPO + fixed AVOCADO in the native SigmaRL simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from utilities.avocado_marl.benchmark import (
    A4_SUPPORTED_PLANNERS,
    run_a4_benchmark,
    run_live_a4_case,
)
from utilities.avocado_marl.config import A4ExperimentConfig
from utilities.constants import SCENARIOS


DEFAULT_CONFIG_FILE = Path("configs/avocado_marl/a4_base_avocado.json")

# Select the scenario used by main_testing_avocado_marl.py in --render mode.
TEST_SCENARIO_TYPE = (
    "intersection_2"
    # "roundabout_1"
    # "CPM_entire"
    # "CPM_mixed"
    # "on_ramp_1"
)


def _print_metrics(metrics) -> None:
    bridge = metrics.bridge
    print(f"case: {metrics.case}")
    print(f"planner: {metrics.planner}")
    print(f"checkpoint: {metrics.checkpoint}")
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
        "nominal / executed speed: "
        f"{bridge.mean_nominal_speed_mps:.3f} / "
        f"{bridge.mean_executed_speed_mps:.3f} m/s"
    )
    print(
        "AVOCADO action intervention rate: "
        f"{100.0 * bridge.action_intervention_rate:.2f}%"
    )
    print(
        "no-conflict pass-through rate: "
        f"{100.0 * bridge.no_conflict_passthrough_rate:.2f}%"
    )
    print(
        "conflict intervention rate: "
        f"{100.0 * bridge.conflict_intervention_rate:.2f}%"
    )
    print(
        "nominal/executed speed correlation: "
        f"{bridge.nominal_executed_speed_correlation:.3f}"
    )
    print(
        "TTC shield intervention rate: "
        f"{100.0 * bridge.shield_intervention_rate:.2f}%"
    )


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    *,
    render_live: bool = False,
    scenario_type: Optional[str] = None,
    case_name: Optional[str] = None,
    planner: str = "base_mappo_avocado",
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    output_directory: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> None:
    config = A4ExperimentConfig.from_json(config_file)
    if render_live:
        selected_scenario = scenario_type
        if selected_scenario is None and case_name is None:
            selected_scenario = TEST_SCENARIO_TYPE
        print(
            "[INFO] Starting A4 real-time rendering: frozen Base-MAPPO "
            "+ fixed AVOCADO-KB."
        )
        metrics = run_live_a4_case(
            config,
            case_name=case_name,
            scenario_type=selected_scenario,
            planner=planner,
            run_directory=run_directory,
            checkpoint=checkpoint,
            max_steps_override=max_steps_override,
        )
        print("\nA4 real-time rollout complete")
        _print_metrics(metrics)
        return

    result = run_a4_benchmark(
        config,
        output_directory=output_directory,
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
    )
    print("\nA4 Base-MAPPO + AVOCADO benchmark")
    for metrics in result.metrics:
        print()
        _print_metrics(metrics)
    print(f"\nArtifacts: {result.output_directory}")
    print(f"Validation gate: {'PASSED' if result.passed else 'FAILED'}")
    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate A4 action-level MARL--AVOCADO coupling."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--render", action="store_true")
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--scenario", choices=tuple(SCENARIOS), default=None
    )
    environment.add_argument("--case", type=str, default=None)
    parser.add_argument(
        "--planner",
        choices=A4_SUPPORTED_PLANNERS,
        default="base_mappo_avocado",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    arguments = parser.parse_args()
    if arguments.render and arguments.episodes not in (None, 1):
        parser.error("--render requires --episodes 1 (or omit --episodes).")
    try:
        main(
            arguments.config,
            render_live=arguments.render,
            scenario_type=arguments.scenario,
            case_name=arguments.case,
            planner=arguments.planner,
            run_directory=arguments.run_dir,
            checkpoint=arguments.checkpoint,
            output_directory=arguments.output_dir,
            episodes_override=arguments.episodes,
            max_steps_override=arguments.max_steps,
        )
    except KeyboardInterrupt:
        print("\n[INFO] A4 visualization stopped by user.")
