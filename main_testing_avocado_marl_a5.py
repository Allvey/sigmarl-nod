"""Validate A5 strict-zero YCorrectionNet against A4 step by step."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from utilities.avocado_marl.a5_benchmark import (
    run_a5_validation,
    run_live_a5_case,
)
from utilities.avocado_marl.a5_config import A5ExperimentConfig
from utilities.constants import SCENARIOS


DEFAULT_CONFIG_FILE = Path("configs/avocado_marl/a5_zero_correction.json")

TEST_SCENARIO_TYPE = (
    "intersection_2"
    # "roundabout_1"
    # "CPM_entire"
    # "CPM_mixed"
    # "on_ramp_1"
)


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    *,
    render_live: bool = False,
    scenario_type: Optional[str] = None,
    case_name: Optional[str] = None,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    output_directory: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> None:
    config = A5ExperimentConfig.from_json(config_file)
    if render_live:
        selected_scenario = scenario_type
        if selected_scenario is None and case_name is None:
            selected_scenario = TEST_SCENARIO_TYPE
        result = run_live_a5_case(
            config,
            case_name=case_name,
            scenario_type=selected_scenario,
            run_directory=run_directory,
            checkpoint=checkpoint,
            max_steps_override=max_steps_override,
        )
        print("\nA5 real-time rollout complete")
        print(f"case: {result.metrics.case}")
        print(f"steps: {result.metrics.executed_steps}")
        print(
            "maximum |Delta y|: "
            f"{result.diagnostics.maximum_absolute_correction:.9f}"
        )
        print(
            "maximum |y_fused - y_heuristic|: "
            f"{result.diagnostics.maximum_fusion_error:.9f}"
        )
        return

    result = run_a5_validation(
        config,
        output_directory=output_directory,
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
    )
    print("\nA5 strict-zero equivalence validation")
    for case_result in result.results:
        print(
            f"{case_result.case}: "
            f"{'PASSED' if case_result.passed else 'FAILED'}"
        )
        for name, difference in case_result.maximum_differences.items():
            print(f"  max diff {name}: {difference:.9g}")
        for failure in case_result.failures:
            print(f"  [FAIL] {failure}")
    print(f"Artifacts: {result.output_directory}")
    print(f"Validation gate: {'PASSED' if result.passed else 'FAILED'}")
    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate A5 zero-correction equivalence with A4."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--render", action="store_true")
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--scenario", choices=tuple(SCENARIOS), default=None
    )
    environment.add_argument("--case", type=str, default=None)
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
            run_directory=arguments.run_dir,
            checkpoint=arguments.checkpoint,
            output_directory=arguments.output_dir,
            episodes_override=arguments.episodes,
            max_steps_override=arguments.max_steps,
        )
    except KeyboardInterrupt:
        print("\n[INFO] A5 visualization stopped by user.")
