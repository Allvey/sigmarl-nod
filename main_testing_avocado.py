"""Run the strict, non-learning A0-A2 AVOCADO validation environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from utilities.avocado.benchmark import run_benchmark
from utilities.avocado.config import AVOCADOExperimentConfig


DEFAULT_CONFIG_FILE = Path("configs/avocado/a2_strict_benchmark.json")


def main(
    config_file: Path = DEFAULT_CONFIG_FILE,
    *,
    output_directory: Optional[Path] = None,
    save_plots: bool = True,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    fail_on_gate: bool = True,
) -> None:
    config = AVOCADOExperimentConfig.from_json(config_file)
    result = run_benchmark(
        config,
        output_directory=output_directory,
        save_plots=save_plots,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
    )
    print("\nA2 strict AVOCADO benchmark")
    print(
        "case                         planner     "
        "success collision timeout min_clearance"
    )
    for item in result.metrics:
        print(
            f"{item.case:<28} {item.planner:<11} "
            f"{item.success_rate:>7.3f} {item.collision_rate:>9.3f} "
            f"{item.timeout_rate:>7.3f} {item.minimum_clearance_meters:>13.4f}"
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
            "Validate AVOCADO in its original holonomic-disc setting without "
            "MARL or the SigmaRL bicycle model."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write summary.json and trajectory plots to this exact directory.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the configured vectorized episode count (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the configured horizon (useful for smoke tests).",
    )
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="Return zero even when an A2 validation gate fails.",
    )
    arguments = parser.parse_args()
    main(
        arguments.config,
        output_directory=arguments.output_dir,
        save_plots=not arguments.no_plots,
        episodes_override=arguments.episodes,
        max_steps_override=arguments.max_steps,
        fail_on_gate=not arguments.allow_gate_failure,
    )
