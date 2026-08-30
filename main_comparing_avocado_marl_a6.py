"""Run paired multi-environment-seed A5/A6 checkpoint evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from utilities.avocado_marl.a6_benchmark import run_a5_a6_comparison
from utilities.constants import SCENARIOS


DEFAULT_CONFIG = Path("configs/avocado_marl/a6_comparison.json")


def main(
    config: Path = DEFAULT_CONFIG,
    *,
    checkpoint: Path = None,
    output_directory: Path = None,
    scenarios=None,
    seeds=None,
    parallel_environments: int = None,
    max_steps: int = None,
) -> Path:
    result = run_a5_a6_comparison(
        config,
        checkpoint=checkpoint,
        output_directory=output_directory,
        scenarios_override=scenarios,
        seeds_override=seeds,
        parallel_environments_override=parallel_environments,
        max_steps_override=max_steps,
    )
    print(f"A5/A6 comparison artifacts: {result}")
    print(f"Markdown report: {result / 'report.md'}")
    print(f"Machine-readable summary: {result / 'summary.json'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare A5 and one A6 checkpoint with paired scenarios, "
            "environment seeds, and budgets."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=None,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Paired environment seeds used for both A5 and the selected A6 checkpoint.",
    )
    parser.add_argument("--parallel-envs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    arguments = parser.parse_args()
    main(
        arguments.config,
        checkpoint=arguments.checkpoint,
        output_directory=arguments.output_dir,
        scenarios=arguments.scenarios,
        seeds=arguments.seeds,
        parallel_environments=arguments.parallel_envs,
        max_steps=arguments.max_steps,
    )
