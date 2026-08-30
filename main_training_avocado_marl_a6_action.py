"""Train the A6-Action preferred-action correction with one-step PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from utilities.avocado_marl.a6_action_trainer import train_a6_action


DEFAULT_CONFIG = Path("configs/avocado_marl/a6_action.json")


def main(
    config: Path = DEFAULT_CONFIG,
    iterations: int = None,
    resume: Path = None,
) -> Path:
    run_directory = train_a6_action(
        config,
        iterations_override=iterations,
        resume_checkpoint=resume,
    )
    print(f"A6-Action run directory: {run_directory}")
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train an interaction-conditioned preferred-action correction "
            "while keeping Base Actor and AVOCADO safety fixed."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional training-iteration override.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an A6-Action latest_checkpoint.pt.",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.iterations, arguments.resume)
