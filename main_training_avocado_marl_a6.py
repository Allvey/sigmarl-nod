"""Train A6 bounded y-correction with one-step truncated PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from utilities.avocado_marl.a6_trainer import train_a6


DEFAULT_CONFIG = Path("configs/avocado_marl/a6_y_correction.json")


def main(
    config: Path = DEFAULT_CONFIG,
    iterations: int = None,
    resume: Path = None,
) -> Path:
    run_directory = train_a6(
        config,
        iterations_override=iterations,
        resume_checkpoint=resume,
    )
    print(f"A6 run directory: {run_directory}")
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train only YCorrectionNet with one-step PPO."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional short-run override for smoke validation.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an A6 run from latest_checkpoint.pt.",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.iterations, arguments.resume)
