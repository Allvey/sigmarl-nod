"""Validate the independent Opinion-MARL testing configuration."""

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from typing import Optional, Sequence

from utilities.opinion.config import (
    DEFAULT_OPINION_CONFIG_PATH,
    load_opinion_experiment_config,
    OPINION_STAGES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_OPINION_CONFIG_PATH,
        help="Opinion experiment JSON (default: config_opinion.json)",
    )
    parser.add_argument(
        "--stage",
        choices=OPINION_STAGES,
        default=None,
        help="override opinion_config.stage to match the checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Opinion checkpoint produced by main_training_opinion.py",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the resolved M2 configuration without rollout",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_opinion_experiment_config(args.config)
        if args.stage is not None:
            loaded = replace(loaded, opinion=replace(loaded.opinion, stage=args.stage))
    except Exception as error:
        print(f"[FAIL] Opinion testing configuration: {error}", file=sys.stderr)
        return 1

    summary = (
        f"stage={loaded.opinion.stage} rho_c={loaded.opinion.rho_c:g} "
        f"source={loaded.source_path}"
    )
    if args.validate_only:
        print(f"[PASS] Opinion testing configuration valid: {summary}")
        return 0

    if args.checkpoint is None:
        print("[FAIL] Opinion testing requires --checkpoint", file=sys.stderr)
        return 1
    print(f"[OK] Opinion testing configuration valid: {summary}")
    try:
        from utilities.opinion.evaluation import evaluate_opinion_checkpoint

        metrics = evaluate_opinion_checkpoint(
            loaded,
            checkpoint=args.checkpoint,
            steps=args.steps,
            smoke=args.smoke,
            output_path=args.output,
        )
    except Exception as error:
        print(f"[FAIL] Opinion testing: {error}", file=sys.stderr)
        return 1
    print(f"[PASS] Opinion testing metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
