"""Validate the independent Opinion-MARL training configuration."""

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
        help="override opinion_config.stage without editing JSON",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run two tiny CPU training iterations and save a test checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the Opinion output directory",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="initialize this stage from a previous Opinion checkpoint",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the resolved M2 configuration without training",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_opinion_experiment_config(args.config)
        if args.stage is not None:
            loaded = replace(loaded, opinion=replace(loaded.opinion, stage=args.stage))
    except Exception as error:
        print(f"[FAIL] Opinion training configuration: {error}", file=sys.stderr)
        return 1

    summary = (
        f"stage={loaded.opinion.stage} rho_c={loaded.opinion.rho_c:g} "
        f"source={loaded.source_path}"
    )
    if args.validate_only:
        print(f"[PASS] Opinion training configuration valid: {summary}")
        return 0

    print(f"[OK] Opinion training configuration valid: {summary}")
    trainer = None
    try:
        from utilities.opinion.trainer import build_opinion_trainer

        trainer = build_opinion_trainer(
            loaded, smoke=args.smoke, output_dir=args.output_dir
        )
        if args.resume is not None:
            trainer.load_stage_weights(args.resume)
        checkpoint = trainer.fit()
    except Exception as error:
        print(f"[FAIL] Opinion training: {error}", file=sys.stderr)
        return 1
    finally:
        if trainer is not None:
            trainer.env.close()
    print(f"[PASS] Opinion training checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
