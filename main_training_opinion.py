"""Validate the independent Opinion-MARL training configuration."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from utilities.opinion.config import (
    DEFAULT_OPINION_CONFIG_PATH,
    load_opinion_experiment_config,
)


NOT_IMPLEMENTED_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_OPINION_CONFIG_PATH,
        help="Opinion experiment JSON (default: config_opinion.json)",
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
    print(
        "[NOT IMPLEMENTED] M2-M5 configuration, math, ConflictGraph, and "
        "single-step policy components are available, but stateful collection "
        "and Opinion training integration start in later milestones.",
        file=sys.stderr,
    )
    return NOT_IMPLEMENTED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
