"""Run a reproducible Base-MAPPO or frozen TSC training baseline."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from utilities.baseline_config import (
    BASELINE_NAMES,
    REPO_ROOT,
    load_baseline_config,
    materialize_metrics,
    validate_baseline_artifacts,
    write_resolved_config,
)
from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINE_NAMES, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a deterministic two-iteration CPU smoke without changing JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if Path.cwd().resolve() != REPO_ROOT.resolve():
            raise RuntimeError(
                f"run this command from the repository root: {REPO_ROOT}"
            )
        resolved = load_baseline_config(args.baseline, smoke=args.smoke)
        snapshot = write_resolved_config(resolved)
        os.environ["WANDB_MODE"] = "disabled"
        parameters = Parameters.from_dict(resolved)
        mappo_cavs(parameters=parameters)
        output_dir = Path(parameters.where_to_save)
        materialize_metrics(output_dir)
        summary = validate_baseline_artifacts(args.baseline, output_dir, resolved)
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print(f"[PASS] baseline={args.baseline} snapshot={snapshot}")
    print(f"[PASS] metrics={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
