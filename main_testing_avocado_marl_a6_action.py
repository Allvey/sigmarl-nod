"""Evaluate an A6-Action checkpoint deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utilities.avocado_marl.a6_action_config import A6ActionExperimentConfig
from utilities.avocado_marl.a6_action_trainer import (
    evaluate_a6_action,
    resolve_latest_a6_action_checkpoint,
    verify_a6_action_zero_equivalence,
)
from utilities.constants import SCENARIOS


DEFAULT_CONFIG = Path("configs/avocado_marl/a6_action.json")


def main(
    config_path: Path = DEFAULT_CONFIG,
    *,
    checkpoint: Path = None,
    max_steps: int = None,
    render_live: bool = False,
    scenario_type: str = None,
    seed: int = None,
    verify_zero: bool = False,
):
    if verify_zero:
        result = verify_a6_action_zero_equivalence(
            config_path,
            steps=4 if max_steps is None else max_steps,
            scenario_type=scenario_type,
            seed=seed,
        ).to_dict()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["passed"]:
            raise RuntimeError("A6-Action zero-equivalence verification failed.")
        return result
    config = A6ActionExperimentConfig.from_json(config_path)
    if checkpoint is None:
        checkpoint = resolve_latest_a6_action_checkpoint(config.output_root)
    result = evaluate_a6_action(
        config_path,
        checkpoint,
        max_steps=max_steps,
        render_live=render_live,
        scenario_type=scenario_type,
        seed=seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained A6-Action checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--verify-zero",
        action="store_true",
        help="Verify the untrained A6-Action closed loop is exactly A5.",
    )
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic evaluation-seed override.",
    )
    arguments = parser.parse_args()
    try:
        main(
            arguments.config,
            checkpoint=arguments.checkpoint,
            max_steps=arguments.max_steps,
            render_live=arguments.render,
            scenario_type=arguments.scenario,
            seed=arguments.seed,
            verify_zero=arguments.verify_zero,
        )
    except KeyboardInterrupt:
        print("\n[INFO] A6-Action visualization stopped by user.")
