"""Evaluate a trained A6 y-correction checkpoint deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utilities.avocado_marl.a6_config import A6ExperimentConfig
from utilities.avocado_marl.a6_trainer import (
    evaluate_a6,
    resolve_latest_a6_checkpoint,
)
from utilities.constants import SCENARIOS


DEFAULT_CONFIG = Path("configs/avocado_marl/a6_y_correction.json")


def main(
    config_path: Path = DEFAULT_CONFIG,
    *,
    checkpoint: Path = None,
    max_steps: int = None,
    render_live: bool = False,
    scenario_type: str = None,
):
    config = A6ExperimentConfig.from_json(config_path)
    if checkpoint is None:
        checkpoint = resolve_latest_a6_checkpoint(config.output_root)
    result = evaluate_a6(
        config_path,
        checkpoint,
        max_steps=max_steps,
        render_live=render_live,
        scenario_type=scenario_type,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an A6 checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default=None)
    arguments = parser.parse_args()
    try:
        main(
            arguments.config,
            checkpoint=arguments.checkpoint,
            max_steps=arguments.max_steps,
            render_live=arguments.render,
            scenario_type=arguments.scenario,
        )
    except KeyboardInterrupt:
        print("\n[INFO] A6 visualization stopped by user.")
