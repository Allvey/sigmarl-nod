"""Typed configuration for A6 one-step y-correction PPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from utilities.avocado.config import (
    AVOCADOConfigError,
    _exact_keys,
    _integer,
    _number,
    _object,
)
from utilities.avocado_marl.a5_config import A5YCorrectionConfig
from utilities.avocado_marl.config import _string
from utilities.constants import SCENARIOS


@dataclass(frozen=True)
class A6BasePolicyConfig:
    run_directory: Path
    policy_checkpoint: Path
    critic_checkpoint: Path

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A6BasePolicyConfig":
        raw = _object(raw, "base_policy")
        _exact_keys(raw, set(cls.__dataclass_fields__), "base_policy")
        result = cls(
            run_directory=Path(_string(raw["run_directory"], "base_policy.run_directory")),
            policy_checkpoint=Path(
                _string(raw["policy_checkpoint"], "base_policy.policy_checkpoint")
            ),
            critic_checkpoint=Path(
                _string(raw["critic_checkpoint"], "base_policy.critic_checkpoint")
            ),
        )
        for value, label, kind in (
            (result.run_directory, "run_directory", "directory"),
            (result.policy_checkpoint, "policy_checkpoint", "file"),
            (result.critic_checkpoint, "critic_checkpoint", "file"),
        ):
            exists = value.is_dir() if kind == "directory" else value.is_file()
            if not exists:
                raise AVOCADOConfigError(
                    f"base_policy.{label} does not exist: {value}"
                )
        if result.policy_checkpoint.parent.resolve() != result.run_directory.resolve():
            raise AVOCADOConfigError(
                "base_policy.policy_checkpoint must belong to run_directory."
            )
        if result.critic_checkpoint.parent.resolve() != result.run_directory.resolve():
            raise AVOCADOConfigError(
                "base_policy.critic_checkpoint must belong to run_directory."
            )
        return result


@dataclass(frozen=True)
class A6ResidualConfig:
    opinion_scale: float
    gain: float
    maximum_absolute_residual: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A6ResidualConfig":
        raw = _object(raw, "opinion_residual")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion_residual")
        result = cls(
            opinion_scale=_number(raw["opinion_scale"], "opinion_residual.opinion_scale", strict=True),
            gain=_number(raw["gain"], "opinion_residual.gain", strict=True),
            maximum_absolute_residual=_number(
                raw["maximum_absolute_residual"],
                "opinion_residual.maximum_absolute_residual",
                strict=True,
            ),
        )
        if result.gain > result.maximum_absolute_residual:
            raise AVOCADOConfigError(
                "opinion_residual.gain must not exceed maximum_absolute_residual."
            )
        if result.maximum_absolute_residual > 1.0:
            raise AVOCADOConfigError(
                "opinion_residual.maximum_absolute_residual must not exceed 1."
            )
        return result


@dataclass(frozen=True)
class A6TrainingConfig:
    scenario_type: str
    n_agents: int
    seed: int
    iterations: int
    parallel_environments: int
    rollout_steps: int
    epochs: int
    minibatch_size: int
    y_learning_rate: float
    critic_learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coefficient: float
    correction_regularization: float
    saturation_regularization: float
    soft_fusion_limit: float
    maximum_gradient_norm: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A6TrainingConfig":
        raw = _object(raw, "training")
        _exact_keys(raw, set(cls.__dataclass_fields__), "training")
        result = cls(
            scenario_type=_string(raw["scenario_type"], "training.scenario_type"),
            n_agents=_integer(raw["n_agents"], "training.n_agents"),
            seed=_integer(raw["seed"], "training.seed", minimum=0),
            iterations=_integer(raw["iterations"], "training.iterations"),
            parallel_environments=_integer(
                raw["parallel_environments"], "training.parallel_environments"
            ),
            rollout_steps=_integer(raw["rollout_steps"], "training.rollout_steps"),
            epochs=_integer(raw["epochs"], "training.epochs"),
            minibatch_size=_integer(raw["minibatch_size"], "training.minibatch_size"),
            y_learning_rate=_number(
                raw["y_learning_rate"], "training.y_learning_rate", strict=True
            ),
            critic_learning_rate=_number(
                raw["critic_learning_rate"],
                "training.critic_learning_rate",
                strict=True,
            ),
            gamma=_number(raw["gamma"], "training.gamma", strict=True),
            gae_lambda=_number(
                raw["gae_lambda"], "training.gae_lambda", strict=True
            ),
            clip_epsilon=_number(
                raw["clip_epsilon"], "training.clip_epsilon", strict=True
            ),
            entropy_coefficient=_number(
                raw["entropy_coefficient"], "training.entropy_coefficient"
            ),
            correction_regularization=_number(
                raw["correction_regularization"],
                "training.correction_regularization",
            ),
            saturation_regularization=_number(
                raw["saturation_regularization"],
                "training.saturation_regularization",
            ),
            soft_fusion_limit=_number(
                raw["soft_fusion_limit"], "training.soft_fusion_limit", strict=True
            ),
            maximum_gradient_norm=_number(
                raw["maximum_gradient_norm"],
                "training.maximum_gradient_norm",
                strict=True,
            ),
        )
        if not 0.0 < result.gamma <= 1.0:
            raise AVOCADOConfigError("training.gamma must be in (0, 1].")
        if not 0.0 < result.gae_lambda <= 1.0:
            raise AVOCADOConfigError("training.gae_lambda must be in (0, 1].")
        if not 0.0 < result.clip_epsilon < 1.0:
            raise AVOCADOConfigError("training.clip_epsilon must be in (0, 1).")
        if not 0.0 < result.soft_fusion_limit <= 1.0:
            raise AVOCADOConfigError("training.soft_fusion_limit must be in (0, 1].")
        if result.scenario_type not in SCENARIOS:
            raise AVOCADOConfigError(
                f"Unknown training.scenario_type: {result.scenario_type}"
            )
        expected_agents = int(SCENARIOS[result.scenario_type]["n_agents"])
        if result.n_agents != expected_agents:
            raise AVOCADOConfigError(
                "training.n_agents must match the selected scenario: "
                f"expected {expected_agents}, got {result.n_agents}."
            )
        frame_count = result.parallel_environments * result.rollout_steps
        if result.minibatch_size > frame_count:
            raise AVOCADOConfigError(
                "training.minibatch_size must not exceed one rollout batch."
            )
        return result


@dataclass(frozen=True)
class A6ExperimentConfig:
    a5_config: Path
    output_root: str
    base_policy: A6BasePolicyConfig
    y_correction: A5YCorrectionConfig
    opinion_residual: A6ResidualConfig
    training: A6TrainingConfig

    @classmethod
    def from_json(cls, path: Path) -> "A6ExperimentConfig":
        with path.open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "a5_config",
            "output_root",
            "base_policy",
            "y_correction",
            "opinion_residual",
            "training",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != 1:
            raise AVOCADOConfigError("A6 schema_version must be 1.")
        if raw["method"] != "avocado_marl" or raw["stage"] != "a6":
            raise AVOCADOConfigError(
                "Expected method='avocado_marl' and stage='a6'."
            )
        a5_config = Path(_string(raw["a5_config"], "a5_config"))
        if not a5_config.is_file():
            raise AVOCADOConfigError(f"A5 config does not exist: {a5_config}")
        correction = A5YCorrectionConfig.from_dict(
            raw["y_correction"], require_frozen_zero=False
        )
        if correction.strict_zero or correction.freeze:
            raise AVOCADOConfigError(
                "A6 requires y_correction.strict_zero=false and freeze=false."
            )
        return cls(
            a5_config=a5_config,
            output_root=_string(raw["output_root"], "output_root"),
            base_policy=A6BasePolicyConfig.from_dict(raw["base_policy"]),
            y_correction=correction,
            opinion_residual=A6ResidualConfig.from_dict(raw["opinion_residual"]),
            training=A6TrainingConfig.from_dict(raw["training"]),
        )
