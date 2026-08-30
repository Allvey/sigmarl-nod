"""Typed configuration for A6-Action preferred-action PPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from utilities.avocado.config import (
    AVOCADOConfigError,
    _exact_keys,
    _integer,
    _number,
    _object,
)
from utilities.avocado_marl.a6_config import A6BasePolicyConfig
from utilities.avocado_marl.config import _boolean, _string
from utilities.avocado_marl.y_correction import Y_CORRECTION_FEATURE_DIM
from utilities.constants import SCENARIOS


@dataclass(frozen=True)
class A6ActionPolicyConfig:
    feature_dim: int
    hidden_sizes: Tuple[int, ...]
    maximum_loc_correction: Tuple[float, float]
    candidate_count: int
    zero_initialization: bool
    freeze: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A6ActionPolicyConfig":
        raw = _object(raw, "action_policy")
        _exact_keys(raw, set(cls.__dataclass_fields__), "action_policy")
        hidden = raw["hidden_sizes"]
        maximum = raw["maximum_loc_correction"]
        if not isinstance(hidden, list) or not hidden:
            raise AVOCADOConfigError(
                "action_policy.hidden_sizes must be a non-empty list."
            )
        if not isinstance(maximum, list) or len(maximum) != 2:
            raise AVOCADOConfigError(
                "action_policy.maximum_loc_correction must contain two values."
            )
        result = cls(
            feature_dim=_integer(
                raw["feature_dim"], "action_policy.feature_dim"
            ),
            hidden_sizes=tuple(
                _integer(value, "action_policy.hidden_sizes[]")
                for value in hidden
            ),
            maximum_loc_correction=tuple(
                _number(
                    value,
                    "action_policy.maximum_loc_correction[]",
                    strict=True,
                )
                for value in maximum
            ),
            candidate_count=_integer(
                raw["candidate_count"], "action_policy.candidate_count"
            ),
            zero_initialization=_boolean(
                raw["zero_initialization"],
                "action_policy.zero_initialization",
            ),
            freeze=_boolean(raw["freeze"], "action_policy.freeze"),
        )
        if result.feature_dim != Y_CORRECTION_FEATURE_DIM:
            raise AVOCADOConfigError(
                "A6-Action must reuse the 14-D local pair feature contract."
            )
        if any(value > 5.0 for value in result.maximum_loc_correction):
            raise AVOCADOConfigError(
                "action_policy.maximum_loc_correction values must not exceed 5."
            )
        if result.freeze:
            raise AVOCADOConfigError("A6-Action requires action_policy.freeze=false.")
        if not result.zero_initialization:
            raise AVOCADOConfigError(
                "A6-Action requires zero_initialization=true so iteration 0 "
                "strictly recovers A5."
            )
        return result


@dataclass(frozen=True)
class A6ActionTrainingConfig:
    scenario_type: str
    n_agents: int
    seed: int
    iterations: int
    parallel_environments: int
    rollout_steps: int
    epochs: int
    minibatch_size: int
    action_learning_rate: float
    critic_learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coefficient: float
    action_regularization: float
    maximum_gradient_norm: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A6ActionTrainingConfig":
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
            minibatch_size=_integer(
                raw["minibatch_size"], "training.minibatch_size"
            ),
            action_learning_rate=_number(
                raw["action_learning_rate"],
                "training.action_learning_rate",
                strict=True,
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
            action_regularization=_number(
                raw["action_regularization"], "training.action_regularization"
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
class A6ActionExperimentConfig:
    a5_config: Path
    output_root: str
    base_policy: A6BasePolicyConfig
    action_policy: A6ActionPolicyConfig
    training: A6ActionTrainingConfig

    @classmethod
    def from_json(cls, path: Path) -> "A6ActionExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "a5_config",
            "output_root",
            "base_policy",
            "action_policy",
            "training",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != 1:
            raise AVOCADOConfigError("A6-Action schema_version must be 1.")
        if raw["method"] != "avocado_marl" or raw["stage"] != "a6_action":
            raise AVOCADOConfigError(
                "Expected method='avocado_marl' and stage='a6_action'."
            )
        a5_config = Path(_string(raw["a5_config"], "a5_config"))
        if not a5_config.is_file():
            raise AVOCADOConfigError(f"A5 config does not exist: {a5_config}")
        return cls(
            a5_config=a5_config,
            output_root=_string(raw["output_root"], "output_root"),
            base_policy=A6BasePolicyConfig.from_dict(raw["base_policy"]),
            action_policy=A6ActionPolicyConfig.from_dict(raw["action_policy"]),
            training=A6ActionTrainingConfig.from_dict(raw["training"]),
        )
