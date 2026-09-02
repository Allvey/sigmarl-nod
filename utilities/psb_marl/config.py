"""Strict experiment configuration for PSB-MARL stages."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class PSBConfigError(ValueError):
    """Raised when a PSB experiment violates its stage contract."""


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PSBConfigError(f"{location} must be a JSON object.")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PSBConfigError(
            f"{location} has invalid keys: missing={missing}, extra={extra}."
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PSBConfigError(f"{location} must be a non-empty string.")
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PSBConfigError(f"{location} must be a finite number.")
    result = float(value)
    if strictly_positive and result <= minimum:
        raise PSBConfigError(f"{location} must be greater than {minimum}.")
    if not strictly_positive and result < minimum:
        raise PSBConfigError(
            f"{location} must be greater than or equal to {minimum}."
        )
    return result


def _integer(value: Any, location: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PSBConfigError(
            f"{location} must be an integer greater than or equal to {minimum}."
        )
    return value


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise PSBConfigError(f"{location} must be a boolean.")
    return value


def _hidden_sizes(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PSBConfigError(f"{location} must be a non-empty integer list.")
    return tuple(
        _integer(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    )


def _integer_tuple(
    value: Any,
    location: str,
    *,
    minimum_length: int = 1,
    minimum_value: int = 0,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise PSBConfigError(
            f"{location} must contain at least {minimum_length} integers."
        )
    result = tuple(
        _integer(item, f"{location}[{index}]", minimum=minimum_value)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise PSBConfigError(f"{location} must not contain duplicate values.")
    return result


def _resolve_existing_path(
    value: Any,
    location: str,
    config_path: Path,
    *,
    kind: str,
) -> Path:
    raw = Path(_string(value, location)).expanduser()
    candidates = (raw, config_path.parent / raw)
    selected = next(
        (
            candidate.resolve()
            for candidate in candidates
            if (candidate.is_file() if kind == "file" else candidate.is_dir())
        ),
        None,
    )
    if selected is None:
        raise PSBConfigError(f"{location} {kind} does not exist: {raw}")
    return selected


@dataclass(frozen=True)
class PSBBaseSourceConfig:
    """Exact Base run and policy/critic pair wrapped by P0."""

    run_directory: Path
    policy_checkpoint: Path
    critic_checkpoint: Path

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        config_path: Path,
    ) -> "PSBBaseSourceConfig":
        raw = _object(raw, "base")
        _exact_keys(raw, set(cls.__dataclass_fields__), "base")
        result = cls(
            run_directory=_resolve_existing_path(
                raw["run_directory"],
                "base.run_directory",
                config_path,
                kind="directory",
            ),
            policy_checkpoint=_resolve_existing_path(
                raw["policy_checkpoint"],
                "base.policy_checkpoint",
                config_path,
                kind="file",
            ),
            critic_checkpoint=_resolve_existing_path(
                raw["critic_checkpoint"],
                "base.critic_checkpoint",
                config_path,
                kind="file",
            ),
        )
        if result.policy_checkpoint.parent != result.run_directory:
            raise PSBConfigError(
                "base.policy_checkpoint must belong directly to base.run_directory."
            )
        if result.critic_checkpoint.parent != result.run_directory:
            raise PSBConfigError(
                "base.critic_checkpoint must belong directly to base.run_directory."
            )
        if not result.policy_checkpoint.name.endswith("_policy.pth"):
            raise PSBConfigError(
                "base.policy_checkpoint must end with '_policy.pth'."
            )
        prefix = result.policy_checkpoint.name[: -len("_policy.pth")]
        expected_critic = f"{prefix}_critic.pth"
        if result.critic_checkpoint.name != expected_critic:
            raise PSBConfigError(
                "base policy and critic checkpoints must share the same prefix: "
                f"expected {expected_critic}."
            )
        return result


@dataclass(frozen=True)
class PSBConflictGraphConfig:
    """P1 local conflict information emitted by the existing road scenario."""

    emit_pair_info: bool
    candidate_count: int
    pair_feature_dim: int
    prediction_horizon_seconds: float
    conflict_distance_meters: float
    sensing_distance_meters: float
    cpa_epsilon: float
    urgency_time_scale_seconds: float
    urgency_distance_scale_meters: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBConflictGraphConfig":
        raw = _object(raw, "conflict_graph")
        _exact_keys(raw, set(cls.__dataclass_fields__), "conflict_graph")
        if type(raw["emit_pair_info"]) is not bool:
            raise PSBConfigError("conflict_graph.emit_pair_info must be a boolean.")
        result = cls(
            emit_pair_info=raw["emit_pair_info"],
            candidate_count=_integer(
                raw["candidate_count"], "conflict_graph.candidate_count"
            ),
            pair_feature_dim=_integer(
                raw["pair_feature_dim"], "conflict_graph.pair_feature_dim"
            ),
            prediction_horizon_seconds=_number(
                raw["prediction_horizon_seconds"],
                "conflict_graph.prediction_horizon_seconds",
                strictly_positive=True,
            ),
            conflict_distance_meters=_number(
                raw["conflict_distance_meters"],
                "conflict_graph.conflict_distance_meters",
                strictly_positive=True,
            ),
            sensing_distance_meters=_number(
                raw["sensing_distance_meters"],
                "conflict_graph.sensing_distance_meters",
                strictly_positive=True,
            ),
            cpa_epsilon=_number(
                raw["cpa_epsilon"],
                "conflict_graph.cpa_epsilon",
                strictly_positive=True,
            ),
            urgency_time_scale_seconds=_number(
                raw["urgency_time_scale_seconds"],
                "conflict_graph.urgency_time_scale_seconds",
                strictly_positive=True,
            ),
            urgency_distance_scale_meters=_number(
                raw["urgency_distance_scale_meters"],
                "conflict_graph.urgency_distance_scale_meters",
                strictly_positive=True,
            ),
        )
        if not result.emit_pair_info:
            raise PSBConfigError("P1 requires conflict_graph.emit_pair_info=true.")
        if result.pair_feature_dim != 10:
            raise PSBConfigError("P1 requires conflict_graph.pair_feature_dim=10.")
        if result.conflict_distance_meters >= result.sensing_distance_meters:
            raise PSBConfigError(
                "conflict_distance_meters must be smaller than sensing_distance_meters."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PSBProximalConfig:
    """Fixed P1 proximal dynamics satisfying the global uniqueness bound."""

    kappa: float
    nu: float
    alpha: float
    rho_max: float
    tau_z: float
    b_max: float
    residual_tolerance: float
    max_iterations: int
    zero_threshold: float

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        require_zero_control: bool,
    ) -> "PSBProximalConfig":
        raw = _object(raw, "proximal")
        _exact_keys(raw, set(cls.__dataclass_fields__), "proximal")
        result = cls(
            kappa=_number(
                raw["kappa"], "proximal.kappa", strictly_positive=True
            ),
            nu=_number(raw["nu"], "proximal.nu", strictly_positive=True),
            alpha=_number(
                raw["alpha"], "proximal.alpha", strictly_positive=True
            ),
            rho_max=_number(
                raw["rho_max"], "proximal.rho_max", strictly_positive=True
            ),
            tau_z=_number(
                raw["tau_z"], "proximal.tau_z", strictly_positive=True
            ),
            b_max=_number(raw["b_max"], "proximal.b_max"),
            residual_tolerance=_number(
                raw["residual_tolerance"],
                "proximal.residual_tolerance",
                strictly_positive=True,
            ),
            max_iterations=_integer(
                raw["max_iterations"], "proximal.max_iterations"
            ),
            zero_threshold=_number(
                raw["zero_threshold"],
                "proximal.zero_threshold",
                strictly_positive=True,
            ),
        )
        if require_zero_control and result.b_max != 0.0:
            raise PSBConfigError("P1 zero-control equivalence requires b_max=0.")
        if not require_zero_control and result.b_max <= 0.0:
            raise PSBConfigError("P2 requires proximal.b_max > 0.")
        if result.rho_max <= result.critical_rho:
            raise PSBConfigError(
                "proximal.rho_max must exceed rho_c so P1 covers both sides of "
                "the bifurcation point."
            )
        return result

    @property
    def critical_rho(self) -> float:
        return self.kappa / (self.nu * self.alpha)

    def h_z(self, dt: float) -> float:
        return float(dt) / self.tau_z

    def convexity_margin(self, dt: float) -> float:
        h_z = self.h_z(dt)
        return 1.0 / h_z + self.kappa - self.rho_max * self.nu * self.alpha

    def to_runtime_dict(self, dt: float) -> Dict[str, object]:
        return {
            "kappa": self.kappa,
            "nu": self.nu,
            "alpha": self.alpha,
            "rho_max": self.rho_max,
            "tau_z": self.tau_z,
            "h_z": self.h_z(dt),
            "b_max": self.b_max,
            "residual_tolerance": self.residual_tolerance,
            "max_iterations": self.max_iterations,
            "zero_threshold": self.zero_threshold,
            "rho_c": self.critical_rho,
            "convexity_margin": self.convexity_margin(dt),
        }


@dataclass(frozen=True)
class PSBControlConfig:
    """Trainable bounded antisymmetric P2 bifurcation controller."""

    hidden_sizes: tuple[int, ...]
    temperature: float
    support_power: float
    critical_gate_enabled: bool
    critical_width: float
    critical_floor: float
    final_layer_gain: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBControlConfig":
        raw = _object(raw, "control")
        _exact_keys(raw, set(cls.__dataclass_fields__), "control")
        result = cls(
            hidden_sizes=_hidden_sizes(raw["hidden_sizes"], "control.hidden_sizes"),
            temperature=_number(
                raw["temperature"], "control.temperature", strictly_positive=True
            ),
            support_power=_number(
                raw["support_power"], "control.support_power", strictly_positive=True
            ),
            critical_gate_enabled=_boolean(
                raw["critical_gate_enabled"], "control.critical_gate_enabled"
            ),
            critical_width=_number(
                raw["critical_width"],
                "control.critical_width",
                strictly_positive=True,
            ),
            critical_floor=_number(
                raw["critical_floor"], "control.critical_floor"
            ),
            final_layer_gain=_number(
                raw["final_layer_gain"],
                "control.final_layer_gain",
                strictly_positive=True,
            ),
        )
        if result.critical_floor > 1.0:
            raise PSBConfigError("control.critical_floor must not exceed 1.")
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "temperature": self.temperature,
            "support_power": self.support_power,
            "critical_gate_enabled": self.critical_gate_enabled,
            "critical_width": self.critical_width,
            "critical_floor": self.critical_floor,
            "final_layer_gain": self.final_layer_gain,
        }


@dataclass(frozen=True)
class PSBBranchAdapterConfig:
    """P2 branch context and zero-initialized distribution adapter."""

    pair_hidden_sizes: tuple[int, ...]
    context_dim: int
    adapter_hidden_sizes: tuple[int, ...]
    z_scale: float
    max_delta_loc: float
    max_delta_log_scale: float
    conditioning_mode: str = "general"
    action_projection: str = "full"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBBranchAdapterConfig":
        raw = _object(raw, "branch_adapter")
        optional_keys = {"conditioning_mode", "action_projection"}
        required_keys = set(cls.__dataclass_fields__) - optional_keys
        actual_keys = set(raw)
        if not required_keys.issubset(actual_keys) or not actual_keys.issubset(
            set(cls.__dataclass_fields__)
        ):
            missing = sorted(required_keys - actual_keys)
            extra = sorted(actual_keys - set(cls.__dataclass_fields__))
            raise PSBConfigError(
                "branch_adapter has invalid keys: "
                f"missing={missing}, extra={extra}."
            )
        conditioning_mode = _string(
            raw.get("conditioning_mode", "general"),
            "branch_adapter.conditioning_mode",
        )
        if conditioning_mode not in {
            "general",
            "causal_q_gate",
            "sector_q_gate",
            "supported_sector_q_gate",
        }:
            raise PSBConfigError(
                "branch_adapter.conditioning_mode must be 'general', "
                "'causal_q_gate', 'sector_q_gate', or "
                "'supported_sector_q_gate'."
            )
        action_projection = _string(
            raw.get("action_projection", "full"),
            "branch_adapter.action_projection",
        )
        if action_projection not in {"full", "longitudinal_only"}:
            raise PSBConfigError(
                "branch_adapter.action_projection must be 'full' or "
                "'longitudinal_only'."
            )
        result = cls(
            pair_hidden_sizes=_hidden_sizes(
                raw["pair_hidden_sizes"], "branch_adapter.pair_hidden_sizes"
            ),
            context_dim=_integer(
                raw["context_dim"], "branch_adapter.context_dim"
            ),
            adapter_hidden_sizes=_hidden_sizes(
                raw["adapter_hidden_sizes"],
                "branch_adapter.adapter_hidden_sizes",
            ),
            z_scale=_number(
                raw["z_scale"], "branch_adapter.z_scale", strictly_positive=True
            ),
            max_delta_loc=_number(
                raw["max_delta_loc"],
                "branch_adapter.max_delta_loc",
                strictly_positive=True,
            ),
            max_delta_log_scale=_number(
                raw["max_delta_log_scale"],
                "branch_adapter.max_delta_log_scale",
            ),
            conditioning_mode=conditioning_mode,
            action_projection=action_projection,
        )
        if result.action_projection == "longitudinal_only" and (
            result.conditioning_mode
            not in {
                "causal_q_gate",
                "sector_q_gate",
                "supported_sector_q_gate",
            }
            or result.max_delta_log_scale != 0.0
        ):
            raise PSBConfigError(
                "longitudinal_only requires a causal or sector q gate and "
                "max_delta_log_scale=0."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        result = {
            "pair_hidden_sizes": list(self.pair_hidden_sizes),
            "context_dim": self.context_dim,
            "adapter_hidden_sizes": list(self.adapter_hidden_sizes),
            "z_scale": self.z_scale,
            "max_delta_loc": self.max_delta_loc,
            "max_delta_log_scale": self.max_delta_log_scale,
        }
        # Preserve the exact legacy runtime/checkpoint contract when the field
        # is omitted. P2.1-C/S write an explicit, semantically distinct mode.
        if self.conditioning_mode != "general":
            result["conditioning_mode"] = self.conditioning_mode
        if self.action_projection != "full":
            result["action_projection"] = self.action_projection
        return result


@dataclass(frozen=True)
class PSBP2TrainingConfig:
    """Frozen-Base P2 sequence PPO and trust-region settings."""

    iterations: int
    chunk_length: int
    control_learning_rate_scale: float
    adapter_learning_rate_scale: float
    critic_learning_rate_scale: float
    energy_coefficient: float
    control_trust_region_coefficient: float
    saturation_coefficient: float
    saturation_fraction: float
    checkpoint_interval: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBP2TrainingConfig":
        raw = _object(raw, "training")
        _exact_keys(raw, set(cls.__dataclass_fields__), "training")
        result = cls(
            iterations=_integer(raw["iterations"], "training.iterations"),
            chunk_length=_integer(
                raw["chunk_length"], "training.chunk_length", minimum=2
            ),
            control_learning_rate_scale=_number(
                raw["control_learning_rate_scale"],
                "training.control_learning_rate_scale",
                strictly_positive=True,
            ),
            adapter_learning_rate_scale=_number(
                raw["adapter_learning_rate_scale"],
                "training.adapter_learning_rate_scale",
                strictly_positive=True,
            ),
            critic_learning_rate_scale=_number(
                raw["critic_learning_rate_scale"],
                "training.critic_learning_rate_scale",
                strictly_positive=True,
            ),
            energy_coefficient=_number(
                raw["energy_coefficient"], "training.energy_coefficient"
            ),
            control_trust_region_coefficient=_number(
                raw["control_trust_region_coefficient"],
                "training.control_trust_region_coefficient",
            ),
            saturation_coefficient=_number(
                raw["saturation_coefficient"],
                "training.saturation_coefficient",
            ),
            saturation_fraction=_number(
                raw["saturation_fraction"],
                "training.saturation_fraction",
                strictly_positive=True,
            ),
            checkpoint_interval=_integer(
                raw["checkpoint_interval"], "training.checkpoint_interval"
            ),
        )
        if result.saturation_fraction > 1.0:
            raise PSBConfigError("training.saturation_fraction must not exceed 1.")
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PSBPromotionConfig:
    """Conservative paired non-inferiority gate for candidate deployment."""

    minimum_paired_seeds: int
    confidence_z: float
    reward_margin: float
    collision_margin: float
    lane_collision_margin: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBPromotionConfig":
        raw = _object(raw, "promotion")
        optional_keys = {"lane_collision_margin"}
        required_keys = set(cls.__dataclass_fields__) - optional_keys
        actual_keys = set(raw)
        if not required_keys.issubset(actual_keys) or not actual_keys.issubset(
            set(cls.__dataclass_fields__)
        ):
            missing = sorted(required_keys - actual_keys)
            extra = sorted(actual_keys - set(cls.__dataclass_fields__))
            raise PSBConfigError(
                "promotion has invalid keys: "
                f"missing={missing}, extra={extra}."
            )
        lane_collision_margin = raw.get("lane_collision_margin")
        result = cls(
            minimum_paired_seeds=_integer(
                raw["minimum_paired_seeds"],
                "promotion.minimum_paired_seeds",
                minimum=2,
            ),
            confidence_z=_number(
                raw["confidence_z"],
                "promotion.confidence_z",
                strictly_positive=True,
            ),
            reward_margin=_number(raw["reward_margin"], "promotion.reward_margin"),
            collision_margin=_number(
                raw["collision_margin"], "promotion.collision_margin"
            ),
            lane_collision_margin=(
                None
                if lane_collision_margin is None
                else _number(
                    lane_collision_margin,
                    "promotion.lane_collision_margin",
                )
            ),
        )
        if min(result.reward_margin, result.collision_margin) < 0.0 or (
            result.lane_collision_margin is not None
            and result.lane_collision_margin < 0.0
        ):
            raise PSBConfigError("promotion margins must be non-negative.")
        return result

    def to_dict(self) -> Dict[str, object]:
        result = {
            "minimum_paired_seeds": self.minimum_paired_seeds,
            "confidence_z": self.confidence_z,
            "reward_margin": self.reward_margin,
            "collision_margin": self.collision_margin,
        }
        if self.lane_collision_margin is not None:
            result["lane_collision_margin"] = self.lane_collision_margin
        return result


@dataclass(frozen=True)
class PSBP3PairedRolloutConfig:
    """Read-only P3.0 paired Candidate/Base collection contract."""

    common_random_numbers: bool
    learning_enabled: bool
    require_exact_source_equivalence: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBP3PairedRolloutConfig":
        raw = _object(raw, "paired_rollout")
        _exact_keys(raw, set(cls.__dataclass_fields__), "paired_rollout")
        result = cls(
            common_random_numbers=_boolean(
                raw["common_random_numbers"],
                "paired_rollout.common_random_numbers",
            ),
            learning_enabled=_boolean(
                raw["learning_enabled"], "paired_rollout.learning_enabled"
            ),
            require_exact_source_equivalence=_boolean(
                raw["require_exact_source_equivalence"],
                "paired_rollout.require_exact_source_equivalence",
            ),
        )
        if not result.common_random_numbers:
            raise PSBConfigError("P3.0 requires common_random_numbers=true.")
        if result.learning_enabled:
            raise PSBConfigError("P3.0 requires learning_enabled=false.")
        if not result.require_exact_source_equivalence:
            raise PSBConfigError(
                "P3.0 requires require_exact_source_equivalence=true."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PSBP31DifferentialCriticConfig:
    """Actor-frozen paired supervision for the P3.1 vector critic."""

    actor_learning_enabled: bool
    dual_learning_enabled: bool
    collection_scenario: str
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    max_steps: int
    episodes: int
    gamma: float
    embedding_dim: int
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    epochs: int
    minibatch_size: int
    huber_delta: float
    target_scale_floor: float
    lane_safety_margin: float
    gradient_clip_norm: float
    early_stopping_patience: int
    required_relative_improvement: float
    minimum_target_std: float
    minimum_channel_explained_variance: float

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> "PSBP31DifferentialCriticConfig":
        raw = _object(raw, "differential_critic")
        _exact_keys(raw, set(cls.__dataclass_fields__), "differential_critic")
        result = cls(
            actor_learning_enabled=_boolean(
                raw["actor_learning_enabled"],
                "differential_critic.actor_learning_enabled",
            ),
            dual_learning_enabled=_boolean(
                raw["dual_learning_enabled"],
                "differential_critic.dual_learning_enabled",
            ),
            collection_scenario=_string(
                raw["collection_scenario"],
                "differential_critic.collection_scenario",
            ),
            training_seeds=_integer_tuple(
                raw["training_seeds"],
                "differential_critic.training_seeds",
                minimum_length=2,
            ),
            validation_seeds=_integer_tuple(
                raw["validation_seeds"],
                "differential_critic.validation_seeds",
                minimum_length=1,
            ),
            max_steps=_integer(
                raw["max_steps"], "differential_critic.max_steps", minimum=2
            ),
            episodes=_integer(
                raw["episodes"], "differential_critic.episodes"
            ),
            gamma=_number(
                raw["gamma"],
                "differential_critic.gamma",
                strictly_positive=True,
            ),
            embedding_dim=_integer(
                raw["embedding_dim"], "differential_critic.embedding_dim"
            ),
            hidden_sizes=_hidden_sizes(
                raw["hidden_sizes"], "differential_critic.hidden_sizes"
            ),
            learning_rate=_number(
                raw["learning_rate"],
                "differential_critic.learning_rate",
                strictly_positive=True,
            ),
            weight_decay=_number(
                raw["weight_decay"], "differential_critic.weight_decay"
            ),
            epochs=_integer(raw["epochs"], "differential_critic.epochs"),
            minibatch_size=_integer(
                raw["minibatch_size"], "differential_critic.minibatch_size"
            ),
            huber_delta=_number(
                raw["huber_delta"],
                "differential_critic.huber_delta",
                strictly_positive=True,
            ),
            target_scale_floor=_number(
                raw["target_scale_floor"],
                "differential_critic.target_scale_floor",
                strictly_positive=True,
            ),
            lane_safety_margin=_number(
                raw["lane_safety_margin"],
                "differential_critic.lane_safety_margin",
                strictly_positive=True,
            ),
            gradient_clip_norm=_number(
                raw["gradient_clip_norm"],
                "differential_critic.gradient_clip_norm",
                strictly_positive=True,
            ),
            early_stopping_patience=_integer(
                raw["early_stopping_patience"],
                "differential_critic.early_stopping_patience",
            ),
            required_relative_improvement=_number(
                raw["required_relative_improvement"],
                "differential_critic.required_relative_improvement",
            ),
            minimum_target_std=_number(
                raw["minimum_target_std"],
                "differential_critic.minimum_target_std",
            ),
            minimum_channel_explained_variance=_number(
                raw["minimum_channel_explained_variance"],
                "differential_critic.minimum_channel_explained_variance",
                minimum=-1.0,
            ),
        )
        if result.actor_learning_enabled or result.dual_learning_enabled:
            raise PSBConfigError(
                "P3.1 requires actor_learning_enabled=false and "
                "dual_learning_enabled=false."
            )
        if result.gamma > 1.0:
            raise PSBConfigError("differential_critic.gamma must not exceed 1.")
        if set(result.training_seeds) & set(result.validation_seeds):
            raise PSBConfigError(
                "P3.1 training_seeds and validation_seeds must be disjoint."
            )
        if result.required_relative_improvement >= 1.0:
            raise PSBConfigError(
                "required_relative_improvement must be smaller than 1."
            )
        if result.lane_safety_margin > 1.0:
            raise PSBConfigError(
                "lane_safety_margin must be a normalized value no greater than 1."
            )
        if result.minimum_channel_explained_variance >= 1.0:
            raise PSBConfigError(
                "minimum_channel_explained_variance must be smaller than 1."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        result = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        result["training_seeds"] = list(self.training_seeds)
        result["validation_seeds"] = list(self.validation_seeds)
        result["hidden_sizes"] = list(self.hidden_sizes)
        return result


@dataclass(frozen=True)
class PSBP32PrimalDualConfig:
    """Locked projected-dual settings for P3.2 sequence PPO."""

    iterations: int
    vehicle_budget: float
    lane_budget: float
    vehicle_learning_rate: float
    lane_learning_rate: float
    maximum_multiplier: float
    initial_vehicle_multiplier: float
    initial_lane_multiplier: float
    lane_safety_margin: float
    normalize_constraints: bool = False
    active_constraints: tuple[str, ...] = ("vehicle", "lane")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBP32PrimalDualConfig":
        raw = _object(raw, "primal_dual")
        optional = {"normalize_constraints", "active_constraints"}
        required = set(cls.__dataclass_fields__) - optional
        actual = set(raw)
        if not required.issubset(actual) or not actual.issubset(
            set(cls.__dataclass_fields__)
        ):
            raise PSBConfigError(
                "primal_dual has invalid keys: "
                f"missing={sorted(required - actual)}, "
                f"extra={sorted(actual - set(cls.__dataclass_fields__))}."
            )
        active_constraints = raw.get(
            "active_constraints", ["vehicle", "lane"]
        )
        if not isinstance(active_constraints, list):
            raise PSBConfigError(
                "primal_dual.active_constraints must be a list."
            )
        result = cls(
            iterations=_integer(raw["iterations"], "primal_dual.iterations"),
            vehicle_budget=_number(
                raw["vehicle_budget"], "primal_dual.vehicle_budget"
            ),
            lane_budget=_number(raw["lane_budget"], "primal_dual.lane_budget"),
            vehicle_learning_rate=_number(
                raw["vehicle_learning_rate"],
                "primal_dual.vehicle_learning_rate",
                strictly_positive=True,
            ),
            lane_learning_rate=_number(
                raw["lane_learning_rate"],
                "primal_dual.lane_learning_rate",
                strictly_positive=True,
            ),
            maximum_multiplier=_number(
                raw["maximum_multiplier"],
                "primal_dual.maximum_multiplier",
                strictly_positive=True,
            ),
            initial_vehicle_multiplier=_number(
                raw["initial_vehicle_multiplier"],
                "primal_dual.initial_vehicle_multiplier",
            ),
            initial_lane_multiplier=_number(
                raw["initial_lane_multiplier"],
                "primal_dual.initial_lane_multiplier",
            ),
            lane_safety_margin=_number(
                raw["lane_safety_margin"],
                "primal_dual.lane_safety_margin",
                strictly_positive=True,
            ),
            normalize_constraints=_boolean(
                raw.get("normalize_constraints", False),
                "primal_dual.normalize_constraints",
            ),
            active_constraints=tuple(
                _string(item, f"primal_dual.active_constraints[{index}]")
                for index, item in enumerate(active_constraints)
            ),
        )
        if max(
            result.initial_vehicle_multiplier,
            result.initial_lane_multiplier,
        ) > result.maximum_multiplier:
            raise PSBConfigError(
                "P3.2 initial multipliers exceed maximum_multiplier."
            )
        if result.lane_safety_margin > 1.0:
            raise PSBConfigError("P3.2 lane_safety_margin must not exceed 1.")
        if (
            not result.active_constraints
            or len(set(result.active_constraints))
            != len(result.active_constraints)
            or not set(result.active_constraints).issubset({"vehicle", "lane"})
        ):
            raise PSBConfigError(
                "P3.2 active_constraints must be a non-empty unique subset "
                "of vehicle and lane."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"normalize_constraints", "active_constraints"}
        }
        # Omitting the legacy false value preserves byte-compatible runtime
        # dictionaries for already completed P3.2 runs.
        if self.normalize_constraints:
            result["normalize_constraints"] = True
        if self.active_constraints != ("vehicle", "lane"):
            result["active_constraints"] = list(self.active_constraints)
        return result


@dataclass(frozen=True)
class PSBP33PairedDifferentialConfig:
    """Online paired-control-variate settings for P3.3 sequence PPO."""

    common_random_numbers: bool
    reset_at_each_iteration: bool
    synchronize_episode_boundaries: bool
    online_critic_learning_enabled: bool
    critic_learning_rate_scale: float
    huber_delta: float
    gradient_clip_norm: float
    normalize_advantage: bool
    advantage_scale_floor: float

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> "PSBP33PairedDifferentialConfig":
        raw = _object(raw, "paired_differential")
        _exact_keys(raw, set(cls.__dataclass_fields__), "paired_differential")
        result = cls(
            common_random_numbers=_boolean(
                raw["common_random_numbers"],
                "paired_differential.common_random_numbers",
            ),
            reset_at_each_iteration=_boolean(
                raw["reset_at_each_iteration"],
                "paired_differential.reset_at_each_iteration",
            ),
            synchronize_episode_boundaries=_boolean(
                raw["synchronize_episode_boundaries"],
                "paired_differential.synchronize_episode_boundaries",
            ),
            online_critic_learning_enabled=_boolean(
                raw["online_critic_learning_enabled"],
                "paired_differential.online_critic_learning_enabled",
            ),
            critic_learning_rate_scale=_number(
                raw["critic_learning_rate_scale"],
                "paired_differential.critic_learning_rate_scale",
                strictly_positive=True,
            ),
            huber_delta=_number(
                raw["huber_delta"],
                "paired_differential.huber_delta",
                strictly_positive=True,
            ),
            gradient_clip_norm=_number(
                raw["gradient_clip_norm"],
                "paired_differential.gradient_clip_norm",
                strictly_positive=True,
            ),
            normalize_advantage=_boolean(
                raw["normalize_advantage"],
                "paired_differential.normalize_advantage",
            ),
            advantage_scale_floor=_number(
                raw["advantage_scale_floor"],
                "paired_differential.advantage_scale_floor",
                strictly_positive=True,
            ),
        )
        if not result.common_random_numbers:
            raise PSBConfigError("P3.3 requires common_random_numbers=true.")
        if not result.reset_at_each_iteration:
            raise PSBConfigError("P3.3 requires reset_at_each_iteration=true.")
        if not result.synchronize_episode_boundaries:
            raise PSBConfigError(
                "P3.3 requires synchronize_episode_boundaries=true."
            )
        if not result.online_critic_learning_enabled:
            raise PSBConfigError(
                "P3.3 requires online_critic_learning_enabled=true."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PSBP5JointTrainingConfig:
    """Single-stage joint optimization settings for the final PSB policy."""

    ppo_mode: str
    ppo_epochs: int
    minibatch_size: int
    target_kl: float
    base_actor_learning_rate_scale: float
    absolute_critic_learning_rate_scale: float
    absolute_critic_loss_coefficient: float
    base_anchor_coefficient: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBP5JointTrainingConfig":
        raw = _object(raw, "joint_training")
        _exact_keys(raw, set(cls.__dataclass_fields__), "joint_training")
        result = cls(
            ppo_mode=_string(raw["ppo_mode"], "joint_training.ppo_mode"),
            ppo_epochs=_integer(
                raw["ppo_epochs"], "joint_training.ppo_epochs", minimum=1
            ),
            minibatch_size=_integer(
                raw["minibatch_size"],
                "joint_training.minibatch_size",
                minimum=1,
            ),
            target_kl=_number(
                raw["target_kl"],
                "joint_training.target_kl",
                strictly_positive=True,
            ),
            base_actor_learning_rate_scale=_number(
                raw["base_actor_learning_rate_scale"],
                "joint_training.base_actor_learning_rate_scale",
                strictly_positive=True,
            ),
            absolute_critic_learning_rate_scale=_number(
                raw["absolute_critic_learning_rate_scale"],
                "joint_training.absolute_critic_learning_rate_scale",
                strictly_positive=True,
            ),
            absolute_critic_loss_coefficient=_number(
                raw["absolute_critic_loss_coefficient"],
                "joint_training.absolute_critic_loss_coefficient",
                strictly_positive=True,
            ),
            base_anchor_coefficient=_number(
                raw["base_anchor_coefficient"],
                "joint_training.base_anchor_coefficient",
            ),
        )
        if result.ppo_mode != "transition":
            raise PSBConfigError(
                "P5 requires ppo_mode='transition'; Sequence PPO is disabled."
            )
        if result.base_actor_learning_rate_scale > 1.0:
            raise PSBConfigError(
                "P5 Base Actor learning-rate scale must not exceed 1."
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PSBExperimentConfig:
    """Resolved PSB experiment plus its immutable JSON snapshots."""

    config_path: Path
    source_config: Dict[str, Any]
    base_config_path: Path
    base_source_config: Dict[str, Any]
    base_run_config: Dict[str, Any]
    output_root: str
    base: PSBBaseSourceConfig
    stage: str
    parent_run: Optional[Path] = None
    conflict_graph: Optional[PSBConflictGraphConfig] = None
    proximal: Optional[PSBProximalConfig] = None
    control: Optional[PSBControlConfig] = None
    branch_adapter: Optional[PSBBranchAdapterConfig] = None
    training: Optional[PSBP2TrainingConfig] = None
    promotion: Optional[PSBPromotionConfig] = None
    training_seed: Optional[int] = None
    robustness_summary: Optional[Path] = None
    paired_rollout: Optional[PSBP3PairedRolloutConfig] = None
    differential_critic: Optional[PSBP31DifferentialCriticConfig] = None
    primal_dual: Optional[PSBP32PrimalDualConfig] = None
    paired_differential: Optional[PSBP33PairedDifferentialConfig] = None
    joint_training: Optional[PSBP5JointTrainingConfig] = None
    source_p2_runtime: Optional[Dict[str, object]] = None

    @property
    def base_parameters(self):
        from utilities.helper_training import Parameters

        return Parameters.from_dict(dict(self.base_run_config))

    @property
    def seed(self) -> int:
        return int(self.base_run_config["seed"])

    @property
    def effective_training_seed(self) -> int:
        return self.seed if self.training_seed is None else self.training_seed

    @property
    def dt(self) -> float:
        return float(self.base_run_config["dt"])

    def p1_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p1_zero_control_equivalence":
            raise PSBConfigError("P1 runtime config requested for a non-P1 stage.")
        assert self.conflict_graph is not None and self.proximal is not None
        return {
            "stage": self.stage,
            "n_agents": int(self.base_run_config["n_agents"]),
            "actor_context_gain": 0.0,
            "control_mode": "zero",
            "proximal": self.proximal.to_runtime_dict(self.dt),
        }

    def p2_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p2_frozen_base_bifurcation":
            raise PSBConfigError("P2 runtime config requested for a non-P2 stage.")
        assert self.conflict_graph is not None
        assert self.proximal is not None
        assert self.control is not None
        assert self.branch_adapter is not None
        assert self.training is not None
        assert self.promotion is not None
        result = {
            "stage": self.stage,
            "n_agents": int(self.base_run_config["n_agents"]),
            "control_mode": "learned_antisymmetric",
            "freeze_base_actor": True,
            "base_policy_checkpoint": str(self.base.policy_checkpoint),
            "base_critic_checkpoint": str(self.base.critic_checkpoint),
            "proximal": self.proximal.to_runtime_dict(self.dt),
            "control": self.control.to_dict(),
            "branch_adapter": self.branch_adapter.to_dict(),
            "training": self.training.to_dict(),
            "promotion": self.promotion.to_dict(),
        }
        if self.training_seed is not None:
            result["training_seed"] = self.training_seed
        return result

    def source_p2_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p3_paired_rollout_equivalence":
            raise PSBConfigError(
                "Source P2 runtime requested for a non-P3.0 stage."
            )
        assert self.proximal is not None
        assert self.control is not None
        assert self.branch_adapter is not None
        assert self.training is not None
        assert self.promotion is not None
        assert self.training_seed is not None
        return {
            "stage": "p2_frozen_base_bifurcation",
            "n_agents": int(self.base_run_config["n_agents"]),
            "control_mode": "learned_antisymmetric",
            "freeze_base_actor": True,
            "base_policy_checkpoint": str(self.base.policy_checkpoint),
            "base_critic_checkpoint": str(self.base.critic_checkpoint),
            "proximal": self.proximal.to_runtime_dict(self.dt),
            "control": self.control.to_dict(),
            "branch_adapter": self.branch_adapter.to_dict(),
            "training": self.training.to_dict(),
            "promotion": self.promotion.to_dict(),
            "training_seed": self.training_seed,
        }

    def p3_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p3_paired_rollout_equivalence":
            raise PSBConfigError("P3 runtime requested for a non-P3.0 stage.")
        assert self.parent_run is not None
        assert self.robustness_summary is not None
        assert self.paired_rollout is not None
        return {
            "stage": self.stage,
            "source_p2_run": str(self.parent_run),
            "robustness_summary": str(self.robustness_summary),
            "paired_rollout": self.paired_rollout.to_dict(),
            "source_p2_runtime": self.source_p2_runtime_config(),
        }

    def p31_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p3_differential_critic":
            raise PSBConfigError("P3.1 runtime requested for a non-P3.1 stage.")
        assert self.parent_run is not None
        assert self.differential_critic is not None
        assert self.source_p2_runtime is not None
        assert self.training_seed is not None
        return {
            "stage": self.stage,
            "training_seed": self.training_seed,
            "source_p3_run": str(self.parent_run),
            "source_p2_runtime": dict(self.source_p2_runtime),
            "differential_critic": self.differential_critic.to_dict(),
            "target_channels": [
                "augmented_reward_return_delta",
                "vehicle_conflict_risk_return_delta",
                "lane_margin_violation_return_delta",
            ],
        }

    def p32_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p3_primal_dual_ppo":
            raise PSBConfigError("P3.2 runtime requested for a non-P3.2 stage.")
        assert self.parent_run is not None
        assert self.source_p2_runtime is not None
        assert self.primal_dual is not None
        runtime = dict(self.source_p2_runtime)
        training = dict(runtime["training"])
        training["iterations"] = self.primal_dual.iterations
        runtime.update(
            {
                "training": training,
                "training_seed": self.effective_training_seed,
                "p3_stage": self.stage,
                "primal_dual": self.primal_dual.to_dict(),
                "initial_policy_checkpoint": str(
                    self.parent_run / "candidate_policy.pth"
                ),
                "initial_scalar_critic_checkpoint": str(
                    self.parent_run / "source_p2_critic.pth"
                ),
                "p3_differential_critic_checkpoint": str(
                    self.parent_run / "candidate_critic.pth"
                ),
            }
        )
        return runtime

    def p33_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p3_paired_differential_primal_dual_ppo":
            raise PSBConfigError("P3.3 runtime requested for a non-P3.3 stage.")
        assert self.parent_run is not None
        assert self.source_p2_runtime is not None
        assert self.primal_dual is not None
        assert self.paired_differential is not None
        runtime = dict(self.source_p2_runtime)
        training = dict(runtime["training"])
        training["iterations"] = self.primal_dual.iterations
        runtime.update(
            {
                "training": training,
                "training_seed": self.effective_training_seed,
                "p3_stage": self.stage,
                "primal_dual": self.primal_dual.to_dict(),
                "paired_differential": self.paired_differential.to_dict(),
                "initial_policy_checkpoint": str(
                    self.parent_run / "candidate_policy.pth"
                ),
                "initial_scalar_critic_checkpoint": str(
                    self.parent_run / "source_p2_critic.pth"
                ),
                "p3_differential_critic_checkpoint": str(
                    self.parent_run / "candidate_critic.pth"
                ),
            }
        )
        return runtime

    def p5_runtime_config(self) -> Dict[str, object]:
        if self.stage != "p5_joint_psb_marl":
            raise PSBConfigError("P5 runtime requested for a non-P5 stage.")
        assert self.parent_run is not None
        assert self.source_p2_runtime is not None
        assert self.primal_dual is not None
        assert self.paired_differential is not None
        assert self.joint_training is not None
        runtime = dict(self.source_p2_runtime)
        training = dict(runtime["training"])
        training["iterations"] = self.primal_dual.iterations
        runtime.update(
            {
                "training": training,
                "training_seed": self.effective_training_seed,
                "freeze_base_actor": False,
                "p3_stage": "p3_paired_differential_primal_dual_ppo",
                "p5_stage": self.stage,
                "primal_dual": self.primal_dual.to_dict(),
                "paired_differential": self.paired_differential.to_dict(),
                "joint_training": self.joint_training.to_dict(),
                "initial_policy_checkpoint": str(
                    self.parent_run / "candidate_policy.pth"
                ),
                "initial_scalar_critic_checkpoint": str(
                    self.parent_run / "candidate_critic.pth"
                ),
                "p3_differential_critic_checkpoint": str(
                    self.parent_run / "candidate_differential_critic.pth"
                ),
            }
        )
        return runtime


def _validate_base_run(
    base_config_path: Path,
    base: PSBBaseSourceConfig,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    with base_config_path.open("r", encoding="utf-8") as stream:
        source = _object(json.load(stream), "base_config")
    source_dict = dict(source)
    if type(source_dict.get("seed")) is not int or source_dict["seed"] < 0:
        raise PSBConfigError("base_config.seed must be a non-negative integer.")

    resolved_path = base.run_directory / "config_resolved.json"
    if not resolved_path.is_file():
        raise PSBConfigError(
            "base.run_directory is missing config_resolved.json: "
            f"{resolved_path}"
        )
    with resolved_path.open("r", encoding="utf-8") as stream:
        resolved = _object(json.load(stream), "base run config_resolved.json")
    resolved_dict = dict(resolved)
    if type(resolved_dict.get("seed")) is not int or resolved_dict["seed"] < 0:
        raise PSBConfigError(
            "Base run config_resolved.json has an invalid seed."
        )

    for key, expected in source_dict.items():
        if key == "where_to_save":
            continue
        actual = resolved_dict.get(key)
        if actual != expected:
            raise PSBConfigError(
                "Selected Base run does not match base_config at "
                f"{key!r}: expected {expected!r}, found {actual!r}."
            )
    return source_dict, resolved_dict


def _validate_p0_parent(parent_run: Path, base: PSBBaseSourceConfig) -> None:
    from utilities.psb_marl.checkpoint import sha256_file

    required = {
        "config_source.json",
        "deployment_manifest.json",
        "p0_equivalence.json",
        "p0_manual_validation.json",
        "training_status.json",
        "final_policy.pth",
        "final_critic.pth",
    }
    missing = sorted(name for name in required if not (parent_run / name).is_file())
    if missing:
        raise PSBConfigError(f"P1 parent_run is missing P0 artifacts: {missing}.")

    def load(name: str) -> Mapping[str, Any]:
        with (parent_run / name).open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), f"parent_run/{name}")

    source = load("config_source.json")
    manifest = load("deployment_manifest.json")
    proof = load("p0_equivalence.json")
    validation = load("p0_manual_validation.json")
    status = load("training_status.json")
    if (
        source.get("method") != "psb_marl"
        or source.get("stage") != "p0_base_passthrough"
    ):
        raise PSBConfigError("P1 parent_run must be a P0 run.")
    if status.get("status") != "completed":
        raise PSBConfigError("P1 parent P0 run is not completed.")
    if manifest.get("selected") != "base_passthrough":
        raise PSBConfigError("P1 parent P0 deployment is not Base passthrough.")
    if validation.get("noninferiority_result") != (
        "proven_by_identical_policy_checkpoint"
    ):
        raise PSBConfigError("P1 requires a manually validated P0 parent run.")
    rollouts = validation.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise PSBConfigError("P0 manual validation must contain rollout evidence.")
    if any(not isinstance(item, Mapping) for item in rollouts):
        raise PSBConfigError("P0 rollout evidence entries must be JSON objects.")
    if any(
        item.get("nonfinite_action_count") != 0
        or item.get("nonfinite_reward_count") != 0
        for item in rollouts
    ):
        raise PSBConfigError("P0 manual validation contains non-finite values.")

    policy_hash = sha256_file(parent_run / "final_policy.pth")
    critic_hash = sha256_file(parent_run / "final_critic.pth")
    expected_policy_hash = sha256_file(base.policy_checkpoint)
    expected_critic_hash = sha256_file(base.critic_checkpoint)
    if policy_hash != expected_policy_hash or critic_hash != expected_critic_hash:
        raise PSBConfigError("P1 parent P0 checkpoints no longer match Base.")
    if proof.get("policy_sha256") != policy_hash:
        raise PSBConfigError("P1 parent P0 policy proof is invalid.")
    if proof.get("critic_sha256") != critic_hash:
        raise PSBConfigError("P1 parent P0 critic proof is invalid.")


def _validate_p1_parent(parent_run: Path, base: PSBBaseSourceConfig) -> None:
    from utilities.psb_marl.checkpoint import sha256_file

    required = {
        "config_source.json",
        "deployment_manifest.json",
        "p1_certification.json",
        "p1_equivalence.json",
        "p1_manual_validation.json",
        "training_status.json",
        "final_policy.pth",
        "final_critic.pth",
        "final_psb_layer.pth",
    }
    missing = sorted(name for name in required if not (parent_run / name).is_file())
    if missing:
        raise PSBConfigError(f"P2 parent_run is missing P1 artifacts: {missing}.")

    def load(name: str) -> Mapping[str, Any]:
        with (parent_run / name).open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), f"parent_run/{name}")

    source = load("config_source.json")
    manifest = load("deployment_manifest.json")
    proof = load("p1_equivalence.json")
    certification = load("p1_certification.json")
    validation = load("p1_manual_validation.json")
    status = load("training_status.json")
    if (
        source.get("method") != "psb_marl"
        or source.get("stage") != "p1_zero_control_equivalence"
    ):
        raise PSBConfigError("P2 parent_run must be a P1 run.")
    if status.get("status") != "completed":
        raise PSBConfigError("P2 parent P1 run is not completed.")
    if manifest.get("selected") != "p1_zero_control_sidecar":
        raise PSBConfigError("P2 parent P1 deployment is not the certified sidecar.")
    if certification.get("passed") is not True:
        raise PSBConfigError("P2 requires a certified P1 proximal layer.")
    if validation.get("noninferiority_result") != "proven_by_exact_paired_actions":
        raise PSBConfigError("P2 requires an exactly paired validated P1 run.")
    comparisons = validation.get("paired_comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise PSBConfigError("P1 validation must contain paired comparisons.")
    if any(
        not isinstance(item, Mapping)
        or item.get("actions_exactly_equal") is not True
        or item.get("rewards_exactly_equal") is not True
        for item in comparisons
    ):
        raise PSBConfigError("P1 paired Base equivalence is invalid.")

    policy_hash = sha256_file(parent_run / "final_policy.pth")
    critic_hash = sha256_file(parent_run / "final_critic.pth")
    if policy_hash != sha256_file(base.policy_checkpoint):
        raise PSBConfigError("P2 parent P1 policy no longer matches Base.")
    if critic_hash != sha256_file(base.critic_checkpoint):
        raise PSBConfigError("P2 parent P1 critic no longer matches Base.")
    if proof.get("policy_sha256") != policy_hash:
        raise PSBConfigError("P2 parent P1 policy proof is invalid.")
    if proof.get("critic_sha256") != critic_hash:
        raise PSBConfigError("P2 parent P1 critic proof is invalid.")


def _validate_p3_parent(experiment: PSBExperimentConfig) -> None:
    """Require one P2.1-U candidate certified by the locked P2.2-R summary."""

    from utilities.psb_marl.checkpoint import sha256_file

    assert experiment.parent_run is not None
    assert experiment.robustness_summary is not None
    assert experiment.training is not None
    assert experiment.training_seed is not None
    parent = experiment.parent_run
    required = {
        "config_source.json",
        "deployment_manifest.json",
        "psb_config_resolved.json",
        "training_status.json",
        "candidate_policy.pth",
        "candidate_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
    }
    missing = sorted(name for name in required if not (parent / name).is_file())
    if missing:
        raise PSBConfigError(f"P3.0 parent_run is missing P2 artifacts: {missing}.")

    def load(path: Path, label: str) -> Mapping[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), label)

    source = load(parent / "config_source.json", "P3.0 parent source")
    resolved = load(parent / "psb_config_resolved.json", "P3.0 parent runtime")
    manifest = load(parent / "deployment_manifest.json", "P3.0 parent manifest")
    status = load(parent / "training_status.json", "P3.0 parent status")
    expected_runtime = experiment.source_p2_runtime_config()
    if (
        source.get("method") != "psb_marl"
        or source.get("stage") != "p2_frozen_base_bifurcation"
    ):
        raise PSBConfigError("P3.0 parent_run must be a PSB P2 run.")
    if resolved.get("runtime_config") != expected_runtime:
        raise PSBConfigError("P3.0 source P2 runtime does not match its parent run.")
    if (
        status.get("status") != "completed"
        or status.get("iteration") != experiment.training.iterations
    ):
        raise PSBConfigError("P3.0 parent P2 training is incomplete.")

    policy_hash = sha256_file(parent / "candidate_policy.pth")
    critic_hash = sha256_file(parent / "candidate_critic.pth")
    if manifest.get("candidate_policy_sha256") != policy_hash:
        raise PSBConfigError("P3.0 parent candidate policy proof is invalid.")
    if manifest.get("candidate_critic_sha256") != critic_hash:
        raise PSBConfigError("P3.0 parent candidate critic proof is invalid.")
    if sha256_file(parent / "base_fallback_policy.pth") != sha256_file(
        experiment.base.policy_checkpoint
    ):
        raise PSBConfigError("P3.0 parent Base fallback policy is invalid.")
    if sha256_file(parent / "base_fallback_critic.pth") != sha256_file(
        experiment.base.critic_checkpoint
    ):
        raise PSBConfigError("P3.0 parent Base fallback critic is invalid.")

    branch = expected_runtime["branch_adapter"]
    if (
        branch.get("conditioning_mode") != "supported_sector_q_gate"
        or branch.get("action_projection") != "longitudinal_only"
        or float(branch.get("max_delta_log_scale", -1.0)) != 0.0
    ):
        raise PSBConfigError("P3.0 requires the final P2.1-U actor contract.")

    summary = load(experiment.robustness_summary, "P2.2-R robustness summary")
    training_seed_count = summary.get("training_seed_count")
    if (
        summary.get("method") != "psb_marl_robustness"
        or summary.get("protocol") != "p2_2_r_locked_holdout"
        or summary.get("passed") is not True
        or type(training_seed_count) is not int
        or training_seed_count < 3
    ):
        raise PSBConfigError("P3.0 requires a passed P2.2-R robustness summary.")
    run_results = summary.get("run_results")
    if not isinstance(run_results, list):
        raise PSBConfigError("P2.2-R summary is missing run_results.")
    valid_results = [item for item in run_results if isinstance(item, Mapping)]
    certified_seeds = [item.get("training_seed") for item in valid_results]
    certified_hashes = [
        item.get("candidate_policy_sha256") for item in valid_results
    ]
    if (
        len(valid_results) != len(run_results)
        or len(valid_results) != training_seed_count
        or any(type(seed) is not int or seed < 0 for seed in certified_seeds)
        or len(set(certified_seeds)) != training_seed_count
        or any(
            not isinstance(candidate_hash, str) or not candidate_hash
            for candidate_hash in certified_hashes
        )
        or len(set(certified_hashes)) != training_seed_count
        or any(item.get("passed") is not True for item in valid_results)
    ):
        raise PSBConfigError(
            "P3.0 requires distinct, passed P2.2-R training-seed runs."
        )
    matches = [
        item
        for item in valid_results
        if Path(str(item.get("run_directory", ""))).expanduser().resolve()
        == parent
    ]
    if len(matches) != 1:
        raise PSBConfigError("P3.0 parent is not uniquely certified by P2.2-R.")
    certified = matches[0]
    if (
        certified.get("passed") is not True
        or certified.get("training_seed") != experiment.training_seed
        or certified.get("candidate_policy_sha256") != policy_hash
    ):
        raise PSBConfigError("P3.0 parent P2.2-R certification is invalid.")
    scenarios = summary.get("scenario_summaries")
    if not isinstance(scenarios, Mapping) or not scenarios or any(
        not isinstance(item, Mapping)
        or item.get("all_training_seeds_passed") is not True
        for item in scenarios.values()
    ):
        raise PSBConfigError("P3.0 requires every locked P2.2-R scenario to pass.")


def _validate_p31_parent(
    parent: Path,
    base: PSBBaseSourceConfig,
) -> tuple[Dict[str, object], PSBConflictGraphConfig]:
    """Validate the completed P3.0 bridge used by critic-only P3.1."""

    from utilities.psb_marl.checkpoint import sha256_file

    required = {
        "config_source.json",
        "deployment_manifest.json",
        "p3_0_equivalence.json",
        "p3_0_paired_equivalence.json",
        "psb_config_resolved.json",
        "training_status.json",
        "candidate_policy.pth",
        "candidate_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
        "final_policy.pth",
        "final_critic.pth",
    }
    missing = sorted(name for name in required if not (parent / name).is_file())
    if missing:
        raise PSBConfigError(f"P3.1 parent_run is missing P3.0 artifacts: {missing}.")

    def load(name: str) -> Mapping[str, Any]:
        with (parent / name).open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), f"P3.1 parent {name}")

    source = load("config_source.json")
    manifest = load("deployment_manifest.json")
    proof = load("p3_0_equivalence.json")
    report = load("p3_0_paired_equivalence.json")
    resolved = load("psb_config_resolved.json")
    status = load("training_status.json")
    if (
        source.get("method") != "psb_marl"
        or source.get("stage") != "p3_paired_rollout_equivalence"
        or manifest.get("stage") != "p3_paired_rollout_equivalence"
        or manifest.get("selected") != "base_fallback_p3_pairing_only"
        or manifest.get("learning_enabled") is not False
        or status.get("status") != "completed"
        or status.get("iteration") != 0
    ):
        raise PSBConfigError("P3.1 parent_run is not a completed safe P3.0 run.")
    artifact_checks = report.get("artifact_checks")
    equivalence_items = report.get("source_equivalence")
    paired_batches = report.get("paired_batches")
    if (
        report.get("passed") is not True
        or report.get("source_equivalence_passed") is not True
        or report.get("paired_contract_passed") is not True
        or not isinstance(artifact_checks, Mapping)
        or not artifact_checks
        or any(value is not True for value in artifact_checks.values())
        or not isinstance(equivalence_items, list)
        or len(equivalence_items) < 2
        or any(
            not isinstance(item, Mapping)
            or item.get("actions_exactly_equal") is not True
            or item.get("rewards_exactly_equal") is not True
            for item in equivalence_items
        )
        or not isinstance(paired_batches, list)
        or len(paired_batches) != len(equivalence_items)
        or any(
            not isinstance(item, Mapping) or item.get("finite") is not True
            for item in paired_batches
        )
    ):
        raise PSBConfigError("P3.1 requires a passed P3.0 paired report.")

    candidate_policy_hash = sha256_file(parent / "candidate_policy.pth")
    candidate_critic_hash = sha256_file(parent / "candidate_critic.pth")
    base_policy_hash = sha256_file(base.policy_checkpoint)
    base_critic_hash = sha256_file(base.critic_checkpoint)
    hash_checks = (
        manifest.get("candidate_policy_sha256") == candidate_policy_hash,
        manifest.get("candidate_critic_sha256") == candidate_critic_hash,
        proof.get("candidate_policy_sha256") == candidate_policy_hash,
        proof.get("candidate_critic_sha256") == candidate_critic_hash,
        sha256_file(parent / "base_fallback_policy.pth") == base_policy_hash,
        sha256_file(parent / "base_fallback_critic.pth") == base_critic_hash,
        sha256_file(parent / "final_policy.pth") == base_policy_hash,
        sha256_file(parent / "final_critic.pth") == base_critic_hash,
    )
    if not all(hash_checks):
        raise PSBConfigError("P3.1 parent checkpoint integrity is invalid.")

    runtime = resolved.get("runtime_config")
    if not isinstance(runtime, Mapping):
        raise PSBConfigError("P3.1 parent is missing its P3.0 runtime.")
    source_p2_runtime = runtime.get("source_p2_runtime")
    if not isinstance(source_p2_runtime, Mapping):
        raise PSBConfigError("P3.1 parent is missing its source P2 runtime.")
    source_p2_runtime = dict(source_p2_runtime)
    branch = source_p2_runtime.get("branch_adapter")
    if (
        source_p2_runtime.get("stage") != "p2_frozen_base_bifurcation"
        or source_p2_runtime.get("freeze_base_actor") is not True
        or source_p2_runtime.get("base_policy_checkpoint")
        != str(base.policy_checkpoint)
        or source_p2_runtime.get("base_critic_checkpoint")
        != str(base.critic_checkpoint)
        or not isinstance(branch, Mapping)
        or branch.get("conditioning_mode") != "supported_sector_q_gate"
        or branch.get("action_projection") != "longitudinal_only"
        or float(branch.get("max_delta_log_scale", -1.0)) != 0.0
    ):
        raise PSBConfigError("P3.1 source P2 runtime violates the P2.1-U contract.")
    conflict_graph = PSBConflictGraphConfig.from_dict(source["conflict_graph"])
    return source_p2_runtime, conflict_graph


def _validate_p32_parent(
    parent: Path,
    base: PSBBaseSourceConfig,
) -> tuple[Dict[str, object], PSBConflictGraphConfig]:
    """Require a certified and manually validated P3.1 parent."""

    from utilities.psb_marl.checkpoint import sha256_file

    required = {
        "config_source.json",
        "deployment_manifest.json",
        "psb_config_resolved.json",
        "p3_1_certification.json",
        "p3_1_manual_validation.json",
        "training_status.json",
        "candidate_policy.pth",
        "candidate_critic.pth",
        "source_p2_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
    }
    missing = sorted(name for name in required if not (parent / name).is_file())
    if missing:
        raise PSBConfigError(f"P3.2 parent is missing artifacts: {missing}.")

    def load(name: str) -> Mapping[str, Any]:
        with (parent / name).open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), f"P3.2 parent {name}")

    source = load("config_source.json")
    manifest = load("deployment_manifest.json")
    resolved = load("psb_config_resolved.json")
    certification = load("p3_1_certification.json")
    validation = load("p3_1_manual_validation.json")
    status = load("training_status.json")
    if (
        source.get("stage") != "p3_differential_critic"
        or manifest.get("stage") != "p3_differential_critic"
        or manifest.get("selected") != "base_fallback_p3_critic_only"
        or certification.get("passed") is not True
        or validation.get("passed") is not True
        or validation.get("critic_passed") is not True
        or status.get("status") != "completed"
    ):
        raise PSBConfigError("P3.2 requires a passed P3.1 parent run.")
    if (
        manifest.get("candidate_policy_sha256")
        != sha256_file(parent / "candidate_policy.pth")
        or manifest.get("candidate_critic_sha256")
        != sha256_file(parent / "candidate_critic.pth")
        or manifest.get("source_p2_critic_sha256")
        != sha256_file(parent / "source_p2_critic.pth")
    ):
        raise PSBConfigError("P3.2 parent learned artifact integrity failed.")
    channel_quality = validation.get("critic_channel_quality")
    if not isinstance(channel_quality, Mapping) or channel_quality.get(
        "passed"
    ) is not True:
        raise PSBConfigError("P3.2 requires every P3.1 critic head to pass.")
    runtime = resolved.get("runtime_config")
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get("source_p2_runtime"), Mapping
    ):
        raise PSBConfigError("P3.2 parent is missing source P2 runtime.")
    if sha256_file(parent / "base_fallback_policy.pth") != sha256_file(
        base.policy_checkpoint
    ) or sha256_file(parent / "base_fallback_critic.pth") != sha256_file(
        base.critic_checkpoint
    ):
        raise PSBConfigError("P3.2 parent Base fallback integrity failed.")
    p30_parent = Path(str(source["parent_run"])).expanduser().resolve()
    with (p30_parent / "config_source.json").open(
        "r", encoding="utf-8"
    ) as stream:
        p30_source = _object(json.load(stream), "P3.0 source")
    conflict_graph = PSBConflictGraphConfig.from_dict(
        p30_source["conflict_graph"]
    )
    return dict(runtime["source_p2_runtime"]), conflict_graph


def _validate_p5_parent(
    parent: Path,
    base: PSBBaseSourceConfig,
) -> tuple[Dict[str, object], PSBConflictGraphConfig]:
    """Require a completed P3.3 run while preserving the immutable Base."""

    from utilities.psb_marl.checkpoint import sha256_file

    required = {
        "config_source.json",
        "deployment_manifest.json",
        "psb_config_resolved.json",
        "training_status.json",
        "candidate_policy.pth",
        "candidate_critic.pth",
        "candidate_differential_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
    }
    missing = sorted(name for name in required if not (parent / name).is_file())
    if missing:
        raise PSBConfigError(f"P5 parent is missing artifacts: {missing}.")

    def load(name: str) -> Mapping[str, Any]:
        with (parent / name).open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), f"P5 parent {name}")

    source = load("config_source.json")
    manifest = load("deployment_manifest.json")
    resolved = load("psb_config_resolved.json")
    status = load("training_status.json")
    runtime = resolved.get("runtime_config")
    if (
        source.get("stage") != "p3_paired_differential_primal_dual_ppo"
        or manifest.get("stage")
        != "p3_paired_differential_primal_dual_ppo"
        or manifest.get("paired_differential_learning_enabled") is not True
        or manifest.get("paired_episode_boundaries_synchronized") is not True
        or status.get("status") != "completed"
        or not isinstance(runtime, Mapping)
    ):
        raise PSBConfigError("P5 requires a completed synchronized P3.3 parent.")
    hash_checks = (
        manifest.get("candidate_policy_sha256")
        == sha256_file(parent / "candidate_policy.pth"),
        manifest.get("candidate_critic_sha256")
        == sha256_file(parent / "candidate_critic.pth"),
        manifest.get("candidate_differential_critic_sha256")
        == sha256_file(parent / "candidate_differential_critic.pth"),
        sha256_file(parent / "base_fallback_policy.pth")
        == sha256_file(base.policy_checkpoint),
        sha256_file(parent / "base_fallback_critic.pth")
        == sha256_file(base.critic_checkpoint),
    )
    if not all(hash_checks):
        raise PSBConfigError("P5 parent checkpoint integrity is invalid.")
    p31_parent = Path(str(source["parent_run"])).expanduser().resolve()
    source_p2_runtime, conflict_graph = _validate_p32_parent(p31_parent, base)
    branch = source_p2_runtime.get("branch_adapter")
    if (
        not isinstance(branch, Mapping)
        or branch.get("conditioning_mode") != "supported_sector_q_gate"
        or branch.get("action_projection") != "longitudinal_only"
        or float(branch.get("max_delta_log_scale", -1.0)) != 0.0
    ):
        raise PSBConfigError("P5 requires the locked P2.1-U branch contract.")
    return source_p2_runtime, conflict_graph


def load_psb_experiment(path: Path) -> PSBExperimentConfig:
    """Load and fully validate one PSB stage configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"PSB config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = _object(json.load(stream), "root")
    required_discriminator_keys = {"schema_version", "method", "stage"}
    missing_discriminators = sorted(required_discriminator_keys - set(raw))
    if missing_discriminators:
        raise PSBConfigError(
            f"root is missing stage discriminator keys: {missing_discriminators}."
        )
    if raw["schema_version"] != 1:
        raise PSBConfigError("PSB schema_version must be 1.")
    if raw["method"] != "psb_marl":
        raise PSBConfigError("PSB method must be 'psb_marl'.")
    stage = _string(raw["stage"], "stage")
    common_keys = {
        "schema_version",
        "method",
        "stage",
        "base_config",
        "output_root",
        "base",
    }
    if stage == "p0_base_passthrough":
        _exact_keys(raw, common_keys, "root")
    elif stage == "p1_zero_control_equivalence":
        _exact_keys(
            raw,
            common_keys | {"parent_run", "conflict_graph", "proximal"},
            "root",
        )
    elif stage == "p2_frozen_base_bifurcation":
        required = common_keys | {
            "parent_run",
            "conflict_graph",
            "proximal",
            "control",
            "branch_adapter",
            "training",
            "promotion",
        }
        allowed = required | {"training_seed"}
        actual = set(raw)
        if not required.issubset(actual) or not actual.issubset(allowed):
            raise PSBConfigError(
                "root has invalid keys: "
                f"missing={sorted(required - actual)}, "
                f"extra={sorted(actual - allowed)}."
            )
    elif stage == "p3_paired_rollout_equivalence":
        _exact_keys(
            raw,
            common_keys
            | {
                "parent_run",
                "robustness_summary",
                "training_seed",
                "conflict_graph",
                "proximal",
                "control",
                "branch_adapter",
                "training",
                "promotion",
                "paired_rollout",
            },
            "root",
        )
    elif stage == "p3_differential_critic":
        _exact_keys(
            raw,
            common_keys
            | {
                "parent_run",
                "training_seed",
                "differential_critic",
            },
            "root",
        )
    elif stage == "p3_primal_dual_ppo":
        _exact_keys(
            raw,
            common_keys
            | {"parent_run", "training_seed", "primal_dual"},
            "root",
        )
    elif stage == "p3_paired_differential_primal_dual_ppo":
        _exact_keys(
            raw,
            common_keys
            | {
                "parent_run",
                "training_seed",
                "primal_dual",
                "paired_differential",
            },
            "root",
        )
    elif stage == "p5_joint_psb_marl":
        _exact_keys(
            raw,
            common_keys
            | {
                "parent_run",
                "training_seed",
                "primal_dual",
                "paired_differential",
                "joint_training",
            },
            "root",
        )
    else:
        raise PSBConfigError(
            "Supported PSB stages are P0 Base passthrough, P1 zero-control "
            "equivalence, P2 frozen-Base bifurcation, and P3.0 paired "
            "rollout equivalence, P3.1 differential critic, P3.2 "
            "primal-dual PPO, P3.3 paired differential primal-dual PPO, "
            "and P5 joint PSB-MARL."
        )
    base_config_path = _resolve_existing_path(
        raw["base_config"], "base_config", config_path, kind="file"
    )
    base = PSBBaseSourceConfig.from_dict(raw["base"], config_path)
    base_source, base_resolved = _validate_base_run(base_config_path, base)
    output_root = _string(raw["output_root"], "output_root")
    parent_run = None
    conflict_graph = None
    proximal = None
    control = None
    branch_adapter = None
    training = None
    promotion = None
    training_seed = None
    robustness_summary = None
    paired_rollout = None
    differential_critic = None
    primal_dual = None
    paired_differential = None
    joint_training = None
    source_p2_runtime = None
    if stage in {
        "p3_primal_dual_ppo",
        "p3_paired_differential_primal_dual_ppo",
    }:
        parent_run = _resolve_existing_path(
            raw["parent_run"], "parent_run", config_path, kind="directory"
        )
        source_p2_runtime, conflict_graph = _validate_p32_parent(
            parent_run, base
        )
        training_seed = _integer(
            raw["training_seed"], "training_seed", minimum=0
        )
        primal_dual = PSBP32PrimalDualConfig.from_dict(raw["primal_dual"])
        if stage == "p3_paired_differential_primal_dual_ppo":
            paired_differential = PSBP33PairedDifferentialConfig.from_dict(
                raw["paired_differential"]
            )
        if conflict_graph.candidate_count != int(
            base_resolved["n_nearing_agents_observed"]
        ):
            raise PSBConfigError(
                f"{stage} conflict graph does not match the Base config."
            )
    if stage == "p5_joint_psb_marl":
        parent_run = _resolve_existing_path(
            raw["parent_run"], "parent_run", config_path, kind="directory"
        )
        source_p2_runtime, conflict_graph = _validate_p5_parent(
            parent_run, base
        )
        training_seed = _integer(
            raw["training_seed"], "training_seed", minimum=0
        )
        primal_dual = PSBP32PrimalDualConfig.from_dict(raw["primal_dual"])
        paired_differential = PSBP33PairedDifferentialConfig.from_dict(
            raw["paired_differential"]
        )
        joint_training = PSBP5JointTrainingConfig.from_dict(
            raw["joint_training"]
        )
        if conflict_graph.candidate_count != int(
            base_resolved["n_nearing_agents_observed"]
        ):
            raise PSBConfigError(
                "P5 conflict graph does not match the Base config."
            )
    if stage == "p3_differential_critic":
        parent_run = _resolve_existing_path(
            raw["parent_run"], "parent_run", config_path, kind="directory"
        )
        source_p2_runtime, conflict_graph = _validate_p31_parent(
            parent_run, base
        )
        training_seed = _integer(
            raw["training_seed"], "training_seed", minimum=0
        )
        differential_critic = PSBP31DifferentialCriticConfig.from_dict(
            raw["differential_critic"]
        )
        from utilities.constants import SCENARIOS

        if differential_critic.collection_scenario not in SCENARIOS:
            raise PSBConfigError(
                "P3.1 differential_critic.collection_scenario is unknown."
            )
        if conflict_graph.candidate_count != int(
            base_resolved["n_nearing_agents_observed"]
        ):
            raise PSBConfigError(
                "P3.1 parent conflict graph does not match the Base config."
            )
        if differential_critic.gamma != float(base_resolved["gamma"]):
            raise PSBConfigError(
                "P3.1 differential_critic.gamma must equal the Base gamma."
            )
    if stage in {
        "p1_zero_control_equivalence",
        "p2_frozen_base_bifurcation",
        "p3_paired_rollout_equivalence",
    }:
        parent_run = _resolve_existing_path(
            raw["parent_run"], "parent_run", config_path, kind="directory"
        )
        if stage == "p1_zero_control_equivalence":
            _validate_p0_parent(parent_run, base)
        elif stage == "p2_frozen_base_bifurcation":
            _validate_p1_parent(parent_run, base)
        conflict_graph = PSBConflictGraphConfig.from_dict(raw["conflict_graph"])
        proximal = PSBProximalConfig.from_dict(
            raw["proximal"],
            require_zero_control=stage == "p1_zero_control_equivalence",
        )
        if conflict_graph.candidate_count != int(
            base_resolved["n_nearing_agents_observed"]
        ):
            raise PSBConfigError(
                "conflict_graph.candidate_count must equal the Base local "
                "neighbor count."
            )
        dt = _number(base_resolved.get("dt"), "base run dt", strictly_positive=True)
        if proximal.convexity_margin(dt) <= 0.0:
            raise PSBConfigError(
                f"{stage} violates proximal uniqueness: 1/h_z + kappa must be "
                "greater than rho_max*nu*alpha."
            )
        if stage in {
            "p2_frozen_base_bifurcation",
            "p3_paired_rollout_equivalence",
        }:
            if "training_seed" in raw:
                training_seed = _integer(
                    raw["training_seed"], "training_seed", minimum=0
                )
            control = PSBControlConfig.from_dict(raw["control"])
            branch_adapter = PSBBranchAdapterConfig.from_dict(
                raw["branch_adapter"]
            )
            training = PSBP2TrainingConfig.from_dict(raw["training"])
            promotion = PSBPromotionConfig.from_dict(raw["promotion"])
            if training.chunk_length > int(base_resolved["max_steps"]):
                raise PSBConfigError(
                    "training.chunk_length must not exceed Base max_steps."
                )
            if training.chunk_length > int(base_resolved["minibatch_size"]):
                raise PSBConfigError(
                    "training.chunk_length must not exceed Base minibatch_size."
                )
            if bool(base_resolved.get("is_prb", False)):
                raise PSBConfigError(
                    "P2 sequence PPO does not support prioritized replay."
                )
            if bool(base_resolved.get("is_using_prioritized_marl", False)):
                raise PSBConfigError(
                    "P2 does not support the separate MARL priority policy."
                )
            if stage == "p3_paired_rollout_equivalence":
                robustness_summary = _resolve_existing_path(
                    raw["robustness_summary"],
                    "robustness_summary",
                    config_path,
                    kind="file",
                )
                paired_rollout = PSBP3PairedRolloutConfig.from_dict(
                    raw["paired_rollout"]
                )
    result = PSBExperimentConfig(
        config_path=config_path,
        source_config=dict(raw),
        base_config_path=base_config_path,
        base_source_config=base_source,
        base_run_config=base_resolved,
        output_root=output_root,
        base=base,
        stage=stage,
        parent_run=parent_run,
        conflict_graph=conflict_graph,
        proximal=proximal,
        control=control,
        branch_adapter=branch_adapter,
        training=training,
        promotion=promotion,
        training_seed=training_seed,
        robustness_summary=robustness_summary,
        paired_rollout=paired_rollout,
        differential_critic=differential_critic,
        primal_dual=primal_dual,
        paired_differential=paired_differential,
        joint_training=joint_training,
        source_p2_runtime=source_p2_runtime,
    )
    if stage == "p3_paired_rollout_equivalence":
        _validate_p3_parent(result)
    return result
