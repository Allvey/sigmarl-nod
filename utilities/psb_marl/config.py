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
    else:
        raise PSBConfigError(
            "Supported PSB stages are P0 Base passthrough, P1 zero-control "
            "equivalence, and P2 frozen-Base bifurcation."
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
    if stage in {
        "p1_zero_control_equivalence",
        "p2_frozen_base_bifurcation",
    }:
        parent_run = _resolve_existing_path(
            raw["parent_run"], "parent_run", config_path, kind="directory"
        )
        if stage == "p1_zero_control_equivalence":
            _validate_p0_parent(parent_run, base)
        else:
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
        if stage == "p2_frozen_base_bifurcation":
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
    return PSBExperimentConfig(
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
    )
