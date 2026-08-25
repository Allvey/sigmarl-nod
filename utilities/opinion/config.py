"""Typed configuration boundary for the independent Opinion-MARL method.

The loader keeps SigmaRL's ``Parameters`` object free of Opinion and historical
TSC fields.  The Base configuration and the Opinion configuration travel next
to each other and are only combined by the future Opinion trainer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from utilities.helper_training import Parameters


OPINION_CONFIG_SCHEMA_VERSION = 1
OPINION_METHOD = "opinion_marl"
OPINION_STAGES = {"base", "evidence", "sequence", "sequence_ppo", "joint"}


class OpinionConfigError(ValueError):
    """Raised before training when an Opinion experiment is inconsistent."""


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OpinionConfigError(f"{location} must be a JSON object.")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set, location: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise OpinionConfigError(f"Invalid keys at {location}: {', '.join(details)}")


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise OpinionConfigError(f"{location} must be a boolean.")
    return value


def _integer(value: Any, location: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise OpinionConfigError(
            f"{location} must be an integer greater than or equal to {minimum}."
        )
    return value


def _number(
    value: Any,
    location: str,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise OpinionConfigError(f"{location} must be a finite number.")
    result = float(value)
    if strictly_positive and result <= minimum:
        raise OpinionConfigError(f"{location} must be greater than {minimum}.")
    if not strictly_positive and result < minimum:
        raise OpinionConfigError(
            f"{location} must be greater than or equal to {minimum}."
        )
    return result


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpinionConfigError(f"{location} must be a non-empty string.")
    return value


def _optional_string(value: Any, location: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, location)


@dataclass(frozen=True)
class ConflictGraphConfig:
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConflictGraphConfig":
        raw = _object(raw, "opinion.conflict_graph")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.conflict_graph")
        result = cls(
            emit_pair_info=_boolean(
                raw["emit_pair_info"],
                "opinion.conflict_graph.emit_pair_info",
            ),
            candidate_count=_integer(
                raw["candidate_count"], "opinion.conflict_graph.candidate_count"
            ),
            pair_feature_dim=_integer(
                raw["pair_feature_dim"], "opinion.conflict_graph.pair_feature_dim"
            ),
            prediction_horizon_seconds=_number(
                raw["prediction_horizon_seconds"],
                "opinion.conflict_graph.prediction_horizon_seconds",
                strictly_positive=True,
            ),
            conflict_distance_meters=_number(
                raw["conflict_distance_meters"],
                "opinion.conflict_graph.conflict_distance_meters",
                strictly_positive=True,
            ),
            sensing_distance_meters=_number(
                raw["sensing_distance_meters"],
                "opinion.conflict_graph.sensing_distance_meters",
                strictly_positive=True,
            ),
            cpa_epsilon=_number(
                raw["cpa_epsilon"],
                "opinion.conflict_graph.cpa_epsilon",
                strictly_positive=True,
            ),
            urgency_time_scale_seconds=_number(
                raw["urgency_time_scale_seconds"],
                "opinion.conflict_graph.urgency_time_scale_seconds",
                strictly_positive=True,
            ),
            urgency_distance_scale_meters=_number(
                raw["urgency_distance_scale_meters"],
                "opinion.conflict_graph.urgency_distance_scale_meters",
                strictly_positive=True,
            ),
        )
        if result.candidate_count != 2:
            raise OpinionConfigError(
                "The first method version requires conflict_graph.candidate_count=2."
            )
        if result.pair_feature_dim != 10:
            raise OpinionConfigError(
                "The first method version requires conflict_graph.pair_feature_dim=10."
            )
        if result.conflict_distance_meters >= result.sensing_distance_meters:
            raise OpinionConfigError(
                "conflict_distance_meters must be smaller than sensing_distance_meters."
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceConfig:
    hidden_sizes: Tuple[int, ...]
    b_max: float
    temperature: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvidenceConfig":
        raw = _object(raw, "opinion.evidence")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.evidence")
        hidden_raw = raw["hidden_sizes"]
        if not isinstance(hidden_raw, list) or not hidden_raw:
            raise OpinionConfigError(
                "opinion.evidence.hidden_sizes must be a non-empty integer list."
            )
        hidden_sizes = tuple(
            _integer(value, f"opinion.evidence.hidden_sizes[{index}]")
            for index, value in enumerate(hidden_raw)
        )
        return cls(
            hidden_sizes=hidden_sizes,
            b_max=_number(
                raw["b_max"], "opinion.evidence.b_max", strictly_positive=True
            ),
            temperature=_number(
                raw["temperature"],
                "opinion.evidence.temperature",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class DynamicsConfig:
    response_rate: float
    decay_rate: float
    self_reinforcement: float
    nonlinear_sensitivity: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DynamicsConfig":
        raw = _object(raw, "opinion.dynamics")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.dynamics")
        return cls(
            response_rate=_number(
                raw["response_rate"],
                "opinion.dynamics.response_rate",
                strictly_positive=True,
            ),
            decay_rate=_number(
                raw["decay_rate"],
                "opinion.dynamics.decay_rate",
                strictly_positive=True,
            ),
            self_reinforcement=_number(
                raw["self_reinforcement"],
                "opinion.dynamics.self_reinforcement",
            ),
            nonlinear_sensitivity=_number(
                raw["nonlinear_sensitivity"],
                "opinion.dynamics.nonlinear_sensitivity",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class ResidualConfig:
    opinion_scale: float
    gain: float
    max_abs: float
    action_index: int
    apply_to_action: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResidualConfig":
        raw = _object(raw, "opinion.residual")
        raw = dict(raw)
        raw.setdefault("apply_to_action", True)
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.residual")
        result = cls(
            opinion_scale=_number(
                raw["opinion_scale"],
                "opinion.residual.opinion_scale",
                strictly_positive=True,
            ),
            gain=_number(
                raw["gain"], "opinion.residual.gain", strictly_positive=True
            ),
            max_abs=_number(
                raw["max_abs"], "opinion.residual.max_abs", strictly_positive=True
            ),
            action_index=_integer(
                raw["action_index"], "opinion.residual.action_index", minimum=0
            ),
            apply_to_action=_boolean(
                raw["apply_to_action"],
                "opinion.residual.apply_to_action",
            ),
        )
        if result.action_index != 0:
            raise OpinionConfigError(
                "The first method version only modifies action_index=0 (speed)."
            )
        if result.gain > result.max_abs or result.max_abs > 1.0:
            raise OpinionConfigError(
                "residual gain must be <= max_abs, and max_abs must be <= 1."
            )
        return result


@dataclass(frozen=True)
class PolicyBridgeConfig:
    enabled: bool
    mode: str
    base_output_root: Optional[str]
    freeze_base_actor: bool
    visualize_agent_id: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyBridgeConfig":
        raw = _object(raw, "opinion.policy_bridge")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.policy_bridge")
        result = cls(
            enabled=_boolean(raw["enabled"], "opinion.policy_bridge.enabled"),
            mode=_string(raw["mode"], "opinion.policy_bridge.mode").lower(),
            base_output_root=_optional_string(
                raw["base_output_root"],
                "opinion.policy_bridge.base_output_root",
            ),
            freeze_base_actor=_boolean(
                raw["freeze_base_actor"],
                "opinion.policy_bridge.freeze_base_actor",
            ),
            visualize_agent_id=_integer(
                raw["visualize_agent_id"],
                "opinion.policy_bridge.visualize_agent_id",
                minimum=0,
            ),
        )
        if result.mode not in {"direct_evidence", "stateful_opinion"}:
            raise OpinionConfigError(
                "policy_bridge.mode must be 'direct_evidence' or "
                "'stateful_opinion'."
            )
        return result


@dataclass(frozen=True)
class StatefulOpinionConfig:
    enabled: bool
    evidence_output_root: Optional[str]
    freeze_evidence: bool
    zero_threshold: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StatefulOpinionConfig":
        raw = _object(raw, "opinion.stateful")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.stateful")
        result = cls(
            enabled=_boolean(raw["enabled"], "opinion.stateful.enabled"),
            evidence_output_root=_optional_string(
                raw["evidence_output_root"],
                "opinion.stateful.evidence_output_root",
            ),
            freeze_evidence=_boolean(
                raw["freeze_evidence"], "opinion.stateful.freeze_evidence"
            ),
            zero_threshold=_number(
                raw["zero_threshold"],
                "opinion.stateful.zero_threshold",
                strictly_positive=True,
            ),
        )
        if not result.enabled and result.evidence_output_root is not None:
            raise OpinionConfigError(
                "Disabled stateful opinion requires evidence_output_root=null."
            )
        return result


@dataclass(frozen=True)
class SequencePPOConfig:
    enabled: bool
    train_evidence: bool
    source_output_root: Optional[str]
    chunk_length: int
    evidence_learning_rate_scale: float
    neutral_loss_coefficient: float
    magnitude_loss_coefficient: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SequencePPOConfig":
        raw = _object(raw, "opinion.sequence_ppo")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.sequence_ppo")
        result = cls(
            enabled=_boolean(
                raw["enabled"], "opinion.sequence_ppo.enabled"
            ),
            train_evidence=_boolean(
                raw["train_evidence"], "opinion.sequence_ppo.train_evidence"
            ),
            source_output_root=_optional_string(
                raw["source_output_root"],
                "opinion.sequence_ppo.source_output_root",
            ),
            chunk_length=_integer(
                raw["chunk_length"], "opinion.sequence_ppo.chunk_length", minimum=2
            ),
            evidence_learning_rate_scale=_number(
                raw["evidence_learning_rate_scale"],
                "opinion.sequence_ppo.evidence_learning_rate_scale",
                strictly_positive=True,
            ),
            neutral_loss_coefficient=_number(
                raw["neutral_loss_coefficient"],
                "opinion.sequence_ppo.neutral_loss_coefficient",
            ),
            magnitude_loss_coefficient=_number(
                raw["magnitude_loss_coefficient"],
                "opinion.sequence_ppo.magnitude_loss_coefficient",
            ),
        )
        if result.evidence_learning_rate_scale > 1.0:
            raise OpinionConfigError(
                "evidence_learning_rate_scale must be <= 1 through M5."
            )
        if not result.enabled and result.source_output_root is not None:
            raise OpinionConfigError(
                "Disabled sequence_ppo requires source_output_root=null."
            )
        if result.train_evidence and not result.enabled:
            raise OpinionConfigError(
                "sequence_ppo.train_evidence=true requires sequence_ppo.enabled=true."
            )
        return result


@dataclass(frozen=True)
class OpinionTrainerConfig:
    enabled: bool
    mode: str
    initialization: str
    source_output_root: Optional[str]
    evidence_warmup_iterations: int
    base_actor_learning_rate_scale: float
    critic_learning_rate_scale: float
    base_anchor_coefficient: float
    checkpoint_interval: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpinionTrainerConfig":
        raw = _object(raw, "opinion.trainer")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.trainer")
        result = cls(
            enabled=_boolean(raw["enabled"], "opinion.trainer.enabled"),
            mode=_string(raw["mode"], "opinion.trainer.mode").lower(),
            initialization=_string(
                raw["initialization"], "opinion.trainer.initialization"
            ).lower(),
            source_output_root=_optional_string(
                raw["source_output_root"],
                "opinion.trainer.source_output_root",
            ),
            evidence_warmup_iterations=_integer(
                raw["evidence_warmup_iterations"],
                "opinion.trainer.evidence_warmup_iterations",
                minimum=0,
            ),
            base_actor_learning_rate_scale=_number(
                raw["base_actor_learning_rate_scale"],
                "opinion.trainer.base_actor_learning_rate_scale",
                strictly_positive=True,
            ),
            critic_learning_rate_scale=_number(
                raw["critic_learning_rate_scale"],
                "opinion.trainer.critic_learning_rate_scale",
                strictly_positive=True,
            ),
            base_anchor_coefficient=_number(
                raw["base_anchor_coefficient"],
                "opinion.trainer.base_anchor_coefficient",
            ),
            checkpoint_interval=_integer(
                raw["checkpoint_interval"],
                "opinion.trainer.checkpoint_interval",
                minimum=1,
            ),
        )
        if result.mode not in {
            "disabled",
            "evidence_only",
            "joint",
            "warmup_then_joint",
        }:
            raise OpinionConfigError(
                "trainer.mode must be disabled, evidence_only, joint, or "
                "warmup_then_joint."
            )
        if result.initialization not in {"none", "base", "opinion"}:
            raise OpinionConfigError(
                "trainer.initialization must be none, base, or opinion."
            )
        for name, value in (
            ("base_actor_learning_rate_scale", result.base_actor_learning_rate_scale),
            ("critic_learning_rate_scale", result.critic_learning_rate_scale),
        ):
            if value > 1.0:
                raise OpinionConfigError(f"{name} must be <= 1.")
        if result.base_anchor_coefficient < 0.0:
            raise OpinionConfigError("base_anchor_coefficient must be >= 0.")
        if not result.enabled:
            if result.mode != "disabled" or result.initialization != "none":
                raise OpinionConfigError(
                    "Disabled trainer requires mode='disabled' and "
                    "initialization='none'."
                )
            if result.source_output_root is not None:
                raise OpinionConfigError(
                    "Disabled trainer requires source_output_root=null."
                )
            if result.evidence_warmup_iterations != 0:
                raise OpinionConfigError(
                    "Disabled trainer requires evidence_warmup_iterations=0."
                )
            return result
        if result.mode == "disabled":
            raise OpinionConfigError(
                "Enabled trainer requires an active mode."
            )
        if result.initialization == "none":
            if result.mode != "joint":
                raise OpinionConfigError(
                    "From-scratch initialization is only supported by "
                    "trainer.mode='joint'."
                )
            if result.source_output_root is not None:
                raise OpinionConfigError(
                    "From-scratch initialization requires source_output_root=null."
                )
            if result.base_anchor_coefficient != 0.0:
                raise OpinionConfigError(
                    "From-scratch initialization requires "
                    "base_anchor_coefficient=0."
                )
        if result.initialization == "opinion" and result.source_output_root is None:
            raise OpinionConfigError(
                "Opinion initialization requires trainer.source_output_root."
            )
        if result.initialization == "base" and result.source_output_root is not None:
            raise OpinionConfigError(
                "Base initialization uses policy_bridge.base_output_root and "
                "requires trainer.source_output_root=null."
            )
        if result.mode == "warmup_then_joint":
            if result.evidence_warmup_iterations < 1:
                raise OpinionConfigError(
                    "warmup_then_joint requires evidence_warmup_iterations >= 1."
                )
        elif result.evidence_warmup_iterations != 0:
            raise OpinionConfigError(
                "Only warmup_then_joint may set evidence_warmup_iterations."
            )
        return result


@dataclass(frozen=True)
class OpinionConfig:
    conflict_graph: ConflictGraphConfig
    evidence: EvidenceConfig
    dynamics: DynamicsConfig
    residual: ResidualConfig
    policy_bridge: PolicyBridgeConfig
    stateful: StatefulOpinionConfig
    sequence_ppo: SequencePPOConfig
    trainer: OpinionTrainerConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpinionConfig":
        raw = _object(raw, "opinion")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion")
        return cls(
            conflict_graph=ConflictGraphConfig.from_dict(raw["conflict_graph"]),
            evidence=EvidenceConfig.from_dict(raw["evidence"]),
            dynamics=DynamicsConfig.from_dict(raw["dynamics"]),
            residual=ResidualConfig.from_dict(raw["residual"]),
            policy_bridge=PolicyBridgeConfig.from_dict(raw["policy_bridge"]),
            stateful=StatefulOpinionConfig.from_dict(raw["stateful"]),
            sequence_ppo=SequencePPOConfig.from_dict(raw["sequence_ppo"]),
            trainer=OpinionTrainerConfig.from_dict(raw["trainer"]),
        )


@dataclass(frozen=True)
class OpinionExperimentConfig:
    schema_version: int
    method: str
    stage: str
    use_opinion_marl: bool
    base_config: str
    output_root: str
    opinion: OpinionConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpinionExperimentConfig":
        raw = _object(raw, "root")
        _exact_keys(raw, set(cls.__dataclass_fields__), "root")
        schema_version = _integer(raw["schema_version"], "schema_version")
        if schema_version != OPINION_CONFIG_SCHEMA_VERSION:
            raise OpinionConfigError(
                f"schema_version must be {OPINION_CONFIG_SCHEMA_VERSION}."
            )
        method = _string(raw["method"], "method")
        if method != OPINION_METHOD:
            raise OpinionConfigError(f"method must be '{OPINION_METHOD}'.")
        stage = _string(raw["stage"], "stage").lower()
        if stage not in OPINION_STAGES:
            raise OpinionConfigError(
                f"stage must be one of {sorted(OPINION_STAGES)}."
            )
        use_opinion_marl = _boolean(raw["use_opinion_marl"], "use_opinion_marl")
        if (stage == "base") != (not use_opinion_marl):
            raise OpinionConfigError(
                "stage='base' requires use_opinion_marl=false; non-Base stages "
                "require use_opinion_marl=true."
            )
        policy_bridge = _object(raw["opinion"], "opinion").get("policy_bridge")
        if isinstance(policy_bridge, dict):
            bridge_enabled = policy_bridge.get("enabled")
            if stage == "base" and bridge_enabled is not False:
                raise OpinionConfigError(
                    "stage='base' requires policy_bridge.enabled=false."
                )
            if stage in {"evidence", "sequence", "sequence_ppo", "joint"} and bridge_enabled is not True:
                raise OpinionConfigError(
                    "Non-Base stages require policy_bridge.enabled=true."
                )
        opinion = OpinionConfig.from_dict(raw["opinion"])
        if stage in {"base", "evidence"} and opinion.sequence_ppo.enabled:
            raise OpinionConfigError(
                "Base/evidence stages require sequence_ppo.enabled=false."
            )
        if stage == "sequence" and not opinion.sequence_ppo.enabled:
            raise OpinionConfigError(
                "stage='sequence' requires sequence_ppo.enabled=true."
            )
        if stage == "sequence" and opinion.sequence_ppo.train_evidence:
            raise OpinionConfigError(
                "M7 stage='sequence' requires sequence_ppo.train_evidence=false."
            )
        if stage == "sequence_ppo":
            if not opinion.sequence_ppo.enabled:
                raise OpinionConfigError(
                    "stage='sequence_ppo' requires sequence_ppo.enabled=true."
                )
            if not opinion.sequence_ppo.train_evidence:
                raise OpinionConfigError(
                    "stage='sequence_ppo' requires train_evidence=true."
                )
            if not opinion.stateful.enabled or opinion.stateful.freeze_evidence:
                raise OpinionConfigError(
                    "M8 requires stateful.enabled=true and freeze_evidence=false."
                )
            if opinion.trainer.enabled:
                raise OpinionConfigError("M8 requires trainer.enabled=false.")
        if stage == "joint":
            trainer = opinion.trainer
            if not trainer.enabled:
                raise OpinionConfigError("M9 stage='joint' requires trainer.enabled=true.")
            if (
                not opinion.sequence_ppo.enabled
                or not opinion.sequence_ppo.train_evidence
                or not opinion.stateful.enabled
                or opinion.stateful.freeze_evidence
            ):
                raise OpinionConfigError(
                    "M9 requires stateful differentiable sequence PPO with "
                    "train_evidence=true."
                )
            should_freeze_base = trainer.mode == "evidence_only"
            if opinion.policy_bridge.freeze_base_actor != should_freeze_base:
                raise OpinionConfigError(
                    "M9 evidence_only must freeze Base Actor; joint modes must "
                    "set freeze_base_actor=false."
                )
            if trainer.initialization == "base":
                if opinion.policy_bridge.base_output_root is None:
                    raise OpinionConfigError(
                        "M9 Base initialization requires policy_bridge."
                        "base_output_root."
                    )
                if opinion.sequence_ppo.source_output_root is not None:
                    raise OpinionConfigError(
                        "M9 Base initialization requires sequence source=null."
                    )
                if opinion.stateful.evidence_output_root is not None:
                    raise OpinionConfigError(
                        "M9 Base initialization requires evidence_output_root=null."
                    )
            elif trainer.initialization == "opinion":
                if opinion.policy_bridge.base_output_root is None:
                    raise OpinionConfigError(
                        "M9 Opinion initialization currently requires "
                        "policy_bridge.base_output_root."
                    )
                if opinion.sequence_ppo.source_output_root is not None:
                    raise OpinionConfigError(
                        "M9 Opinion initialization uses trainer.source_output_root; "
                        "sequence_ppo.source_output_root must be null."
                    )
            else:
                if opinion.policy_bridge.base_output_root is not None:
                    raise OpinionConfigError(
                        "M9 from-scratch initialization requires "
                        "policy_bridge.base_output_root=null."
                    )
                if opinion.stateful.evidence_output_root is not None:
                    raise OpinionConfigError(
                        "M9 from-scratch initialization requires "
                        "evidence_output_root=null."
                    )
                if opinion.sequence_ppo.source_output_root is not None:
                    raise OpinionConfigError(
                        "M9 from-scratch initialization requires "
                        "sequence_ppo.source_output_root=null."
                    )
        if (
            stage != "joint"
            and opinion.policy_bridge.enabled
            and opinion.policy_bridge.base_output_root is None
        ):
            raise OpinionConfigError(
                "M4-M8 Opinion stages require policy_bridge.base_output_root."
            )
        if (
            opinion.stateful.enabled
            and stage not in {"sequence_ppo", "joint"}
            and not opinion.stateful.freeze_evidence
        ):
            raise OpinionConfigError(
                "Stateful M6/M7 stages require freeze_evidence=true."
            )
        if stage != "joint" and opinion.trainer.enabled:
            raise OpinionConfigError("Only M9 stage='joint' may enable trainer.")
        if stage != "joint" and opinion.policy_bridge.enabled and not opinion.policy_bridge.freeze_base_actor:
            raise OpinionConfigError("M4-M8 Opinion stages require a frozen Base Actor.")
        if (
            stage != "joint"
            and opinion.stateful.enabled
            and opinion.stateful.evidence_output_root is None
        ):
            raise OpinionConfigError(
                "M6-M8 stateful stages require evidence_output_root."
            )
        if (
            stage in {"sequence", "sequence_ppo"}
            and opinion.sequence_ppo.source_output_root is None
        ):
            raise OpinionConfigError(
                "M7/M8 require sequence_ppo.source_output_root."
            )
        return cls(
            schema_version=schema_version,
            method=method,
            stage=stage,
            use_opinion_marl=use_opinion_marl,
            base_config=_string(raw["base_config"], "base_config"),
            output_root=_string(raw["output_root"], "output_root"),
            opinion=opinion,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedOpinionExperiment:
    config_path: Path
    base_config_path: Path
    source_config: Dict[str, Any]
    base_source_config: Dict[str, Any]
    config: OpinionExperimentConfig
    parameters: Parameters

    def resolved_opinion_config(self) -> Dict[str, Any]:
        resolved = self.config.to_dict()
        resolved["base_config"] = str(self.base_config_path)
        resolved["output_root"] = str(
            Path(self.config.output_root).expanduser().resolve()
        )
        return resolved


def _validate_base_contract(parameters: Parameters, opinion: OpinionConfig) -> None:
    for field_name, value in vars(parameters).items():
        if field_name.startswith("is_") and type(value) is not bool:
            raise OpinionConfigError(f"Base {field_name} must be a boolean.")
    for field_name in (
        "n_agents",
        "seed",
        "n_iters",
        "frames_per_batch",
        "num_epochs",
        "minibatch_size",
        "max_steps",
        "n_nearing_agents_observed",
    ):
        value = getattr(parameters, field_name)
        minimum = 0 if field_name == "seed" else 1
        if type(value) is not int or value < minimum:
            raise OpinionConfigError(
                f"Base {field_name} must be an integer >= {minimum}."
            )
    for field_name in (
        "dt",
        "lr",
        "lr_min",
        "max_grad_norm",
        "clip_epsilon",
        "gamma",
        "lmbda",
        "entropy_eps",
    ):
        value = getattr(parameters, field_name)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise OpinionConfigError(f"Base {field_name} must be a finite number.")
        if float(value) <= 0:
            raise OpinionConfigError(f"Base {field_name} must be greater than zero.")
    if parameters.scenario_name != "road_traffic":
        raise OpinionConfigError("Base scenario_name must remain 'road_traffic'.")
    if parameters.scenario_type != "CPM_mixed" or parameters.n_agents != 4:
        raise OpinionConfigError(
            "The first method version requires CPM_mixed with exactly 4 agents."
        )
    if not parameters.is_partial_observation:
        raise OpinionConfigError("Opinion-MARL requires local partial observations.")
    if parameters.n_nearing_agents_observed != opinion.conflict_graph.candidate_count:
        raise OpinionConfigError(
            "n_nearing_agents_observed must equal conflict_graph.candidate_count."
        )
    if opinion.policy_bridge.enabled and not opinion.conflict_graph.emit_pair_info:
        raise OpinionConfigError(
            "An enabled policy bridge requires conflict_graph.emit_pair_info=true."
        )
    is_stateful_mode = opinion.policy_bridge.mode == "stateful_opinion"
    if opinion.stateful.enabled != is_stateful_mode:
        raise OpinionConfigError(
            "stateful.enabled must be true exactly when "
            "policy_bridge.mode='stateful_opinion'."
        )
    if is_stateful_mode and not opinion.policy_bridge.enabled:
        raise OpinionConfigError(
            "The stateful policy mode requires policy_bridge.enabled=true."
        )
    if opinion.sequence_ppo.enabled and not opinion.stateful.enabled:
        raise OpinionConfigError(
            "Sequence Buffer requires stateful opinion to be enabled."
        )
    if opinion.policy_bridge.visualize_agent_id >= parameters.n_agents:
        raise OpinionConfigError(
            "policy_bridge.visualize_agent_id must be a valid global agent ID."
        )
    if parameters.is_using_opponent_modeling:
        raise OpinionConfigError("Opponent modeling is forbidden in Opinion-MARL.")
    if parameters.is_using_prioritized_marl or parameters.is_prb:
        raise OpinionConfigError("Priority MARL and prioritized replay are forbidden.")
    if (
        parameters.is_load_model
        or parameters.is_load_final_model
        or parameters.is_continue_train
    ):
        raise OpinionConfigError(
            "Milestone entrypoints start a new stage; loading/resume is not "
            "available through M7."
        )
    if parameters.is_testing_mode:
        raise OpinionConfigError("The referenced Base config must be a training config.")
    if parameters.frames_per_batch % parameters.max_steps != 0:
        raise OpinionConfigError("frames_per_batch must be divisible by max_steps.")
    if parameters.frames_per_batch % parameters.minibatch_size != 0:
        raise OpinionConfigError("frames_per_batch must be divisible by minibatch_size.")
    chunk_length = opinion.sequence_ppo.chunk_length
    if chunk_length > parameters.max_steps or parameters.max_steps % chunk_length != 0:
        raise OpinionConfigError(
            "sequence_ppo.chunk_length must divide max_steps without remainder."
        )
    if (
        opinion.sequence_ppo.enabled
        and parameters.minibatch_size % chunk_length != 0
    ):
        raise OpinionConfigError(
            "M7 minibatch_size must be divisible by sequence_ppo.chunk_length."
        )


def load_opinion_experiment(config_path: Path) -> LoadedOpinionExperiment:
    """Load, validate and resolve one Opinion experiment without side effects."""

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        source_config = json.load(file)
    config = OpinionExperimentConfig.from_dict(source_config)

    base_config_path = (config_path.parent / config.base_config).resolve()
    if not base_config_path.is_file():
        raise OpinionConfigError(f"Base config does not exist: {base_config_path}")
    with base_config_path.open("r", encoding="utf-8") as file:
        base_source_config = json.load(file)
    try:
        parameters = Parameters.from_dict(base_source_config)
    except (TypeError, ValueError) as error:
        raise OpinionConfigError(f"Invalid Base Parameters: {error}") from error

    parameters.where_to_save = config.output_root
    if (
        config.opinion.policy_bridge.enabled
        and config.opinion.policy_bridge.base_output_root is not None
        and (
            Path(config.opinion.policy_bridge.base_output_root)
            .expanduser()
            .resolve()
            == Path(config.output_root).expanduser().resolve()
        )
    ):
        raise OpinionConfigError(
            "policy_bridge.base_output_root and output_root must be isolated."
        )
    if config.opinion.stateful.evidence_output_root is not None and (
        Path(config.opinion.stateful.evidence_output_root).expanduser().resolve()
        == Path(config.output_root).expanduser().resolve()
    ):
        raise OpinionConfigError(
            "stateful.evidence_output_root and output_root must be isolated."
        )
    if config.opinion.sequence_ppo.source_output_root is not None and (
        Path(config.opinion.sequence_ppo.source_output_root)
        .expanduser()
        .resolve()
        == Path(config.output_root).expanduser().resolve()
    ):
        raise OpinionConfigError(
            "sequence_ppo.source_output_root and output_root must be isolated."
        )
    if config.opinion.trainer.source_output_root is not None and (
        Path(config.opinion.trainer.source_output_root).expanduser().resolve()
        == Path(config.output_root).expanduser().resolve()
    ):
        raise OpinionConfigError(
            "trainer.source_output_root and output_root must be isolated."
        )
    _validate_base_contract(parameters, config.opinion)
    return LoadedOpinionExperiment(
        config_path=config_path,
        base_config_path=base_config_path,
        source_config=dict(source_config),
        base_source_config=dict(base_source_config),
        config=config,
        parameters=parameters,
    )


def require_m5_supported_mode(experiment: LoadedOpinionExperiment) -> None:
    """Allow the Base/M4 path and the M5 stateless Evidence path only."""

    stage = experiment.config.stage
    bridge_enabled = experiment.config.opinion.policy_bridge.enabled
    if stage == "base" and not bridge_enabled:
        return
    if (
        stage == "evidence"
        and bridge_enabled
        and experiment.config.opinion.policy_bridge.mode == "direct_evidence"
        and not experiment.config.opinion.stateful.enabled
    ):
        return
    raise NotImplementedError(
        "M5 supports stage='base' without the bridge, or stage='evidence' "
        "with the direct-evidence bridge. Stateful/joint execution starts in "
        "M6-M9."
    )


def require_m6_supported_mode(experiment: LoadedOpinionExperiment) -> None:
    """Allow Base/M4, M5 direct evidence, and M6 stateful rollout modes."""

    stage = experiment.config.stage
    bridge = experiment.config.opinion.policy_bridge
    stateful = experiment.config.opinion.stateful
    if stage == "base" and not bridge.enabled and not stateful.enabled:
        return
    if stage == "evidence" and bridge.enabled:
        if bridge.mode == "direct_evidence" and not stateful.enabled:
            return
        if bridge.mode == "stateful_opinion" and stateful.enabled:
            return
    raise NotImplementedError(
        "M6 supports Base/M4, M5 direct-evidence, or M6 stateful-opinion "
        "execution. Joint and sequence-PPO optimization start in M7-M9."
    )


def require_m7_supported_mode(experiment: LoadedOpinionExperiment) -> None:
    """Allow all implemented paths through M7 Sequence Buffer."""

    stage = experiment.config.stage
    bridge = experiment.config.opinion.policy_bridge
    stateful = experiment.config.opinion.stateful
    sequence = experiment.config.opinion.sequence_ppo
    if stage == "base" and not bridge.enabled and not stateful.enabled:
        return
    if stage == "evidence" and bridge.enabled and not sequence.enabled:
        if bridge.mode == "direct_evidence" and not stateful.enabled:
            return
        if bridge.mode == "stateful_opinion" and stateful.enabled:
            return
    if (
        stage == "sequence"
        and bridge.enabled
        and bridge.mode == "stateful_opinion"
        and stateful.enabled
        and sequence.enabled
    ):
        return
    raise NotImplementedError(
        "M7 supports Base/M4, M5 Direct-Evidence, M6 Stateful, or M7 "
        "Sequence-Buffer execution. Evidence sequence gradients begin in M8."
    )


def require_m8_supported_mode(experiment: LoadedOpinionExperiment) -> None:
    """Allow all implemented paths through M8 differentiable sequence PPO."""

    try:
        require_m7_supported_mode(experiment)
        return
    except NotImplementedError:
        pass
    bridge = experiment.config.opinion.policy_bridge
    stateful = experiment.config.opinion.stateful
    sequence = experiment.config.opinion.sequence_ppo
    if (
        experiment.config.stage == "sequence_ppo"
        and bridge.enabled
        and bridge.mode == "stateful_opinion"
        and stateful.enabled
        and not stateful.freeze_evidence
        and sequence.enabled
        and sequence.train_evidence
    ):
        return
    raise NotImplementedError(
        "M8 supports the existing M4-M7 modes or stage='sequence_ppo' with "
        "stateful truncated-BPTT evidence training."
    )


def require_m9_supported_mode(experiment: LoadedOpinionExperiment) -> None:
    """Allow existing modes plus the configurable M9 trainer."""

    try:
        require_m8_supported_mode(experiment)
        return
    except NotImplementedError:
        pass
    opinion = experiment.config.opinion
    if (
        experiment.config.stage == "joint"
        and opinion.trainer.enabled
        and opinion.policy_bridge.enabled
        and opinion.policy_bridge.mode == "stateful_opinion"
        and opinion.stateful.enabled
        and opinion.sequence_ppo.enabled
        and opinion.sequence_ppo.train_evidence
    ):
        return
    raise NotImplementedError(
        "M9 supports existing M4-M8 modes or stage='joint' with the unified "
        "evidence/joint trainer."
    )


def require_base_noop_mode(experiment: LoadedOpinionExperiment) -> None:
    """Compatibility name retained for M2-M4 callers."""

    if (
        experiment.config.stage != "base"
        or experiment.config.use_opinion_marl
        or experiment.config.opinion.policy_bridge.enabled
    ):
        raise NotImplementedError(
            "This caller only supports the Base/M4 information-only path."
        )


# Compatibility for sessions or scripts created during M2.
require_m2_base_mode = require_base_noop_mode
