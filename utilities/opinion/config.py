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
from typing import Any, Dict, Mapping, Tuple

from utilities.helper_training import Parameters


OPINION_CONFIG_SCHEMA_VERSION = 1
OPINION_METHOD = "opinion_marl"
OPINION_STAGES = {"base", "evidence", "joint"}


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


@dataclass(frozen=True)
class ConflictGraphConfig:
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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResidualConfig":
        raw = _object(raw, "opinion.residual")
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
class SequencePPOConfig:
    chunk_length: int
    evidence_learning_rate_scale: float
    neutral_loss_coefficient: float
    magnitude_loss_coefficient: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SequencePPOConfig":
        raw = _object(raw, "opinion.sequence_ppo")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion.sequence_ppo")
        return cls(
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


@dataclass(frozen=True)
class OpinionConfig:
    conflict_graph: ConflictGraphConfig
    evidence: EvidenceConfig
    dynamics: DynamicsConfig
    residual: ResidualConfig
    sequence_ppo: SequencePPOConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpinionConfig":
        raw = _object(raw, "opinion")
        _exact_keys(raw, set(cls.__dataclass_fields__), "opinion")
        return cls(
            conflict_graph=ConflictGraphConfig.from_dict(raw["conflict_graph"]),
            evidence=EvidenceConfig.from_dict(raw["evidence"]),
            dynamics=DynamicsConfig.from_dict(raw["dynamics"]),
            residual=ResidualConfig.from_dict(raw["residual"]),
            sequence_ppo=SequencePPOConfig.from_dict(raw["sequence_ppo"]),
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
                "stage='base' requires use_opinion_marl=false; evidence/joint "
                "require use_opinion_marl=true."
            )
        return cls(
            schema_version=schema_version,
            method=method,
            stage=stage,
            use_opinion_marl=use_opinion_marl,
            base_config=_string(raw["base_config"], "base_config"),
            output_root=_string(raw["output_root"], "output_root"),
            opinion=OpinionConfig.from_dict(raw["opinion"]),
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
            "Milestone entrypoints start a new stage; loading/resume is not available in M2."
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
    _validate_base_contract(parameters, config.opinion)
    return LoadedOpinionExperiment(
        config_path=config_path,
        base_config_path=base_config_path,
        source_config=dict(source_config),
        base_source_config=dict(base_source_config),
        config=config,
        parameters=parameters,
    )


def require_base_noop_mode(experiment: LoadedOpinionExperiment) -> None:
    """Fail explicitly instead of silently treating an enabled method as Base."""

    if experiment.config.use_opinion_marl or experiment.config.stage != "base":
        raise NotImplementedError(
            "M3 provides pure math modules; active Opinion execution is "
            "introduced in M5-M9. For the current trainable no-op path, use "
            "stage='base' with use_opinion_marl=false."
        )


# Compatibility for sessions or scripts created during M2.
require_m2_base_mode = require_base_noop_mode
