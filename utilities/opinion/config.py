"""Strict configuration contract for the independent Opinion-MARL path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from utilities.baseline_config import (
    BASELINE_CONFIG_FIELDS,
    REPO_ROOT,
    validate_baseline_config,
)
from utilities.helper_training import Parameters


DEFAULT_OPINION_CONFIG_PATH = REPO_ROOT / "config_opinion.json"
OPINION_STAGES = ("base", "evidence", "joint")
POSITIVE_INTEGER_FIELDS = (
    "n_candidates",
    "chunk_length",
    "chunks_per_minibatch",
    "evidence_hidden_dim",
    "evidence_num_layers",
    "n_substeps",
)
BOOLEAN_FIELDS = ("include_z_in_critic", "log_pair_diagnostics")
POSITIVE_NUMERIC_FIELDS = (
    "b_max",
    "b_temperature",
    "kappa",
    "nu",
    "alpha",
    "eta",
    "z0",
    "z_clip",
    "lr_actor",
    "lr_evidence",
    "lr_critic",
    "ttc_horizon",
    "urgency_time_scale",
    "urgency_distance_temperature",
)
NON_NEGATIVE_NUMERIC_FIELDS = (
    "residual_scale_start",
    "residual_scale_target",
    "neutral_loss_weight",
    "magnitude_loss_weight",
    "safe_distance",
)
UNIT_INTERVAL_FIELDS = ("residual_warmup_fraction",)
OPINION_CONFIG_FIELDS = frozenset(
    ("stage",)
    + POSITIVE_INTEGER_FIELDS
    + BOOLEAN_FIELDS
    + POSITIVE_NUMERIC_FIELDS
    + NON_NEGATIVE_NUMERIC_FIELDS
    + UNIT_INTERVAL_FIELDS
)
ROOT_CONFIG_FIELDS = frozenset(
    set(BASELINE_CONFIG_FIELDS) | {"use_opinion_marl", "opinion_config"}
)


def _raise_schema_error(
    label: str,
    expected: frozenset,
    supplied: Mapping[str, Any],
) -> None:
    missing = sorted(expected.difference(supplied))
    unknown = sorted(set(supplied).difference(expected))
    if not missing and not unknown:
        return
    details = []
    if missing:
        details.append(f"missing fields: {missing}")
    if unknown:
        details.append(f"unknown fields: {unknown}")
    raise ValueError(f"{label} has " + "; ".join(details))


def _require_finite_number(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"opinion_config.{field} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"opinion_config.{field} must be finite")
    return float(value)


@dataclass(frozen=True)
class OpinionConfig:
    """Validated configuration for the fixed-dynamics Opinion-MARL method."""

    stage: str
    n_candidates: int
    chunk_length: int
    chunks_per_minibatch: int
    evidence_hidden_dim: int
    evidence_num_layers: int
    b_max: float
    b_temperature: float
    kappa: float
    nu: float
    alpha: float
    eta: float
    z0: float
    z_clip: float
    n_substeps: int
    residual_scale_start: float
    residual_scale_target: float
    residual_warmup_fraction: float
    lr_actor: float
    lr_evidence: float
    lr_critic: float
    neutral_loss_weight: float
    magnitude_loss_weight: float
    ttc_horizon: float
    safe_distance: float
    urgency_time_scale: float
    urgency_distance_temperature: float
    include_z_in_critic: bool
    log_pair_diagnostics: bool

    def __post_init__(self) -> None:
        if type(self.stage) is not str or self.stage not in OPINION_STAGES:
            raise ValueError(
                f"opinion_config.stage must be one of {OPINION_STAGES}, "
                f"got {self.stage!r}"
            )
        for field in POSITIVE_INTEGER_FIELDS:
            value = getattr(self, field)
            if type(value) is not int:
                raise ValueError(f"opinion_config.{field} must have type int")
            if value <= 0:
                raise ValueError(f"opinion_config.{field} must be positive")
        for field in BOOLEAN_FIELDS:
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"opinion_config.{field} must have type bool")
        for field in POSITIVE_NUMERIC_FIELDS:
            value = _require_finite_number(field, getattr(self, field))
            if value <= 0:
                raise ValueError(f"opinion_config.{field} must be positive")
        for field in NON_NEGATIVE_NUMERIC_FIELDS:
            value = _require_finite_number(field, getattr(self, field))
            if value < 0:
                raise ValueError(f"opinion_config.{field} must be non-negative")
        for field in UNIT_INTERVAL_FIELDS:
            value = _require_finite_number(field, getattr(self, field))
            if not 0 <= value <= 1:
                raise ValueError(f"opinion_config.{field} must be in [0, 1]")
        if self.residual_scale_start > self.residual_scale_target:
            raise ValueError(
                "opinion_config.residual_scale_start must not exceed "
                "residual_scale_target"
            )
        if self.nu * self.alpha <= self.kappa:
            raise ValueError(
                "opinion_config must satisfy nu * alpha > kappa so rho_c is in (0, 1)"
            )

    @property
    def rho_c(self) -> float:
        """Critical urgency where the neutral opinion changes stability."""
        return self.kappa / (self.nu * self.alpha)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rho_c"] = self.rho_c
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpinionConfig":
        if not isinstance(data, Mapping):
            raise ValueError("opinion_config must be an object")
        _raise_schema_error("opinion_config", OPINION_CONFIG_FIELDS, data)
        return cls(**dict(data))


@dataclass(frozen=True)
class LoadedOpinionExperimentConfig:
    """Validated root Parameters plus the typed Opinion configuration."""

    parameters: Parameters
    opinion: OpinionConfig
    source_path: Path

    def to_dict(self) -> Dict[str, Any]:
        """Return a resolved, serializable config including derived values."""
        data = dict(self.parameters.to_dict())
        data["opinion_config"] = self.opinion.to_dict()
        return data


def load_opinion_experiment_config(
    path: Union[str, Path] = DEFAULT_OPINION_CONFIG_PATH,
) -> LoadedOpinionExperimentConfig:
    import json

    source_path = Path(path).resolve()
    try:
        with source_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read Opinion config '{source_path}': {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("Opinion experiment config must be a JSON object")
    _raise_schema_error("Opinion experiment config", ROOT_CONFIG_FIELDS, raw)

    if type(raw["use_opinion_marl"]) is not bool or not raw["use_opinion_marl"]:
        raise ValueError("use_opinion_marl must be true for the Opinion entrypoints")

    base_config = {field: raw[field] for field in BASELINE_CONFIG_FIELDS}
    validate_baseline_config("base_mappo", base_config)
    opinion = OpinionConfig.from_dict(raw["opinion_config"])

    stability_factor = raw["dt"] * opinion.eta * opinion.kappa
    if not 0 < stability_factor < 2:
        raise ValueError(
            "Opinion dynamics must satisfy 0 < dt * eta * kappa < 2; "
            f"got {stability_factor}"
        )
    if opinion.n_candidates > raw["n_agents"] - 1:
        raise ValueError(
            "opinion_config.n_candidates must be at most n_agents - 1; "
            f"got n_candidates={opinion.n_candidates}, n_agents={raw['n_agents']}"
        )

    parameters = Parameters.from_dict(dict(raw))
    return LoadedOpinionExperimentConfig(
        parameters=parameters,
        opinion=opinion,
        source_path=source_path,
    )
