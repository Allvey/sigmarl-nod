"""Typed configuration for the A4 MARL--AVOCADO action bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from utilities.avocado.config import (
    AVOCADOConfigError,
    _exact_keys,
    _number,
    _object,
    _string,
)


A4_CONFIG_SCHEMA_VERSION = 1
A4_METHOD = "avocado_marl"
A4_STAGE = "a4"


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise AVOCADOConfigError(f"{location} must be a boolean.")
    return value


def _optional_string(value: Any, location: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, location)


@dataclass(frozen=True)
class A4BasePolicyConfig:
    output_root: str
    run_directory: Optional[str]
    checkpoint: Optional[str]
    deterministic: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A4BasePolicyConfig":
        raw = _object(raw, "base_policy")
        _exact_keys(raw, set(cls.__dataclass_fields__), "base_policy")
        return cls(
            output_root=_string(raw["output_root"], "base_policy.output_root"),
            run_directory=_optional_string(
                raw["run_directory"], "base_policy.run_directory"
            ),
            checkpoint=_optional_string(
                raw["checkpoint"], "base_policy.checkpoint"
            ),
            deterministic=_boolean(
                raw["deterministic"], "base_policy.deterministic"
            ),
        )


@dataclass(frozen=True)
class A4DiagnosticsConfig:
    speed_intervention_tolerance_mps: float
    steering_intervention_tolerance_degrees: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A4DiagnosticsConfig":
        raw = _object(raw, "diagnostics")
        _exact_keys(raw, set(cls.__dataclass_fields__), "diagnostics")
        return cls(
            speed_intervention_tolerance_mps=_number(
                raw["speed_intervention_tolerance_mps"],
                "diagnostics.speed_intervention_tolerance_mps",
                strict=True,
            ),
            steering_intervention_tolerance_degrees=_number(
                raw["steering_intervention_tolerance_degrees"],
                "diagnostics.steering_intervention_tolerance_degrees",
                strict=True,
            ),
        )


@dataclass(frozen=True)
class A4CouplingConfig:
    velocity_continuity_weight: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A4CouplingConfig":
        raw = _object(raw, "coupling")
        _exact_keys(raw, set(cls.__dataclass_fields__), "coupling")
        return cls(
            velocity_continuity_weight=_number(
                raw["velocity_continuity_weight"],
                "coupling.velocity_continuity_weight",
            )
        )


@dataclass(frozen=True)
class A4ExperimentConfig:
    a3_config: Path
    output_root: str
    base_policy: A4BasePolicyConfig
    coupling: A4CouplingConfig
    diagnostics: A4DiagnosticsConfig

    @classmethod
    def from_json(cls, path: Path) -> "A4ExperimentConfig":
        with path.open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "a3_config",
            "output_root",
            "base_policy",
            "coupling",
            "diagnostics",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != A4_CONFIG_SCHEMA_VERSION:
            raise AVOCADOConfigError(
                f"schema_version must be {A4_CONFIG_SCHEMA_VERSION}."
            )
        if raw["method"] != A4_METHOD or raw["stage"] != A4_STAGE:
            raise AVOCADOConfigError(
                f"Expected method={A4_METHOD!r} and stage={A4_STAGE!r}."
            )
        a3_config = Path(_string(raw["a3_config"], "a3_config"))
        if not a3_config.is_file():
            raise AVOCADOConfigError(f"A3 config does not exist: {a3_config}")
        return cls(
            a3_config=a3_config,
            output_root=_string(raw["output_root"], "output_root"),
            base_policy=A4BasePolicyConfig.from_dict(raw["base_policy"]),
            coupling=A4CouplingConfig.from_dict(raw["coupling"]),
            diagnostics=A4DiagnosticsConfig.from_dict(raw["diagnostics"]),
        )
