"""Typed configuration for the independent A0-A2 AVOCADO benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from utilities.avocado.core import AVOCADOParameters


AVOCADO_CONFIG_SCHEMA_VERSION = 1
AVOCADO_METHOD = "avocado_strict"
AVOCADO_STAGE = "a2"
SUPPORTED_PLANNERS = ("preferred", "orca", "avocado")
SUPPORTED_LAYOUTS = (
    "head_on_noncooperative",
    "head_on_cooperative",
    "circle_cooperative",
    "crossing_mixed",
)


class AVOCADOConfigError(ValueError):
    """Raised before a benchmark starts when its contract is inconsistent."""


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AVOCADOConfigError(f"{location} must be a JSON object.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set, location: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise AVOCADOConfigError(f"Invalid keys at {location}: {', '.join(details)}")


def _number(
    value: Any,
    location: str,
    *,
    minimum: float = 0.0,
    strict: bool = False,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise AVOCADOConfigError(f"{location} must be a finite number.")
    result = float(value)
    if (strict and result <= minimum) or (not strict and result < minimum):
        relation = "greater than" if strict else "greater than or equal to"
        raise AVOCADOConfigError(f"{location} must be {relation} {minimum}.")
    return result


def _integer(value: Any, location: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise AVOCADOConfigError(
            f"{location} must be an integer greater than or equal to {minimum}."
        )
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AVOCADOConfigError(f"{location} must be a non-empty string.")
    return value


@dataclass(frozen=True)
class EntityConfig:
    robot_radius: float
    agent_radius: float
    avoidance_radius_scale: float
    robot_max_speed: float
    agent_max_speed: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EntityConfig":
        raw = _object(raw, "entities")
        _exact_keys(raw, set(cls.__dataclass_fields__), "entities")
        result = cls(
            robot_radius=_number(
                raw["robot_radius"], "entities.robot_radius", strict=True
            ),
            agent_radius=_number(
                raw["agent_radius"], "entities.agent_radius", strict=True
            ),
            avoidance_radius_scale=_number(
                raw["avoidance_radius_scale"],
                "entities.avoidance_radius_scale",
                strict=True,
            ),
            robot_max_speed=_number(
                raw["robot_max_speed"], "entities.robot_max_speed", strict=True
            ),
            agent_max_speed=_number(
                raw["agent_max_speed"], "entities.agent_max_speed", strict=True
            ),
        )
        if result.avoidance_radius_scale < 1.0:
            raise AVOCADOConfigError(
                "entities.avoidance_radius_scale must be at least 1.0."
            )
        return result


@dataclass(frozen=True)
class SimulationConfig:
    device: str
    episodes: int
    max_steps: int
    goal_tolerance: float
    layout_seed: int
    controller_seed: int
    position_jitter: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SimulationConfig":
        raw = _object(raw, "simulation")
        _exact_keys(raw, set(cls.__dataclass_fields__), "simulation")
        return cls(
            device=_string(raw["device"], "simulation.device"),
            episodes=_integer(raw["episodes"], "simulation.episodes"),
            max_steps=_integer(raw["max_steps"], "simulation.max_steps"),
            goal_tolerance=_number(
                raw["goal_tolerance"], "simulation.goal_tolerance", strict=True
            ),
            layout_seed=_integer(
                raw["layout_seed"], "simulation.layout_seed", minimum=0
            ),
            controller_seed=_integer(
                raw["controller_seed"], "simulation.controller_seed", minimum=0
            ),
            position_jitter=_number(raw["position_jitter"], "simulation.position_jitter"),
        )


@dataclass(frozen=True)
class CaseConfig:
    name: str
    layout: str
    n_agents: int
    controlled_agents: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], index: int) -> "CaseConfig":
        raw = _object(raw, f"cases[{index}]")
        _exact_keys(raw, set(cls.__dataclass_fields__), f"cases[{index}]")
        result = cls(
            name=_string(raw["name"], f"cases[{index}].name"),
            layout=_string(raw["layout"], f"cases[{index}].layout"),
            n_agents=_integer(
                raw["n_agents"], f"cases[{index}].n_agents", minimum=2
            ),
            controlled_agents=_integer(
                raw["controlled_agents"],
                f"cases[{index}].controlled_agents",
            ),
        )
        if result.controlled_agents > result.n_agents:
            raise AVOCADOConfigError(
                f"cases[{index}].controlled_agents cannot exceed n_agents."
            )
        if result.layout not in SUPPORTED_LAYOUTS:
            raise AVOCADOConfigError(
                f"cases[{index}].layout must be one of {SUPPORTED_LAYOUTS}."
            )
        return result


@dataclass(frozen=True)
class ValidationConfig:
    required_case: str
    minimum_avocado_success_rate: float
    maximum_avocado_collision_rate: float
    minimum_preferred_collision_rate: float
    minimum_collision_rate_improvement: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ValidationConfig":
        raw = _object(raw, "validation")
        _exact_keys(raw, set(cls.__dataclass_fields__), "validation")
        values = {
            key: _number(raw[key], f"validation.{key}")
            for key in (
                "minimum_avocado_success_rate",
                "maximum_avocado_collision_rate",
                "minimum_preferred_collision_rate",
                "minimum_collision_rate_improvement",
            )
        }
        for key, value in values.items():
            if value > 1:
                raise AVOCADOConfigError(f"validation.{key} must not exceed 1.")
        return cls(
            required_case=_string(raw["required_case"], "validation.required_case"),
            **values,
        )


@dataclass(frozen=True)
class AVOCADOExperimentConfig:
    output_root: str
    simulation: SimulationConfig
    entities: EntityConfig
    parameters: AVOCADOParameters
    planners: Tuple[str, ...]
    cases: Tuple[CaseConfig, ...]
    validation: ValidationConfig

    @classmethod
    def from_json(cls, path: Path) -> "AVOCADOExperimentConfig":
        with path.open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "output_root",
            "simulation",
            "entities",
            "avocado",
            "planners",
            "cases",
            "validation",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != AVOCADO_CONFIG_SCHEMA_VERSION:
            raise AVOCADOConfigError(
                f"schema_version must be {AVOCADO_CONFIG_SCHEMA_VERSION}."
            )
        if raw["method"] != AVOCADO_METHOD or raw["stage"] != AVOCADO_STAGE:
            raise AVOCADOConfigError(
                f"Expected method={AVOCADO_METHOD!r} and stage={AVOCADO_STAGE!r}."
            )

        avocado_raw = _object(raw["avocado"], "avocado")
        _exact_keys(
            avocado_raw,
            set(AVOCADOParameters.__dataclass_fields__),
            "avocado",
        )
        try:
            parameters = AVOCADOParameters(
                **{
                    key: _number(
                        value,
                        f"avocado.{key}",
                        minimum=-math.inf if key == "opinion_bias" else 0.0,
                    )
                    for key, value in avocado_raw.items()
                }
            )
        except (TypeError, ValueError) as error:
            raise AVOCADOConfigError(f"Invalid avocado parameters: {error}") from error

        planners_raw = raw["planners"]
        if not isinstance(planners_raw, list) or not planners_raw:
            raise AVOCADOConfigError("planners must be a non-empty list.")
        planners = tuple(
            _string(value, "planners[]").lower() for value in planners_raw
        )
        if len(set(planners)) != len(planners):
            raise AVOCADOConfigError("planners must not contain duplicates.")
        unsupported = sorted(set(planners) - set(SUPPORTED_PLANNERS))
        if unsupported:
            raise AVOCADOConfigError(f"Unsupported planners: {unsupported}.")
        if "preferred" not in planners or "avocado" not in planners:
            raise AVOCADOConfigError(
                "A2 validation requires both 'preferred' and 'avocado'."
            )

        cases_raw = raw["cases"]
        if not isinstance(cases_raw, list) or not cases_raw:
            raise AVOCADOConfigError("cases must be a non-empty list.")
        cases = tuple(
            CaseConfig.from_dict(value, index)
            for index, value in enumerate(cases_raw)
        )
        if len({case.name for case in cases}) != len(cases):
            raise AVOCADOConfigError("Case names must be unique.")
        validation = ValidationConfig.from_dict(raw["validation"])
        if validation.required_case not in {case.name for case in cases}:
            raise AVOCADOConfigError("validation.required_case is not present in cases.")
        return cls(
            output_root=_string(raw["output_root"], "output_root"),
            simulation=SimulationConfig.from_dict(raw["simulation"]),
            entities=EntityConfig.from_dict(raw["entities"]),
            parameters=parameters,
            planners=planners,
            cases=cases,
            validation=validation,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": AVOCADO_CONFIG_SCHEMA_VERSION,
            "method": AVOCADO_METHOD,
            "stage": AVOCADO_STAGE,
            "output_root": self.output_root,
            "simulation": self.simulation.__dict__,
            "entities": self.entities.__dict__,
            "avocado": self.parameters.__dict__,
            "planners": list(self.planners),
            "cases": [case.__dict__ for case in self.cases],
            "validation": self.validation.__dict__,
        }
