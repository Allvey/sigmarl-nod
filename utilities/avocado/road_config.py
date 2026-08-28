"""Typed A3 configuration for AVOCADO-KB in SigmaRL road traffic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from utilities.avocado.config import (
    AVOCADOConfigError,
    _exact_keys,
    _integer,
    _number,
    _object,
    _string,
)
from utilities.avocado.core import AVOCADOParameters
from utilities.constants import AGENTS, SCENARIOS


A3_CONFIG_SCHEMA_VERSION = 2
A3_METHOD = "avocado_kb"
A3_STAGE = "a3"
A3_SUPPORTED_PLANNERS = ("path_following", "orca_kb", "avocado_kb")


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise AVOCADOConfigError(f"{location} must be a boolean.")
    return value


@dataclass(frozen=True)
class RoadSimulationConfig:
    device: str
    episodes: int
    max_steps: int
    seed: int
    video_stride: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoadSimulationConfig":
        raw = _object(raw, "simulation")
        _exact_keys(raw, set(cls.__dataclass_fields__), "simulation")
        return cls(
            device=_string(raw["device"], "simulation.device"),
            episodes=_integer(raw["episodes"], "simulation.episodes"),
            max_steps=_integer(raw["max_steps"], "simulation.max_steps"),
            seed=_integer(raw["seed"], "simulation.seed", minimum=0),
            video_stride=_integer(
                raw["video_stride"], "simulation.video_stride"
            ),
        )


@dataclass(frozen=True)
class RoadVehicleConfig:
    cruise_speed: float
    avoidance_radius_scale: float
    minimum_speed_ratio: float
    path_tracking_gain: float
    path_tracking_softening_speed: float
    maximum_path_correction_degrees: float
    maximum_path_deviation_degrees: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoadVehicleConfig":
        raw = _object(raw, "vehicle")
        _exact_keys(raw, set(cls.__dataclass_fields__), "vehicle")
        result = cls(
            cruise_speed=_number(
                raw["cruise_speed"], "vehicle.cruise_speed", strict=True
            ),
            avoidance_radius_scale=_number(
                raw["avoidance_radius_scale"],
                "vehicle.avoidance_radius_scale",
                strict=True,
            ),
            minimum_speed_ratio=_number(
                raw["minimum_speed_ratio"], "vehicle.minimum_speed_ratio"
            ),
            path_tracking_gain=_number(
                raw["path_tracking_gain"],
                "vehicle.path_tracking_gain",
                strict=True,
            ),
            path_tracking_softening_speed=_number(
                raw["path_tracking_softening_speed"],
                "vehicle.path_tracking_softening_speed",
            ),
            maximum_path_correction_degrees=_number(
                raw["maximum_path_correction_degrees"],
                "vehicle.maximum_path_correction_degrees",
                strict=True,
            ),
            maximum_path_deviation_degrees=_number(
                raw["maximum_path_deviation_degrees"],
                "vehicle.maximum_path_deviation_degrees",
                strict=True,
            ),
        )
        if result.avoidance_radius_scale < 1.0:
            raise AVOCADOConfigError(
                "vehicle.avoidance_radius_scale must be at least 1.0."
            )
        if result.minimum_speed_ratio > 1.0:
            raise AVOCADOConfigError(
                "vehicle.minimum_speed_ratio must not exceed 1.0."
            )
        if result.maximum_path_deviation_degrees >= 90.0:
            raise AVOCADOConfigError(
                "vehicle.maximum_path_deviation_degrees must be below 90."
            )
        if result.maximum_path_correction_degrees > 90.0:
            raise AVOCADOConfigError(
                "vehicle.maximum_path_correction_degrees must not exceed 90."
            )
        if result.cruise_speed > float(AGENTS["max_speed"]):
            raise AVOCADOConfigError(
                "vehicle.cruise_speed cannot exceed the road vehicle max speed."
            )
        return result


@dataclass(frozen=True)
class RoadSafetyConfig:
    complementary_responsibility: bool
    ttc_braking_shield_enabled: bool
    minimum_ttc_seconds: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoadSafetyConfig":
        raw = _object(raw, "safety")
        _exact_keys(raw, set(cls.__dataclass_fields__), "safety")
        return cls(
            complementary_responsibility=_boolean(
                raw["complementary_responsibility"],
                "safety.complementary_responsibility",
            ),
            ttc_braking_shield_enabled=_boolean(
                raw["ttc_braking_shield_enabled"],
                "safety.ttc_braking_shield_enabled",
            ),
            minimum_ttc_seconds=_number(
                raw["minimum_ttc_seconds"],
                "safety.minimum_ttc_seconds",
                strict=True,
            ),
        )


@dataclass(frozen=True)
class RoadCaseConfig:
    name: str
    scenario_type: str
    n_agents: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], index: int) -> "RoadCaseConfig":
        location = f"cases[{index}]"
        raw = _object(raw, location)
        _exact_keys(raw, set(cls.__dataclass_fields__), location)
        result = cls(
            name=_string(raw["name"], f"{location}.name"),
            scenario_type=_string(
                raw["scenario_type"], f"{location}.scenario_type"
            ),
            n_agents=_integer(raw["n_agents"], f"{location}.n_agents", minimum=2),
        )
        if result.scenario_type not in SCENARIOS:
            raise AVOCADOConfigError(
                f"{location}.scenario_type is not defined in SCENARIOS."
            )
        maximum_agents = int(SCENARIOS[result.scenario_type]["n_agents"])
        if result.n_agents > maximum_agents:
            raise AVOCADOConfigError(
                f"{location}.n_agents cannot exceed the scenario default "
                f"({maximum_agents})."
            )
        return result


@dataclass(frozen=True)
class RoadValidationConfig:
    maximum_agent_collision_events_per_1000_steps: float
    maximum_lane_collision_events_per_1000_steps: float
    maximum_mean_tracking_error_mps: float
    maximum_p95_reference_distance_meters: float
    maximum_steering_saturation_rate: float
    minimum_route_completion_events_per_1000_steps: float
    minimum_maximum_attention: float
    minimum_agent_collision_improvement: float
    maximum_shield_intervention_rate: float
    maximum_post_shield_unsafe_pair_events_per_1000_steps: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoadValidationConfig":
        raw = _object(raw, "validation")
        _exact_keys(raw, set(cls.__dataclass_fields__), "validation")
        result = cls(
            **{
                key: _number(raw[key], f"validation.{key}")
                for key in cls.__dataclass_fields__
            }
        )
        if result.maximum_steering_saturation_rate > 1.0:
            raise AVOCADOConfigError(
                "validation.maximum_steering_saturation_rate must not exceed 1."
            )
        if result.minimum_maximum_attention > 1.0:
            raise AVOCADOConfigError(
                "validation.minimum_maximum_attention must not exceed 1."
            )
        if result.maximum_shield_intervention_rate > 1.0:
            raise AVOCADOConfigError(
                "validation.maximum_shield_intervention_rate must not exceed 1."
            )
        return result


@dataclass(frozen=True)
class A3RoadExperimentConfig:
    output_root: str
    simulation: RoadSimulationConfig
    vehicle: RoadVehicleConfig
    safety: RoadSafetyConfig
    parameters: AVOCADOParameters
    planners: Tuple[str, ...]
    cases: Tuple[RoadCaseConfig, ...]
    validation: RoadValidationConfig

    @classmethod
    def from_json(cls, path: Path) -> "A3RoadExperimentConfig":
        with path.open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "output_root",
            "simulation",
            "vehicle",
            "safety",
            "avocado",
            "planners",
            "cases",
            "validation",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != A3_CONFIG_SCHEMA_VERSION:
            raise AVOCADOConfigError(
                f"schema_version must be {A3_CONFIG_SCHEMA_VERSION}."
            )
        if raw["method"] != A3_METHOD or raw["stage"] != A3_STAGE:
            raise AVOCADOConfigError(
                f"Expected method={A3_METHOD!r} and stage={A3_STAGE!r}."
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
                        minimum=(
                            float("-inf") if key == "opinion_bias" else 0.0
                        ),
                    )
                    for key, value in avocado_raw.items()
                }
            )
        except (TypeError, ValueError) as error:
            raise AVOCADOConfigError(f"Invalid avocado parameters: {error}") from error

        simulation = RoadSimulationConfig.from_dict(raw["simulation"])
        if abs(parameters.dt - 0.05) > 1e-9:
            raise AVOCADOConfigError(
                "A3 currently requires dt=0.05 to match the road environment."
            )
        planners_raw = raw["planners"]
        if not isinstance(planners_raw, list) or not planners_raw:
            raise AVOCADOConfigError("planners must be a non-empty list.")
        planners = tuple(
            _string(value, "planners[]").lower() for value in planners_raw
        )
        if len(set(planners)) != len(planners):
            raise AVOCADOConfigError("planners must not contain duplicates.")
        unsupported = sorted(set(planners) - set(A3_SUPPORTED_PLANNERS))
        if unsupported:
            raise AVOCADOConfigError(f"Unsupported A3 planners: {unsupported}.")
        if "path_following" not in planners or "avocado_kb" not in planners:
            raise AVOCADOConfigError(
                "A3 validation requires path_following and avocado_kb."
            )

        cases_raw = raw["cases"]
        if not isinstance(cases_raw, list) or not cases_raw:
            raise AVOCADOConfigError("cases must be a non-empty list.")
        cases = tuple(
            RoadCaseConfig.from_dict(value, index)
            for index, value in enumerate(cases_raw)
        )
        if len({case.name for case in cases}) != len(cases):
            raise AVOCADOConfigError("Case names must be unique.")
        return cls(
            output_root=_string(raw["output_root"], "output_root"),
            simulation=simulation,
            vehicle=RoadVehicleConfig.from_dict(raw["vehicle"]),
            safety=RoadSafetyConfig.from_dict(raw["safety"]),
            parameters=parameters,
            planners=planners,
            cases=cases,
            validation=RoadValidationConfig.from_dict(raw["validation"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": A3_CONFIG_SCHEMA_VERSION,
            "method": A3_METHOD,
            "stage": A3_STAGE,
            "output_root": self.output_root,
            "simulation": self.simulation.__dict__,
            "vehicle": self.vehicle.__dict__,
            "safety": self.safety.__dict__,
            "avocado": self.parameters.__dict__,
            "planners": list(self.planners),
            "cases": [case.__dict__ for case in self.cases],
            "validation": self.validation.__dict__,
        }
