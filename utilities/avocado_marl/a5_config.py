"""Typed A5 configuration for strict-zero y-correction equivalence."""

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
from utilities.avocado_marl.config import _boolean, _string
from utilities.avocado_marl.y_correction import Y_CORRECTION_FEATURE_DIM


@dataclass(frozen=True)
class A5YCorrectionConfig:
    feature_dim: int
    hidden_sizes: Tuple[int, ...]
    maximum_correction: float
    temperature: float
    candidate_count: int
    strict_zero: bool
    freeze: bool

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        require_frozen_zero: bool = True,
    ) -> "A5YCorrectionConfig":
        raw = _object(raw, "y_correction")
        _exact_keys(raw, set(cls.__dataclass_fields__), "y_correction")
        hidden = raw["hidden_sizes"]
        if not isinstance(hidden, list) or not hidden:
            raise AVOCADOConfigError(
                "y_correction.hidden_sizes must be a non-empty list."
            )
        result = cls(
            feature_dim=_integer(
                raw["feature_dim"], "y_correction.feature_dim"
            ),
            hidden_sizes=tuple(
                _integer(value, "y_correction.hidden_sizes[]")
                for value in hidden
            ),
            maximum_correction=_number(
                raw["maximum_correction"],
                "y_correction.maximum_correction",
                strict=True,
            ),
            temperature=_number(
                raw["temperature"], "y_correction.temperature", strict=True
            ),
            candidate_count=_integer(
                raw["candidate_count"], "y_correction.candidate_count"
            ),
            strict_zero=_boolean(
                raw["strict_zero"], "y_correction.strict_zero"
            ),
            freeze=_boolean(raw["freeze"], "y_correction.freeze"),
        )
        if result.feature_dim != Y_CORRECTION_FEATURE_DIM:
            raise AVOCADOConfigError(
                f"A5 feature_dim must be {Y_CORRECTION_FEATURE_DIM}."
            )
        if result.maximum_correction > 0.5:
            raise AVOCADOConfigError(
                "y_correction.maximum_correction must not exceed 0.5."
            )
        if require_frozen_zero and (not result.strict_zero or not result.freeze):
            raise AVOCADOConfigError(
                "A5 requires strict_zero=true and freeze=true."
            )
        return result


@dataclass(frozen=True)
class A5ExperimentConfig:
    a4_config: Path
    output_root: str
    y_correction: A5YCorrectionConfig

    @classmethod
    def from_json(cls, path: Path) -> "A5ExperimentConfig":
        with path.open("r", encoding="utf-8") as stream:
            raw = _object(json.load(stream), "root")
        expected = {
            "schema_version",
            "method",
            "stage",
            "a4_config",
            "output_root",
            "y_correction",
        }
        _exact_keys(raw, expected, "root")
        if raw["schema_version"] != 1:
            raise AVOCADOConfigError("A5 schema_version must be 1.")
        if raw["method"] != "avocado_marl" or raw["stage"] != "a5":
            raise AVOCADOConfigError(
                "Expected method='avocado_marl' and stage='a5'."
            )
        a4_path = Path(_string(raw["a4_config"], "a4_config"))
        if not a4_path.is_file():
            raise AVOCADOConfigError(f"A4 config does not exist: {a4_path}")
        return cls(
            a4_config=a4_path,
            output_root=_string(raw["output_root"], "output_root"),
            y_correction=A5YCorrectionConfig.from_dict(raw["y_correction"]),
        )
