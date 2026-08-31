"""Strict experiment configuration for PSB-MARL stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


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

    @property
    def base_parameters(self):
        from utilities.helper_training import Parameters

        return Parameters.from_dict(dict(self.base_run_config))

    @property
    def seed(self) -> int:
        return int(self.base_run_config["seed"])


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


def load_psb_experiment(path: Path) -> PSBExperimentConfig:
    """Load and fully validate one PSB stage configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"PSB config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = _object(json.load(stream), "root")
    expected = {
        "schema_version",
        "method",
        "stage",
        "base_config",
        "output_root",
        "base",
    }
    _exact_keys(raw, expected, "root")
    if raw["schema_version"] != 1:
        raise PSBConfigError("PSB schema_version must be 1.")
    if raw["method"] != "psb_marl":
        raise PSBConfigError("PSB method must be 'psb_marl'.")
    stage = _string(raw["stage"], "stage")
    if stage != "p0_base_passthrough":
        raise PSBConfigError(
            "The current implementation supports only stage "
            "'p0_base_passthrough'."
        )
    base_config_path = _resolve_existing_path(
        raw["base_config"], "base_config", config_path, kind="file"
    )
    base = PSBBaseSourceConfig.from_dict(raw["base"], config_path)
    base_source, base_resolved = _validate_base_run(base_config_path, base)
    output_root = _string(raw["output_root"], "output_root")
    return PSBExperimentConfig(
        config_path=config_path,
        source_config=dict(raw),
        base_config_path=base_config_path,
        base_source_config=base_source,
        base_run_config=base_resolved,
        output_root=output_root,
        base=base,
        stage=stage,
    )
