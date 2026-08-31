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
    def from_dict(cls, raw: Mapping[str, Any]) -> "PSBProximalConfig":
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
        if result.b_max != 0.0:
            raise PSBConfigError("P1 zero-control equivalence requires b_max=0.")
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

    @property
    def base_parameters(self):
        from utilities.helper_training import Parameters

        return Parameters.from_dict(dict(self.base_run_config))

    @property
    def seed(self) -> int:
        return int(self.base_run_config["seed"])

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
    else:
        raise PSBConfigError(
            "Supported PSB stages are 'p0_base_passthrough' and "
            "'p1_zero_control_equivalence'."
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
    if stage == "p1_zero_control_equivalence":
        parent_run = _resolve_existing_path(
            raw["parent_run"], "parent_run", config_path, kind="directory"
        )
        _validate_p0_parent(parent_run, base)
        conflict_graph = PSBConflictGraphConfig.from_dict(raw["conflict_graph"])
        proximal = PSBProximalConfig.from_dict(raw["proximal"])
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
                "P1 violates proximal uniqueness: 1/h_z + kappa must be "
                "greater than rho_max*nu*alpha."
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
    )
