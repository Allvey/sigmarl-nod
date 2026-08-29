"""A5 strict-zero rollout and stepwise A4 equivalence validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from utilities.avocado.road_benchmark import A3ScenarioRoadTraffic
from utilities.avocado.road_config import A3RoadExperimentConfig, RoadCaseConfig
from utilities.avocado_marl.a5_bridge import A5ActionBridge
from utilities.avocado_marl.a5_config import A5ExperimentConfig
from utilities.avocado_marl.benchmark import (
    A4RolloutMetrics,
    _select_case,
    run_a4_rollout,
)
from utilities.avocado_marl.bridge import A4ActionBridge, A4BridgeTrace
from utilities.avocado_marl.config import A4ExperimentConfig
from utilities.experiment_artifacts import atomic_write_json


@dataclass(frozen=True)
class A5RolloutDiagnostics:
    maximum_absolute_correction: float
    maximum_fusion_error: float
    network_parameter_count: int
    trainable_parameter_count: int
    valid_pair_samples: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class A5RolloutResult:
    metrics: A4RolloutMetrics
    diagnostics: A5RolloutDiagnostics


@dataclass(frozen=True)
class A5EquivalenceResult:
    case: str
    episodes: int
    executed_steps: int
    passed: bool
    checks: Tuple[str, ...]
    failures: Tuple[str, ...]
    maximum_differences: Dict[str, float]
    a4_metrics: A4RolloutMetrics
    a5_metrics: A4RolloutMetrics
    a5_diagnostics: A5RolloutDiagnostics
    trace_file: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "case": self.case,
            "episodes": self.episodes,
            "executed_steps": self.executed_steps,
            "passed": self.passed,
            "checks": list(self.checks),
            "failures": list(self.failures),
            "maximum_differences": self.maximum_differences,
            "a4_metrics": self.a4_metrics.to_dict(),
            "a5_metrics": self.a5_metrics.to_dict(),
            "a5_diagnostics": self.a5_diagnostics.to_dict(),
            "trace_file": self.trace_file,
        }


@dataclass(frozen=True)
class A5ValidationRun:
    output_directory: Path
    results: Tuple[A5EquivalenceResult, ...]
    passed: bool


def _a4_bridge_factory(
    a4_config: A4ExperimentConfig,
    captured: list[A4ActionBridge],
):
    def factory(policy, scenario, a3_config):
        bridge = A4ActionBridge(
            policy,
            scenario,
            a3_config,
            use_avocado=True,
            deterministic=a4_config.base_policy.deterministic,
            velocity_continuity_weight=(
                a4_config.coupling.velocity_continuity_weight
            ),
            speed_intervention_tolerance_mps=(
                a4_config.diagnostics.speed_intervention_tolerance_mps
            ),
            steering_intervention_tolerance_degrees=(
                a4_config.diagnostics.steering_intervention_tolerance_degrees
            ),
        )
        captured.append(bridge)
        return bridge

    return factory


def _a5_bridge_factory(
    a4_config: A4ExperimentConfig,
    a5_config: A5ExperimentConfig,
    captured: list[A5ActionBridge],
):
    def factory(policy, scenario, a3_config):
        bridge = A5ActionBridge(
            policy,
            scenario,
            a3_config,
            a5_config.y_correction,
            deterministic=a4_config.base_policy.deterministic,
            velocity_continuity_weight=(
                a4_config.coupling.velocity_continuity_weight
            ),
            speed_intervention_tolerance_mps=(
                a4_config.diagnostics.speed_intervention_tolerance_mps
            ),
            steering_intervention_tolerance_degrees=(
                a4_config.diagnostics.steering_intervention_tolerance_degrees
            ),
        )
        captured.append(bridge)
        return bridge

    return factory


def _a5_diagnostics(bridge: A5ActionBridge) -> A5RolloutDiagnostics:
    trace = bridge.trace()
    network = bridge.y_correction_net
    return A5RolloutDiagnostics(
        maximum_absolute_correction=float(trace.estimate_correction.abs().max()),
        maximum_fusion_error=float(
            (trace.fused_estimate - trace.heuristic_estimate).abs().max()
        ),
        network_parameter_count=sum(
            parameter.numel() for parameter in network.parameters()
        ),
        trainable_parameter_count=sum(
            parameter.numel()
            for parameter in network.parameters()
            if parameter.requires_grad
        ),
        valid_pair_samples=int(trace.pair_mask.sum()),
    )


def run_a5_rollout(
    config: A5ExperimentConfig,
    case: RoadCaseConfig,
    *,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    render_live: bool = False,
) -> A5RolloutResult:
    a4_config = A4ExperimentConfig.from_json(config.a4_config)
    captured: list[A5ActionBridge] = []
    metrics = run_a4_rollout(
        a4_config,
        case,
        "base_mappo_avocado",
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
        render_live=render_live,
        bridge_factory=_a5_bridge_factory(a4_config, config, captured),
    )
    if len(captured) != 1:
        raise RuntimeError("A5 bridge factory did not capture exactly one bridge.")
    return A5RolloutResult(metrics, _a5_diagnostics(captured[0]))


def _maximum_difference(first: Tensor, second: Tensor) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float((first - second).abs().max())


def run_a5_equivalence(
    config: A5ExperimentConfig,
    case: RoadCaseConfig,
    *,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    trace_output_path: Optional[Path] = None,
) -> A5EquivalenceResult:
    """Run A4 and A5 from equal seeds and compare every physical step."""

    a4_config = A4ExperimentConfig.from_json(config.a4_config)
    a4_captured: list[A4ActionBridge] = []
    a4_metrics = run_a4_rollout(
        a4_config,
        case,
        "base_mappo_avocado",
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
        bridge_factory=_a4_bridge_factory(a4_config, a4_captured),
    )
    a5_captured: list[A5ActionBridge] = []
    a5_metrics = run_a4_rollout(
        a4_config,
        case,
        "base_mappo_avocado",
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=episodes_override,
        max_steps_override=max_steps_override,
        bridge_factory=_a5_bridge_factory(a4_config, config, a5_captured),
    )
    if len(a4_captured) != 1 or len(a5_captured) != 1:
        raise RuntimeError("Equivalence rollout did not capture both bridges.")
    a4_trace: A4BridgeTrace = a4_captured[0].trace()
    a5_trace: A4BridgeTrace = a5_captured[0].trace()
    compared = {
        "nominal_action": (a4_trace.nominal_action, a5_trace.nominal_action),
        "executed_action": (a4_trace.executed_action, a5_trace.executed_action),
        "heuristic_estimate": (
            a4_trace.heuristic_estimate,
            a5_trace.heuristic_estimate,
        ),
        "fused_estimate": (a4_trace.fused_estimate, a5_trace.fused_estimate),
        "opinion": (a4_trace.opinion, a5_trace.opinion),
        "attention": (a4_trace.attention, a5_trace.attention),
        "pair_mask": (
            a4_trace.pair_mask.to(torch.int8),
            a5_trace.pair_mask.to(torch.int8),
        ),
    }
    maximum_differences = {
        name: _maximum_difference(first, second)
        for name, (first, second) in compared.items()
    }
    diagnostics = _a5_diagnostics(a5_captured[0])
    checks = []
    failures = []
    if diagnostics.maximum_absolute_correction == 0.0:
        checks.append("YCorrectionNet output is exactly zero.")
    else:
        failures.append("YCorrectionNet output was not exactly zero.")
    if diagnostics.maximum_fusion_error == 0.0:
        checks.append("A5 fused estimate equals heuristic estimate exactly.")
    else:
        failures.append("A5 fused estimate differs from heuristic estimate.")
    if diagnostics.trainable_parameter_count == 0:
        checks.append("All A5 YCorrectionNet parameters are frozen.")
    else:
        failures.append("A5 YCorrectionNet contains trainable parameters.")
    for name, difference in maximum_differences.items():
        if difference == 0.0:
            checks.append(f"A4/A5 {name} is stepwise identical.")
        else:
            failures.append(
                f"A4/A5 {name} maximum difference is {difference:.9g}."
            )
    trace_file = None
    if trace_output_path is not None:
        trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "method": "avocado_marl",
                "stage": "a5",
                "case": case.name,
                "attention": a5_trace.attention,
                "pair_mask": a5_trace.pair_mask,
                "y_heuristic": a5_trace.heuristic_estimate,
                "delta_y": a5_trace.estimate_correction,
                "y_fused": a5_trace.fused_estimate,
                "opinion_z": a5_trace.opinion,
                "nominal_action": a5_trace.nominal_action,
                "executed_action": a5_trace.executed_action,
            },
            trace_output_path,
        )
        trace_file = str(trace_output_path)
    return A5EquivalenceResult(
        case=case.name,
        episodes=a5_metrics.episodes,
        executed_steps=a5_metrics.executed_steps,
        passed=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
        maximum_differences=maximum_differences,
        a4_metrics=a4_metrics,
        a5_metrics=a5_metrics,
        a5_diagnostics=diagnostics,
        trace_file=trace_file,
    )


def run_live_a5_case(
    config: A5ExperimentConfig,
    *,
    case_name: Optional[str] = None,
    scenario_type: Optional[str] = None,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    max_steps_override: Optional[int] = None,
) -> A5RolloutResult:
    a4_config = A4ExperimentConfig.from_json(config.a4_config)
    a3_config = A3RoadExperimentConfig.from_json(a4_config.a3_config)
    case = _select_case(
        a3_config, case_name=case_name, scenario_type=scenario_type
    )
    return run_a5_rollout(
        config,
        case,
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=1,
        max_steps_override=max_steps_override,
        render_live=True,
    )


def run_a5_validation(
    config: A5ExperimentConfig,
    *,
    output_directory: Optional[Path] = None,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> A5ValidationRun:
    a4_config = A4ExperimentConfig.from_json(config.a4_config)
    a3_config = A3RoadExperimentConfig.from_json(a4_config.a3_config)
    if output_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path(config.output_root) / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    collected = []
    for case in a3_config.cases:
        collected.append(
            run_a5_equivalence(
                config,
                case,
                run_directory=run_directory,
                checkpoint=checkpoint,
                episodes_override=episodes_override,
                max_steps_override=max_steps_override,
                trace_output_path=output_directory / f"trace_{case.name}.pt",
            )
        )
    results = tuple(collected)
    passed = all(result.passed for result in results)
    atomic_write_json(
        output_directory / "summary.json",
        {
            "schema_version": 1,
            "method": "avocado_marl",
            "stage": "a5",
            "passed": passed,
            "results": [result.to_dict() for result in results],
        },
    )
    return A5ValidationRun(output_directory, results, passed)
