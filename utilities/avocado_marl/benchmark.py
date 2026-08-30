"""A4 rollout and benchmark in SigmaRL's native road environment."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchrl.envs.utils import step_mdp

from utilities.avocado.road_benchmark import A3ScenarioRoadTraffic
from utilities.avocado.road_config import (
    A3RoadExperimentConfig,
    RoadCaseConfig,
)
from utilities.avocado_marl.bridge import A4ActionBridge, A4BridgeMetrics
from utilities.avocado_marl.config import A4ExperimentConfig
from utilities.constants import SCENARIOS
from utilities.experiment_artifacts import (
    atomic_write_json,
    resolve_latest_testable_run,
    resolve_policy_checkpoint,
)
from utilities.helper_training import Parameters, SaveData
from utilities.mappo_cavs import mappo_cavs


A4_SUPPORTED_PLANNERS = ("base_mappo", "base_mappo_avocado")


@dataclass(frozen=True)
class A4RolloutMetrics:
    case: str
    scenario_type: str
    planner: str
    episodes: int
    executed_steps: int
    checkpoint: str
    seed: int
    mean_reward_per_agent_step: float
    agent_collision_events_per_1000_steps: float
    lane_collision_events_per_1000_steps: float
    wrong_entry_events_per_1000_steps: float
    route_completion_events_per_1000_steps: float
    mean_reference_distance_meters: float
    p95_reference_distance_meters: float
    mean_measured_speed_mps: float
    bridge: A4BridgeMetrics

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        return result


@dataclass(frozen=True)
class A4BenchmarkRun:
    output_directory: Path
    metrics: Tuple[A4RolloutMetrics, ...]
    passed: bool
    checks: Tuple[str, ...]
    failures: Tuple[str, ...]


def _load_run_parameters(run_directory: Path) -> Parameters:
    resolved = run_directory / "config_resolved.json"
    if resolved.is_file():
        with resolved.open("r", encoding="utf-8") as stream:
            return Parameters.from_dict(json.load(stream))
    legacy = sorted(run_directory.glob("reward*_data.json"))
    if not legacy:
        raise FileNotFoundError(
            f"No config_resolved.json or legacy reward data found in {run_directory}."
        )
    with legacy[-1].open("r", encoding="utf-8") as stream:
        return SaveData.from_dict(json.load(stream)).parameters


def resolve_a4_policy_source(
    config: A4ExperimentConfig,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
) -> tuple[Path, Path]:
    policy_config = config.base_policy
    configured_run = (
        Path(policy_config.run_directory)
        if policy_config.run_directory is not None
        else None
    )
    configured_checkpoint = (
        Path(policy_config.checkpoint)
        if policy_config.checkpoint is not None
        else None
    )
    selected_checkpoint = checkpoint or configured_checkpoint
    selected_run = run_directory or configured_run
    if selected_checkpoint is not None and selected_run is None:
        selected_run = selected_checkpoint.expanduser().resolve().parent
    elif selected_run is None:
        selected_run = resolve_latest_testable_run(policy_config.output_root)
    else:
        selected_run = selected_run.expanduser().resolve()
        if not selected_run.is_dir():
            raise FileNotFoundError(
                f"Base-MAPPO run directory does not exist: {selected_run}"
            )
    selected_checkpoint = resolve_policy_checkpoint(
        selected_run, selected_checkpoint
    )
    return selected_run, selected_checkpoint


def _testing_parameters(
    run_directory: Path,
    a3_config: A3RoadExperimentConfig,
    case: RoadCaseConfig,
    episodes: int,
    max_steps: int,
    *,
    render_live: bool,
) -> Parameters:
    parameters = _load_run_parameters(run_directory)
    parameters.where_to_save = str(run_directory) + os.sep
    parameters.device = a3_config.simulation.device
    parameters.seed = a3_config.simulation.seed
    parameters.scenario_type = case.scenario_type
    parameters.n_agents = case.n_agents
    parameters.n_nearing_agents_observed = min(
        parameters.n_nearing_agents_observed, case.n_agents - 1
    )
    parameters.num_vmas_envs = episodes
    parameters.frames_per_batch = episodes * max_steps
    parameters.max_steps = max_steps
    parameters.dt = a3_config.parameters.dt
    parameters.artifact_logging_enabled = False
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = render_live
    parameters.is_save_eval_results = False
    parameters.is_load_model = True
    parameters.is_load_final_model = True
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.is_save_simulation_video = False
    parameters.is_visualize_short_term_path = True
    parameters.is_visualize_lane_boundary = True
    parameters.is_visualize_extra_info = True
    parameters.render_title = f"A4 MARL--AVOCADO | {case.name}"
    return parameters


def _reset_mask(scenario: A3ScenarioRoadTraffic) -> Tensor:
    return (
        scenario.a3_last_agent_collisions
        | scenario.a3_last_lane_collisions
        | scenario.a3_last_wrong_entries
        | scenario.a3_last_route_completions
    )


def _set_live_diagnostics(
    scenario: A3ScenarioRoadTraffic,
    bridge: A4ActionBridge,
    step: int,
) -> None:
    diagnostic = bridge.last
    if diagnostic is None:
        return
    nominal = diagnostic.nominal_action[0, 0]
    executed = diagnostic.executed_action[0, 0]
    lines = [
        f"stage: {bridge.stage_label}",
        f"step: {step}",
        f"MARL nominal: {float(nominal[0]):.2f} m/s, "
        f"{math.degrees(float(nominal[1])):.1f} deg",
        f"executed: {float(executed[0]):.2f} m/s, "
        f"{math.degrees(float(executed[1])):.1f} deg",
        f"action modified: {bool(diagnostic.intervention_mask[0, 0])}",
        f"active conflict: {bool(diagnostic.conflict_mask[0, 0])}",
        (
            "ego path error: "
            f"{float(scenario.a3_last_reference_distances[0, 0]):.3f} m"
        ),
    ]
    controller = bridge.controller
    if controller is not None:
        active = controller.last_neighbor_mask[0]
        opinions = controller.opinion[0][active]
        mean_opinion = float(opinions.abs().mean()) if opinions.numel() else 0.0
        lines.extend(
            (
                f"attention max: {float(controller.attention[0].max()):.3f}",
                f"mean |opinion|: {mean_opinion:.3f}",
                f"active VOs: {int(controller.last_active_vo_count[0].sum())}",
                f"joint infeasible: {int(controller.last_infeasible[0].sum())}",
            )
        )
        if bridge.last is not None and bridge.last.estimate_correction is not None:
            fusion_error = float(
                (
                    bridge.last.fused_estimate
                    - bridge.last.heuristic_estimate
                )
                .abs()
                .max()
            )
            lines.extend(
                (
                    "max |Delta y|: "
                    f"{float(bridge.last.estimate_correction.abs().max()):.4f}",
                    f"max |yF-yH|: {fusion_error:.4f}",
                )
            )
    if diagnostic.shield_result is not None:
        lines.append(
            "TTC shield interventions: "
            f"{int(diagnostic.shield_result.intervention_mask[0].sum())}"
        )
    lines.append(f"resets this step: {int(_reset_mask(scenario)[0].sum())}")
    scenario.set_opinion_visualization(lines)


def run_a4_rollout(
    config: A4ExperimentConfig,
    case: RoadCaseConfig,
    planner: str = "base_mappo_avocado",
    *,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    render_live: bool = False,
    bridge_factory: Optional[
        Callable[[torch.nn.Module, A3ScenarioRoadTraffic, A3RoadExperimentConfig], A4ActionBridge]
    ] = None,
) -> A4RolloutMetrics:
    if planner not in A4_SUPPORTED_PLANNERS:
        raise ValueError(f"Unsupported A4 planner: {planner}")
    a3_config = A3RoadExperimentConfig.from_json(config.a3_config)
    if seed_override is not None:
        if type(seed_override) is not int or seed_override < 0:
            raise ValueError("seed_override must be a non-negative integer.")
        a3_config = replace(
            a3_config,
            simulation=replace(a3_config.simulation, seed=seed_override),
        )
    episodes = (
        a3_config.simulation.episodes
        if episodes_override is None
        else episodes_override
    )
    max_steps = (
        a3_config.simulation.max_steps
        if max_steps_override is None
        else max_steps_override
    )
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive.")
    if render_live and episodes != 1:
        raise ValueError("Live rendering requires one environment.")
    selected_run, selected_checkpoint = resolve_a4_policy_source(
        config, run_directory, checkpoint
    )
    parameters = _testing_parameters(
        selected_run,
        a3_config,
        case,
        episodes,
        max_steps,
        render_live=render_live,
    )
    random.seed(a3_config.simulation.seed)
    np.random.seed(a3_config.simulation.seed)
    torch.manual_seed(a3_config.simulation.seed)
    scenario = A3ScenarioRoadTraffic()
    env, policy, priority_module, _ = mappo_cavs(
        parameters=parameters,
        policy_checkpoint_path=selected_checkpoint,
        scenario_override=scenario,
    )
    if priority_module is not None:
        raise NotImplementedError(
            "A4 currently supports the standard Base-MAPPO actor, not prioritized AP."
        )
    if bridge_factory is None:
        bridge = A4ActionBridge(
            policy,
            scenario,
            a3_config,
            use_avocado=planner == "base_mappo_avocado",
            deterministic=config.base_policy.deterministic,
            velocity_continuity_weight=(
                config.coupling.velocity_continuity_weight
            ),
            speed_intervention_tolerance_mps=(
                config.diagnostics.speed_intervention_tolerance_mps
            ),
            steering_intervention_tolerance_degrees=(
                config.diagnostics.steering_intervention_tolerance_degrees
            ),
        )
    else:
        if planner != "base_mappo_avocado":
            raise ValueError("A custom bridge requires base_mappo_avocado planner.")
        bridge = bridge_factory(policy, scenario, a3_config)

    agent_collisions = 0
    lane_collisions = 0
    wrong_entries = 0
    route_completions = 0
    reference_values = []
    measured_speed_sum = 0.0
    reward_sum = 0.0
    action_samples = 0
    tensordict = env.reset()
    for step in range(max_steps):
        tensordict = bridge(tensordict)
        step_tensordict = env.step(tensordict)
        reward_sum += float(
            step_tensordict.get(("next", "agents", "reward")).sum()
        )
        reset_mask = _reset_mask(scenario)
        agent_collisions += int(scenario.a3_last_agent_collisions.sum())
        lane_collisions += int(scenario.a3_last_lane_collisions.sum())
        wrong_entries += int(scenario.a3_last_wrong_entries.sum())
        route_completions += int(scenario.a3_last_route_completions.sum())
        reference_values.append(
            scenario.a3_last_reference_distances.detach().cpu().reshape(-1)
        )
        measured = torch.stack(
            [agent.state.vel for agent in scenario.world.agents], dim=1
        )
        measured_speed_sum += float(
            torch.linalg.vector_norm(measured, dim=-1).sum()
        )
        action_samples += episodes * case.n_agents
        if render_live:
            _set_live_diagnostics(scenario, bridge, step + 1)
            env.render(mode="rgb_array", visualize_when_rgb=True)
        bridge.reset_agents(reset_mask)
        if step + 1 < max_steps:
            tensordict = step_mdp(
                step_tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=env.reward_keys,
                action_keys=env.action_keys,
                done_keys=env.done_keys,
            )

    all_reference = torch.cat(reference_values)
    scale = 1000.0 / max(action_samples, 1)
    result = A4RolloutMetrics(
        case=case.name,
        scenario_type=case.scenario_type,
        planner=planner,
        episodes=episodes,
        executed_steps=max_steps,
        checkpoint=str(selected_checkpoint),
        seed=a3_config.simulation.seed,
        mean_reward_per_agent_step=reward_sum / max(action_samples, 1),
        agent_collision_events_per_1000_steps=agent_collisions * scale,
        lane_collision_events_per_1000_steps=lane_collisions * scale,
        wrong_entry_events_per_1000_steps=wrong_entries * scale,
        route_completion_events_per_1000_steps=route_completions * scale,
        mean_reference_distance_meters=float(all_reference.mean()),
        p95_reference_distance_meters=float(
            torch.quantile(all_reference, 0.95)
        ),
        mean_measured_speed_mps=measured_speed_sum / max(action_samples, 1),
        bridge=bridge.metrics(),
    )
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return result


def _select_case(
    a3_config: A3RoadExperimentConfig,
    *,
    case_name: Optional[str],
    scenario_type: Optional[str],
) -> RoadCaseConfig:
    if case_name is not None and scenario_type is not None:
        raise ValueError("Select either case_name or scenario_type, not both.")
    if scenario_type is not None:
        if scenario_type not in SCENARIOS:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        return RoadCaseConfig(
            name=f"{scenario_type}_{SCENARIOS[scenario_type]['n_agents']}",
            scenario_type=scenario_type,
            n_agents=int(SCENARIOS[scenario_type]["n_agents"]),
        )
    selected = a3_config.cases[0].name if case_name is None else case_name
    for case in a3_config.cases:
        if case.name == selected:
            return case
    raise ValueError(f"Unknown configured A4 case: {selected}")


def run_live_a4_case(
    config: A4ExperimentConfig,
    *,
    case_name: Optional[str] = None,
    scenario_type: Optional[str] = None,
    planner: str = "base_mappo_avocado",
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    max_steps_override: Optional[int] = None,
) -> A4RolloutMetrics:
    a3_config = A3RoadExperimentConfig.from_json(config.a3_config)
    case = _select_case(
        a3_config, case_name=case_name, scenario_type=scenario_type
    )
    return run_a4_rollout(
        config,
        case,
        planner,
        run_directory=run_directory,
        checkpoint=checkpoint,
        episodes_override=1,
        max_steps_override=max_steps_override,
        render_live=True,
    )


def run_a4_benchmark(
    config: A4ExperimentConfig,
    *,
    output_directory: Optional[Path] = None,
    run_directory: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> A4BenchmarkRun:
    a3_config = A3RoadExperimentConfig.from_json(config.a3_config)
    if output_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path(config.output_root) / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = []
    for case in a3_config.cases:
        for planner in A4_SUPPORTED_PLANNERS:
            metrics.append(
                run_a4_rollout(
                    config,
                    case,
                    planner,
                    run_directory=run_directory,
                    checkpoint=checkpoint,
                    episodes_override=episodes_override,
                    max_steps_override=max_steps_override,
                )
            )

    checks = []
    failures = []
    for item in metrics:
        bridge = item.bridge
        if bridge.nonfinite_action_count == 0:
            checks.append(f"{item.case}/{item.planner}: all actions finite.")
        else:
            failures.append(
                f"{item.case}/{item.planner}: "
                f"{bridge.nonfinite_action_count} nonfinite actions."
            )
        if item.planner == "base_mappo":
            if bridge.action_intervention_rate == 0.0:
                checks.append(
                    f"{item.case}: raw Base-MAPPO action is passed through exactly."
                )
            else:
                failures.append(
                    f"{item.case}: raw Base-MAPPO unexpectedly modified actions."
                )
        elif bridge.action_samples > 0:
            checks.append(
                f"{item.case}: MARL nominal and AVOCADO executed actions audited."
            )
    atomic_write_json(
        output_directory / "summary.json",
        {
            "schema_version": 1,
            "method": "avocado_marl",
            "stage": "a4",
            "passed": not failures,
            "checks": checks,
            "failures": failures,
            "metrics": [item.to_dict() for item in metrics],
        },
    )
    return A4BenchmarkRun(
        output_directory=output_directory,
        metrics=tuple(metrics),
        passed=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
    )
