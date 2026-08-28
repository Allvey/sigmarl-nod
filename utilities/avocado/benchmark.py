"""A2: reproducible AVOCADO effectiveness benchmark and validation gates."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor
import vmas

from scenarios.avocado_holonomic import Scenario
from utilities.avocado.config import AVOCADOExperimentConfig, CaseConfig
from utilities.avocado.controller import AVOCADOController, fixed_orca_actions
from utilities.avocado.core import goal_preferred_velocity


@dataclass(frozen=True)
class BenchmarkMetrics:
    case: str
    planner: str
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_time_to_goal_seconds: Optional[float]
    mean_path_length_meters: float
    minimum_clearance_meters: float
    mean_controller_time_microseconds: float
    infeasible_projection_rate: float
    maximum_attention: float
    mean_absolute_opinion: float


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    checks: Tuple[str, ...]
    failures: Tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkRun:
    output_directory: Path
    metrics: Tuple[BenchmarkMetrics, ...]
    validation: ValidationResult


def _state(scenario: Scenario) -> Tuple[Tensor, Tensor]:
    positions = torch.stack(
        [agent.state.pos for agent in scenario.world.agents], dim=1
    )
    velocities = torch.stack(
        [agent.state.vel for agent in scenario.world.agents], dim=1
    )
    return positions, velocities


def _pair_clearance(
    positions: Tensor, radii: Tensor, controlled_mask: Tensor
) -> Tuple[Tensor, Tensor]:
    batch_size, entity_count, _ = positions.shape
    differences = positions[:, :, None, :] - positions[:, None, :, :]
    distances = torch.linalg.vector_norm(differences, dim=-1)
    combined = radii[None, :, None] + radii[None, None, :]
    clearances = distances - combined
    upper = torch.triu(
        torch.ones(
            entity_count,
            entity_count,
            dtype=torch.bool,
            device=positions.device,
        ),
        diagonal=1,
    )
    controlled_pairs = controlled_mask[:, None] | controlled_mask[None, :]
    relevant = upper & controlled_pairs
    selected = clearances[:, relevant]
    minimum = selected.min(dim=-1).values
    collision = torch.any(selected <= 0.0, dim=-1)
    return minimum, collision


def run_case(
    config: AVOCADOExperimentConfig,
    case: CaseConfig,
    planner: str,
    *,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> Tuple[BenchmarkMetrics, Dict[str, Tensor]]:
    """Run one vectorized planner/case combination."""

    simulation = config.simulation
    entities = config.entities
    episodes = simulation.episodes if episodes_override is None else episodes_override
    max_steps = (
        simulation.max_steps if max_steps_override is None else max_steps_override
    )
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive.")
    scenario = Scenario()
    env = vmas.make_env(
        scenario=scenario,
        num_envs=episodes,
        device=simulation.device,
        continuous_actions=True,
        max_steps=None,
        clamp_actions=True,
        grad_enabled=False,
        layout=case.layout,
        n_agents=case.n_agents,
        controlled_agents=case.controlled_agents,
        dt=config.parameters.dt,
        robot_radius=entities.robot_radius,
        agent_radius=entities.agent_radius,
        avoidance_radius_scale=entities.avoidance_radius_scale,
        robot_max_speed=entities.robot_max_speed,
        agent_max_speed=entities.agent_max_speed,
        goal_tolerance=simulation.goal_tolerance,
        position_jitter=simulation.position_jitter,
        layout_seed=simulation.layout_seed,
    )
    env.reset(seed=simulation.layout_seed)
    device = scenario.world.device
    controlled_mask = scenario.controlled_mask
    radii = scenario.radii
    maximum_speeds = scenario.maximum_speeds
    controller = None
    if planner == "avocado":
        controller = AVOCADOController(
            config.parameters,
            batch_size=episodes,
            entity_count=case.n_agents,
            controlled_mask=controlled_mask,
            security_radii=scenario.security_radii,
            maximum_speeds=maximum_speeds,
            seed=simulation.controller_seed,
            device=device,
        )

    positions, velocities = _state(scenario)
    previous_positions = positions.clone()
    reached = torch.zeros(
        episodes, case.n_agents, dtype=torch.bool, device=device
    )
    collision_ever = torch.zeros(episodes, dtype=torch.bool, device=device)
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    success = torch.zeros(episodes, dtype=torch.bool, device=device)
    finish_time = torch.full((episodes,), torch.nan, device=device)
    path_length = torch.zeros(episodes, case.n_agents, device=device)
    minimum_clearance = torch.full((episodes,), torch.inf, device=device)
    controller_seconds = 0.0
    controller_calls = 0
    infeasible_count = 0
    projection_count = 0
    maximum_attention = 0.0
    opinion_accumulator = 0.0
    opinion_samples = 0
    trajectory_positions: List[Tensor] = [positions[0].detach().cpu().clone()]
    trajectory_actions: List[Tensor] = []

    for step in range(max_steps):
        distances_to_goal = torch.linalg.vector_norm(
            scenario.goals - positions, dim=-1
        )
        reached |= distances_to_goal <= simulation.goal_tolerance
        preferred = goal_preferred_velocity(
            positions, scenario.goals, maximum_speeds[None, :]
        )
        preferred[reached] = 0.0
        preferred[finished] = 0.0

        start = time.perf_counter()
        if planner == "preferred":
            actions = preferred
        elif planner == "orca":
            actions = fixed_orca_actions(
                positions,
                velocities,
                preferred,
                controlled_mask=controlled_mask,
                security_radii=scenario.security_radii,
                maximum_speeds=maximum_speeds,
                perception_radius=config.parameters.perception_radius,
                time_horizon=config.parameters.velocity_obstacle_horizon,
            )
        elif planner == "avocado":
            assert controller is not None
            actions = controller.step(
                positions,
                velocities,
                preferred,
                active_environment_mask=~finished,
            )
            infeasible_count += int(controller.last_infeasible.sum())
            projection_count += int((~finished).sum()) * int(controlled_mask.sum())
            maximum_attention = max(
                maximum_attention, float(controller.attention.max())
            )
            active_opinions = controller.opinion[controller.last_neighbor_mask]
            if active_opinions.numel():
                opinion_accumulator += float(active_opinions.abs().sum())
                opinion_samples += int(active_opinions.numel())
        else:
            raise ValueError(f"Unknown planner: {planner}.")
        controller_seconds += time.perf_counter() - start
        controller_calls += int((~finished).sum()) * max(
            int(controlled_mask.sum()), 1
        )
        actions[finished] = 0.0
        trajectory_actions.append(actions[0].detach().cpu().clone())

        env.step([actions[:, index] for index in range(case.n_agents)])
        positions, velocities = _state(scenario)
        path_length += torch.linalg.vector_norm(
            positions - previous_positions, dim=-1
        )
        previous_positions = positions.clone()
        clearance, colliding = _pair_clearance(
            positions, radii, controlled_mask
        )
        minimum_clearance = torch.minimum(minimum_clearance, clearance)
        new_collision = colliding & ~finished
        collision_ever |= new_collision

        distances_to_goal = torch.linalg.vector_norm(
            scenario.goals - positions, dim=-1
        )
        reached |= distances_to_goal <= simulation.goal_tolerance
        reached_controlled = torch.all(reached[:, controlled_mask], dim=-1)
        new_success = reached_controlled & ~collision_ever & ~finished
        finish_time[new_success] = (step + 1) * config.parameters.dt
        success |= new_success
        finished |= new_success | new_collision
        trajectory_positions.append(positions[0].detach().cpu().clone())
        if bool(torch.all(finished)):
            break

    timeout = ~finished
    controlled_path = path_length[:, controlled_mask].mean(dim=-1)
    successful_times = finish_time[success]
    mean_time = (
        float(successful_times.mean()) if successful_times.numel() else None
    )
    metrics = BenchmarkMetrics(
        case=case.name,
        planner=planner,
        episodes=episodes,
        success_rate=float(success.float().mean()),
        collision_rate=float(collision_ever.float().mean()),
        timeout_rate=float(timeout.float().mean()),
        mean_time_to_goal_seconds=mean_time,
        mean_path_length_meters=float(controlled_path.mean()),
        minimum_clearance_meters=float(minimum_clearance.min()),
        mean_controller_time_microseconds=(
            controller_seconds * 1e6 / max(controller_calls, 1)
        ),
        infeasible_projection_rate=(
            infeasible_count / max(projection_count, 1)
        ),
        maximum_attention=maximum_attention,
        mean_absolute_opinion=(
            opinion_accumulator / max(opinion_samples, 1)
        ),
    )
    trajectory = {
        "positions": torch.stack(trajectory_positions),
        "actions": torch.stack(trajectory_actions),
        "goals": scenario.goals[0].detach().cpu(),
        "radii": radii.detach().cpu(),
        "controlled_mask": controlled_mask.detach().cpu(),
    }
    return metrics, trajectory


def validate_metrics(
    config: AVOCADOExperimentConfig,
    metrics: Tuple[BenchmarkMetrics, ...],
) -> ValidationResult:
    """Evaluate the explicit A2 go/no-go criteria from the JSON config."""

    lookup = {(item.case, item.planner): item for item in metrics}
    checks = []
    failures = []
    validation = config.validation
    required_avocado = lookup[(validation.required_case, "avocado")]
    required_preferred = lookup[(validation.required_case, "preferred")]

    def check(condition: bool, success_message: str, failure_message: str) -> None:
        if condition:
            checks.append(success_message)
        else:
            failures.append(failure_message)

    for case in config.cases:
        avocado = lookup[(case.name, "avocado")]
        check(
            avocado.success_rate >= validation.minimum_avocado_success_rate,
            f"{case.name}: AVOCADO success_rate={avocado.success_rate:.3f} passed.",
            f"{case.name}: AVOCADO success_rate={avocado.success_rate:.3f} is below "
            f"{validation.minimum_avocado_success_rate:.3f}.",
        )
        check(
            avocado.collision_rate <= validation.maximum_avocado_collision_rate,
            f"{case.name}: AVOCADO collision_rate={avocado.collision_rate:.3f} passed.",
            f"{case.name}: AVOCADO collision_rate={avocado.collision_rate:.3f} exceeds "
            f"{validation.maximum_avocado_collision_rate:.3f}.",
        )
    check(
        required_preferred.collision_rate
        >= validation.minimum_preferred_collision_rate,
        f"{validation.required_case}: preferred baseline exposes the conflict.",
        f"{validation.required_case}: preferred collision_rate="
        f"{required_preferred.collision_rate:.3f} is below the required "
        f"{validation.minimum_preferred_collision_rate:.3f}.",
    )
    improvement = (
        required_preferred.collision_rate - required_avocado.collision_rate
    )
    check(
        improvement >= validation.minimum_collision_rate_improvement,
        f"{validation.required_case}: collision-rate improvement={improvement:.3f} passed.",
        f"{validation.required_case}: collision-rate improvement={improvement:.3f} is below "
        f"{validation.minimum_collision_rate_improvement:.3f}.",
    )
    return ValidationResult(not failures, tuple(checks), tuple(failures))


def _save_trajectory_plot(
    output_path: Path,
    case: CaseConfig,
    trajectories: Mapping[str, Mapping[str, Tensor]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planners = list(trajectories)
    figure, axes = plt.subplots(
        1, len(planners), figsize=(5 * len(planners), 4.5), squeeze=False
    )
    for axis, planner in zip(axes[0], planners):
        trajectory = trajectories[planner]
        positions = trajectory["positions"].numpy()
        goals = trajectory["goals"].numpy()
        controlled = trajectory["controlled_mask"].numpy()
        for index in range(positions.shape[1]):
            color = "tab:blue" if controlled[index] else "tab:orange"
            axis.plot(
                positions[:, index, 0],
                positions[:, index, 1],
                color=color,
                linewidth=1.5,
            )
            axis.scatter(
                positions[0, index, 0],
                positions[0, index, 1],
                color=color,
                marker="o",
                s=24,
            )
            axis.scatter(
                goals[index, 0], goals[index, 1], color=color, marker="x", s=36
            )
        axis.set_title(planner)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    figure.suptitle(f"A2 strict AVOCADO benchmark: {case.name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_benchmark(
    config: AVOCADOExperimentConfig,
    *,
    output_directory: Optional[Path] = None,
    save_plots: bool = True,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> BenchmarkRun:
    """Run all A2 cases, persist evidence, and evaluate validation gates."""

    if output_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path(config.output_root) / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    all_metrics: List[BenchmarkMetrics] = []
    for case in config.cases:
        case_trajectories: Dict[str, Mapping[str, Tensor]] = {}
        for planner in config.planners:
            metrics, trajectory = run_case(
                config,
                case,
                planner,
                episodes_override=episodes_override,
                max_steps_override=max_steps_override,
            )
            all_metrics.append(metrics)
            case_trajectories[planner] = trajectory
        if save_plots:
            _save_trajectory_plot(
                output_directory / f"trajectories_{case.name}.png",
                case,
                case_trajectories,
            )

    metrics_tuple = tuple(all_metrics)
    validation = validate_metrics(config, metrics_tuple)
    summary: Dict[str, Any] = {
        "config": config.to_dict(),
        "metrics": [asdict(item) for item in metrics_tuple],
        "validation": asdict(validation),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return BenchmarkRun(output_directory, metrics_tuple, validation)
