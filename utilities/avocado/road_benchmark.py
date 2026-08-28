"""A3 benchmark: AVOCADO-KB inside SigmaRL's road-traffic environment."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import vmas

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.avocado.bicycle import (
    BicycleAdapterParameters,
    path_velocity_cone_constraints,
    stanley_path_preferred_velocity,
    vector_velocity_to_bicycle_action,
)
from utilities.avocado.controller import AVOCADOController, fixed_orca_actions
from utilities.avocado.road_config import A3RoadExperimentConfig, RoadCaseConfig
from utilities.avocado.road_safety import (
    TTCSafetyShieldResult,
    apply_ttc_braking_shield,
)
from utilities.constants import AGENTS, SCENARIOS
from utilities.helper_training import Parameters


@dataclass(frozen=True)
class RoadBenchmarkMetrics:
    case: str
    planner: str
    episodes: int
    executed_steps: int
    collision_free_episode_rate: float
    agent_collision_events_per_1000_steps: float
    lane_collision_events_per_1000_steps: float
    wrong_entry_events_per_1000_steps: float
    route_completion_events_per_1000_steps: float
    mean_reward_per_agent_step: float
    mean_reference_distance_meters: float
    rms_reference_distance_meters: float
    p95_reference_distance_meters: float
    maximum_reference_distance_meters: float
    minimum_lane_clearance_meters: float
    minimum_vehicle_clearance_meters: float
    mean_tracking_error_mps: float
    steering_saturation_rate: float
    mean_controller_time_microseconds: float
    infeasible_projection_rate: float
    maximum_attention: float
    mean_absolute_opinion: float
    shield_interventions_per_1000_steps: float
    shield_intervention_rate: float
    post_shield_unsafe_pair_events_per_1000_steps: float
    nonfinite_action_count: int


@dataclass(frozen=True)
class RoadValidationResult:
    passed: bool
    checks: Tuple[str, ...]
    failures: Tuple[str, ...]


@dataclass(frozen=True)
class RoadBenchmarkRun:
    output_directory: Path
    metrics: Tuple[RoadBenchmarkMetrics, ...]
    validation: RoadValidationResult


class A3ScenarioRoadTraffic(ScenarioRoadTraffic):
    """Road scenario with pre-reset event snapshots for A3 diagnostics only."""

    def done(self) -> Tensor:
        self.a3_last_reference_distances = self.distances.ref_paths.clone()
        self.a3_last_lane_clearances = self.distances.boundaries.clone()
        self.a3_last_agent_collisions = self.collisions.with_agents.any(
            dim=-1
        ).clone()
        self.a3_last_lane_collisions = self.collisions.with_lanelets.clone()
        self.a3_last_wrong_entries = self.collisions.with_entry_segments.clone()
        self.a3_last_route_completions = self.collisions.with_exit_segments.clone()
        return super().done()


def _road_state(scenario: ScenarioRoadTraffic) -> Tuple[Tensor, Tensor, Tensor]:
    positions = torch.stack(
        [agent.state.pos for agent in scenario.world.agents], dim=1
    )
    velocities = torch.stack(
        [agent.state.vel for agent in scenario.world.agents], dim=1
    )
    yaws = torch.stack(
        [agent.state.rot for agent in scenario.world.agents], dim=1
    )
    return positions, velocities, yaws


def _build_road_environment(
    config: A3RoadExperimentConfig,
    case: RoadCaseConfig,
    episodes: int,
    max_steps: int,
    *,
    real_time_rendering: bool = False,
) -> Tuple[object, A3ScenarioRoadTraffic]:
    simulation = config.simulation
    random.seed(simulation.seed)
    np.random.seed(simulation.seed)
    torch.manual_seed(simulation.seed)
    parameters = Parameters(
        n_agents=case.n_agents,
        dt=config.parameters.dt,
        device=simulation.device,
        seed=simulation.seed,
        frames_per_batch=episodes * max_steps,
        max_steps=max_steps,
        scenario_type=case.scenario_type,
        is_testing_mode=True,
        is_partial_observation=True,
        n_nearing_agents_observed=min(2, case.n_agents - 1),
        is_add_noise=False,
        is_apply_mask=False,
        is_use_mtv_distance=True,
        is_challenging_initial_state_buffer=False,
        is_real_time_rendering=real_time_rendering,
        is_visualize_short_term_path=True,
        is_visualize_lane_boundary=True,
        is_visualize_extra_info=True,
        render_title=f"A3 {case.name}",
        is_save_eval_results=False,
        artifact_logging_enabled=False,
    )
    scenario = A3ScenarioRoadTraffic()
    scenario.parameters = parameters
    environment = vmas.make_env(
        scenario=scenario,
        num_envs=episodes,
        device=simulation.device,
        continuous_actions=True,
        max_steps=None,
        n_agents=case.n_agents,
    )
    environment.reset(seed=simulation.seed)
    return environment, scenario


def _render_road_environment(environment: object) -> None:
    """Render one VMAS frame in a visible window."""

    environment.render(mode="human")


def _set_live_diagnostics(
    scenario: A3ScenarioRoadTraffic,
    planner: str,
    actions: Tensor,
    controller: Optional[AVOCADOController],
    shield_result: Optional[TTCSafetyShieldResult],
) -> None:
    """Expose controller state through the road scenario's native overlay."""

    lines = [
        f"planner: {planner}",
        f"ego speed: {float(actions[0, 0, 0]):.2f} m/s",
        f"ego steer: {math.degrees(float(actions[0, 0, 1])):.1f} deg",
        (
            "ego path error: "
            f"{float(scenario.a3_last_reference_distances[0, 0]):.3f} m"
        ),
        (
            "ego lane clearance: "
            f"{float(scenario.a3_last_lane_clearances[0, 0]):.3f} m"
        ),
    ]
    if controller is not None:
        active = controller.last_neighbor_mask[0]
        active_opinions = controller.opinion[0][active]
        mean_opinion = (
            float(active_opinions.abs().mean())
            if active_opinions.numel()
            else 0.0
        )
        lines.extend(
            (
                f"attention max: {float(controller.attention[0].max()):.3f}",
                f"mean |opinion|: {mean_opinion:.3f}",
                f"active VOs: {int(controller.last_active_vo_count[0].sum())}",
                f"joint infeasible: {int(controller.last_infeasible[0].sum())}",
            )
        )
        finite_ttc = (
            controller.last_neighbor_mask[0]
            & torch.isfinite(controller.last_time_to_collision[0])
        )
        if bool(finite_ttc.any()):
            masked_ttc = controller.last_time_to_collision[0].masked_fill(
                ~finite_ttc, torch.inf
            )
            flat_index = int(masked_ttc.argmin())
            robot = flat_index // controller.entity_count
            agent = flat_index % controller.entity_count
            lines.extend(
                (
                    "critical measured TTC: "
                    f"{float(masked_ttc[robot, agent]):.2f} s ({robot}->{agent})",
                    "critical responsibility: "
                    f"{float(controller.last_responsibility[0, robot, agent]):.2f}",
                )
            )
    if shield_result is not None:
        minimum_before = float(shield_result.minimum_ttc_before[0])
        minimum_label = f"{minimum_before:.2f} s" if math.isfinite(
            minimum_before
        ) else "inf"
        lines.extend(
            (
                f"executable TTC before shield: {minimum_label}",
                "TTC shield interventions: "
                f"{int(shield_result.intervention_mask[0].sum())}",
                "unsafe pairs after shield: "
                f"{int(shield_result.unsafe_pair_count_after[0])}",
            )
        )
    reset_reason_count = (
        int(scenario.a3_last_agent_collisions[0].sum())
        + int(scenario.a3_last_lane_collisions[0].sum())
        + int(scenario.a3_last_wrong_entries[0].sum())
        + int(scenario.a3_last_route_completions[0].sum())
    )
    lines.append(f"resets this step: {reset_reason_count}")
    scenario.set_opinion_visualization(lines)


def _adapter_parameters(scenario: ScenarioRoadTraffic) -> BicycleAdapterParameters:
    dynamics = scenario.world.agents[0].dynamics
    return BicycleAdapterParameters(
        front_length=float(dynamics.l_f),
        rear_length=float(dynamics.l_r),
        maximum_speed=float(scenario.max_speed),
        maximum_steering_angle=float(scenario.max_steering_angle),
    )


def _minimum_vehicle_clearance(scenario: ScenarioRoadTraffic) -> Tensor:
    entity_count = scenario.n_agents
    upper = torch.triu(
        torch.ones(
            entity_count,
            entity_count,
            dtype=torch.bool,
            device=scenario.world.device,
        ),
        diagonal=1,
    )
    return scenario.distances.agents[:, upper].min(dim=-1).values


def run_road_case(
    config: A3RoadExperimentConfig,
    case: RoadCaseConfig,
    planner: str,
    *,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    capture_video: bool = False,
    render_live: bool = False,
) -> Tuple[RoadBenchmarkMetrics, Dict[str, Any]]:
    """Run one deterministic controller in the actual road-traffic scenario."""

    simulation = config.simulation
    episodes = simulation.episodes if episodes_override is None else episodes_override
    max_steps = (
        simulation.max_steps if max_steps_override is None else max_steps_override
    )
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive.")
    if render_live and episodes != 1:
        raise ValueError("Live rendering requires exactly one environment.")
    if planner not in config.planners:
        raise ValueError(f"Planner {planner!r} is not enabled by the A3 config.")

    environment, scenario = _build_road_environment(
        config,
        case,
        episodes,
        max_steps,
        real_time_rendering=render_live,
    )
    if render_live:
        scenario.parameters.render_title = f"A3 {planner} | {case.name}"
    device = scenario.world.device
    controlled_mask = torch.ones(case.n_agents, dtype=torch.bool, device=device)
    maximum_speeds = torch.full(
        (case.n_agents,), float(scenario.max_speed), device=device
    )
    circumscribed_radius = 0.5 * math.hypot(
        float(AGENTS["length"]), float(AGENTS["width"])
    )
    security_radii = torch.full(
        (case.n_agents,),
        circumscribed_radius * config.vehicle.avoidance_radius_scale,
        device=device,
    )
    adapter = _adapter_parameters(scenario)
    adapter = BicycleAdapterParameters(
        front_length=adapter.front_length,
        rear_length=adapter.rear_length,
        maximum_speed=adapter.maximum_speed,
        maximum_steering_angle=adapter.maximum_steering_angle,
        minimum_speed_ratio=config.vehicle.minimum_speed_ratio,
    )
    controller = None
    if planner == "avocado_kb":
        controller = AVOCADOController(
            config.parameters,
            batch_size=episodes,
            entity_count=case.n_agents,
            controlled_mask=controlled_mask,
            security_radii=security_radii,
            maximum_speeds=maximum_speeds,
            seed=simulation.seed,
            device=device,
            complementary_responsibility=(
                config.safety.complementary_responsibility
            ),
        )

    positions, velocities, yaws = _road_state(scenario)
    initial_reference_paths = scenario.ref_paths_agent_related.long_term[
        0
    ].detach().cpu().clone()
    initial_reference_counts = scenario.ref_paths_agent_related.n_points_long_term[
        0
    ].detach().cpu().clone()
    trajectory_positions: List[Tensor] = [positions[0].detach().cpu().clone()]
    trajectory_reset_masks: List[Tensor] = []
    video_positions: List[Tensor] = []
    video_yaws: List[Tensor] = []
    video_reference_paths: List[Tensor] = []
    collision_ever = torch.zeros(episodes, dtype=torch.bool, device=device)
    agent_collision_events = 0
    lane_collision_events = 0
    wrong_entry_events = 0
    route_completion_events = 0
    reward_sum = 0.0
    reference_distance_sum = 0.0
    reference_distance_squared_sum = 0.0
    reference_distance_samples = 0
    reference_distance_values: List[Tensor] = []
    minimum_lane_clearance = math.inf
    minimum_clearance = torch.full((episodes,), torch.inf, device=device)
    tracking_error_sum = 0.0
    tracking_error_samples = 0
    steering_saturated_count = 0
    action_count = 0
    controller_seconds = 0.0
    controller_calls = 0
    infeasible_count = 0
    projection_count = 0
    maximum_attention = 0.0
    opinion_sum = 0.0
    opinion_samples = 0
    nonfinite_action_count = 0
    shield_intervention_count = 0
    post_shield_unsafe_pair_count = 0

    if capture_video:
        video_positions.append(positions[0].detach().cpu().clone())
        video_yaws.append(yaws[0].detach().cpu().clone())
        video_reference_paths.append(
            scenario.ref_paths_agent_related.short_term[0]
            .detach()
            .cpu()
            .clone()
        )

    for step in range(max_steps):
        reference_paths = scenario.ref_paths_agent_related.short_term
        heading_directions = torch.cat(
            (torch.cos(yaws), torch.sin(yaws)), dim=-1
        )
        preferred_velocity = stanley_path_preferred_velocity(
            positions,
            reference_paths,
            config.vehicle.cruise_speed,
            cross_track_gain=config.vehicle.path_tracking_gain,
            softening_speed=config.vehicle.path_tracking_softening_speed,
            maximum_correction_angle=math.radians(
                config.vehicle.maximum_path_correction_degrees
            ),
            terminal_fallback_directions=heading_directions,
        )
        path_half_plane_normals, path_half_plane_offsets = (
            path_velocity_cone_constraints(
                preferred_velocity,
                math.radians(config.vehicle.maximum_path_deviation_degrees),
            )
        )
        start = time.perf_counter()
        if planner == "path_following":
            desired_velocity = preferred_velocity
        elif planner == "orca_kb":
            desired_velocity = fixed_orca_actions(
                positions,
                velocities,
                preferred_velocity,
                controlled_mask=controlled_mask,
                security_radii=security_radii,
                maximum_speeds=maximum_speeds,
                perception_radius=config.parameters.perception_radius,
                time_horizon=config.parameters.velocity_obstacle_horizon,
                additional_half_plane_normals=path_half_plane_normals,
                additional_half_plane_offsets=path_half_plane_offsets,
            )
        elif planner == "avocado_kb":
            assert controller is not None
            desired_velocity = controller.step(
                positions,
                velocities,
                preferred_velocity,
                additional_half_plane_normals=path_half_plane_normals,
                additional_half_plane_offsets=path_half_plane_offsets,
            )
            infeasible_count += int(controller.last_infeasible.sum())
            projection_count += episodes * case.n_agents
            maximum_attention = max(
                maximum_attention, float(controller.attention.max())
            )
            active_opinions = controller.opinion[controller.last_neighbor_mask]
            if active_opinions.numel():
                opinion_sum += float(active_opinions.abs().sum())
                opinion_samples += int(active_opinions.numel())
        else:
            raise ValueError(f"Unknown A3 planner: {planner}.")
        bicycle_result = vector_velocity_to_bicycle_action(
            desired_velocity,
            yaws,
            adapter,
        )
        actions = bicycle_result.action
        shield_result = None
        if (
            planner == "avocado_kb"
            and config.safety.ttc_braking_shield_enabled
        ):
            assert controller is not None
            shield_result = apply_ttc_braking_shield(
                positions,
                actions,
                yaws,
                security_radii,
                adapter,
                minimum_ttc_seconds=config.safety.minimum_ttc_seconds,
                responsibility=controller.last_responsibility,
            )
            actions = shield_result.action
            shield_intervention_count += int(
                shield_result.intervention_mask.sum()
            )
            post_shield_unsafe_pair_count += int(
                shield_result.unsafe_pair_count_after.sum()
            )
        controller_seconds += time.perf_counter() - start
        controller_calls += episodes * case.n_agents
        steering_saturated_count += int(bicycle_result.steering_saturated.sum())
        action_count += actions.shape[0] * actions.shape[1]
        nonfinite = ~torch.isfinite(actions).all(dim=-1)
        nonfinite_action_count += int(nonfinite.sum())
        if bool(nonfinite.any()):
            actions = torch.nan_to_num(actions)

        _, rewards, _, _ = environment.step(
            [actions[:, index] for index in range(case.n_agents)]
        )
        if render_live:
            _set_live_diagnostics(
                scenario, planner, actions, controller, shield_result
            )
            _render_road_environment(environment)
        next_positions, next_velocities, next_yaws = _road_state(scenario)
        collision_agents = scenario.a3_last_agent_collisions
        lane_collisions = scenario.a3_last_lane_collisions
        wrong_entries = scenario.a3_last_wrong_entries
        route_completions = scenario.a3_last_route_completions
        reference_distances = scenario.a3_last_reference_distances
        lane_clearances = scenario.a3_last_lane_clearances
        reset_mask = (
            collision_agents | lane_collisions | wrong_entries | route_completions
        )

        collision_ever |= collision_agents.any(dim=-1)
        agent_collision_events += int(collision_agents.sum())
        lane_collision_events += int(lane_collisions.sum())
        wrong_entry_events += int(wrong_entries.sum())
        route_completion_events += int(route_completions.sum())
        reward_sum += float(torch.stack(rewards, dim=1).sum())
        reference_distance_sum += float(reference_distances.sum())
        reference_distance_squared_sum += float(
            reference_distances.square().sum()
        )
        reference_distance_samples += episodes * case.n_agents
        reference_distance_values.append(
            reference_distances.detach().cpu().reshape(-1)
        )
        minimum_lane_clearance = min(
            minimum_lane_clearance,
            float(lane_clearances.min()),
        )
        minimum_clearance = torch.minimum(
            minimum_clearance,
            _minimum_vehicle_clearance(scenario),
        )
        valid_tracking = ~reset_mask
        tracking_errors = torch.linalg.vector_norm(
            next_velocities - desired_velocity, dim=-1
        )
        tracking_error_sum += float(tracking_errors[valid_tracking].sum())
        tracking_error_samples += int(valid_tracking.sum())

        if controller is not None and bool(reset_mask.any()):
            controller.reset_agents(reset_mask)
        plotted_position = next_positions[0].detach().cpu().clone()
        plotted_position[reset_mask[0].detach().cpu()] = torch.nan
        trajectory_positions.append(plotted_position)
        trajectory_reset_masks.append(reset_mask[0].detach().cpu().clone())
        if capture_video and (step + 1) % simulation.video_stride == 0:
            video_positions.append(next_positions[0].detach().cpu().clone())
            video_yaws.append(next_yaws[0].detach().cpu().clone())
            video_reference_paths.append(
                scenario.ref_paths_agent_related.short_term[0]
                .detach()
                .cpu()
                .clone()
            )
        positions, velocities, yaws = next_positions, next_velocities, next_yaws

    agent_steps = episodes * case.n_agents * max_steps
    scale = 1000.0 / agent_steps
    all_reference_distances = torch.cat(reference_distance_values)
    metrics = RoadBenchmarkMetrics(
        case=case.name,
        planner=planner,
        episodes=episodes,
        executed_steps=max_steps,
        collision_free_episode_rate=float((~collision_ever).float().mean()),
        agent_collision_events_per_1000_steps=agent_collision_events * scale,
        lane_collision_events_per_1000_steps=lane_collision_events * scale,
        wrong_entry_events_per_1000_steps=wrong_entry_events * scale,
        route_completion_events_per_1000_steps=route_completion_events * scale,
        mean_reward_per_agent_step=reward_sum / agent_steps,
        mean_reference_distance_meters=(
            reference_distance_sum / max(reference_distance_samples, 1)
        ),
        rms_reference_distance_meters=math.sqrt(
            reference_distance_squared_sum / max(reference_distance_samples, 1)
        ),
        p95_reference_distance_meters=float(
            torch.quantile(all_reference_distances, 0.95)
        ),
        maximum_reference_distance_meters=float(
            all_reference_distances.max()
        ),
        minimum_lane_clearance_meters=minimum_lane_clearance,
        minimum_vehicle_clearance_meters=float(minimum_clearance.min()),
        mean_tracking_error_mps=(
            tracking_error_sum / max(tracking_error_samples, 1)
        ),
        steering_saturation_rate=(
            steering_saturated_count / max(action_count, 1)
        ),
        mean_controller_time_microseconds=(
            controller_seconds * 1e6 / max(controller_calls, 1)
        ),
        infeasible_projection_rate=(
            infeasible_count / max(projection_count, 1)
        ),
        maximum_attention=maximum_attention,
        mean_absolute_opinion=opinion_sum / max(opinion_samples, 1),
        shield_interventions_per_1000_steps=(
            shield_intervention_count * scale
        ),
        shield_intervention_rate=(
            shield_intervention_count / max(action_count, 1)
        ),
        post_shield_unsafe_pair_events_per_1000_steps=(
            post_shield_unsafe_pair_count * 1000.0
            / max(episodes * max_steps, 1)
        ),
        nonfinite_action_count=nonfinite_action_count,
    )
    evidence: Dict[str, Any] = {
        "positions": torch.stack(trajectory_positions),
        "reset_masks": torch.stack(trajectory_reset_masks),
        "initial_reference_paths": initial_reference_paths,
        "initial_reference_counts": initial_reference_counts,
        "video_positions": (
            torch.stack(video_positions) if video_positions else None
        ),
        "video_yaws": torch.stack(video_yaws) if video_yaws else None,
        "video_reference_paths": (
            torch.stack(video_reference_paths)
            if video_reference_paths
            else None
        ),
    }
    close = getattr(environment, "close", None)
    if callable(close):
        close()
    return metrics, evidence


def validate_road_metrics(
    config: A3RoadExperimentConfig,
    metrics: Tuple[RoadBenchmarkMetrics, ...],
) -> RoadValidationResult:
    """Evaluate A3 safety, tracking, activity, and comparison gates."""

    lookup = {(item.case, item.planner): item for item in metrics}
    checks: List[str] = []
    failures: List[str] = []
    validation = config.validation

    def check(condition: bool, success: str, failure: str) -> None:
        (checks if condition else failures).append(success if condition else failure)

    for case in config.cases:
        avocado = lookup[(case.name, "avocado_kb")]
        baseline = lookup[(case.name, "path_following")]
        comparisons = (
            (
                avocado.agent_collision_events_per_1000_steps
                <= validation.maximum_agent_collision_events_per_1000_steps,
                "agent collision event rate",
                avocado.agent_collision_events_per_1000_steps,
                validation.maximum_agent_collision_events_per_1000_steps,
            ),
            (
                avocado.lane_collision_events_per_1000_steps
                <= validation.maximum_lane_collision_events_per_1000_steps,
                "lane collision event rate",
                avocado.lane_collision_events_per_1000_steps,
                validation.maximum_lane_collision_events_per_1000_steps,
            ),
            (
                avocado.mean_tracking_error_mps
                <= validation.maximum_mean_tracking_error_mps,
                "mean tracking error",
                avocado.mean_tracking_error_mps,
                validation.maximum_mean_tracking_error_mps,
            ),
            (
                avocado.p95_reference_distance_meters
                <= validation.maximum_p95_reference_distance_meters,
                "p95 reference distance",
                avocado.p95_reference_distance_meters,
                validation.maximum_p95_reference_distance_meters,
            ),
            (
                avocado.steering_saturation_rate
                <= validation.maximum_steering_saturation_rate,
                "steering saturation rate",
                avocado.steering_saturation_rate,
                validation.maximum_steering_saturation_rate,
            ),
            (
                avocado.route_completion_events_per_1000_steps
                >= validation.minimum_route_completion_events_per_1000_steps,
                "route completion event rate",
                avocado.route_completion_events_per_1000_steps,
                validation.minimum_route_completion_events_per_1000_steps,
            ),
            (
                avocado.maximum_attention
                >= validation.minimum_maximum_attention,
                "maximum attention",
                avocado.maximum_attention,
                validation.minimum_maximum_attention,
            ),
            (
                avocado.shield_intervention_rate
                <= validation.maximum_shield_intervention_rate,
                "TTC shield intervention rate",
                avocado.shield_intervention_rate,
                validation.maximum_shield_intervention_rate,
            ),
            (
                avocado.post_shield_unsafe_pair_events_per_1000_steps
                <= validation.maximum_post_shield_unsafe_pair_events_per_1000_steps,
                "post-shield unsafe-pair event rate",
                avocado.post_shield_unsafe_pair_events_per_1000_steps,
                validation.maximum_post_shield_unsafe_pair_events_per_1000_steps,
            ),
        )
        for condition, label, value, threshold in comparisons:
            relation = "passed threshold"
            check(
                condition,
                f"{case.name}: {label}={value:.4f} {relation}.",
                f"{case.name}: {label}={value:.4f} failed threshold "
                f"{threshold:.4f}.",
            )
        improvement = (
            baseline.agent_collision_events_per_1000_steps
            - avocado.agent_collision_events_per_1000_steps
        )
        check(
            improvement >= validation.minimum_agent_collision_improvement,
            f"{case.name}: agent-collision improvement={improvement:.4f} passed.",
            f"{case.name}: agent-collision improvement={improvement:.4f} is below "
            f"{validation.minimum_agent_collision_improvement:.4f}.",
        )
        check(
            avocado.nonfinite_action_count == 0,
            f"{case.name}: all AVOCADO-KB actions were finite.",
            f"{case.name}: nonfinite_action_count="
            f"{avocado.nonfinite_action_count}.",
        )
    return RoadValidationResult(not failures, tuple(checks), tuple(failures))


def _save_road_trajectory_plot(
    output_path: Path,
    case: RoadCaseConfig,
    trajectories: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planners = list(trajectories)
    figure, axes = plt.subplots(
        1,
        len(planners),
        figsize=(5 * len(planners), 4.5),
        squeeze=False,
    )
    colors = plt.get_cmap("tab10")
    scenario_spec = SCENARIOS[case.scenario_type]
    for axis, planner in zip(axes[0], planners):
        trajectory = trajectories[planner]
        positions = trajectory["positions"].numpy()
        references = trajectory["initial_reference_paths"].numpy()
        counts = trajectory["initial_reference_counts"].numpy()
        for index in range(case.n_agents):
            count = int(counts[index])
            axis.plot(
                references[index, :count, 0],
                references[index, :count, 1],
                color="0.82",
                linewidth=0.8,
                zorder=1,
            )
            axis.plot(
                positions[:, index, 0],
                positions[:, index, 1],
                color=colors(index % 10),
                linewidth=1.2,
                zorder=2,
            )
            axis.scatter(
                positions[0, index, 0],
                positions[0, index, 1],
                color=colors(index % 10),
                s=18,
                zorder=3,
            )
        axis.set_title(planner)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(scenario_spec["x_dim_min"], scenario_spec["x_dim_max"])
        axis.set_ylim(scenario_spec["y_dim_min"], scenario_spec["y_dim_max"])
        axis.grid(True, alpha=0.2)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    figure.suptitle(f"A3 AVOCADO-KB in road_traffic: {case.name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_road_trajectory_video(
    output_stem: Path,
    case: RoadCaseConfig,
    trajectory: Mapping[str, Any],
    fps: int,
) -> Path:
    """Render a headless MP4, or a GIF when FFmpeg is unavailable."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    positions = trajectory["video_positions"]
    yaws = trajectory["video_yaws"]
    references = trajectory["video_reference_paths"]
    if positions is None or yaws is None or references is None:
        raise ValueError("Video trajectory state was not captured.")
    positions = positions.numpy()
    yaws = yaws.squeeze(-1).numpy()
    references = references.numpy()
    scenario_spec = SCENARIOS[case.scenario_type]
    colors = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(scenario_spec["x_dim_min"], scenario_spec["x_dim_max"])
    axis.set_ylim(scenario_spec["y_dim_min"], scenario_spec["y_dim_max"])
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(True, alpha=0.2)
    axis.set_title(f"A3 AVOCADO-KB: {case.name}")
    reference_lines = []
    heading_lines = []
    for index in range(case.n_agents):
        color = colors(index % 10)
        reference_line, = axis.plot([], [], color=color, alpha=0.35, linewidth=1.0)
        heading_line, = axis.plot([], [], color=color, linewidth=2.0)
        reference_lines.append(reference_line)
        heading_lines.append(heading_line)
    scatter = axis.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        c=[colors(index % 10) for index in range(case.n_agents)],
        s=38,
        zorder=3,
    )
    time_text = axis.text(0.02, 0.97, "", transform=axis.transAxes, va="top")

    def update(frame_index: int):
        current_positions = positions[frame_index]
        scatter.set_offsets(current_positions)
        for index in range(case.n_agents):
            reference_lines[index].set_data(
                references[frame_index, index, :, 0],
                references[frame_index, index, :, 1],
            )
            heading = yaws[frame_index, index]
            length = float(AGENTS["length"])
            heading_lines[index].set_data(
                [
                    current_positions[index, 0],
                    current_positions[index, 0] + length * math.cos(heading),
                ],
                [
                    current_positions[index, 1],
                    current_positions[index, 1] + length * math.sin(heading),
                ],
            )
        time_text.set_text(f"frame {frame_index + 1}/{positions.shape[0]}")
        return [scatter, time_text, *reference_lines, *heading_lines]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=positions.shape[0],
        interval=1000 / fps,
        blit=True,
    )
    if animation.writers.is_available("ffmpeg"):
        output_path = output_stem.with_suffix(".mp4")
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    else:
        output_path = output_stem.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    movie.save(output_path, writer=writer, dpi=140)
    plt.close(figure)
    return output_path


def run_live_road_case(
    config: A3RoadExperimentConfig,
    *,
    case_name: Optional[str] = None,
    scenario_type: Optional[str] = None,
    planner: str = "avocado_kb",
    max_steps_override: Optional[int] = None,
) -> RoadBenchmarkMetrics:
    """Run one A3 controller in the native VMAS real-time viewer."""

    if case_name is not None and scenario_type is not None:
        raise ValueError(
            "Select either a configured case or a scenario type, not both."
        )
    if scenario_type is not None:
        if scenario_type not in SCENARIOS:
            available = ", ".join(SCENARIOS)
            raise ValueError(
                f"Unknown scenario type {scenario_type!r}; available scenarios: "
                f"{available}."
            )
        n_agents = int(SCENARIOS[scenario_type]["n_agents"])
        selected_case = RoadCaseConfig(
            name=f"{scenario_type}_{n_agents}",
            scenario_type=scenario_type,
            n_agents=n_agents,
        )
    else:
        selected_name = config.cases[0].name if case_name is None else case_name
        matching_cases = [
            case for case in config.cases if case.name == selected_name
        ]
        if not matching_cases:
            available = ", ".join(case.name for case in config.cases)
            raise ValueError(
                f"Unknown A3 road case {selected_name!r}; available cases: "
                f"{available}."
            )
        selected_case = matching_cases[0]
    metrics, _ = run_road_case(
        config,
        selected_case,
        planner,
        episodes_override=1,
        max_steps_override=max_steps_override,
        render_live=True,
    )
    return metrics


def run_road_benchmark(
    config: A3RoadExperimentConfig,
    *,
    output_directory: Optional[Path] = None,
    save_plots: bool = True,
    save_videos: bool = False,
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> RoadBenchmarkRun:
    """Run all A3 comparisons and persist metrics and visual evidence."""

    if output_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path(config.output_root) / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    all_metrics: List[RoadBenchmarkMetrics] = []
    for case in config.cases:
        case_trajectories: Dict[str, Mapping[str, Any]] = {}
        for planner in config.planners:
            metrics, evidence = run_road_case(
                config,
                case,
                planner,
                episodes_override=episodes_override,
                max_steps_override=max_steps_override,
                capture_video=save_videos and planner == "avocado_kb",
            )
            all_metrics.append(metrics)
            case_trajectories[planner] = evidence
            if evidence["video_positions"] is not None:
                video_stem = output_directory / f"video_{case.name}_{planner}"
                fps = max(
                    1,
                    round(
                        1.0
                        / (
                            config.parameters.dt
                            * config.simulation.video_stride
                        )
                    ),
                )
                _save_road_trajectory_video(
                    video_stem,
                    case,
                    evidence,
                    fps,
                )
        if save_plots:
            _save_road_trajectory_plot(
                output_directory / f"trajectories_{case.name}.png",
                case,
                case_trajectories,
            )

    metrics_tuple = tuple(all_metrics)
    validation = validate_road_metrics(config, metrics_tuple)
    summary = {
        "config": config.to_dict(),
        "metrics": [asdict(item) for item in metrics_tuple],
        "validation": asdict(validation),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return RoadBenchmarkRun(output_directory, metrics_tuple, validation)
