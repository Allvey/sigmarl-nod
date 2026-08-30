"""Auditable action bridge from a frozen Base-MAPPO actor to AVOCADO-KB."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
from torchrl.envs.utils import ExplorationType, set_exploration_type

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.avocado.bicycle import (
    BicycleAdapterParameters,
    continuity_regularized_velocity_target,
    path_velocity_cone_constraints,
    stanley_path_preferred_velocity,
    vector_velocity_to_bicycle_action,
)
from utilities.avocado.controller import AVOCADOController
from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.avocado.road_safety import (
    TTCSafetyShieldResult,
    apply_ttc_braking_shield,
    bicycle_action_velocity,
)
from utilities.constants import AGENTS


@dataclass(frozen=True)
class A4StepDiagnostics:
    nominal_action: Tensor
    pre_shield_action: Tensor
    executed_action: Tensor
    nominal_velocity: Tensor
    safe_velocity: Tensor
    conflict_mask: Tensor
    intervention_mask: Tensor
    shield_result: Optional[TTCSafetyShieldResult]
    heuristic_estimate: Optional[Tensor]
    estimate_correction: Optional[Tensor]
    fused_estimate: Optional[Tensor]
    opinion: Optional[Tensor]


@dataclass(frozen=True)
class A4BridgeTrace:
    nominal_action: Tensor
    pre_shield_action: Tensor
    executed_action: Tensor
    conflict_mask: Tensor
    intervention_mask: Tensor
    shield_mask: Tensor
    reset_mask: Tensor
    heuristic_estimate: Tensor
    estimate_correction: Tensor
    fused_estimate: Tensor
    opinion: Tensor
    attention: Tensor
    pair_mask: Tensor


@dataclass(frozen=True)
class A4BridgeMetrics:
    action_samples: int
    conflict_agent_rate: float
    action_intervention_rate: float
    no_conflict_passthrough_rate: float
    conflict_intervention_rate: float
    shield_intervention_rate: float
    mean_nominal_speed_mps: float
    mean_executed_speed_mps: float
    mean_absolute_speed_change_mps: float
    mean_absolute_steering_change_degrees: float
    nominal_executed_speed_correlation: float
    nominal_executed_steering_correlation: float
    maximum_attention: float
    infeasible_projection_rate: float
    nonfinite_action_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _correlation(first: Tensor, second: Tensor) -> float:
    if first.numel() < 2:
        return 1.0
    first = first.float()
    second = second.float()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = torch.sqrt(
        first_centered.square().sum() * second_centered.square().sum()
    )
    if float(denominator) <= 1e-12:
        return 1.0 if torch.allclose(first, second, atol=1e-6) else 0.0
    return float((first_centered * second_centered).sum() / denominator)


class A4ActionBridge:
    """Run a frozen actor, then optionally project its action with A3 safety.

    In A4, ``opinion_bias`` remains fixed and no learned correction of the
    AVOCADO cooperation estimate is present.  The class therefore isolates
    action-level coupling from the later A5/A6 learning experiments.
    """

    action_key = ("agents", "action")
    stage_label = "A4 MARL nominal + fixed AVOCADO"

    def __init__(
        self,
        base_policy: torch.nn.Module,
        scenario: ScenarioRoadTraffic,
        a3_config: A3RoadExperimentConfig,
        *,
        use_avocado: bool,
        deterministic: bool,
        velocity_continuity_weight: float,
        speed_intervention_tolerance_mps: float,
        steering_intervention_tolerance_degrees: float,
    ) -> None:
        self.base_policy = base_policy
        self.scenario = scenario
        self.config = a3_config
        self.use_avocado = bool(use_avocado)
        self.deterministic = bool(deterministic)
        self.velocity_continuity_weight = float(velocity_continuity_weight)
        self.speed_tolerance = float(speed_intervention_tolerance_mps)
        self.steering_tolerance = math.radians(
            float(steering_intervention_tolerance_degrees)
        )

        self.base_policy.eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

        device = scenario.world.device
        batch_size = scenario.world.batch_dim
        entity_count = scenario.n_agents
        controlled_mask = torch.ones(
            entity_count, dtype=torch.bool, device=device
        )
        maximum_speeds = torch.full(
            (entity_count,), float(scenario.max_speed), device=device
        )
        circumscribed_radius = 0.5 * math.hypot(
            float(AGENTS["length"]), float(AGENTS["width"])
        )
        self.security_radii = torch.full(
            (entity_count,),
            circumscribed_radius * a3_config.vehicle.avoidance_radius_scale,
            device=device,
        )
        dynamics = scenario.world.agents[0].dynamics
        self.adapter = BicycleAdapterParameters(
            front_length=float(dynamics.l_f),
            rear_length=float(dynamics.l_r),
            maximum_speed=float(scenario.max_speed),
            maximum_steering_angle=float(scenario.max_steering_angle),
            minimum_speed_ratio=a3_config.vehicle.minimum_speed_ratio,
        )
        self.controller = (
            AVOCADOController(
                a3_config.parameters,
                batch_size=batch_size,
                entity_count=entity_count,
                controlled_mask=controlled_mask,
                security_radii=self.security_radii,
                maximum_speeds=maximum_speeds,
                seed=a3_config.simulation.seed,
                device=device,
                complementary_responsibility=(
                    a3_config.safety.complementary_responsibility
                ),
            )
            if self.use_avocado
            else None
        )
        self.last: Optional[A4StepDiagnostics] = None
        self._nominal_actions = []
        self._pre_shield_actions = []
        self._executed_actions = []
        self._conflict_masks = []
        self._intervention_masks = []
        self._shield_masks = []
        self._reset_masks = []
        self._heuristic_estimates = []
        self._estimate_corrections = []
        self._fused_estimates = []
        self._opinions = []
        self._attentions = []
        self._pair_masks = []
        self._infeasible_count = 0
        self._projection_count = 0
        self._maximum_attention = 0.0
        self._nonfinite_count = 0

    def _road_state(self) -> tuple[Tensor, Tensor, Tensor]:
        positions = torch.stack(
            [agent.state.pos for agent in self.scenario.world.agents], dim=1
        )
        velocities = torch.stack(
            [agent.state.vel for agent in self.scenario.world.agents], dim=1
        )
        yaws = torch.stack(
            [agent.state.rot for agent in self.scenario.world.agents], dim=1
        )
        return positions, velocities, yaws

    def opinion_estimate_correction(
        self,
        positions: Tensor,
        velocities: Tensor,
        yaws: Tensor,
    ) -> Optional[Tensor]:
        """Optional A5+ hook; A4 preserves the heuristic estimate exactly."""

        return None

    @torch.no_grad()
    def __call__(self, tensordict):
        exploration = (
            ExplorationType.MODE
            if self.deterministic
            else ExplorationType.RANDOM
        )
        with set_exploration_type(exploration):
            tensordict = self.base_policy(tensordict)
        nominal_action = tensordict.get(self.action_key).detach().clone()
        if nominal_action.shape[-1] != 2:
            raise ValueError("Base-MAPPO action must be [speed, steering].")

        positions, velocities, yaws = self._road_state()
        nominal_velocity = bicycle_action_velocity(
            nominal_action, yaws, self.adapter
        )
        pre_shield_action = nominal_action
        safe_velocity = nominal_velocity
        conflict_mask = torch.zeros(
            nominal_action.shape[:-1],
            dtype=torch.bool,
            device=nominal_action.device,
        )
        shield_result = None

        if self.controller is not None:
            reference_paths = self.scenario.ref_paths_agent_related.short_term
            heading_directions = torch.cat(
                (torch.cos(yaws), torch.sin(yaws)), dim=-1
            )
            path_velocity = stanley_path_preferred_velocity(
                positions,
                reference_paths,
                self.config.vehicle.cruise_speed,
                cross_track_gain=self.config.vehicle.path_tracking_gain,
                softening_speed=(
                    self.config.vehicle.path_tracking_softening_speed
                ),
                maximum_correction_angle=math.radians(
                    self.config.vehicle.maximum_path_correction_degrees
                ),
                terminal_fallback_directions=heading_directions,
            )
            path_normals, path_offsets = path_velocity_cone_constraints(
                path_velocity,
                math.radians(
                    self.config.vehicle.maximum_path_deviation_degrees
                ),
            )
            optimization_velocity = continuity_regularized_velocity_target(
                nominal_velocity,
                velocities,
                self.velocity_continuity_weight,
            )
            estimate_correction = self.opinion_estimate_correction(
                positions, velocities, yaws
            )
            safe_velocity = self.controller.step(
                positions,
                velocities,
                optimization_velocity,
                additional_half_plane_normals=path_normals,
                additional_half_plane_offsets=path_offsets,
                estimated_opinion_correction=estimate_correction,
            )
            bicycle_result = vector_velocity_to_bicycle_action(
                safe_velocity, yaws, self.adapter
            )
            pre_shield_action = bicycle_result.action
            conflict_mask = self.controller.last_active_vo_count > 0
            self._infeasible_count += int(self.controller.last_infeasible.sum())
            self._projection_count += nominal_action.shape[0] * nominal_action.shape[1]
            self._maximum_attention = max(
                self._maximum_attention,
                float(self.controller.attention.max()),
            )
            executed_action = pre_shield_action
            if self.config.safety.ttc_braking_shield_enabled:
                shield_result = apply_ttc_braking_shield(
                    positions,
                    executed_action,
                    yaws,
                    self.security_radii,
                    self.adapter,
                    minimum_ttc_seconds=(
                        self.config.safety.minimum_ttc_seconds
                    ),
                    responsibility=self.controller.last_responsibility,
                )
                executed_action = shield_result.action
        else:
            executed_action = nominal_action

        nonfinite = ~torch.isfinite(executed_action).all(dim=-1)
        self._nonfinite_count += int(nonfinite.sum())
        if bool(nonfinite.any()):
            executed_action = torch.nan_to_num(executed_action)
        speed_change = (executed_action[..., 0] - nominal_action[..., 0]).abs()
        steering_change = (
            executed_action[..., 1] - nominal_action[..., 1]
        ).abs()
        intervention_mask = (speed_change > self.speed_tolerance) | (
            steering_change > self.steering_tolerance
        )
        shield_mask = (
            shield_result.intervention_mask
            if shield_result is not None
            else torch.zeros_like(intervention_mask)
        )

        tensordict.set(self.action_key, executed_action)
        self.last = A4StepDiagnostics(
            nominal_action=nominal_action,
            pre_shield_action=pre_shield_action.detach().clone(),
            executed_action=executed_action.detach().clone(),
            nominal_velocity=nominal_velocity.detach().clone(),
            safe_velocity=safe_velocity.detach().clone(),
            conflict_mask=conflict_mask.detach().clone(),
            intervention_mask=intervention_mask.detach().clone(),
            shield_result=shield_result,
            heuristic_estimate=(
                self.controller.last_estimated_opinion.detach().clone()
                if self.controller is not None
                else None
            ),
            estimate_correction=(
                self.controller.last_estimate_correction.detach().clone()
                if self.controller is not None
                else None
            ),
            fused_estimate=(
                self.controller.last_fused_estimated_opinion.detach().clone()
                if self.controller is not None
                else None
            ),
            opinion=(
                self.controller.opinion.detach().clone()
                if self.controller is not None
                else None
            ),
        )
        self._nominal_actions.append(nominal_action.cpu())
        self._pre_shield_actions.append(pre_shield_action.detach().cpu())
        self._executed_actions.append(executed_action.detach().cpu())
        self._conflict_masks.append(conflict_mask.cpu())
        self._intervention_masks.append(intervention_mask.cpu())
        self._shield_masks.append(shield_mask.cpu())
        if self.controller is not None:
            self._heuristic_estimates.append(
                self.controller.last_estimated_opinion.detach().cpu().clone()
            )
            self._estimate_corrections.append(
                self.controller.last_estimate_correction.detach().cpu().clone()
            )
            self._fused_estimates.append(
                self.controller.last_fused_estimated_opinion.detach().cpu().clone()
            )
            self._opinions.append(self.controller.opinion.detach().cpu().clone())
            self._attentions.append(
                self.controller.attention.detach().cpu().clone()
            )
            self._pair_masks.append(
                self.controller.last_neighbor_mask.detach().cpu().clone()
            )
        return tensordict

    def reset_agents(self, reset_mask: Tensor) -> None:
        self._reset_masks.append(reset_mask.detach().cpu().clone())
        if self.controller is not None and bool(reset_mask.any()):
            self.controller.reset_agents(reset_mask)

    def metrics(self) -> A4BridgeMetrics:
        if not self._nominal_actions:
            raise RuntimeError("No A4 actions have been evaluated.")
        nominal = torch.cat([item.reshape(-1, 2) for item in self._nominal_actions])
        executed = torch.cat([item.reshape(-1, 2) for item in self._executed_actions])
        conflict = torch.cat([item.reshape(-1) for item in self._conflict_masks])
        intervention = torch.cat(
            [item.reshape(-1) for item in self._intervention_masks]
        )
        shield = torch.cat([item.reshape(-1) for item in self._shield_masks])
        no_conflict = ~conflict
        no_conflict_passthrough = (
            float((~intervention[no_conflict]).float().mean())
            if bool(no_conflict.any())
            else 1.0
        )
        conflict_intervention = (
            float(intervention[conflict].float().mean())
            if bool(conflict.any())
            else 0.0
        )
        return A4BridgeMetrics(
            action_samples=int(nominal.shape[0]),
            conflict_agent_rate=float(conflict.float().mean()),
            action_intervention_rate=float(intervention.float().mean()),
            no_conflict_passthrough_rate=no_conflict_passthrough,
            conflict_intervention_rate=conflict_intervention,
            shield_intervention_rate=float(shield.float().mean()),
            mean_nominal_speed_mps=float(nominal[:, 0].abs().mean()),
            mean_executed_speed_mps=float(executed[:, 0].abs().mean()),
            mean_absolute_speed_change_mps=float(
                (executed[:, 0] - nominal[:, 0]).abs().mean()
            ),
            mean_absolute_steering_change_degrees=float(
                torch.rad2deg((executed[:, 1] - nominal[:, 1]).abs()).mean()
            ),
            nominal_executed_speed_correlation=_correlation(
                nominal[:, 0], executed[:, 0]
            ),
            nominal_executed_steering_correlation=_correlation(
                nominal[:, 1], executed[:, 1]
            ),
            maximum_attention=self._maximum_attention,
            infeasible_projection_rate=(
                self._infeasible_count / max(self._projection_count, 1)
            ),
            nonfinite_action_count=self._nonfinite_count,
        )

    def trace(self) -> A4BridgeTrace:
        if not self._nominal_actions or not self._heuristic_estimates:
            raise RuntimeError("An AVOCADO bridge rollout is required for a trace.")
        return A4BridgeTrace(
            nominal_action=torch.stack(self._nominal_actions),
            pre_shield_action=torch.stack(self._pre_shield_actions),
            executed_action=torch.stack(self._executed_actions),
            conflict_mask=torch.stack(self._conflict_masks),
            intervention_mask=torch.stack(self._intervention_masks),
            shield_mask=torch.stack(self._shield_masks),
            reset_mask=torch.stack(self._reset_masks),
            heuristic_estimate=torch.stack(self._heuristic_estimates),
            estimate_correction=torch.stack(self._estimate_corrections),
            fused_estimate=torch.stack(self._fused_estimates),
            opinion=torch.stack(self._opinions),
            attention=torch.stack(self._attentions),
            pair_mask=torch.stack(self._pair_masks),
        )
