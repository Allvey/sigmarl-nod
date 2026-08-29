"""Stateful strict AVOCADO controller for A1-A2 holonomic experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from utilities.avocado.core import (
    AVOCADOParameters,
    HalfPlane,
    attention_reference_step,
    build_oca_half_plane,
    collision_time,
    finite_velocity_obstacle_correction,
    opinion_euler_step,
    opinion_to_cooperation,
    projection_estimator,
    solve_closest_admissible_velocity,
)


class AVOCADOController:
    """Algorithm 1 applied independently by every controlled entity."""

    def __init__(
        self,
        parameters: AVOCADOParameters,
        *,
        batch_size: int,
        entity_count: int,
        controlled_mask: Tensor,
        security_radii: Tensor,
        maximum_speeds: Tensor,
        seed: int,
        device: torch.device | str = "cpu",
        complementary_responsibility: bool = False,
    ) -> None:
        if batch_size <= 0 or entity_count <= 1:
            raise ValueError(
                "batch_size must be positive and entity_count must exceed one."
            )
        self.parameters = parameters
        self.batch_size = int(batch_size)
        self.entity_count = int(entity_count)
        self.device = torch.device(device)
        self.controlled_mask = controlled_mask.to(self.device, dtype=torch.bool)
        self.security_radii = security_radii.to(self.device, dtype=torch.float32)
        self.maximum_speeds = maximum_speeds.to(self.device, dtype=torch.float32)
        self.complementary_responsibility = bool(complementary_responsibility)
        if self.controlled_mask.shape != (entity_count,):
            raise ValueError("controlled_mask must have shape [entity_count].")
        if self.security_radii.shape != (entity_count,):
            raise ValueError("security_radii must have shape [entity_count].")
        if self.maximum_speeds.shape != (entity_count,):
            raise ValueError("maximum_speeds must have shape [entity_count].")

        matrix_shape = (batch_size, entity_count, entity_count)
        self.attention = torch.zeros(matrix_shape, device=self.device)
        self.opinion = torch.full(
            matrix_shape,
            parameters.initial_opinion,
            device=self.device,
        )
        self.previous_observed_velocity = torch.zeros(
            batch_size, entity_count, 2, device=self.device
        )
        self.previous_correction = torch.zeros(
            batch_size, entity_count, entity_count, 2, device=self.device
        )
        self.last_neighbor_mask = torch.zeros(
            matrix_shape, dtype=torch.bool, device=self.device
        )
        self.last_time_to_collision = torch.full(
            matrix_shape, torch.inf, device=self.device
        )
        self.last_estimated_opinion = torch.zeros(matrix_shape, device=self.device)
        self.last_estimate_correction = torch.zeros(
            matrix_shape, device=self.device
        )
        self.last_fused_estimated_opinion = torch.zeros(
            matrix_shape, device=self.device
        )
        initial_cooperation = (parameters.initial_opinion + 1.0) / 2.0
        self.last_cooperation = torch.full(
            matrix_shape,
            min(max(initial_cooperation, 0.0), 1.0),
            device=self.device,
        )
        self.last_infeasible = torch.zeros(
            batch_size, entity_count, dtype=torch.bool, device=self.device
        )
        self.last_active_vo_count = torch.zeros(
            batch_size, entity_count, dtype=torch.int64, device=self.device
        )
        self.last_constraint_count = torch.zeros(
            batch_size, entity_count, dtype=torch.int64, device=self.device
        )
        self.last_responsibility = torch.full(
            matrix_shape, 0.5, device=self.device
        )
        self.last_vo_active = torch.zeros(
            matrix_shape, dtype=torch.bool, device=self.device
        )
        self.last_vo_correction = torch.zeros(
            batch_size, entity_count, entity_count, 2, device=self.device
        )
        self.last_vo_normal = torch.zeros_like(self.last_vo_correction)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def reset(self) -> None:
        self.attention.zero_()
        self.opinion.fill_(self.parameters.initial_opinion)
        self.previous_observed_velocity.zero_()
        self.previous_correction.zero_()
        self.last_neighbor_mask.zero_()
        self.last_time_to_collision.fill_(torch.inf)
        self.last_estimated_opinion.zero_()
        self.last_estimate_correction.zero_()
        self.last_fused_estimated_opinion.zero_()
        initial_cooperation = (self.parameters.initial_opinion + 1.0) / 2.0
        self.last_cooperation.fill_(min(max(initial_cooperation, 0.0), 1.0))
        self.last_infeasible.zero_()
        self.last_active_vo_count.zero_()
        self.last_constraint_count.zero_()
        self.last_responsibility.fill_(0.5)
        self.last_vo_active.zero_()
        self.last_vo_correction.zero_()
        self.last_vo_normal.zero_()

    def reset_agents(self, reset_mask: Tensor) -> None:
        """Clear all incoming/outgoing pair state for reset road vehicles."""

        expected = (self.batch_size, self.entity_count)
        reset_mask = reset_mask.to(self.device, dtype=torch.bool)
        if reset_mask.shape != expected:
            raise ValueError(f"reset_mask must have shape {expected}.")
        pair_mask = reset_mask[:, :, None] | reset_mask[:, None, :]
        initial_opinion = self.parameters.initial_opinion
        initial_cooperation = min(max((initial_opinion + 1.0) / 2.0, 0.0), 1.0)
        self.attention[pair_mask] = 0.0
        self.opinion[pair_mask] = initial_opinion
        self.previous_correction[pair_mask] = 0.0
        self.last_neighbor_mask[pair_mask] = False
        self.last_time_to_collision[pair_mask] = torch.inf
        self.last_estimated_opinion[pair_mask] = 0.0
        self.last_estimate_correction[pair_mask] = 0.0
        self.last_fused_estimated_opinion[pair_mask] = 0.0
        self.last_cooperation[pair_mask] = initial_cooperation
        self.previous_observed_velocity[reset_mask] = 0.0
        self.last_infeasible[reset_mask] = False
        self.last_active_vo_count[reset_mask] = 0
        self.last_constraint_count[reset_mask] = 0
        self.last_responsibility[pair_mask] = 0.5
        self.last_vo_active[pair_mask] = False
        self.last_vo_correction[pair_mask] = 0.0
        self.last_vo_normal[pair_mask] = 0.0

    def step(
        self,
        positions: Tensor,
        measured_velocities: Tensor,
        preferred_velocities: Tensor,
        *,
        active_environment_mask: Optional[Tensor] = None,
        additional_half_plane_normals: Optional[Tensor] = None,
        additional_half_plane_offsets: Optional[Tensor] = None,
        estimated_opinion_correction: Optional[Tensor] = None,
    ) -> Tensor:
        """Update all pair states and return directly executable 2-D velocities."""

        expected = (self.batch_size, self.entity_count, 2)
        for value, name in (
            (positions, "positions"),
            (measured_velocities, "measured_velocities"),
            (preferred_velocities, "preferred_velocities"),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {value.shape}."
                )
        if active_environment_mask is None:
            active_environment_mask = torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            )
        else:
            active_environment_mask = active_environment_mask.to(
                self.device, dtype=torch.bool
            )
        if (additional_half_plane_normals is None) != (
            additional_half_plane_offsets is None
        ):
            raise ValueError(
                "additional half-plane normals and offsets must be supplied together."
            )
        if additional_half_plane_normals is not None:
            expected_prefix = (self.batch_size, self.entity_count)
            if (
                additional_half_plane_normals.ndim != 4
                or additional_half_plane_normals.shape[:2] != expected_prefix
                or additional_half_plane_normals.shape[-1] != 2
            ):
                raise ValueError(
                    "additional_half_plane_normals must have shape [batch, entity, K, 2]."
                )
            if additional_half_plane_offsets.shape != (
                *additional_half_plane_normals.shape[:-1],
            ):
                raise ValueError(
                    "additional_half_plane_offsets must have shape [batch, entity, K]."
                )
            additional_half_plane_normals = additional_half_plane_normals.to(
                self.device
            )
            additional_half_plane_offsets = additional_half_plane_offsets.to(
                self.device
            )
        if estimated_opinion_correction is not None:
            expected_correction = (
                self.batch_size,
                self.entity_count,
                self.entity_count,
            )
            if estimated_opinion_correction.shape != expected_correction:
                raise ValueError(
                    "estimated_opinion_correction must have shape "
                    f"{expected_correction}."
                )
            if not bool(torch.isfinite(estimated_opinion_correction).all()):
                raise ValueError("estimated_opinion_correction must be finite.")
            estimated_opinion_correction = estimated_opinion_correction.to(
                self.device
            )

        positions = positions.to(self.device)
        measured_velocities = measured_velocities.to(self.device)
        preferred_velocities = preferred_velocities.to(self.device)
        actions = preferred_velocities.clone()
        pair_distance = torch.cdist(positions, positions)
        identity = torch.eye(
            self.entity_count, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        neighbor_mask = (
            (pair_distance < self.parameters.perception_radius)
            & ~identity
            & active_environment_mask[:, None, None]
        )
        neighbor_mask &= self.controlled_mask[None, :, None]
        self.last_neighbor_mask.copy_(neighbor_mask)
        self.last_time_to_collision.fill_(torch.inf)
        self.last_estimated_opinion.zero_()
        self.last_estimate_correction.zero_()
        self.last_fused_estimated_opinion.zero_()
        self.last_infeasible.zero_()
        self.last_active_vo_count.zero_()
        self.last_constraint_count.zero_()
        self.last_responsibility.fill_(0.5)
        self.last_vo_active.zero_()
        self.last_vo_correction.zero_()
        self.last_vo_normal.zero_()

        random_noise = torch.rand(
            (
                self.batch_size,
                self.entity_count,
                self.entity_count,
                2,
            ),
            generator=self.generator,
            device="cpu",
        ).to(self.device)
        random_noise = (
            random_noise * 2.0 - 1.0
        ) * self.parameters.noise_sigma

        # First update every directed pair.  Keeping this separate from the
        # projection pass lets the road extension normalize both sides of a
        # pair from states measured at the same instant.
        for batch in range(self.batch_size):
            if not bool(active_environment_mask[batch]):
                actions[batch].zero_()
                continue
            for robot in range(self.entity_count):
                if not bool(self.controlled_mask[robot]):
                    continue
                for agent in range(self.entity_count):
                    if robot == agent:
                        continue
                    if not bool(neighbor_mask[batch, robot, agent]):
                        continue

                    combined_radius = (
                        self.security_radii[robot] + self.security_radii[agent]
                    )
                    time_to_collision = collision_time(
                        positions[batch, robot],
                        measured_velocities[batch, robot],
                        positions[batch, agent],
                        measured_velocities[batch, agent],
                        combined_radius,
                    )
                    self.last_time_to_collision[batch, robot, agent] = (
                        time_to_collision
                    )
                    new_attention = attention_reference_step(
                        self.attention[batch, robot, agent],
                        time_to_collision,
                        delta=self.parameters.attention_decay,
                        kappa=self.parameters.kappa,
                    )
                    self.attention[batch, robot, agent] = new_attention

                    delta_velocity = (
                        measured_velocities[batch, agent]
                        - self.previous_observed_velocity[batch, agent]
                    )
                    estimated = projection_estimator(
                        delta_velocity,
                        self.previous_correction[batch, robot, agent],
                        self.parameters.epsilon,
                    )
                    self.last_estimated_opinion[batch, robot, agent] = estimated
                    correction = (
                        estimated_opinion_correction[batch, robot, agent]
                        if estimated_opinion_correction is not None
                        else torch.zeros_like(estimated)
                    )
                    fused_estimated = (estimated + correction).clamp(-1.0, 1.0)
                    self.last_estimate_correction[
                        batch, robot, agent
                    ] = correction
                    self.last_fused_estimated_opinion[
                        batch, robot, agent
                    ] = fused_estimated
                    new_opinion = opinion_euler_step(
                        self.opinion[batch, robot, agent],
                        new_attention,
                        fused_estimated,
                        dt=self.parameters.dt,
                        decay=self.parameters.opinion_decay,
                        self_weight=self.parameters.opinion_self_weight,
                        estimate_weight=self.parameters.opinion_estimate_weight,
                        bias=self.parameters.opinion_bias,
                    )
                    self.opinion[batch, robot, agent] = new_opinion
                    cooperation = opinion_to_cooperation(new_opinion).clamp(
                        0.0, 1.0
                    )
                    self.last_cooperation[batch, robot, agent] = cooperation

                    perturbed_agent_velocity = measured_velocities[
                        batch, agent
                    ] + (1.0 - new_attention) * random_noise[
                        batch, robot, agent
                    ]
                    relative_position = (
                        positions[batch, agent] - positions[batch, robot]
                    )
                    relative_velocity = (
                        measured_velocities[batch, robot]
                        - perturbed_agent_velocity
                    )
                    vo = finite_velocity_obstacle_correction(
                        relative_position,
                        relative_velocity,
                        combined_radius,
                        self.parameters.velocity_obstacle_horizon,
                        collision_time_step=self.parameters.dt,
                    )
                    self.previous_correction[batch, robot, agent] = vo.correction
                    responsibility = 1.0 - cooperation
                    self.last_responsibility[batch, robot, agent] = responsibility
                    vo_active = vo.active | vo.already_colliding
                    self.last_vo_active[batch, robot, agent] = vo_active
                    self.last_vo_correction[batch, robot, agent] = vo.correction
                    self.last_vo_normal[batch, robot, agent] = vo.normal
                    if bool(vo_active):
                        self.last_active_vo_count[batch, robot] += 1

        if self.complementary_responsibility:
            for batch in range(self.batch_size):
                if not bool(active_environment_mask[batch]):
                    continue
                for robot in range(self.entity_count):
                    for agent in range(robot + 1, self.entity_count):
                        if not bool(
                            neighbor_mask[batch, robot, agent]
                            & neighbor_mask[batch, agent, robot]
                        ):
                            continue
                        robot_raw = self.last_responsibility[
                            batch, robot, agent
                        ]
                        agent_raw = self.last_responsibility[
                            batch, agent, robot
                        ]
                        total = robot_raw + agent_raw
                        robot_share = torch.where(
                            total > 1e-8,
                            robot_raw / total,
                            torch.full_like(total, 0.5),
                        )
                        self.last_responsibility[batch, robot, agent] = (
                            robot_share
                        )
                        self.last_responsibility[batch, agent, robot] = (
                            1.0 - robot_share
                        )
            self.last_cooperation[neighbor_mask] = (
                1.0 - self.last_responsibility[neighbor_mask]
            )

        # Then solve the joint VO, speed-circle, and optional road constraints.
        for batch in range(self.batch_size):
            if not bool(active_environment_mask[batch]):
                continue
            for robot in range(self.entity_count):
                if not bool(self.controlled_mask[robot]):
                    continue
                half_planes = []
                for agent in range(self.entity_count):
                    if robot == agent or not bool(
                        neighbor_mask[batch, robot, agent]
                    ):
                        continue
                    responsibility = self.last_responsibility[
                        batch, robot, agent
                    ]
                    # Match AVOCADO/RVO2: every perceived neighbor contributes
                    # an anticipatory OCA line. ``last_vo_active`` only reports
                    # whether the current relative velocity is already in VO.
                    if float(responsibility) > 0.0:
                        half_planes.append(
                            build_oca_half_plane(
                                measured_velocities[batch, robot],
                                self.last_vo_correction[batch, robot, agent],
                                self.last_vo_normal[batch, robot, agent],
                                1.0 - responsibility,
                            )
                        )

                if additional_half_plane_normals is not None:
                    for constraint in range(
                        additional_half_plane_normals.shape[-2]
                    ):
                        normal = additional_half_plane_normals[
                            batch, robot, constraint
                        ]
                        offset = additional_half_plane_offsets[
                            batch, robot, constraint
                        ]
                        normal_sq = torch.dot(normal, normal).clamp_min(1e-8)
                        half_planes.append(
                            HalfPlane(
                                normal=normal,
                                offset=offset,
                                point=offset * normal / normal_sq,
                            )
                        )

                solution = solve_closest_admissible_velocity(
                    preferred_velocities[batch, robot],
                    half_planes,
                    maximum_speed=float(self.maximum_speeds[robot]),
                )
                actions[batch, robot] = solution.velocity
                self.last_infeasible[batch, robot] = not solution.feasible
                self.last_constraint_count[batch, robot] = (
                    solution.active_constraint_count
                )
        self.previous_observed_velocity.copy_(measured_velocities)
        return actions


def fixed_orca_actions(
    positions: Tensor,
    measured_velocities: Tensor,
    preferred_velocities: Tensor,
    *,
    controlled_mask: Tensor,
    security_radii: Tensor,
    maximum_speeds: Tensor,
    perception_radius: float,
    time_horizon: float,
    cooperation: float = 0.5,
    additional_half_plane_normals: Optional[Tensor] = None,
    additional_half_plane_offsets: Optional[Tensor] = None,
) -> Tensor:
    """Fixed-responsibility ORCA baseline using the same VO/OCA geometry."""

    batch_size, entity_count, dimension = positions.shape
    if dimension != 2:
        raise ValueError("positions must have shape [batch, entity, 2].")
    if (additional_half_plane_normals is None) != (
        additional_half_plane_offsets is None
    ):
        raise ValueError(
            "additional half-plane normals and offsets must be supplied together."
        )
    if additional_half_plane_normals is not None:
        if additional_half_plane_normals.shape[:2] != (
            batch_size,
            entity_count,
        ) or additional_half_plane_normals.shape[-1] != 2:
            raise ValueError(
                "additional_half_plane_normals must have shape [batch, entity, K, 2]."
            )
        if additional_half_plane_offsets.shape != (
            *additional_half_plane_normals.shape[:-1],
        ):
            raise ValueError(
                "additional_half_plane_offsets must have shape [batch, entity, K]."
            )
    actions = preferred_velocities.clone()
    for batch in range(batch_size):
        for robot in range(entity_count):
            if not bool(controlled_mask[robot]):
                continue
            half_planes = []
            for agent in range(entity_count):
                if robot == agent:
                    continue
                relative_position = (
                    positions[batch, agent] - positions[batch, robot]
                )
                if (
                    float(torch.linalg.vector_norm(relative_position))
                    >= perception_radius
                ):
                    continue
                relative_velocity = (
                    measured_velocities[batch, robot]
                    - measured_velocities[batch, agent]
                )
                vo = finite_velocity_obstacle_correction(
                    relative_position,
                    relative_velocity,
                    security_radii[robot] + security_radii[agent],
                    time_horizon,
                )
                half_planes.append(
                    build_oca_half_plane(
                        measured_velocities[batch, robot],
                        vo.correction,
                        vo.normal,
                        torch.as_tensor(
                            cooperation,
                            device=positions.device,
                            dtype=positions.dtype,
                        ),
                    )
                )
            if additional_half_plane_normals is not None:
                for constraint in range(additional_half_plane_normals.shape[-2]):
                    normal = additional_half_plane_normals[
                        batch, robot, constraint
                    ]
                    offset = additional_half_plane_offsets[
                        batch, robot, constraint
                    ]
                    normal_sq = torch.dot(normal, normal).clamp_min(1e-8)
                    half_planes.append(
                        HalfPlane(
                            normal=normal,
                            offset=offset,
                            point=offset * normal / normal_sq,
                        )
                    )
            solution = solve_closest_admissible_velocity(
                preferred_velocities[batch, robot],
                half_planes,
                maximum_speed=float(maximum_speeds[robot]),
            )
            actions[batch, robot] = solution.velocity
    return actions
