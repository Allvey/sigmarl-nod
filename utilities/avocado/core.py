"""A0: equation-level implementation of AVOCADO.

The equation numbers in this module refer to Martínez-Baselga et al.,
"AVOCADO: Adaptive Optimal Collision Avoidance driven by Opinion".  The code
keeps the paper's original assumptions: two-dimensional holonomic velocities,
disc geometry, pair-wise opinions, forward Euler integration, and direct
projection of a preferred velocity onto OCA half-spaces.

Source and licensing notes for the RVO2-style geometry and linear-programming
fallback are recorded in ``utilities/avocado/NOTICE.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor


_EPSILON = 1e-8


@dataclass(frozen=True)
class AVOCADOParameters:
    """Parameters appearing in AVOCADO Algorithm 1 and the experiments."""

    dt: float = 0.05
    perception_radius: float = 2.5
    velocity_obstacle_horizon: float = 2.5
    epsilon: float = 3.22
    kappa: float = 14.15
    opinion_decay: float = 2.0
    opinion_self_weight: float = 0.3
    opinion_estimate_weight: float = 0.7
    attention_decay: float = 0.57
    opinion_bias: float = 0.0
    noise_sigma: float = 0.0005

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "perception_radius": self.perception_radius,
            "velocity_obstacle_horizon": self.velocity_obstacle_horizon,
            "epsilon": self.epsilon,
            "kappa": self.kappa,
            "opinion_decay": self.opinion_decay,
            "opinion_self_weight": self.opinion_self_weight,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            not math.isfinite(self.opinion_estimate_weight)
            or self.opinion_estimate_weight < 0
        ):
            raise ValueError(
                "opinion_estimate_weight must be finite and non-negative."
            )
        if not 0 <= self.attention_decay < 1:
            raise ValueError("attention_decay must be in [0, 1).")
        if not math.isfinite(self.opinion_bias) or not -1 <= self.opinion_bias <= 1:
            raise ValueError("opinion_bias must be finite and in [-1, 1].")
        if not math.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and non-negative.")

    @property
    def initial_opinion(self) -> float:
        """The equilibrium ``x=b/d`` before attention is activated."""

        return self.opinion_bias / self.opinion_decay


@dataclass(frozen=True)
class VelocityObstacleCorrection:
    """Minimum displacement from relative velocity to a finite VO boundary."""

    correction: Tensor
    normal: Tensor
    line_direction: Tensor
    boundary_velocity: Tensor
    active: Tensor
    already_colliding: Tensor


@dataclass(frozen=True)
class HalfPlane:
    """OCA constraint ``normal dot velocity >= offset`` (paper Eq. 7)."""

    normal: Tensor
    offset: Tensor
    point: Tensor


@dataclass(frozen=True)
class ProjectionResult:
    """Result of the two-dimensional OCA projection in paper Eq. 8."""

    velocity: Tensor
    feasible: bool
    active_constraint_count: int


def _require_vectors(value: Tensor, name: str) -> None:
    if value.shape[-1:] != (2,):
        raise ValueError(f"{name} must have final dimension 2, got {value.shape}.")


def _cross_2d(first: Tensor, second: Tensor) -> Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _clamp_norm(vector: Tensor, maximum: float) -> Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = torch.clamp(float(maximum) / norm.clamp_min(_EPSILON), max=1.0)
    return vector * scale


def goal_preferred_velocity(
    position: Tensor,
    goal: Tensor,
    max_speed: Tensor | float,
) -> Tensor:
    """Goal-directed velocity used by the official AVOCADO simulator.

    It is the displacement to the goal clipped to ``max_speed``.  Therefore it
    slows within one meter of the goal, matching the public reference code.
    """

    _require_vectors(position, "position")
    _require_vectors(goal, "goal")
    displacement = goal - position
    distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    speed = torch.as_tensor(max_speed, dtype=position.dtype, device=position.device)
    while speed.ndim < displacement.ndim:
        speed = speed.unsqueeze(-1)
    scale = torch.clamp(speed / distance.clamp_min(_EPSILON), max=1.0)
    return torch.where(
        distance > _EPSILON,
        displacement * scale,
        torch.zeros_like(displacement),
    )


def collision_time(
    robot_position: Tensor,
    preferred_velocity: Tensor,
    agent_position: Tensor,
    agent_velocity: Tensor,
    combined_radius: Tensor | float,
) -> Tensor:
    """Compute the first contact time from paper Eqs. 12-14.

    Expanding Eq. 13 gives the constant coefficient
    ``||p_r-p_i||^2-R^2``.  The PDF prints its negative in the definition of
    beta_3; using that printed sign would not solve Eq. 13.  This function uses
    the geometrically correct expansion and follows the paper's root cases.
    ``inf`` means no future contact and zero means the discs already overlap.
    """

    for value, name in (
        (robot_position, "robot_position"),
        (preferred_velocity, "preferred_velocity"),
        (agent_position, "agent_position"),
        (agent_velocity, "agent_velocity"),
    ):
        _require_vectors(value, name)

    relative_position = robot_position - agent_position
    relative_velocity = preferred_velocity - agent_velocity
    radius = torch.as_tensor(
        combined_radius,
        dtype=relative_position.dtype,
        device=relative_position.device,
    )
    beta_1 = (relative_velocity * relative_velocity).sum(dim=-1)
    beta_2 = 2.0 * (relative_velocity * relative_position).sum(dim=-1)
    beta_3 = (relative_position * relative_position).sum(dim=-1) - radius.square()

    infinity = torch.full_like(beta_1, torch.inf)
    already_colliding = beta_3 <= 0
    stationary = beta_1 <= _EPSILON
    discriminant = beta_2.square() - 4.0 * beta_1 * beta_3
    has_roots = (~stationary) & (discriminant >= 0)
    square_root = torch.sqrt(discriminant.clamp_min(0.0))
    denominator = 2.0 * beta_1.clamp_min(_EPSILON)
    first = (-beta_2 - square_root) / denominator
    second = (-beta_2 + square_root) / denominator
    positive_first = torch.where(first >= 0, first, infinity)
    positive_second = torch.where(second >= 0, second, infinity)
    first_contact = torch.minimum(positive_first, positive_second)
    result = torch.where(has_roots, first_contact, infinity)
    return torch.where(already_colliding, torch.zeros_like(result), result)


def attention_euler_step(
    attention: Tensor,
    time_to_collision: Tensor,
    *,
    dt: float,
    delta: float,
    kappa: float,
) -> Tensor:
    """Forward-Euler update of the attention law in paper Eq. 11."""

    inverse_time = torch.where(
        torch.isfinite(time_to_collision) & (time_to_collision > 0),
        time_to_collision.reciprocal(),
        torch.where(
            time_to_collision == 0,
            torch.full_like(time_to_collision, torch.inf),
            torch.zeros_like(time_to_collision),
        ),
    )
    stimulus = torch.tanh(float(kappa) * inverse_time)
    derivative = -float(delta) * attention + (1.0 - float(delta)) * stimulus
    return attention + float(dt) * derivative


def attention_reference_step(
    attention: Tensor,
    time_to_collision: Tensor,
    *,
    delta: float,
    kappa: float,
) -> Tensor:
    """Attention update used by the authors' public AVOCADO implementation.

    This discrete filter has equilibrium ``tanh(kappa/tau)`` and differs from
    applying forward Euler literally to Eq. 11 in the supplied preprint PDF.
    Keeping both functions makes the discrepancy directly testable.
    """

    inverse_time = torch.where(
        torch.isfinite(time_to_collision) & (time_to_collision > 0),
        time_to_collision.reciprocal(),
        torch.where(
            time_to_collision == 0,
            torch.full_like(time_to_collision, torch.inf),
            torch.zeros_like(time_to_collision),
        ),
    )
    target = torch.tanh(float(kappa) * inverse_time)
    return (1.0 - float(delta)) * attention + float(delta) * target


def projection_estimator(
    delta_velocity: Tensor,
    correction: Tensor,
    epsilon: float,
) -> Tensor:
    """Sensor-only shifted-cooperation estimate ``y_i`` from paper Eq. 15."""

    _require_vectors(delta_velocity, "delta_velocity")
    _require_vectors(correction, "correction")
    correction_norm_sq = (correction * correction).sum(dim=-1)
    projected_ratio = (delta_velocity * correction).sum(dim=-1).abs()
    projected_ratio = torch.where(
        correction_norm_sq > _EPSILON,
        projected_ratio / correction_norm_sq.clamp_min(_EPSILON),
        torch.zeros_like(projected_ratio),
    )
    return torch.tanh(float(epsilon) * (projected_ratio - 0.5))


def opinion_euler_step(
    opinion: Tensor,
    attention: Tensor,
    estimated_opinion: Tensor,
    *,
    dt: float,
    decay: float,
    self_weight: float,
    estimate_weight: float,
    bias: float,
) -> Tensor:
    """Forward-Euler nonlinear opinion update from paper Eq. 10."""

    derivative = (
        -float(decay) * opinion
        + float(decay)
        * attention
        * torch.tanh(
            float(self_weight) * opinion
            + float(estimate_weight) * estimated_opinion
        )
        + float(bias)
    )
    return opinion + float(dt) * derivative


def opinion_to_cooperation(opinion: Tensor) -> Tensor:
    """Map shifted opinion to cooperation degree using paper Eq. 9."""

    return (opinion + 1.0) / 2.0


def finite_velocity_obstacle_correction(
    relative_position: Tensor,
    relative_velocity: Tensor,
    combined_radius: Tensor | float,
    time_horizon: float,
    *,
    collision_time_step: Optional[float] = None,
) -> VelocityObstacleCorrection:
    """Return the minimum correction ``u_i`` in paper Eqs. 6 and 18.

    This is the standard exact two-dimensional projection onto the boundary of
    a truncated disc velocity obstacle.  The returned normal points out of the
    velocity obstacle.  A correction is active only when the supplied relative
    velocity lies inside the finite-horizon VO.
    """

    _require_vectors(relative_position, "relative_position")
    _require_vectors(relative_velocity, "relative_velocity")
    if not math.isfinite(time_horizon) or time_horizon <= 0:
        raise ValueError("time_horizon must be finite and positive.")

    radius = torch.as_tensor(
        combined_radius,
        dtype=relative_position.dtype,
        device=relative_position.device,
    )
    distance_sq = (relative_position * relative_position).sum(dim=-1)
    radius_sq = radius.square()
    already_colliding = distance_sq <= radius_sq

    origin = torch.zeros_like(relative_position)
    contact_time = collision_time(
        origin,
        relative_velocity,
        relative_position,
        origin,
        radius,
    )
    active = (~already_colliding) & (contact_time <= float(time_horizon))

    inverse_horizon = 1.0 / float(time_horizon)
    cutoff_relative = relative_velocity - inverse_horizon * relative_position
    cutoff_sq = (cutoff_relative * cutoff_relative).sum(dim=-1)
    cutoff_dot = (cutoff_relative * relative_position).sum(dim=-1)
    use_cutoff_circle = (
        (cutoff_dot < 0)
        & (cutoff_dot.square() > radius_sq * cutoff_sq)
    )

    cutoff_norm = torch.sqrt(cutoff_sq.clamp_min(_EPSILON))
    cutoff_unit = cutoff_relative / cutoff_norm.unsqueeze(-1)
    cutoff_direction = torch.stack(
        (cutoff_unit[..., 1], -cutoff_unit[..., 0]), dim=-1
    )
    cutoff_u = (
        radius * inverse_horizon - cutoff_norm
    ).unsqueeze(-1) * cutoff_unit

    leg = torch.sqrt((distance_sq - radius_sq).clamp_min(0.0))
    determinant = _cross_2d(relative_position, cutoff_relative)
    left_direction = torch.stack(
        (
            relative_position[..., 0] * leg
            - relative_position[..., 1] * radius,
            relative_position[..., 0] * radius
            + relative_position[..., 1] * leg,
        ),
        dim=-1,
    ) / distance_sq.clamp_min(_EPSILON).unsqueeze(-1)
    right_direction = -torch.stack(
        (
            relative_position[..., 0] * leg
            + relative_position[..., 1] * radius,
            -relative_position[..., 0] * radius
            + relative_position[..., 1] * leg,
        ),
        dim=-1,
    ) / distance_sq.clamp_min(_EPSILON).unsqueeze(-1)
    leg_direction = torch.where(
        (determinant > 0).unsqueeze(-1), left_direction, right_direction
    )
    leg_projection = (relative_velocity * leg_direction).sum(dim=-1, keepdim=True)
    leg_u = leg_projection * leg_direction - relative_velocity

    correction = torch.where(use_cutoff_circle.unsqueeze(-1), cutoff_u, leg_u)
    line_direction = torch.where(
        use_cutoff_circle.unsqueeze(-1), cutoff_direction, leg_direction
    )

    if collision_time_step is not None:
        if not math.isfinite(collision_time_step) or collision_time_step <= 0:
            raise ValueError("collision_time_step must be finite and positive.")
        inverse_step = 1.0 / float(collision_time_step)
        collision_w = relative_velocity - inverse_step * relative_position
        collision_norm = torch.linalg.vector_norm(
            collision_w, dim=-1, keepdim=True
        ).clamp_min(_EPSILON)
        collision_unit = collision_w / collision_norm
        collision_direction = torch.stack(
            (collision_unit[..., 1], -collision_unit[..., 0]), dim=-1
        )
        collision_u = (
            radius * inverse_step - collision_norm.squeeze(-1)
        ).unsqueeze(-1) * collision_unit
        correction = torch.where(
            already_colliding.unsqueeze(-1), collision_u, correction
        )
        line_direction = torch.where(
            already_colliding.unsqueeze(-1),
            collision_direction,
            line_direction,
        )

    normal = torch.stack(
        (-line_direction[..., 1], line_direction[..., 0]), dim=-1
    )
    normal = normal / torch.linalg.vector_norm(
        normal, dim=-1, keepdim=True
    ).clamp_min(_EPSILON)
    return VelocityObstacleCorrection(
        correction=correction,
        normal=normal,
        line_direction=line_direction,
        boundary_velocity=relative_velocity + correction,
        active=active,
        already_colliding=already_colliding,
    )


def build_oca_half_plane(
    anchor_velocity: Tensor,
    correction: Tensor,
    normal: Tensor,
    cooperation: Tensor | float,
) -> HalfPlane:
    """Build the robot's admissible half-space from paper Eq. 7."""

    _require_vectors(anchor_velocity, "anchor_velocity")
    _require_vectors(correction, "correction")
    _require_vectors(normal, "normal")
    alpha = torch.as_tensor(
        cooperation,
        dtype=anchor_velocity.dtype,
        device=anchor_velocity.device,
    )
    point = anchor_velocity + (1.0 - alpha).unsqueeze(-1) * correction
    offset = (point * normal).sum(dim=-1)
    return HalfPlane(normal=normal, offset=offset, point=point)


def _satisfies_constraints(
    velocity: Tensor,
    normals: Tensor,
    offsets: Tensor,
    maximum_speed: Optional[float],
    tolerance: float,
) -> bool:
    if normals.numel() and bool(
        torch.any(normals @ velocity < offsets - tolerance).item()
    ):
        return False
    if maximum_speed is not None and float(torch.linalg.vector_norm(velocity)) > (
        maximum_speed + tolerance
    ):
        return False
    return True


def _orca_linear_program_fallback(
    preferred_velocity: Tensor,
    half_planes: Sequence[HalfPlane],
    maximum_speed: float,
    tolerance: float,
) -> Tensor:
    """RVO2-style priority fallback for an infeasible OCA intersection."""

    points = torch.stack([plane.point for plane in half_planes])
    normals = torch.stack([plane.normal for plane in half_planes])
    directions = torch.stack((normals[:, 1], -normals[:, 0]), dim=-1)

    def linear_program_one(
        line_points: Tensor,
        line_directions: Tensor,
        line_number: int,
        optimum: Tensor,
        direction_optimum: bool,
    ) -> tuple[bool, Tensor]:
        point = line_points[line_number]
        direction = line_directions[line_number]
        point_dot_direction = torch.dot(point, direction)
        discriminant = (
            point_dot_direction.square()
            + maximum_speed**2
            - torch.dot(point, point)
        )
        if float(discriminant) < 0:
            return False, optimum
        root = torch.sqrt(discriminant.clamp_min(0.0))
        left = -point_dot_direction - root
        right = -point_dot_direction + root
        for index in range(line_number):
            denominator = _cross_2d(direction, line_directions[index])
            numerator = _cross_2d(
                line_directions[index], point - line_points[index]
            )
            if abs(float(denominator)) <= tolerance:
                if float(numerator) < 0:
                    return False, optimum
                continue
            parameter = numerator / denominator
            if float(denominator) >= 0:
                right = torch.minimum(right, parameter)
            else:
                left = torch.maximum(left, parameter)
            if float(left) > float(right):
                return False, optimum
        if direction_optimum:
            parameter = right if torch.dot(optimum, direction) > 0 else left
        else:
            parameter = torch.dot(direction, optimum - point).clamp(left, right)
        return True, point + parameter * direction

    def linear_program_two(
        line_points: Tensor,
        line_directions: Tensor,
        optimum: Tensor,
        direction_optimum: bool,
    ) -> tuple[Tensor, int]:
        if direction_optimum:
            result = optimum * maximum_speed
        else:
            result = _clamp_norm(optimum, maximum_speed)
        for line_number in range(line_points.shape[0]):
            violation = _cross_2d(
                line_directions[line_number],
                line_points[line_number] - result,
            )
            if float(violation) > tolerance:
                previous = result
                succeeded, result = linear_program_one(
                    line_points,
                    line_directions,
                    line_number,
                    optimum,
                    direction_optimum,
                )
                if not succeeded:
                    return previous, line_number
        return result, line_points.shape[0]

    result, failed_line = linear_program_two(
        points, directions, preferred_velocity, False
    )
    if failed_line >= points.shape[0]:
        return result

    distance = torch.zeros((), device=result.device, dtype=result.dtype)
    for line_number in range(failed_line, points.shape[0]):
        violation = _cross_2d(
            directions[line_number], points[line_number] - result
        )
        if float(violation) <= float(distance):
            continue
        projected_points = []
        projected_directions = []
        for previous_line in range(line_number):
            determinant = _cross_2d(
                directions[line_number], directions[previous_line]
            )
            if abs(float(determinant)) <= tolerance:
                if torch.dot(
                    directions[line_number], directions[previous_line]
                ) > 0:
                    continue
                projected_point = 0.5 * (
                    points[line_number] + points[previous_line]
                )
            else:
                projected_point = points[line_number] + (
                    _cross_2d(
                        directions[previous_line],
                        points[line_number] - points[previous_line],
                    )
                    / determinant
                ) * directions[line_number]
            projected_direction = (
                directions[previous_line] - directions[line_number]
            )
            projected_direction = projected_direction / torch.linalg.vector_norm(
                projected_direction
            ).clamp_min(_EPSILON)
            projected_points.append(projected_point)
            projected_directions.append(projected_direction)
        if projected_points:
            projected_points_tensor = torch.stack(projected_points)
            projected_directions_tensor = torch.stack(projected_directions)
        else:
            projected_points_tensor = torch.empty(
                (0, 2), device=result.device, dtype=result.dtype
            )
            projected_directions_tensor = torch.empty_like(
                projected_points_tensor
            )
        previous_result = result
        optimum_direction = torch.stack(
            (-directions[line_number, 1], directions[line_number, 0])
        )
        result, projected_failure = linear_program_two(
            projected_points_tensor,
            projected_directions_tensor,
            optimum_direction,
            True,
        )
        if projected_failure < projected_points_tensor.shape[0]:
            result = previous_result
        distance = violation
    return result


def solve_closest_admissible_velocity(
    preferred_velocity: Tensor,
    half_planes: Sequence[HalfPlane],
    *,
    maximum_speed: Optional[float] = None,
    tolerance: float = 1e-6,
) -> ProjectionResult:
    """Solve paper Eq. 8 exactly in two dimensions by active-set enumeration.

    The optional speed disc implements the ``v_max`` bound used in the paper's
    experiments.  With a feasible intersection, the Euclidean optimum is the
    preferred point itself, a projection onto one active line, an intersection
    of two active lines, or an intersection between an active line and the
    speed circle.  Enumerating those candidates therefore gives the exact 2-D
    solution without an external optimization dependency.
    """

    _require_vectors(preferred_velocity, "preferred_velocity")
    if preferred_velocity.ndim != 1:
        raise ValueError("The OCA solver accepts one 2-D preferred velocity.")
    if maximum_speed is not None and (
        not math.isfinite(maximum_speed) or maximum_speed <= 0
    ):
        raise ValueError("maximum_speed must be finite and positive when set.")

    device = preferred_velocity.device
    dtype = preferred_velocity.dtype
    if half_planes:
        normals = torch.stack(
            [plane.normal.to(device=device, dtype=dtype) for plane in half_planes]
        )
        offsets = torch.stack(
            [plane.offset.to(device=device, dtype=dtype) for plane in half_planes]
        )
    else:
        normals = torch.empty((0, 2), device=device, dtype=dtype)
        offsets = torch.empty((0,), device=device, dtype=dtype)

    candidates: list[Tensor] = []

    def add_candidate(candidate: Tensor) -> None:
        if torch.isfinite(candidate).all() and _satisfies_constraints(
            candidate, normals, offsets, maximum_speed, tolerance
        ):
            candidates.append(candidate)

    initial = preferred_velocity
    if maximum_speed is not None:
        initial = _clamp_norm(initial, maximum_speed)
    add_candidate(initial)
    add_candidate(torch.zeros(2, device=device, dtype=dtype))

    for index in range(normals.shape[0]):
        normal = normals[index]
        offset = offsets[index]
        normal_sq = torch.dot(normal, normal).clamp_min(_EPSILON)
        projected = preferred_velocity + (
            (offset - torch.dot(normal, preferred_velocity)) / normal_sq
        ) * normal
        add_candidate(projected)

        if maximum_speed is not None:
            unit_normal = normal / torch.sqrt(normal_sq)
            signed_offset = offset / torch.sqrt(normal_sq)
            if abs(float(signed_offset)) <= maximum_speed + tolerance:
                tangent = torch.stack((-unit_normal[1], unit_normal[0]))
                tangent_scale = torch.sqrt(
                    torch.clamp(
                        torch.as_tensor(maximum_speed**2, device=device, dtype=dtype)
                        - signed_offset.square(),
                        min=0.0,
                    )
                )
                add_candidate(signed_offset * unit_normal + tangent_scale * tangent)
                add_candidate(signed_offset * unit_normal - tangent_scale * tangent)

    for first in range(normals.shape[0]):
        for second in range(first + 1, normals.shape[0]):
            determinant = _cross_2d(normals[first], normals[second])
            if abs(float(determinant)) <= tolerance:
                continue
            intersection = torch.stack(
                (
                    (
                        offsets[first] * normals[second, 1]
                        - normals[first, 1] * offsets[second]
                    )
                    / determinant,
                    (
                        normals[first, 0] * offsets[second]
                        - offsets[first] * normals[second, 0]
                    )
                    / determinant,
                )
            )
            add_candidate(intersection)

    if not candidates:
        # Eq. 8 has no exact solution for an inconsistent set.  Match the
        # reference RVO2 implementation's ordered least-violation fallback but
        # keep feasible=False so diagnostics never claim a safety guarantee.
        if maximum_speed is None:
            fallback = initial
        else:
            fallback = _orca_linear_program_fallback(
                preferred_velocity,
                half_planes,
                maximum_speed,
                tolerance,
            )
        return ProjectionResult(fallback, False, len(half_planes))

    distances = torch.stack(
        [
            torch.sum((candidate - preferred_velocity).square())
            for candidate in candidates
        ]
    )
    best = candidates[int(torch.argmin(distances))]
    return ProjectionResult(best, True, len(half_planes))
