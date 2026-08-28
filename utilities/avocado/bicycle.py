"""A3 adapters from holonomic AVOCADO velocities to bicycle commands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


_EPSILON = 1e-8


@dataclass(frozen=True)
class BicycleAdapterParameters:
    """Geometry and limits of SigmaRL's kinematic bicycle interface."""

    front_length: float
    rear_length: float
    maximum_speed: float
    maximum_steering_angle: float
    minimum_speed_ratio: float = 0.2

    def __post_init__(self) -> None:
        positive = {
            "front_length": self.front_length,
            "rear_length": self.rear_length,
            "maximum_speed": self.maximum_speed,
            "maximum_steering_angle": self.maximum_steering_angle,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.minimum_speed_ratio <= 1:
            raise ValueError("minimum_speed_ratio must be in [0, 1].")


@dataclass(frozen=True)
class BicycleActionResult:
    """Native ``[speed, steering]`` command plus adapter diagnostics."""

    action: Tensor
    desired_heading: Tensor
    heading_error: Tensor
    commanded_slip_angle: Tensor
    steering_saturated: Tensor


def wrap_to_pi(angle: Tensor) -> Tensor:
    """Wrap angles to ``[-pi, pi]`` without branch discontinuities."""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def constrain_velocity_to_path(
    desired_velocity: Tensor,
    path_velocity: Tensor,
    maximum_deviation_angle: float,
) -> Tensor:
    """Limit a collision-avoidance velocity's heading deviation from the lane."""

    if desired_velocity.shape != path_velocity.shape:
        raise ValueError("desired_velocity and path_velocity must have equal shape.")
    if desired_velocity.shape[-1:] != (2,):
        raise ValueError("velocities must have final dimension 2.")
    if (
        not math.isfinite(maximum_deviation_angle)
        or maximum_deviation_angle <= 0
        or maximum_deviation_angle > math.pi
    ):
        raise ValueError("maximum_deviation_angle must be in (0, pi].")
    desired_speed = torch.linalg.vector_norm(desired_velocity, dim=-1)
    path_speed = torch.linalg.vector_norm(path_velocity, dim=-1)
    desired_heading = torch.atan2(
        desired_velocity[..., 1], desired_velocity[..., 0]
    )
    path_heading = torch.atan2(path_velocity[..., 1], path_velocity[..., 0])
    deviation = wrap_to_pi(desired_heading - path_heading).clamp(
        -float(maximum_deviation_angle),
        float(maximum_deviation_angle),
    )
    constrained_heading = path_heading + deviation
    constrained = torch.stack(
        (torch.cos(constrained_heading), torch.sin(constrained_heading)), dim=-1
    ) * desired_speed.unsqueeze(-1)
    inactive = (desired_speed <= _EPSILON) | (path_speed <= _EPSILON)
    return torch.where(inactive.unsqueeze(-1), desired_velocity, constrained)


def path_velocity_cone_constraints(
    path_velocity: Tensor,
    maximum_deviation_angle: float,
) -> Tuple[Tensor, Tensor]:
    """Return two half-planes defining a forward path-heading velocity cone."""

    if path_velocity.shape[-1:] != (2,):
        raise ValueError("path_velocity must have final dimension 2.")
    if (
        not math.isfinite(maximum_deviation_angle)
        or maximum_deviation_angle <= 0
        or maximum_deviation_angle >= math.pi / 2
    ):
        raise ValueError("maximum_deviation_angle must be in (0, pi/2).")
    path_norm = torch.linalg.vector_norm(
        path_velocity, dim=-1, keepdim=True
    )
    if bool((path_norm <= _EPSILON).any()):
        raise ValueError("path_velocity must be nonzero for cone constraints.")
    tangent = path_velocity / path_norm
    tx = tangent[..., 0]
    ty = tangent[..., 1]
    sine = math.sin(maximum_deviation_angle)
    cosine = math.cos(maximum_deviation_angle)
    lower_normal = torch.stack(
        (sine * tx - cosine * ty, cosine * tx + sine * ty),
        dim=-1,
    )
    upper_normal = torch.stack(
        (sine * tx + cosine * ty, -cosine * tx + sine * ty),
        dim=-1,
    )
    normals = torch.stack((lower_normal, upper_normal), dim=-2)
    offsets = torch.zeros_like(normals[..., 0])
    return normals, offsets


def reference_path_preferred_velocity(
    positions: Tensor,
    short_term_reference_paths: Tensor,
    cruise_speed: float,
    *,
    terminal_fallback_directions: Optional[Tensor] = None,
) -> Tensor:
    """Construct a path-tangent preferred velocity for every road vehicle."""

    if positions.shape[-1:] != (2,):
        raise ValueError("positions must have final dimension 2.")
    if short_term_reference_paths.shape[:-2] != positions.shape[:-1]:
        raise ValueError(
            "short_term_reference_paths must match positions' batch/entity axes."
        )
    if short_term_reference_paths.shape[-1] != 2:
        raise ValueError("short_term_reference_paths must have final dimension 2.")
    if short_term_reference_paths.shape[-2] < 1:
        raise ValueError("At least one short-term reference point is required.")
    if not math.isfinite(cruise_speed) or cruise_speed <= 0:
        raise ValueError("cruise_speed must be finite and positive.")

    direction = short_term_reference_paths[..., -1, :] - positions
    direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if short_term_reference_paths.shape[-2] > 1:
        path_span = (
            short_term_reference_paths[..., -1, :]
            - short_term_reference_paths[..., 0, :]
        )
    else:
        path_span = direction
    terminal_path = (
        torch.linalg.vector_norm(path_span, dim=-1, keepdim=True) <= _EPSILON
    )
    if terminal_fallback_directions is None:
        fallback = path_span
    else:
        if terminal_fallback_directions.shape != positions.shape:
            raise ValueError(
                "terminal_fallback_directions must have the same shape as positions."
            )
        fallback = terminal_fallback_directions
        direction = torch.where(terminal_path, fallback, direction)
        direction_norm = torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        )
    fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
    direction = torch.where(direction_norm > _EPSILON, direction, fallback)
    direction_norm = torch.where(
        direction_norm > _EPSILON,
        direction_norm,
        fallback_norm,
    )
    unit_direction = direction / direction_norm.clamp_min(_EPSILON)
    return torch.where(
        direction_norm > _EPSILON,
        unit_direction * float(cruise_speed),
        torch.zeros_like(direction),
    )


def stanley_path_preferred_velocity(
    positions: Tensor,
    short_term_reference_paths: Tensor,
    cruise_speed: float,
    *,
    cross_track_gain: float,
    softening_speed: float,
    maximum_correction_angle: float,
    terminal_fallback_directions: Optional[Tensor] = None,
) -> Tensor:
    """Build a path-tangent velocity with Stanley-style lateral feedback."""

    if positions.shape[-1:] != (2,):
        raise ValueError("positions must have final dimension 2.")
    if short_term_reference_paths.shape[:-2] != positions.shape[:-1]:
        raise ValueError(
            "short_term_reference_paths must match positions' batch/entity axes."
        )
    if short_term_reference_paths.shape[-2] < 2:
        raise ValueError("Stanley tracking requires at least two reference points.")
    if short_term_reference_paths.shape[-1] != 2:
        raise ValueError("short_term_reference_paths must have final dimension 2.")
    positive = {
        "cruise_speed": cruise_speed,
        "cross_track_gain": cross_track_gain,
        "maximum_correction_angle": maximum_correction_angle,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(softening_speed) or softening_speed < 0:
        raise ValueError("softening_speed must be finite and nonnegative.")
    if maximum_correction_angle > math.pi / 2:
        raise ValueError("maximum_correction_angle must not exceed pi/2.")

    path_tangent = (
        short_term_reference_paths[..., 1, :]
        - short_term_reference_paths[..., 0, :]
    )
    path_span = (
        short_term_reference_paths[..., -1, :]
        - short_term_reference_paths[..., 0, :]
    )
    tangent_norm = torch.linalg.vector_norm(
        path_tangent, dim=-1, keepdim=True
    )
    path_tangent = torch.where(
        tangent_norm > _EPSILON,
        path_tangent,
        path_span,
    )
    tangent_norm = torch.linalg.vector_norm(
        path_tangent, dim=-1, keepdim=True
    )
    if terminal_fallback_directions is not None:
        if terminal_fallback_directions.shape != positions.shape:
            raise ValueError(
                "terminal_fallback_directions must have the same shape as positions."
            )
        path_tangent = torch.where(
            tangent_norm > _EPSILON,
            path_tangent,
            terminal_fallback_directions,
        )
        tangent_norm = torch.linalg.vector_norm(
            path_tangent, dim=-1, keepdim=True
        )

    tangent = path_tangent / tangent_norm.clamp_min(_EPSILON)
    left_normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    error_to_path = short_term_reference_paths[..., 0, :] - positions
    signed_cross_track_error = (error_to_path * left_normal).sum(dim=-1)
    correction = torch.atan2(
        float(cross_track_gain) * signed_cross_track_error,
        torch.full_like(
            signed_cross_track_error,
            float(cruise_speed + softening_speed),
        ),
    ).clamp(
        -float(maximum_correction_angle),
        float(maximum_correction_angle),
    )
    cos_correction = torch.cos(correction)
    sin_correction = torch.sin(correction)
    corrected_direction = torch.stack(
        (
            tangent[..., 0] * cos_correction
            - tangent[..., 1] * sin_correction,
            tangent[..., 0] * sin_correction
            + tangent[..., 1] * cos_correction,
        ),
        dim=-1,
    )
    velocity = corrected_direction * float(cruise_speed)
    return torch.where(
        tangent_norm > _EPSILON,
        velocity,
        torch.zeros_like(velocity),
    )


def vector_velocity_to_bicycle_action(
    desired_velocity: Tensor,
    yaw: Tensor,
    parameters: BicycleAdapterParameters,
) -> BicycleActionResult:
    """Map a global 2-D velocity to SigmaRL's ``[speed, steering]`` action.

    SigmaRL's bicycle translates along ``yaw + beta``, where
    ``beta = atan(tan(delta) * l_r / (l_f + l_r))``.  This function inverts
    that relation when the requested direction is instantaneously reachable.
    Larger direction changes saturate at the maximum slip/steering angle and
    reduce speed in proportion to the remaining heading error.
    """

    if desired_velocity.shape[-1:] != (2,):
        raise ValueError("desired_velocity must have final dimension 2.")
    if yaw.shape == desired_velocity.shape[:-1] + (1,):
        yaw = yaw.squeeze(-1)
    if yaw.shape != desired_velocity.shape[:-1]:
        raise ValueError("yaw must match desired_velocity's batch/entity axes.")

    desired_speed = torch.linalg.vector_norm(desired_velocity, dim=-1)
    desired_heading = torch.atan2(
        desired_velocity[..., 1], desired_velocity[..., 0]
    )
    heading_error = wrap_to_pi(desired_heading - yaw)

    wheelbase = parameters.front_length + parameters.rear_length
    maximum_slip = math.atan(
        math.tan(parameters.maximum_steering_angle)
        * parameters.rear_length
        / wheelbase
    )
    commanded_slip = heading_error.clamp(-maximum_slip, maximum_slip)
    steering = torch.atan(
        torch.tan(commanded_slip) * wheelbase / parameters.rear_length
    )
    steering = steering.clamp(
        -parameters.maximum_steering_angle,
        parameters.maximum_steering_angle,
    )

    residual_heading_error = wrap_to_pi(heading_error - commanded_slip)
    alignment = torch.cos(residual_heading_error).clamp(0.0, 1.0)
    speed_scale = alignment.clamp_min(parameters.minimum_speed_ratio)
    speed = desired_speed.clamp_max(parameters.maximum_speed) * speed_scale
    stopped = desired_speed <= _EPSILON
    speed = torch.where(stopped, torch.zeros_like(speed), speed)
    steering = torch.where(stopped, torch.zeros_like(steering), steering)
    saturated = (~stopped) & (heading_error.abs() > maximum_slip + 1e-6)

    return BicycleActionResult(
        action=torch.stack((speed, steering), dim=-1),
        desired_heading=desired_heading,
        heading_error=heading_error,
        commanded_slip_angle=commanded_slip,
        steering_saturated=saturated,
    )
