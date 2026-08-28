"""Auditable safety guard for executable A3 bicycle commands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from utilities.avocado.bicycle import BicycleAdapterParameters
from utilities.avocado.core import collision_time


@dataclass(frozen=True)
class TTCSafetyShieldResult:
    """Shielded commands and per-step intervention diagnostics."""

    action: Tensor
    intervention_mask: Tensor
    pair_intervention_count: Tensor
    minimum_ttc_before: Tensor
    minimum_ttc_after: Tensor
    unsafe_pair_count_after: Tensor


def bicycle_action_velocity(
    action: Tensor,
    yaw: Tensor,
    parameters: BicycleAdapterParameters,
) -> Tensor:
    """Convert native ``[speed, steering]`` commands to world velocities."""

    if action.shape[-1:] != (2,):
        raise ValueError("action must have final dimension 2.")
    if yaw.shape == action.shape[:-1] + (1,):
        yaw = yaw.squeeze(-1)
    if yaw.shape != action.shape[:-1]:
        raise ValueError("yaw must match action's batch/entity axes.")
    speed = action[..., 0]
    steering = action[..., 1]
    wheelbase = parameters.front_length + parameters.rear_length
    slip = torch.atan(
        torch.tan(steering) * parameters.rear_length / wheelbase
    )
    travel_heading = yaw + slip
    return torch.stack(
        (speed * torch.cos(travel_heading), speed * torch.sin(travel_heading)),
        dim=-1,
    )


def _effective_ttc(
    first_position: Tensor,
    first_velocity: Tensor,
    second_position: Tensor,
    second_velocity: Tensor,
    combined_radius: Tensor,
) -> Tensor:
    """TTC where an overlapping safety margin moving apart is not a threat."""

    value = collision_time(
        first_position,
        first_velocity,
        second_position,
        second_velocity,
        combined_radius,
    )
    relative_position = first_position - second_position
    relative_velocity = first_velocity - second_velocity
    distance_squared = torch.dot(relative_position, relative_position)
    separating = torch.dot(relative_position, relative_velocity) >= 0
    if bool((distance_squared <= combined_radius.square()) & separating):
        return torch.full_like(value, torch.inf)
    return value


def _pairwise_minimum_ttc(
    positions: Tensor,
    velocities: Tensor,
    security_radii: Tensor,
    threshold: float,
) -> tuple[Tensor, Tensor]:
    batch_size, entity_count, _ = positions.shape
    minima = torch.full(
        (batch_size,), torch.inf, dtype=positions.dtype, device=positions.device
    )
    unsafe_counts = torch.zeros(
        batch_size, dtype=torch.int64, device=positions.device
    )
    for batch in range(batch_size):
        for first in range(entity_count):
            for second in range(first + 1, entity_count):
                value = _effective_ttc(
                    positions[batch, first],
                    velocities[batch, first],
                    positions[batch, second],
                    velocities[batch, second],
                    security_radii[first] + security_radii[second],
                )
                minima[batch] = torch.minimum(minima[batch], value)
                if bool(value < threshold):
                    unsafe_counts[batch] += 1
    return minima, unsafe_counts


def apply_ttc_braking_shield(
    positions: Tensor,
    actions: Tensor,
    yaw: Tensor,
    security_radii: Tensor,
    parameters: BicycleAdapterParameters,
    *,
    minimum_ttc_seconds: float,
    responsibility: Tensor | None = None,
) -> TTCSafetyShieldResult:
    """Brake one responsible vehicle when executable commands violate TTC.

    This operates after the holonomic-to-bicycle adapter, so it guards the
    command the simulator will actually execute.  Opinion responsibility is
    used only to choose which vehicle yields; equal shares use the higher
    entity index as a deterministic tie-break.
    """

    if positions.shape != actions.shape or positions.shape[-1:] != (2,):
        raise ValueError("positions and actions must have shape [B, N, 2].")
    entity_count = positions.shape[1]
    if security_radii.shape != (entity_count,):
        raise ValueError("security_radii must have shape [N].")
    if minimum_ttc_seconds <= 0:
        raise ValueError("minimum_ttc_seconds must be positive.")
    if responsibility is not None and responsibility.shape != (
        positions.shape[0], entity_count, entity_count
    ):
        raise ValueError("responsibility must have shape [B, N, N].")

    shielded = actions.clone()
    intervention_mask = torch.zeros(
        positions.shape[:2], dtype=torch.bool, device=positions.device
    )
    pair_count = torch.zeros(
        positions.shape[0], dtype=torch.int64, device=positions.device
    )
    velocities = bicycle_action_velocity(shielded, yaw, parameters)
    minimum_before, _ = _pairwise_minimum_ttc(
        positions,
        velocities,
        security_radii,
        minimum_ttc_seconds,
    )

    for batch in range(positions.shape[0]):
        # Recompute after every intervention because stopping one car changes
        # its TTC with all other cars.
        for _ in range(entity_count):
            most_urgent = None
            urgent_ttc = torch.inf
            for first in range(entity_count):
                for second in range(first + 1, entity_count):
                    value = _effective_ttc(
                        positions[batch, first],
                        velocities[batch, first],
                        positions[batch, second],
                        velocities[batch, second],
                        security_radii[first] + security_radii[second],
                    )
                    if float(value) < float(urgent_ttc):
                        urgent_ttc = value
                        most_urgent = (first, second)
            if most_urgent is None or float(urgent_ttc) >= minimum_ttc_seconds:
                break
            first, second = most_urgent
            if responsibility is None:
                yielding = second
            else:
                first_share = float(responsibility[batch, first, second])
                second_share = float(responsibility[batch, second, first])
                yielding = first if first_share > second_share else second
            if bool(intervention_mask[batch, yielding]):
                # The selected vehicle is already stopped; stop its counterpart
                # only when the pair is still closing.
                yielding = second if yielding == first else first
                if bool(intervention_mask[batch, yielding]):
                    break
            shielded[batch, yielding, 0] = 0.0
            shielded[batch, yielding, 1] = 0.0
            intervention_mask[batch, yielding] = True
            pair_count[batch] += 1
            velocities[batch, yielding] = 0.0

    minimum_after, unsafe_after = _pairwise_minimum_ttc(
        positions,
        velocities,
        security_radii,
        minimum_ttc_seconds,
    )
    return TTCSafetyShieldResult(
        action=shielded,
        intervention_mask=intervention_mask,
        pair_intervention_count=pair_count,
        minimum_ttc_before=minimum_before,
        minimum_ttc_after=minimum_after,
        unsafe_pair_count_after=unsafe_after,
    )
