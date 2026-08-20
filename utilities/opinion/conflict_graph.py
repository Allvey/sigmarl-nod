"""Current-physics conflict candidates for the independent Opinion-MARL path.

The module is deliberately stateless: it neither owns opinion state nor reads
learned topology outputs.  Closest-approach quantities are deterministic
constant-velocity geometry computed from the current positions and velocities;
they are not simulator future states or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


PAIR_FEATURE_NAMES: Tuple[str, ...] = (
    "relative_position_longitudinal",
    "relative_position_lateral",
    "relative_velocity_longitudinal",
    "relative_velocity_lateral",
    "distance",
    "closing_speed",
    "time_to_closest_approach",
    "distance_at_closest_approach",
    "heading_difference_sin",
    "heading_difference_cos",
    "ego_speed",
    "neighbor_speed",
)


@dataclass(frozen=True)
class ConflictGraphOutput:
    """Fixed-shape candidate edges for every ego agent."""

    pair_features: Tensor
    neighbor_ids: Tensor
    pair_mask: Tensor
    urgency: Tensor
    confidence: Tensor


class ConflictGraph(nn.Module):
    """Rank visible directed pairs using current constant-velocity geometry."""

    feature_names = PAIR_FEATURE_NAMES
    feature_dim = len(PAIR_FEATURE_NAMES)

    def __init__(
        self,
        *,
        n_candidates: int,
        ttc_horizon: float,
        safe_distance: float,
        urgency_time_scale: float,
        urgency_distance_temperature: float,
    ) -> None:
        super().__init__()
        if type(n_candidates) is not int or n_candidates <= 0:
            raise ValueError("n_candidates must be a positive integer")
        for name, value, allow_zero in (
            ("ttc_horizon", ttc_horizon, False),
            ("safe_distance", safe_distance, True),
            ("urgency_time_scale", urgency_time_scale, False),
            (
                "urgency_distance_temperature",
                urgency_distance_temperature,
                False,
            ),
        ):
            tensor_value = torch.as_tensor(value)
            if tensor_value.numel() != 1 or not torch.isfinite(tensor_value):
                raise ValueError(f"{name} must be a finite scalar")
            if (value < 0) if allow_zero else (value <= 0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier}")

        self.n_candidates = n_candidates
        self.ttc_horizon = float(ttc_horizon)
        self.safe_distance = float(safe_distance)
        self.urgency_time_scale = float(urgency_time_scale)
        self.urgency_distance_temperature = float(
            urgency_distance_temperature
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        headings: Tensor,
        visibility_mask: Tensor,
    ) -> ConflictGraphOutput:
        """Build candidates from tensors shaped ``[E, N, ...]``.

        ``visibility_mask[e, i, j]`` is authoritative sensor visibility.  Self
        edges are always removed.  Invalid/padded slots have global ID ``-1``
        and all-zero floating outputs.
        """
        self._validate_inputs(positions, velocities, headings, visibility_mask)
        n_envs, n_agents, _ = positions.shape
        if self.n_candidates > n_agents - 1:
            raise ValueError(
                "n_candidates must be at most n_agents - 1; "
                f"got {self.n_candidates} for {n_agents} agents"
            )

        # Directed convention: pair [i, j] stores neighbor j relative to ego i.
        relative_position = positions[:, None, :, :] - positions[:, :, None, :]
        relative_velocity = velocities[:, None, :, :] - velocities[:, :, None, :]

        ego_heading = headings[:, :, None]
        cos_heading = torch.cos(ego_heading)
        sin_heading = torch.sin(ego_heading)
        rel_pos_long = (
            cos_heading * relative_position[..., 0]
            + sin_heading * relative_position[..., 1]
        )
        rel_pos_lat = (
            -sin_heading * relative_position[..., 0]
            + cos_heading * relative_position[..., 1]
        )
        rel_vel_long = (
            cos_heading * relative_velocity[..., 0]
            + sin_heading * relative_velocity[..., 1]
        )
        rel_vel_lat = (
            -sin_heading * relative_velocity[..., 0]
            + cos_heading * relative_velocity[..., 1]
        )

        eps = torch.finfo(positions.dtype).eps
        distance = torch.linalg.vector_norm(relative_position, dim=-1)
        radial_dot = (relative_position * relative_velocity).sum(dim=-1)
        closing_speed = -radial_dot / distance.clamp_min(eps)
        relative_speed_sq = relative_velocity.square().sum(dim=-1)
        raw_t_cpa = -radial_dot / relative_speed_sq.clamp_min(eps)
        t_cpa = raw_t_cpa.clamp(min=0.0, max=self.ttc_horizon)
        position_at_cpa = (
            relative_position + t_cpa.unsqueeze(-1) * relative_velocity
        )
        distance_at_cpa = torch.linalg.vector_norm(position_at_cpa, dim=-1)

        heading_difference = headings[:, None, :] - headings[:, :, None]
        ego_speed = torch.linalg.vector_norm(velocities, dim=-1)[:, :, None]
        neighbor_speed = torch.linalg.vector_norm(velocities, dim=-1)[:, None, :]
        ego_speed = ego_speed.expand(-1, -1, n_agents)
        neighbor_speed = neighbor_speed.expand(-1, n_agents, -1)

        all_features = torch.stack(
            (
                rel_pos_long,
                rel_pos_lat,
                rel_vel_long,
                rel_vel_lat,
                distance,
                closing_speed,
                t_cpa,
                distance_at_cpa,
                torch.sin(heading_difference),
                torch.cos(heading_difference),
                ego_speed,
                neighbor_speed,
            ),
            dim=-1,
        )

        not_self = ~torch.eye(
            n_agents, device=positions.device, dtype=torch.bool
        ).unsqueeze(0)
        valid_pairs = visibility_mask & not_self
        approaching = (radial_dot < 0) & (raw_t_cpa >= 0)
        currently_unsafe = distance <= self.safe_distance
        active_conflict = valid_pairs & (approaching | currently_unsafe)

        time_score = torch.exp(-t_cpa / self.urgency_time_scale)
        distance_score = torch.sigmoid(
            (self.safe_distance - distance_at_cpa)
            / self.urgency_distance_temperature
        )
        all_confidence = valid_pairs.to(dtype=positions.dtype)
        all_urgency = (
            active_conflict.to(dtype=positions.dtype)
            * time_score
            * distance_score
            * all_confidence
        ).clamp_(0.0, 1.0)

        # Lexicographic ranking: urgency descending, current distance ascending,
        # global neighbor ID ascending. Stable sorts make ties reproducible.
        invalid_distance = torch.full_like(distance, float("inf"))
        ranking_distance = torch.where(valid_pairs, distance, invalid_distance)
        by_distance = torch.argsort(ranking_distance, dim=-1, stable=True)
        urgency_by_distance = torch.gather(all_urgency, -1, by_distance)
        by_urgency = torch.argsort(
            urgency_by_distance, dim=-1, descending=True, stable=True
        )
        ranked_ids = torch.gather(by_distance, -1, by_urgency)
        selected_ids = ranked_ids[..., : self.n_candidates]

        selected_mask = torch.gather(valid_pairs, -1, selected_ids)
        safe_ids = selected_ids.clamp_min(0)
        feature_index = safe_ids.unsqueeze(-1).expand(
            n_envs, n_agents, self.n_candidates, self.feature_dim
        )
        selected_features = torch.gather(all_features, 2, feature_index)
        selected_urgency = torch.gather(all_urgency, -1, safe_ids)
        selected_confidence = torch.gather(all_confidence, -1, safe_ids)

        selected_features = torch.where(
            selected_mask.unsqueeze(-1),
            selected_features,
            torch.zeros_like(selected_features),
        )
        selected_urgency = torch.where(
            selected_mask, selected_urgency, torch.zeros_like(selected_urgency)
        )
        selected_confidence = torch.where(
            selected_mask,
            selected_confidence,
            torch.zeros_like(selected_confidence),
        )
        selected_ids = torch.where(
            selected_mask, selected_ids, torch.full_like(selected_ids, -1)
        )

        return ConflictGraphOutput(
            pair_features=selected_features,
            neighbor_ids=selected_ids,
            pair_mask=selected_mask,
            urgency=selected_urgency,
            confidence=selected_confidence,
        )

    @staticmethod
    def _validate_inputs(
        positions: Tensor,
        velocities: Tensor,
        headings: Tensor,
        visibility_mask: Tensor,
    ) -> None:
        if not torch.is_tensor(positions) or positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [E, N, 2]")
        if not positions.is_floating_point():
            raise TypeError("positions must use a floating dtype")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must have the same [E, N, 2] shape as positions")
        if not velocities.is_floating_point():
            raise TypeError("velocities must use a floating dtype")
        if headings.shape != positions.shape[:2]:
            raise ValueError("headings must have shape [E, N]")
        if not headings.is_floating_point():
            raise TypeError("headings must use a floating dtype")
        expected_visibility_shape = positions.shape[:2] + (positions.shape[1],)
        if visibility_mask.shape != expected_visibility_shape:
            raise ValueError("visibility_mask must have shape [E, N, N]")
        if visibility_mask.dtype is not torch.bool:
            raise TypeError("visibility_mask must have dtype bool")
        tensors = (positions, velocities, headings, visibility_mask)
        if any(tensor.device != positions.device for tensor in tensors):
            raise ValueError("all ConflictGraph inputs must be on the same device")
        if velocities.dtype != positions.dtype or headings.dtype != positions.dtype:
            raise ValueError("positions, velocities, and headings must share a dtype")
        if not all(torch.isfinite(tensor).all() for tensor in tensors[:3]):
            raise ValueError("ConflictGraph floating inputs must be finite")
