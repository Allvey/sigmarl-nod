"""Physical conflict candidates and fixed pair features for Opinion-MARL.

This module is deliberately policy-independent. It converts the current
observable vehicle states and SigmaRL's existing nearest-neighbor IDs into a
fixed tensor contract consumed by later Opinion milestones.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from utilities.opinion.config import ConflictGraphConfig


class ConflictGraphOutput(NamedTuple):
    """Batched directed-pair data with leading shape ``[E, N, K]``."""

    pair_features: torch.Tensor
    neighbor_ids: torch.Tensor
    pair_mask: torch.Tensor
    urgency: torch.Tensor
    confidence: torch.Tensor


class ConflictGraph(nn.Module):
    """Build a constant-velocity closest-approach conflict graph.

    The graph never selects additional vehicles. ``neighbor_ids`` must come
    from the original local-observation top-k selection, so enabling this
    module does not grant the policy a larger sensing field.
    """

    def __init__(self, config: ConflictGraphConfig, max_speed: float) -> None:
        super().__init__()
        if config.pair_feature_dim != 10:
            raise ValueError("ConflictGraph requires pair_feature_dim=10.")
        if max_speed <= 0:
            raise ValueError("max_speed must be positive.")
        self.config = config
        self.max_speed = float(max_speed)

    @staticmethod
    def _validate_inputs(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        yaws: torch.Tensor,
        neighbor_ids: torch.Tensor,
    ) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [E, N, 2].")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must have the same shape as positions.")
        if yaws.ndim == 3 and yaws.shape[-1] == 1:
            yaws = yaws.squeeze(-1)
        if yaws.shape != positions.shape[:2]:
            raise ValueError("yaws must have shape [E, N] or [E, N, 1].")
        if neighbor_ids.ndim != 3 or neighbor_ids.shape[:2] != positions.shape[:2]:
            raise ValueError("neighbor_ids must have shape [E, N, K].")
        if neighbor_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("neighbor_ids must use an integer dtype.")
        return yaws

    @staticmethod
    def _gather_vectors(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        environments, agents, _ = values.shape
        expanded = values.unsqueeze(1).expand(-1, agents, -1, -1)
        indices = ids.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        return torch.gather(expanded, dim=2, index=indices)

    @staticmethod
    def _gather_scalars(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        agents = values.shape[1]
        expanded = values.unsqueeze(1).expand(-1, agents, -1)
        return torch.gather(expanded, dim=2, index=ids)

    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        yaws: torch.Tensor,
        neighbor_ids: torch.Tensor,
    ) -> ConflictGraphOutput:
        yaws = self._validate_inputs(positions, velocities, yaws, neighbor_ids)
        environments, agents, _ = positions.shape
        if neighbor_ids.shape[-1] != self.config.candidate_count:
            raise ValueError(
                "neighbor_ids candidate dimension does not match "
                "conflict_graph.candidate_count."
            )

        ids = neighbor_ids.to(dtype=torch.long)
        valid_ids = (ids >= 0) & (ids < agents)
        safe_ids = ids.clamp(0, agents - 1)
        ego_ids = torch.arange(agents, device=ids.device).view(1, agents, 1)
        valid_ids = valid_ids & (safe_ids != ego_ids)

        neighbor_positions = self._gather_vectors(positions, safe_ids)
        neighbor_velocities = self._gather_vectors(velocities, safe_ids)
        neighbor_yaws = self._gather_scalars(yaws, safe_ids)

        relative_position_global = neighbor_positions - positions.unsqueeze(2)
        relative_velocity_global = neighbor_velocities - velocities.unsqueeze(2)

        ego_yaw = yaws.unsqueeze(-1)
        cosine = torch.cos(ego_yaw)
        sine = torch.sin(ego_yaw)

        def to_ego_frame(vectors: torch.Tensor) -> torch.Tensor:
            x_global, y_global = vectors.unbind(dim=-1)
            return torch.stack(
                (
                    cosine * x_global + sine * y_global,
                    -sine * x_global + cosine * y_global,
                ),
                dim=-1,
            )

        relative_position = to_ego_frame(relative_position_global)
        relative_velocity = to_ego_frame(relative_velocity_global)

        relative_speed_squared = relative_velocity_global.square().sum(dim=-1)
        closing_product = (
            relative_position_global * relative_velocity_global
        ).sum(dim=-1)
        raw_t_cpa = -closing_product / (
            relative_speed_squared + self.config.cpa_epsilon
        )
        t_cpa = raw_t_cpa.clamp(
            min=0.0,
            max=self.config.prediction_horizon_seconds,
        )
        closest_offset = (
            relative_position_global
            + t_cpa.unsqueeze(-1) * relative_velocity_global
        )
        d_cpa = torch.linalg.vector_norm(closest_offset, dim=-1)
        current_distance = torch.linalg.vector_norm(
            relative_position_global, dim=-1
        )

        within_sensing = current_distance <= self.config.sensing_distance_meters
        within_horizon = (raw_t_cpa >= 0.0) & (
            raw_t_cpa <= self.config.prediction_horizon_seconds
        )
        pair_mask = (
            valid_ids
            & within_sensing
            & within_horizon
            & (d_cpa <= self.config.conflict_distance_meters)
        )

        urgency = torch.exp(
            -t_cpa / self.config.urgency_time_scale_seconds
            - d_cpa / self.config.urgency_distance_scale_meters
        ) * pair_mask.to(dtype=positions.dtype)
        confidence = (
            1.0 - current_distance / self.config.sensing_distance_meters
        ).clamp(0.0, 1.0) * valid_ids.to(dtype=positions.dtype)

        ego_speed = (
            torch.linalg.vector_norm(velocities, dim=-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.config.candidate_count)
        )
        neighbor_speed = torch.linalg.vector_norm(neighbor_velocities, dim=-1)
        relative_yaw = neighbor_yaws - ego_yaw
        pair_features = torch.cat(
            (
                (relative_position / self.config.sensing_distance_meters).clamp(
                    -1.0, 1.0
                ),
                (relative_velocity / (2.0 * self.max_speed)).clamp(-1.0, 1.0),
                (ego_speed / self.max_speed).clamp(0.0, 1.0).unsqueeze(-1),
                (neighbor_speed / self.max_speed).clamp(0.0, 1.0).unsqueeze(-1),
                torch.sin(relative_yaw).unsqueeze(-1),
                torch.cos(relative_yaw).unsqueeze(-1),
                (
                    t_cpa / self.config.prediction_horizon_seconds
                ).unsqueeze(-1),
                (
                    d_cpa / self.config.sensing_distance_meters
                ).clamp(0.0, 1.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        pair_features = pair_features * valid_ids.unsqueeze(-1).to(
            dtype=pair_features.dtype
        )

        if pair_features.shape != (
            environments,
            agents,
            self.config.candidate_count,
            self.config.pair_feature_dim,
        ):
            raise RuntimeError("ConflictGraph produced an invalid feature shape.")

        return ConflictGraphOutput(
            pair_features=pair_features,
            neighbor_ids=ids,
            pair_mask=pair_mask,
            urgency=urgency.clamp(0.0, 1.0),
            confidence=confidence,
        )
