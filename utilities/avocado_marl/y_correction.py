"""Bounded correction of AVOCADO's heuristic cooperation estimate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from utilities.avocado.controller import AVOCADOController
from utilities.avocado.core import (
    attention_reference_step,
    collision_time,
    projection_estimator,
)


Y_CORRECTION_FEATURE_DIM = 14


@dataclass(frozen=True)
class YCorrectionFeatures:
    values: Tensor
    pair_mask: Tensor
    confidence: Tensor
    heuristic_estimate: Tensor
    prospective_attention: Tensor


@dataclass(frozen=True)
class YCorrectionOutput:
    logit: Tensor
    raw_correction: Tensor
    correction: Tensor


def build_y_correction_features(
    controller: AVOCADOController,
    positions: Tensor,
    velocities: Tensor,
    yaws: Tensor,
    *,
    candidate_count: int,
) -> YCorrectionFeatures:
    """Build the local 14-D pair contract without reading opinion state ``z``."""

    if positions.shape != velocities.shape or positions.shape[-1:] != (2,):
        raise ValueError("positions and velocities must have shape [B, N, 2].")
    if yaws.shape == positions.shape[:-1] + (1,):
        yaws = yaws.squeeze(-1)
    if yaws.shape != positions.shape[:-1]:
        raise ValueError("yaws must have shape [B, N] or [B, N, 1].")
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("candidate_count must be a positive integer.")
    batch_size, entity_count, _ = positions.shape
    if (batch_size, entity_count) != (
        controller.batch_size,
        controller.entity_count,
    ):
        raise ValueError("state shape does not match the AVOCADO controller.")

    relative_position = positions[:, None, :, :] - positions[:, :, None, :]
    relative_velocity = velocities[:, None, :, :] - velocities[:, :, None, :]
    pair_distance = torch.linalg.vector_norm(relative_position, dim=-1)
    identity = torch.eye(
        entity_count, dtype=torch.bool, device=positions.device
    ).unsqueeze(0)
    perceived = (
        (pair_distance < controller.parameters.perception_radius) & ~identity
    )
    perceived &= controller.controlled_mask[None, :, None]

    selected = torch.zeros_like(perceived)
    nearest_count = min(candidate_count, entity_count - 1)
    masked_distance = pair_distance.masked_fill(~perceived, torch.inf)
    nearest_ids = torch.topk(
        masked_distance, k=nearest_count, dim=-1, largest=False
    ).indices
    selected.scatter_(-1, nearest_ids, True)
    pair_mask = selected & perceived

    combined_radius = (
        controller.security_radii[:, None]
        + controller.security_radii[None, :]
    )
    time_to_collision = collision_time(
        positions[:, :, None, :],
        velocities[:, :, None, :],
        positions[:, None, :, :],
        velocities[:, None, :, :],
        combined_radius[None, :, :],
    )
    prospective_attention = attention_reference_step(
        controller.attention,
        time_to_collision,
        delta=controller.parameters.attention_decay,
        kappa=controller.parameters.kappa,
    )
    delta_neighbor_velocity = (
        velocities[:, None, :, :]
        - controller.previous_observed_velocity[:, None, :, :]
    )
    heuristic_estimate = projection_estimator(
        delta_neighbor_velocity,
        controller.previous_correction,
        controller.parameters.epsilon,
    )

    relative_speed_squared = relative_velocity.square().sum(dim=-1)
    raw_t_cpa = -(
        relative_position * relative_velocity
    ).sum(dim=-1) / relative_speed_squared.clamp_min(1e-8)
    t_cpa = raw_t_cpa.clamp(
        min=0.0,
        max=controller.parameters.velocity_obstacle_horizon,
    )
    closest_offset = relative_position + t_cpa.unsqueeze(-1) * relative_velocity
    d_cpa = torch.linalg.vector_norm(closest_offset, dim=-1)

    cosine = torch.cos(yaws)[:, :, None]
    sine = torch.sin(yaws)[:, :, None]

    def to_ego_frame(vectors: Tensor) -> Tensor:
        x_global, y_global = vectors.unbind(dim=-1)
        return torch.stack(
            (
                cosine * x_global + sine * y_global,
                -sine * x_global + cosine * y_global,
            ),
            dim=-1,
        )

    relative_position_ego = to_ego_frame(relative_position)
    relative_velocity_ego = to_ego_frame(relative_velocity)
    speeds = torch.linalg.vector_norm(velocities, dim=-1)
    ego_speed = speeds[:, :, None].expand(-1, -1, entity_count)
    neighbor_speed = speeds[:, None, :].expand(-1, entity_count, -1)
    relative_yaw = yaws[:, None, :] - yaws[:, :, None]
    maximum_speed = float(controller.maximum_speeds.max())
    sensing_distance = float(controller.parameters.perception_radius)
    horizon = float(controller.parameters.velocity_obstacle_horizon)
    normalized_ttc = torch.where(
        torch.isfinite(time_to_collision),
        time_to_collision.clamp(0.0, horizon) / horizon,
        torch.ones_like(time_to_collision),
    )
    confidence = (1.0 - pair_distance / sensing_distance).clamp(0.0, 1.0)
    correction_norm = torch.linalg.vector_norm(
        controller.previous_correction, dim=-1
    )
    features = torch.cat(
        (
            (relative_position_ego / sensing_distance).clamp(-1.0, 1.0),
            (relative_velocity_ego / (2.0 * maximum_speed)).clamp(-1.0, 1.0),
            (ego_speed / maximum_speed).clamp(0.0, 1.0).unsqueeze(-1),
            (neighbor_speed / maximum_speed).clamp(0.0, 1.0).unsqueeze(-1),
            torch.sin(relative_yaw).unsqueeze(-1),
            torch.cos(relative_yaw).unsqueeze(-1),
            normalized_ttc.unsqueeze(-1),
            (d_cpa / sensing_distance).clamp(0.0, 1.0).unsqueeze(-1),
            prospective_attention.clamp(0.0, 1.0).unsqueeze(-1),
            heuristic_estimate.clamp(-1.0, 1.0).unsqueeze(-1),
            (correction_norm / maximum_speed).clamp(0.0, 1.0).unsqueeze(-1),
            pair_mask.to(dtype=positions.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    if features.shape != (
        batch_size,
        entity_count,
        entity_count,
        Y_CORRECTION_FEATURE_DIM,
    ):
        raise RuntimeError("YCorrectionNet feature contract has an invalid shape.")
    features = features * pair_mask.unsqueeze(-1).to(features.dtype)
    return YCorrectionFeatures(
        values=features,
        pair_mask=pair_mask,
        confidence=confidence * pair_mask.to(confidence.dtype),
        heuristic_estimate=heuristic_estimate,
        prospective_attention=prospective_attention,
    )


class YCorrectionNet(nn.Module):
    """Shared bounded pair network; it deliberately has no ``z`` input."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_sizes: Sequence[int],
        maximum_correction: float,
        temperature: float,
        strict_zero: bool,
        freeze: bool,
    ) -> None:
        super().__init__()
        if feature_dim != Y_CORRECTION_FEATURE_DIM:
            raise ValueError(
                f"YCorrectionNet requires feature_dim={Y_CORRECTION_FEATURE_DIM}."
            )
        if not hidden_sizes or any(
            type(size) is not int or size <= 0 for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers.")
        if not 0 < maximum_correction <= 0.5:
            raise ValueError("maximum_correction must be in (0, 0.5].")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        layers = []
        input_dim = feature_dim
        for hidden_dim in hidden_sizes:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        final_layer = nn.Linear(input_dim, 1)
        if strict_zero:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        else:
            nn.init.xavier_uniform_(final_layer.weight, gain=1e-2)
            nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)
        self.feature_dim = feature_dim
        self.maximum_correction = float(maximum_correction)
        self.temperature = float(temperature)
        self.strict_zero = bool(strict_zero)
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        features: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
    ) -> YCorrectionOutput:
        if features.shape[:-1] != pair_mask.shape:
            raise ValueError("features and pair_mask axes must match.")
        if features.shape[-1] != self.feature_dim:
            raise ValueError("Invalid YCorrectionNet feature dimension.")
        if confidence.shape != pair_mask.shape or pair_mask.dtype != torch.bool:
            raise ValueError("confidence and bool pair_mask must have equal shape.")
        logit = self.network(features).squeeze(-1)
        raw = torch.tanh(logit / self.temperature)
        correction = (
            self.maximum_correction
            * confidence.clamp(0.0, 1.0)
            * pair_mask.to(features.dtype)
            * raw
        )
        return YCorrectionOutput(logit, raw, correction)
