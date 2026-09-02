"""Permutation-equivariant Base-relative vector critic for P3.1."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


TARGET_CHANNELS = (
    "augmented_reward_return_delta",
    "vehicle_conflict_risk_return_delta",
    "lane_margin_violation_return_delta",
)


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers = []
    previous = int(input_dim)
    for width in hidden_sizes:
        width = int(width)
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class BaseRelativeDifferentialCritic(nn.Module):
    """Graph critic for paired reward and dense safety-return differences.

    The same node and edge maps are shared by every agent. Mean aggregation
    makes the critic equivariant to agent relabeling and independent of the
    number of agents used during critic calibration.
    """

    def __init__(
        self,
        *,
        observation_dim: int,
        embedding_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or embedding_dim <= 0 or not hidden_sizes:
            raise ValueError("Invalid P3.1 differential critic dimensions.")
        self.node_encoder = _mlp(
            2 * observation_dim,
            hidden_sizes,
            embedding_dim,
        )
        self.edge_encoder = _mlp(
            2 * embedding_dim + 2,
            hidden_sizes,
            embedding_dim,
        )
        self.head = _mlp(
            3 * embedding_dim,
            hidden_sizes,
            len(TARGET_CHANNELS),
        )
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer(
            "target_center", torch.zeros(len(TARGET_CHANNELS))
        )
        self.register_buffer(
            "target_scale", torch.ones(len(TARGET_CHANNELS))
        )
        self.observation_dim = int(observation_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)

    def set_target_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        expected = (len(TARGET_CHANNELS),)
        if center.shape != expected or scale.shape != expected:
            raise ValueError("P3.1 target normalization has an invalid shape.")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("P3.1 target normalization must be finite.")
        if bool((scale <= 0).any()):
            raise ValueError("P3.1 target scales must be positive.")
        self.target_center.copy_(center.to(self.target_center))
        self.target_scale.copy_(scale.to(self.target_scale))

    def model_config(self) -> dict[str, object]:
        return {
            "observation_dim": self.observation_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "target_channels": list(TARGET_CHANNELS),
        }

    def forward(
        self,
        candidate_observation: torch.Tensor,
        base_observation: torch.Tensor,
        candidate_z: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_observation.shape != base_observation.shape:
            raise ValueError("Candidate/Base critic observations must align.")
        if candidate_observation.shape[-1] != self.observation_dim:
            raise ValueError("P3.1 critic observation dimension does not match.")
        if candidate_observation.ndim < 3:
            raise ValueError("P3.1 critic observations require [...,N,O].")
        n_agents = int(candidate_observation.shape[-2])
        if candidate_z.shape != candidate_observation.shape[:-1] + (n_agents,):
            raise ValueError("candidate_z must have shape [...,N,N].")
        if edge_mask.shape != candidate_z.shape or edge_mask.dtype != torch.bool:
            raise ValueError("edge_mask must be bool with shape [...,N,N].")

        # Critic regression fits value parameters only. It must never alter
        # physical observations or the proximal bifurcation state.
        candidate_observation = candidate_observation.detach()
        base_observation = base_observation.detach()
        candidate_z = candidate_z.detach()
        edge_mask = edge_mask.detach()
        node = self.node_encoder(
            torch.cat((candidate_observation, base_observation), dim=-1)
        )
        source = node.unsqueeze(-2).expand(*node.shape[:-2], n_agents, n_agents, -1)
        target = node.unsqueeze(-3).expand(*node.shape[:-2], n_agents, n_agents, -1)
        edge_input = torch.cat(
            (
                source,
                target,
                candidate_z.unsqueeze(-1),
                candidate_z.abs().unsqueeze(-1),
            ),
            dim=-1,
        )
        messages = self.edge_encoder(edge_input)
        messages = messages * edge_mask.unsqueeze(-1).to(messages.dtype)
        degree = edge_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        aggregate = messages.sum(dim=-2) / degree.to(messages.dtype)
        global_context = (node + aggregate).mean(dim=-2, keepdim=True)
        global_context = global_context.expand_as(node)
        normalized_prediction = self.head(
            torch.cat((node, aggregate, global_context), dim=-1)
        )
        shape = [1] * (normalized_prediction.ndim - 1) + [
            len(TARGET_CHANNELS)
        ]
        return self.target_center.view(*shape) + self.target_scale.view(
            *shape
        ) * normalized_prediction
