"""Augmented P2 centralized critic with stop-gradient bifurcation state."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class AugmentedCentralCritic(nn.Module):
    """Warm-start from Base value and add a zero-initialized global z adapter."""

    def __init__(
        self,
        *,
        base_critic_net: nn.Module,
        n_agents: int,
        observation_dim: int,
        candidate_count: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if (
            n_agents < 2
            or observation_dim <= 0
            or candidate_count <= 0
            or not hidden_sizes
        ):
            raise ValueError("Invalid augmented critic dimensions.")
        self.base_critic_net = base_critic_net
        input_dim = (
            n_agents * observation_dim
            + n_agents * n_agents
            + n_agents * candidate_count
        )
        layers = []
        previous = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, int(width)), nn.Tanh()))
            previous = int(width)
        final_layer = nn.Linear(previous, n_agents)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.adapter = nn.Sequential(*layers)
        self.n_agents = int(n_agents)
        self.observation_dim = int(observation_dim)
        self.candidate_count = int(candidate_count)

    def forward(
        self,
        observation: torch.Tensor,
        z_dense: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        if observation.shape[-2:] != (self.n_agents, self.observation_dim):
            raise ValueError("Observation has an invalid augmented critic shape.")
        if z_dense.shape[-2:] != (self.n_agents, self.n_agents):
            raise ValueError("z_dense has an invalid augmented critic shape.")
        if pair_mask.shape[-2:] != (self.n_agents, self.candidate_count):
            raise ValueError("pair_mask has an invalid augmented critic shape.")
        base_value = self.base_critic_net(observation)
        leading = observation.shape[:-2]
        global_input = torch.cat(
            (
                observation.reshape(*leading, -1),
                z_dense.detach().reshape(*leading, -1),
                pair_mask.detach()
                .to(dtype=observation.dtype)
                .reshape(*leading, -1),
            ),
            dim=-1,
        )
        correction = self.adapter(global_input).reshape(
            *leading, self.n_agents, 1
        )
        return base_value + correction
