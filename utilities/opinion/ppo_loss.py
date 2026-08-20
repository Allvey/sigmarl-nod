"""Contiguous-chunk PPO replay for recurrent opinion state."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from utilities.opinion.collector import (
    apply_opinion_resets,
    decay_dense_opinions,
)
from utilities.opinion.dynamics import (
    gather_candidate_opinions,
    scatter_candidate_opinions,
)
from utilities.opinion.policy import OpinionTanhNormalPolicy
from utilities.opinion.sequence_buffer import SequenceChunk


class OpinionCentralizedCritic(nn.Module):
    """Training-only value network over all agents, optionally detached z."""

    def __init__(
        self,
        *,
        observation_dim: int,
        n_agents: int,
        hidden_dim: int = 256,
        include_z: bool = False,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.n_agents = n_agents
        self.include_z = include_z
        input_dim = n_agents * observation_dim + (n_agents * n_agents if include_z else 0)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_agents),
        )

    def forward(self, observation: Tensor, z_dense: Tensor = None) -> Tensor:
        if observation.shape[-2:] != (self.n_agents, self.observation_dim):
            raise ValueError("critic observation must end with [N, observation_dim]")
        inputs = [observation.reshape(*observation.shape[:-2], -1)]
        if self.include_z:
            if z_dense is None or z_dense.shape[-2:] != (
                self.n_agents,
                self.n_agents,
            ):
                raise ValueError("critic z_dense must end with [N, N]")
            inputs.append(z_dense.detach().reshape(*z_dense.shape[:-2], -1))
        return self.network(torch.cat(inputs, dim=-1))


class SequencePPOOutput(NamedTuple):
    total_loss: Tensor
    actor_loss: Tensor
    critic_loss: Tensor
    ppo_loss: Tensor
    entropy_estimate: Tensor
    neutral_loss: Tensor
    magnitude_loss: Tensor
    new_log_prob: Tensor
    values: Tensor
    final_z_dense: Tensor


class OpinionSequencePPOLoss(nn.Module):
    def __init__(
        self,
        *,
        policy: OpinionTanhNormalPolicy,
        critic: OpinionCentralizedCritic,
        clip_epsilon: float,
        entropy_eps: float,
        neutral_loss_weight: float,
        magnitude_loss_weight: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_eps = float(entropy_eps)
        self.neutral_loss_weight = float(neutral_loss_weight)
        self.magnitude_loss_weight = float(magnitude_loss_weight)

    def forward(
        self, chunk: SequenceChunk, *, residual_scale: float
    ) -> SequencePPOOutput:
        data = chunk.data
        required = {
            "observation",
            "action",
            "old_log_prob",
            "pair_features",
            "neighbor_ids",
            "pair_mask",
            "urgency",
            "confidence",
            "agent_reset_mask",
            "advantage",
            "returns",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"sequence chunk missing fields: {missing}")
        z_dense = chunk.z_init.detach().unsqueeze(0)
        new_log_probs = []
        values = []
        raw_bs = []
        bs = []
        replay_dense = []

        for time_index in range(data["observation"].shape[0]):
            reset_agents = data["agent_reset_mask"][time_index].bool().unsqueeze(0)
            z_dense = apply_opinion_resets(
                z_dense,
                reset_agents,
                torch.zeros(1, dtype=torch.bool, device=z_dense.device),
            )
            dense_before = z_dense
            replay_dense.append(dense_before.squeeze(0))
            mask = data["pair_mask"][time_index].bool().unsqueeze(0)
            ids = data["neighbor_ids"][time_index].long().unsqueeze(0)
            z_prev = gather_candidate_opinions(dense_before, ids, mask)
            policy_output = self.policy(
                data["observation"][time_index].unsqueeze(0),
                data["pair_features"][time_index].unsqueeze(0),
                data["urgency"][time_index].unsqueeze(0),
                data["confidence"][time_index].unsqueeze(0),
                mask,
                z_prev,
                residual_scale=residual_scale,
                action=data["action"][time_index].unsqueeze(0),
            )
            new_log_probs.append(policy_output.log_prob.squeeze(0))
            raw_bs.append(policy_output.core.raw_b.squeeze(0))
            bs.append(policy_output.core.b.squeeze(0))
            decayed = decay_dense_opinions(
                dense_before, self.policy.core.dynamics, dt=self.policy.core.dt
            )
            z_dense = scatter_candidate_opinions(
                decayed, ids, policy_output.core.z_next, mask
            )
            values.append(
                self.critic(
                    data["observation"][time_index].unsqueeze(0),
                    dense_before,
                ).squeeze(0)
            )

        new_log_prob = torch.stack(new_log_probs)
        value_tensor = torch.stack(values)
        raw_b = torch.stack(raw_bs)
        b = torch.stack(bs)
        old_log_prob = data["old_log_prob"]
        advantage = data["advantage"].detach()
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped = ratio * advantage
        clipped = ratio.clamp(
            1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        ) * advantage
        ppo_loss = -torch.minimum(unclipped, clipped).mean()
        entropy_estimate = -new_log_prob.mean()
        mask = data["pair_mask"].bool()
        mask_float = mask.to(dtype=b.dtype)
        denominator = mask_float.sum().clamp_min(1.0)
        neutral_loss = (
            ((1.0 - data["urgency"]) * b.square() * mask_float).sum()
            / denominator
        )
        magnitude_loss = (b.square() * mask_float).sum() / denominator
        actor_loss = (
            ppo_loss
            - self.entropy_eps * entropy_estimate
            + self.neutral_loss_weight * neutral_loss
            + self.magnitude_loss_weight * magnitude_loss
        )
        critic_loss = (value_tensor - data["returns"].detach()).square().mean()
        total_loss = actor_loss + critic_loss
        return SequencePPOOutput(
            total_loss=total_loss,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            ppo_loss=ppo_loss,
            entropy_estimate=entropy_estimate,
            neutral_loss=neutral_loss,
            magnitude_loss=magnitude_loss,
            new_log_prob=new_log_prob,
            values=value_tensor,
            final_z_dense=z_dense,
        )
