"""M8 differentiable truncated sequence PPO for Opinion-MARL."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from utilities.opinion.sequence_buffer import SequenceMiniBatch


class OpinionSequencePPOLoss(nn.Module):
    """Recompute a stateful Opinion policy over contiguous rollout chunks.

    Base Actor and instantaneous EvidenceNet evaluations are vectorized over
    ``[chunk,time]``.  Only the inexpensive opinion state transition loops over
    time, preserving the gradient path from later actions back to earlier
    evidence inside each truncated chunk.
    """

    def __init__(
        self,
        *,
        actor,
        bridge: nn.Module,
        observation_key,
        action_key,
        advantage_key,
        n_agents: int,
        clip_epsilon: float,
        entropy_coefficient: float,
        neutral_loss_coefficient: float,
        magnitude_loss_coefficient: float,
        decay_factor: float,
        zero_threshold: float,
    ) -> None:
        super().__init__()
        if n_agents < 2:
            raise ValueError("n_agents must be >= 2.")
        if not 0.0 < float(clip_epsilon) < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1).")
        if float(entropy_coefficient) < 0.0:
            raise ValueError("entropy_coefficient must be non-negative.")
        if float(neutral_loss_coefficient) < 0.0:
            raise ValueError("neutral_loss_coefficient must be non-negative.")
        if float(magnitude_loss_coefficient) < 0.0:
            raise ValueError("magnitude_loss_coefficient must be non-negative.")
        if not 0.0 <= float(decay_factor) <= 1.0:
            raise ValueError("decay_factor must be in [0, 1].")
        if float(zero_threshold) <= 0.0:
            raise ValueError("zero_threshold must be positive.")
        self.actor = actor
        self.bridge = bridge
        self.observation_key = observation_key
        self.action_key = action_key
        self.advantage_key = advantage_key
        self.n_agents = int(n_agents)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coefficient = float(entropy_coefficient)
        self.neutral_loss_coefficient = float(neutral_loss_coefficient)
        self.magnitude_loss_coefficient = float(magnitude_loss_coefficient)
        self.decay_factor = float(decay_factor)
        self.zero_threshold = float(zero_threshold)

    @staticmethod
    def _optional(tensordict, key) -> Optional[torch.Tensor]:
        try:
            return tensordict.get(key)
        except KeyError:
            return None

    def _candidate_mapping(
        self,
        z_dense: torch.Tensor,
        edge_active: torch.Tensor,
        neighbor_ids: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, agents, _ = neighbor_ids.shape
        if agents != self.n_agents:
            raise ValueError("neighbor_ids agent dimension does not match n_agents.")
        ego_ids = torch.arange(
            agents, device=neighbor_ids.device, dtype=neighbor_ids.dtype
        ).view(1, agents, 1).expand(batch_size, -1, -1)
        ids_in_range = (neighbor_ids >= 0) & (neighbor_ids < self.n_agents)
        safe_ids = torch.where(ids_in_range, neighbor_ids, ego_ids)
        valid_ids = ids_in_range & (safe_ids != ego_ids)
        z_prev = torch.gather(z_dense, 2, safe_ids)
        was_active = torch.gather(edge_active, 2, safe_ids)
        current_active = valid_ids & pair_mask
        new_edges = current_active & ~was_active
        z_prev = torch.where(
            valid_ids & ~new_edges, z_prev, torch.zeros_like(z_prev)
        )
        pending_active = torch.zeros_like(edge_active).scatter(
            2, safe_ids, current_active
        )
        return z_prev, safe_ids, valid_ids, pending_active

    def _prepare_later_step(
        self,
        z_dense: torch.Tensor,
        edge_active: torch.Tensor,
        neighbor_ids: torch.Tensor,
        agent_reset_mask: torch.Tensor,
        environment_done: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if agent_reset_mask.ndim == 3 and agent_reset_mask.shape[-1] == 1:
            agent_reset_mask = agent_reset_mask.squeeze(-1)
        if agent_reset_mask.shape != z_dense.shape[:2]:
            raise ValueError("agent_reset_mask must have shape [B, N].")
        reset = agent_reset_mask.to(dtype=torch.bool)
        reset_pairs = reset.unsqueeze(-1) | reset.unsqueeze(-2)
        z_dense = torch.where(reset_pairs, torch.zeros_like(z_dense), z_dense)
        edge_active = edge_active & ~reset_pairs

        if environment_done is not None:
            done = environment_done.to(dtype=torch.bool)
            done = done.reshape(done.shape[0], -1).any(dim=-1)
            z_dense = torch.where(
                done.view(-1, 1, 1), torch.zeros_like(z_dense), z_dense
            )
            edge_active = edge_active & ~done.view(-1, 1, 1)

        batch_size, agents, _ = neighbor_ids.shape
        ego_ids = torch.arange(
            agents, device=neighbor_ids.device, dtype=neighbor_ids.dtype
        ).view(1, agents, 1).expand(batch_size, -1, -1)
        ids_in_range = (neighbor_ids >= 0) & (neighbor_ids < self.n_agents)
        safe_ids = torch.where(ids_in_range, neighbor_ids, ego_ids)
        valid_ids = ids_in_range & (safe_ids != ego_ids)
        candidate_seen = torch.zeros_like(edge_active).scatter(
            2, safe_ids, valid_ids
        )
        decay = torch.where(
            candidate_seen,
            torch.ones_like(z_dense),
            torch.full_like(z_dense, self.decay_factor),
        )
        return z_dense * decay, edge_active

    def _commit(
        self,
        z_dense: torch.Tensor,
        safe_ids: torch.Tensor,
        valid_ids: torch.Tensor,
        z_next: torch.Tensor,
        pending_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = torch.where(valid_ids, z_next, torch.zeros_like(z_next))
        z_dense = z_dense.scatter(2, safe_ids, values)
        diagonal = torch.arange(self.n_agents, device=z_dense.device)
        diagonal_mask = torch.eye(
            self.n_agents, dtype=torch.bool, device=z_dense.device
        ).unsqueeze(0)
        z_dense = torch.where(diagonal_mask, torch.zeros_like(z_dense), z_dense)
        z_dense = torch.where(
            z_dense.abs() < self.zero_threshold,
            torch.zeros_like(z_dense),
            z_dense,
        )
        edge_active = pending_active.clone()
        edge_active[:, diagonal, diagonal] = False
        return z_dense, edge_active

    def unroll(self, mini_batch: SequenceMiniBatch) -> Dict[str, torch.Tensor]:
        """Return differentiably recomputed policy tensors for one chunk batch."""

        td = mini_batch.tensordict
        if len(td.batch_size) != 2:
            raise ValueError("M8 sequence mini-batch must have [chunk,time] shape.")
        observations = td.get(self.observation_key)
        pair_features = td.get(("agents", "info", "pair_features"))
        urgency = td.get(("agents", "info", "urgency"))
        confidence = td.get(("agents", "info", "confidence"))
        pair_mask = td.get(("agents", "info", "pair_mask")).to(torch.bool)
        neighbor_ids = td.get(("agents", "info", "neighbor_ids")).to(torch.long)
        reset_mask = td.get(("agents", "info", "agent_reset_mask"))
        environment_done = self._optional(td, "done")

        base_loc, base_scale = self.bridge.base_policy_net(observations)
        evidence = self.bridge.evidence_net(
            pair_features=pair_features,
            urgency=urgency,
            confidence=confidence,
            pair_mask=pair_mask,
        )

        z_dense = mini_batch.z_init.detach()
        edge_active = mini_batch.edge_active_init.detach().to(torch.bool)
        z_prev_steps = []
        z_next_steps = []
        residual_steps = []
        final_loc_steps = []
        time_steps = int(td.batch_size[1])
        for time_index in range(time_steps):
            ids_t = neighbor_ids[:, time_index]
            mask_t = pair_mask[:, time_index]
            if time_index > 0:
                done_t = (
                    None
                    if environment_done is None
                    else environment_done[:, time_index]
                )
                z_dense, edge_active = self._prepare_later_step(
                    z_dense=z_dense,
                    edge_active=edge_active,
                    neighbor_ids=ids_t,
                    agent_reset_mask=reset_mask[:, time_index],
                    environment_done=done_t,
                )
            z_prev, safe_ids, valid_ids, pending_active = self._candidate_mapping(
                z_dense=z_dense,
                edge_active=edge_active,
                neighbor_ids=ids_t,
                pair_mask=mask_t,
            )
            z_next = self.bridge.dynamics(
                z_prev=z_prev,
                evidence=evidence.b[:, time_index],
                urgency=urgency[:, time_index],
                pair_mask=mask_t,
                dt=self.bridge.dt,
            )
            residual = self.bridge.residual(
                z=z_next,
                urgency=urgency[:, time_index],
                pair_mask=mask_t,
            ).residual
            final_loc = self.bridge.residual.apply_to_loc(
                base_loc=base_loc[:, time_index], residual=residual
            )
            z_dense, edge_active = self._commit(
                z_dense=z_dense,
                safe_ids=safe_ids,
                valid_ids=valid_ids,
                z_next=z_next,
                pending_active=pending_active,
            )
            z_prev_steps.append(z_prev)
            z_next_steps.append(z_next)
            residual_steps.append(residual)
            final_loc_steps.append(final_loc)

        return {
            "loc": torch.stack(final_loc_steps, dim=1),
            "scale": base_scale,
            "raw_b": evidence.raw_b,
            "b": evidence.b,
            "z_prev": torch.stack(z_prev_steps, dim=1),
            "z_next": torch.stack(z_next_steps, dim=1),
            "residual": torch.stack(residual_steps, dim=1),
        }

    @staticmethod
    def _align_log_prob(
        value: torch.Tensor, reference: torch.Tensor, name: str
    ) -> torch.Tensor:
        if value.shape == reference.shape:
            return value
        if value.shape == reference.shape + (1,):
            return value.squeeze(-1)
        raise ValueError(
            f"{name} shape {tuple(value.shape)} does not match log-prob "
            f"shape {tuple(reference.shape)}."
        )

    def forward(self, mini_batch: SequenceMiniBatch) -> Dict[str, torch.Tensor]:
        td = mini_batch.tensordict
        recomputed = self.unroll(mini_batch)
        distribution_td = td.clone(False)
        distribution_td.set(("agents", "loc"), recomputed["loc"])
        distribution_td.set(("agents", "scale"), recomputed["scale"])
        distribution = self.actor.build_dist_from_params(distribution_td)
        action = td.get(self.action_key)
        if action.requires_grad:
            raise RuntimeError("Stored rollout action must be detached.")
        current_log_prob = distribution.log_prob(action)
        old_log_prob = self._align_log_prob(
            td.get(("agents", "sample_log_prob")),
            current_log_prob,
            "old log-prob",
        )
        if old_log_prob.requires_grad:
            raise RuntimeError("Stored rollout log-prob must be detached.")
        advantage = self._align_log_prob(
            td.get(self.advantage_key), current_log_prob, "advantage"
        ).unsqueeze(-1)

        log_ratio = (current_log_prob - old_log_prob).unsqueeze(-1)
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(
            1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        gain = torch.minimum(ratio * advantage, clipped_ratio * advantage)
        loss_objective = -gain.mean()
        try:
            entropy = distribution.entropy()
        except NotImplementedError:
            entropy_sample = distribution.rsample()
            entropy = -distribution.log_prob(entropy_sample)
        loss_entropy = -self.entropy_coefficient * entropy.mean()

        active = td.get(("agents", "info", "pair_mask")).to(
            dtype=recomputed["b"].dtype
        )
        active_count = active.sum().clamp_min(1.0)
        neutral = (
            (1.0 - td.get(("agents", "info", "urgency")).clamp(0.0, 1.0))
            * recomputed["b"].square()
            * active
        ).sum() / active_count
        magnitude = (recomputed["b"].square() * active).sum() / active_count
        loss_regularization = (
            self.neutral_loss_coefficient * neutral
            + self.magnitude_loss_coefficient * magnitude
        )

        with torch.no_grad():
            collected_z_next = td.get(("agents", "opinion", "z_next"))
            state_replay_error = (
                recomputed["z_next"] - collected_z_next
            ).abs().mean()
            approx_kl = (old_log_prob - current_log_prob).mean()
            clip_fraction = (
                (ratio < 1.0 - self.clip_epsilon)
                | (ratio > 1.0 + self.clip_epsilon)
            ).to(torch.float32).mean()

        return {
            "loss_objective": loss_objective,
            "loss_entropy": loss_entropy,
            "loss_regularization": loss_regularization,
            "neutral_penalty": neutral.detach(),
            "magnitude_penalty": magnitude.detach(),
            "entropy": entropy.mean().detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "log_prob_abs_error": (current_log_prob - old_log_prob)
            .abs()
            .mean()
            .detach(),
            "state_replay_abs_error": state_replay_error.detach(),
        }
