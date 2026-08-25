"""M6 global-ID opinion state for stateful rollout collection."""

from __future__ import annotations

from typing import Optional

import torch


class OpinionStateTracker:
    """Maintain directed ``z_dense[e, ego, neighbor]`` outside the policy.

    Candidate-slot tensors are mapped through global agent IDs. The tracker
    owns rollout state only; it is not a trainable module and is never called
    by PPO loss recomputation.
    """

    def __init__(
        self,
        n_agents: int,
        decay_factor: float,
        zero_threshold: float = 1e-6,
    ) -> None:
        if type(n_agents) is not int or n_agents < 2:
            raise ValueError("n_agents must be an integer >= 2.")
        if not 0.0 <= float(decay_factor) <= 1.0:
            raise ValueError("decay_factor must be in [0, 1].")
        if float(zero_threshold) <= 0.0:
            raise ValueError("zero_threshold must be positive.")
        self.n_agents = n_agents
        self.decay_factor = float(decay_factor)
        self.zero_threshold = float(zero_threshold)
        self.z_dense: Optional[torch.Tensor] = None
        self.edge_active: Optional[torch.Tensor] = None
        self._pending_active: Optional[torch.Tensor] = None

    def _ensure_state(self, reference: torch.Tensor) -> None:
        environments = reference.shape[0]
        target_shape = (environments, self.n_agents, self.n_agents)
        if (
            self.z_dense is None
            or self.z_dense.shape != target_shape
            or self.z_dense.device != reference.device
            or self.z_dense.dtype != reference.dtype
        ):
            self.z_dense = torch.zeros(
                target_shape,
                dtype=reference.dtype,
                device=reference.device,
            )
            self.edge_active = torch.zeros(
                target_shape,
                dtype=torch.bool,
                device=reference.device,
            )
            self._pending_active = None

    def _validate_inputs(
        self,
        neighbor_ids: torch.Tensor,
        pair_mask: torch.Tensor,
        agent_reset_mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if neighbor_ids.ndim != 3 or neighbor_ids.shape[1] != self.n_agents:
            raise ValueError("neighbor_ids must have shape [E, N, K].")
        if neighbor_ids.dtype != torch.long:
            raise ValueError("neighbor_ids must have dtype torch.long.")
        if pair_mask.shape != neighbor_ids.shape or pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool with shape [E, N, K].")
        expected_reset_shape = neighbor_ids.shape[:2]
        if (
            agent_reset_mask.shape != expected_reset_shape
            or agent_reset_mask.dtype != torch.bool
        ):
            raise ValueError(
                "agent_reset_mask must be bool with shape [E, N] after "
                "normalizing an optional trailing singleton dimension."
            )
        if reference.shape != neighbor_ids.shape:
            raise ValueError("reference must have shape [E, N, K].")

    def _clear_resets(
        self,
        agent_reset_mask: torch.Tensor,
        environment_done: Optional[torch.Tensor],
    ) -> None:
        assert self.z_dense is not None and self.edge_active is not None
        reset_pairs = agent_reset_mask.unsqueeze(-1) | agent_reset_mask.unsqueeze(-2)
        self.z_dense.masked_fill_(reset_pairs, 0.0)
        self.edge_active.masked_fill_(reset_pairs, False)
        if environment_done is not None:
            done = environment_done.to(dtype=torch.bool, device=self.z_dense.device)
            if done.ndim == 1:
                done_by_environment = done
            else:
                done_by_environment = done.reshape(done.shape[0], -1).any(dim=-1)
            if done_by_environment.shape != (self.z_dense.shape[0],):
                raise ValueError("environment_done must have leading dimension E.")
            self.z_dense[done_by_environment] = 0.0
            self.edge_active[done_by_environment] = False

    def prepare_step(
        self,
        neighbor_ids: torch.Tensor,
        pair_mask: torch.Tensor,
        agent_reset_mask: torch.Tensor,
        reference: torch.Tensor,
        environment_done: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return candidate-aligned ``z_prev`` before one dynamics update."""

        # VMAS info() returns one scalar per agent, while TorchRL 0.2.1 may
        # preserve that scalar as either [E,N] or [E,N,1] depending on whether
        # this TensorDict comes from reset, fake_tensordict, or a real step.
        # Normalize only the unambiguous singleton representation.
        if (
            agent_reset_mask.ndim == 3
            and agent_reset_mask.shape[-1] == 1
        ):
            agent_reset_mask = agent_reset_mask.squeeze(-1)
        self._validate_inputs(
            neighbor_ids, pair_mask, agent_reset_mask, reference
        )
        self._ensure_state(reference)
        assert self.z_dense is not None and self.edge_active is not None
        self._clear_resets(agent_reset_mask, environment_done)

        environments, agents, _ = neighbor_ids.shape
        ego_ids = torch.arange(
            agents, device=neighbor_ids.device
        ).view(1, agents, 1)
        ids_in_range = (neighbor_ids >= 0) & (neighbor_ids < self.n_agents)
        safe_ids = torch.where(ids_in_range, neighbor_ids, ego_ids)
        valid_ids = ids_in_range & (safe_ids != ego_ids)

        candidate_seen = torch.zeros_like(self.edge_active)
        candidate_seen.scatter_(2, safe_ids, valid_ids)
        # Pairs outside the current K candidates still decay exactly once.
        self.z_dense.mul_(
            torch.where(
                candidate_seen,
                torch.ones_like(self.z_dense),
                torch.full_like(self.z_dense, self.decay_factor),
            )
        )

        z_prev = torch.gather(self.z_dense, 2, safe_ids)
        was_active = torch.gather(self.edge_active, 2, safe_ids)
        current_active_slots = valid_ids & pair_mask
        new_edges = current_active_slots & ~was_active
        z_prev = torch.where(
            valid_ids & ~new_edges,
            z_prev,
            torch.zeros_like(z_prev),
        )

        pending_active = torch.zeros_like(self.edge_active)
        pending_active.scatter_(2, safe_ids, current_active_slots)
        self._pending_active = pending_active
        return z_prev.detach()

    def commit_step(
        self,
        neighbor_ids: torch.Tensor,
        z_next: torch.Tensor,
    ) -> None:
        """Scatter one already-integrated candidate state back by global ID."""

        if self.z_dense is None or self.edge_active is None:
            raise RuntimeError("prepare_step must be called before commit_step.")
        if self._pending_active is None:
            raise RuntimeError("No pending opinion step is available to commit.")
        if neighbor_ids.shape != z_next.shape:
            raise ValueError("neighbor_ids and z_next must have identical shapes.")
        ego_ids = torch.arange(
            self.n_agents, device=neighbor_ids.device
        ).view(1, self.n_agents, 1)
        ids_in_range = (neighbor_ids >= 0) & (neighbor_ids < self.n_agents)
        safe_ids = torch.where(ids_in_range, neighbor_ids, ego_ids)
        valid_ids = ids_in_range & (safe_ids != ego_ids)
        values = torch.where(valid_ids, z_next.detach(), torch.zeros_like(z_next))
        self.z_dense.scatter_(2, safe_ids, values)
        self.edge_active.copy_(self._pending_active)

        diagonal = torch.arange(self.n_agents, device=self.z_dense.device)
        self.z_dense[:, diagonal, diagonal] = 0.0
        self.edge_active[:, diagonal, diagonal] = False
        self.z_dense.masked_fill_(
            self.z_dense.abs() < self.zero_threshold,
            0.0,
        )
        self._pending_active = None

    def reset_all(self) -> None:
        if self.z_dense is not None:
            self.z_dense.zero_()
        if self.edge_active is not None:
            self.edge_active.zero_()
        self._pending_active = None

    def snapshot(self) -> dict:
        return {
            "z_dense": (
                None if self.z_dense is None else self.z_dense.detach().cpu().clone()
            ),
            "edge_active": (
                None
                if self.edge_active is None
                else self.edge_active.detach().cpu().clone()
            ),
        }
