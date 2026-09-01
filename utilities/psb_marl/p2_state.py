"""Rollout state ownership for P2 fixed-universe bifurcation memory."""

from __future__ import annotations

from typing import Optional

import torch


class P2EdgeStateTracker:
    def __init__(self, n_agents: int, zero_threshold: float) -> None:
        if type(n_agents) is not int or n_agents < 2:
            raise ValueError("n_agents must be an integer >= 2.")
        if zero_threshold <= 0.0:
            raise ValueError("zero_threshold must be positive.")
        self.n_agents = n_agents
        self.zero_threshold = float(zero_threshold)
        self.z_dense: Optional[torch.Tensor] = None

    def _ensure_state(self, reference: torch.Tensor) -> None:
        shape = (reference.shape[0], self.n_agents, self.n_agents)
        if (
            self.z_dense is None
            or self.z_dense.shape != shape
            or self.z_dense.dtype != reference.dtype
            or self.z_dense.device != reference.device
        ):
            self.z_dense = torch.zeros(
                shape, dtype=reference.dtype, device=reference.device
            )

    @staticmethod
    def apply_resets(
        z_dense: torch.Tensor,
        agent_reset_mask: torch.Tensor,
        environment_done: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            agent_reset_mask.ndim == z_dense.ndim
            and agent_reset_mask.shape[-1] == 1
        ):
            agent_reset_mask = agent_reset_mask.squeeze(-1)
        if agent_reset_mask.shape != z_dense.shape[:-1]:
            raise ValueError(
                "agent_reset_mask must have shape [..., N] matching "
                "z_dense [..., N, N]."
            )
        reset = agent_reset_mask.to(dtype=torch.bool)
        pair_reset = reset.unsqueeze(-1) | reset.unsqueeze(-2)
        result = torch.where(pair_reset, torch.zeros_like(z_dense), z_dense)
        if environment_done is not None:
            done = environment_done.to(dtype=torch.bool)
            leading_shape = z_dense.shape[:-2]
            while done.ndim > len(leading_shape) and done.shape[-1] == 1:
                done = done.squeeze(-1)
            if done.shape != leading_shape:
                if done.numel() != int(torch.tensor(leading_shape).prod().item()):
                    raise ValueError(
                        "environment_done must match the leading z_dense shape."
                    )
                done = done.reshape(leading_shape)
            result = torch.where(
                done[..., None, None], torch.zeros_like(result), result
            )
        return result

    @torch.no_grad()
    def prepare_step(
        self,
        *,
        reference: torch.Tensor,
        agent_reset_mask: torch.Tensor,
        environment_done: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_state(reference)
        assert self.z_dense is not None
        self.z_dense.copy_(
            self.apply_resets(
                self.z_dense, agent_reset_mask, environment_done
            )
        )
        return self.z_dense.detach().clone()

    @torch.no_grad()
    def commit_step(self, z_next_dense: torch.Tensor) -> None:
        if self.z_dense is None:
            raise RuntimeError("prepare_step must run before commit_step.")
        if z_next_dense.shape != self.z_dense.shape:
            raise ValueError("z_next_dense has an invalid shape.")
        self.z_dense.copy_(z_next_dense.detach())
        self.z_dense.masked_fill_(
            self.z_dense.abs() < self.zero_threshold, 0.0
        )

    def reset_all(self) -> None:
        if self.z_dense is not None:
            self.z_dense.zero_()

    def snapshot(self) -> dict:
        return {
            "z_dense": (
                None if self.z_dense is None else self.z_dense.detach().cpu().clone()
            )
        }
