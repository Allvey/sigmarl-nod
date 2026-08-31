"""P1 fixed-universe antisymmetric bifurcation state."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from utilities.psb_marl.proximal import ProximalSaturatingBifurcation


class P1StateStep(NamedTuple):
    z_prev_dense: torch.Tensor
    z_next_dense: torch.Tensor
    rho_dense: torch.Tensor
    b_dense: torch.Tensor
    residual_dense: torch.Tensor
    denominator_dense: torch.Tensor
    z_next_candidates: torch.Tensor


class P1ZeroControlStateTracker:
    """Maintain one oriented state per unordered pair; P1 never changes action."""

    def __init__(
        self,
        *,
        n_agents: int,
        proximal: ProximalSaturatingBifurcation,
        zero_threshold: float,
    ) -> None:
        if type(n_agents) is not int or n_agents < 2:
            raise ValueError("n_agents must be an integer greater than one.")
        if zero_threshold <= 0.0:
            raise ValueError("zero_threshold must be positive.")
        if proximal.b_max != 0.0:
            raise ValueError("The P1 state tracker requires b_max=0.")
        self.n_agents = n_agents
        self.proximal = proximal
        self.zero_threshold = float(zero_threshold)
        self.z_dense: Optional[torch.Tensor] = None

    def _ensure_state(self, reference: torch.Tensor) -> None:
        shape = (reference.shape[0], self.n_agents, self.n_agents)
        if (
            self.z_dense is None
            or self.z_dense.shape != shape
            or self.z_dense.device != reference.device
            or self.z_dense.dtype != reference.dtype
        ):
            self.z_dense = torch.zeros(
                shape, dtype=reference.dtype, device=reference.device
            )

    @staticmethod
    def _normalize_reset_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        return mask

    def _validate_inputs(
        self,
        neighbor_ids: torch.Tensor,
        pair_mask: torch.Tensor,
        urgency: torch.Tensor,
        agent_reset_mask: torch.Tensor,
    ) -> None:
        if neighbor_ids.ndim != 3 or neighbor_ids.shape[1] != self.n_agents:
            raise ValueError("neighbor_ids must have shape [E, N, K].")
        if neighbor_ids.dtype != torch.long:
            raise ValueError("neighbor_ids must have dtype torch.long.")
        if pair_mask.shape != neighbor_ids.shape or pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool with shape [E, N, K].")
        if urgency.shape != neighbor_ids.shape or not urgency.is_floating_point():
            raise ValueError("urgency must be floating with shape [E, N, K].")
        if agent_reset_mask.shape != neighbor_ids.shape[:2]:
            raise ValueError("agent_reset_mask must have shape [E, N].")
        if agent_reset_mask.dtype != torch.bool:
            raise ValueError("agent_reset_mask must have dtype torch.bool.")

    def _clear_resets(
        self,
        agent_reset_mask: torch.Tensor,
        environment_done: Optional[torch.Tensor],
    ) -> None:
        assert self.z_dense is not None
        pair_reset = agent_reset_mask.unsqueeze(-1) | agent_reset_mask.unsqueeze(-2)
        self.z_dense.masked_fill_(pair_reset, 0.0)
        if environment_done is not None:
            done = environment_done.to(dtype=torch.bool, device=self.z_dense.device)
            done = done.reshape(done.shape[0], -1).any(dim=-1)
            if done.shape != (self.z_dense.shape[0],):
                raise ValueError("environment_done must have leading dimension E.")
            self.z_dense[done] = 0.0

    @torch.no_grad()
    def step(
        self,
        *,
        neighbor_ids: torch.Tensor,
        pair_mask: torch.Tensor,
        urgency: torch.Tensor,
        agent_reset_mask: torch.Tensor,
        environment_done: Optional[torch.Tensor] = None,
    ) -> P1StateStep:
        agent_reset_mask = self._normalize_reset_mask(agent_reset_mask)
        self._validate_inputs(neighbor_ids, pair_mask, urgency, agent_reset_mask)
        self._ensure_state(urgency)
        assert self.z_dense is not None
        self._clear_resets(agent_reset_mask, environment_done)

        environments, agents, candidates = neighbor_ids.shape
        ego_ids = torch.arange(agents, device=neighbor_ids.device).view(1, agents, 1)
        valid = (neighbor_ids >= 0) & (neighbor_ids < agents)
        safe_ids = torch.where(valid, neighbor_ids, ego_ids)
        valid = valid & (safe_ids != ego_ids) & pair_mask
        rho_candidates = (
            self.proximal.rho_max
            * urgency.clamp(0.0, 1.0)
            * valid.to(dtype=urgency.dtype)
        )
        rho_directed = torch.zeros_like(self.z_dense)
        rho_directed.scatter_(2, safe_ids, rho_candidates)
        rho_dense = torch.maximum(rho_directed, rho_directed.transpose(-1, -2))
        diagonal = torch.arange(agents, device=neighbor_ids.device)
        rho_dense[:, diagonal, diagonal] = 0.0

        z_prev_dense = self.z_dense.clone()
        b_dense = torch.zeros_like(z_prev_dense)
        result = self.proximal.solve_with_diagnostics(
            z_prev_dense, rho_dense, b_dense
        )
        upper = torch.triu(result.z_next, diagonal=1)
        z_next_dense = upper - upper.transpose(-1, -2)
        z_next_dense.masked_fill_(z_next_dense.abs() < self.zero_threshold, 0.0)
        residual_dense = (
            (z_next_dense - z_prev_dense) / self.proximal.h_z
            + self.proximal.kappa * z_next_dense
            - rho_dense
            * self.proximal.nu
            * torch.tanh(self.proximal.alpha * z_next_dense)
        )
        tanh_value = torch.tanh(self.proximal.alpha * z_next_dense)
        denominator_dense = (
            1.0 / self.proximal.h_z
            + self.proximal.kappa
            - rho_dense
            * self.proximal.nu
            * self.proximal.alpha
            * (1.0 - tanh_value.square())
        )
        self.z_dense.copy_(z_next_dense)
        z_next_candidates = torch.gather(z_next_dense, 2, safe_ids)
        z_next_candidates = torch.where(
            valid, z_next_candidates, torch.zeros_like(z_next_candidates)
        )
        return P1StateStep(
            z_prev_dense=z_prev_dense,
            z_next_dense=z_next_dense,
            rho_dense=rho_dense,
            b_dense=b_dense,
            residual_dense=residual_dense,
            denominator_dense=denominator_dense,
            z_next_candidates=z_next_candidates,
        )

    def reset_all(self) -> None:
        if self.z_dense is not None:
            self.z_dense.zero_()

    def set_state_for_testing(self, state: torch.Tensor) -> None:
        if state.ndim != 3 or state.shape[1:] != (self.n_agents, self.n_agents):
            raise ValueError("state must have shape [E, N, N].")
        if not torch.allclose(state, -state.transpose(-1, -2)):
            raise ValueError("state must be antisymmetric.")
        self.z_dense = state.detach().clone()
