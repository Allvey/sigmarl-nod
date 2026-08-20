"""State owner for directed dense opinions during environment rollout."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor

from utilities.opinion.dynamics import (
    OpinionDynamics,
    gather_candidate_opinions,
    scatter_candidate_opinions,
)
from utilities.opinion.policy import OpinionTanhNormalPolicy


def apply_opinion_resets(
    z_dense: Tensor, agent_reset_mask: Tensor, environment_done: Tensor
) -> Tensor:
    """Clear full environments and each reset agent's incoming/outgoing edges."""
    if z_dense.ndim != 3 or z_dense.shape[-1] != z_dense.shape[-2]:
        raise ValueError("z_dense must have shape [E, N, N]")
    n_envs, n_agents, _ = z_dense.shape
    if agent_reset_mask.shape != (n_envs, n_agents) or agent_reset_mask.dtype is not torch.bool:
        raise ValueError("agent_reset_mask must be bool [E, N]")
    if environment_done.shape != (n_envs,) or environment_done.dtype is not torch.bool:
        raise ValueError("environment_done must be bool [E]")
    keep_agent_pairs = ~(
        agent_reset_mask.unsqueeze(-1) | agent_reset_mask.unsqueeze(-2)
    )
    keep_environments = ~environment_done[:, None, None]
    return torch.where(
        keep_agent_pairs & keep_environments, z_dense, torch.zeros_like(z_dense)
    )


def decay_dense_opinions(z_dense: Tensor, dynamics: OpinionDynamics, *, dt: float) -> Tensor:
    """Advance every directed edge once with zero urgency and zero evidence."""
    mask = torch.ones_like(z_dense, dtype=torch.bool)
    decayed = dynamics(
        z_dense,
        torch.zeros_like(z_dense),
        torch.zeros_like(z_dense),
        mask,
        dt=dt,
    )
    off_diagonal = ~torch.eye(
        z_dense.shape[-1], device=z_dense.device, dtype=torch.bool
    ).unsqueeze(0)
    return torch.where(off_diagonal, decayed, torch.zeros_like(decayed))


class CollectorStepOutput(NamedTuple):
    action: Tensor
    log_prob: Tensor
    neighbor_ids: Tensor
    pair_mask: Tensor
    z_dense_prev: Tensor
    z_prev: Tensor
    raw_b: Tensor
    b: Tensor
    z_next: Tensor
    q: Tensor
    residual: Tensor
    base_loc: Tensor
    final_loc: Tensor
    scale: Tensor


class OpinionStatefulCollector:
    """Own ``z_dense`` and advance it exactly once per physical step."""

    def __init__(
        self,
        *,
        policy: OpinionTanhNormalPolicy,
        n_envs: int,
        n_agents: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if type(n_envs) is not int or n_envs <= 0:
            raise ValueError("n_envs must be a positive int")
        if type(n_agents) is not int or n_agents <= 1:
            raise ValueError("n_agents must be an int greater than one")
        self.policy = policy
        if device is None:
            device = next(policy.parameters()).device
        self.z_dense = torch.zeros(
            n_envs, n_agents, n_agents, device=device, dtype=dtype
        )
        self.n_envs = n_envs
        self.n_agents = n_agents
        self._last_step_id = None

    def reset_all(self) -> None:
        self.z_dense.zero_()
        self._last_step_id = None

    def _candidate_ids(self, neighbor_ids: Tensor, pair_mask: Tensor) -> Tensor:
        if neighbor_ids.shape != pair_mask.shape:
            raise ValueError("neighbor_ids and pair_mask must have equal shapes")
        if neighbor_ids.is_floating_point():
            active = neighbor_ids[pair_mask]
            if active.numel() and not torch.equal(active, active.round()):
                raise ValueError("active neighbor_ids must be exact integers")
            neighbor_ids = neighbor_ids.round().long()
        elif neighbor_ids.dtype is not torch.long:
            neighbor_ids = neighbor_ids.long()
        return neighbor_ids.to(device=self.z_dense.device)

    def step(
        self,
        *,
        step_id: int,
        observation: Tensor,
        pair_features: Tensor,
        neighbor_ids: Tensor,
        pair_mask: Tensor,
        urgency: Tensor,
        confidence: Tensor,
        agent_reset_mask: Tensor,
        environment_done: Tensor,
        residual_scale: float,
        direction: Optional[Tensor] = None,
        action: Optional[Tensor] = None,
    ) -> CollectorStepOutput:
        if type(step_id) is not int:
            raise ValueError("step_id must be an int")
        if self._last_step_id is not None and step_id <= self._last_step_id:
            raise RuntimeError("opinion state must update exactly once per physical step")
        pair_mask = pair_mask.to(device=self.z_dense.device, dtype=torch.bool)
        candidate_ids = self._candidate_ids(neighbor_ids, pair_mask)
        agent_reset_mask = agent_reset_mask.to(
            device=self.z_dense.device, dtype=torch.bool
        )
        if agent_reset_mask.ndim == 3 and agent_reset_mask.shape[-1] == 1:
            agent_reset_mask = agent_reset_mask.squeeze(-1)
        environment_done = environment_done.to(
            device=self.z_dense.device, dtype=torch.bool
        )
        if environment_done.ndim == 2 and environment_done.shape[-1] == 1:
            environment_done = environment_done.squeeze(-1)

        with torch.no_grad():
            reset_state = apply_opinion_resets(
                self.z_dense, agent_reset_mask, environment_done
            )
            z_dense_prev = reset_state.clone()
            z_prev = gather_candidate_opinions(
                z_dense_prev, candidate_ids, pair_mask
            )
            policy_output = self.policy(
                observation,
                pair_features,
                urgency,
                confidence,
                pair_mask,
                z_prev,
                residual_scale=residual_scale,
                direction=direction,
                action=action,
            )
            decayed = decay_dense_opinions(
                z_dense_prev, self.policy.core.dynamics, dt=self.policy.core.dt
            )
            self.z_dense = scatter_candidate_opinions(
                decayed,
                candidate_ids,
                policy_output.core.z_next,
                pair_mask,
            ).detach()
        self._last_step_id = step_id
        core = policy_output.core
        return CollectorStepOutput(
            action=policy_output.action.detach(),
            log_prob=policy_output.log_prob.detach(),
            neighbor_ids=candidate_ids.detach(),
            pair_mask=pair_mask.detach(),
            z_dense_prev=z_dense_prev.detach(),
            z_prev=z_prev.detach(),
            raw_b=core.raw_b.detach(),
            b=core.b.detach(),
            z_next=core.z_next.detach(),
            q=core.q.detach(),
            residual=core.residual.detach(),
            base_loc=core.base_loc.detach(),
            final_loc=core.final_loc.detach(),
            scale=core.scale.detach(),
        )
