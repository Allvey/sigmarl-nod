"""M5 stateless Direct-Evidence bridge around the frozen Base Actor."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.residual import OpinionResidual
from utilities.opinion.state import OpinionStateTracker


class DirectEvidencePolicyBridge(nn.Module):
    """Modify only the Base speed location using current-frame evidence.

    M5 deliberately defines ``direct_opinion = evidence``. It does not run the
    Opinion ODE and does not retain state between physical steps; stateful
    ``z`` belongs to M6.
    """

    def __init__(
        self,
        base_policy_net: nn.Module,
        evidence_net: OpinionEvidenceNet,
        residual: OpinionResidual,
        freeze_base_actor: bool = True,
    ) -> None:
        super().__init__()
        if not freeze_base_actor:
            raise ValueError("M5 requires a frozen Base Actor.")
        self.base_policy_net = base_policy_net
        self.evidence_net = evidence_net
        self.residual = residual
        for parameter in self.base_policy_net.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        observation: torch.Tensor,
        pair_features: torch.Tensor,
        urgency: torch.Tensor,
        confidence: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        base_loc, base_scale = self.base_policy_net(observation)
        evidence_output = self.evidence_net(
            pair_features=pair_features,
            urgency=urgency,
            confidence=confidence,
            pair_mask=pair_mask,
        )
        direct_opinion = evidence_output.b
        residual_output = self.residual(
            z=direct_opinion,
            urgency=urgency,
            pair_mask=pair_mask,
        )
        final_loc = self.residual.apply_to_loc(
            base_loc=base_loc,
            residual=residual_output.residual,
        )
        return (
            final_loc,
            base_scale,
            base_loc,
            evidence_output.raw_b,
            evidence_output.b,
            direct_opinion,
            residual_output.residual,
        )

    def trainable_parameters(self):
        """Expose the only M5 Actor parameters allowed in the optimizer."""

        return self.evidence_net.parameters()


class StatefulOpinionPolicyBridge(nn.Module):
    """Pure M6 policy mapping from explicit ``z_prev`` to ``z_next``.

    This module never stores temporal state. The rollout-only controller below
    gathers and commits state exactly once per physical step, while PPO can
    recompute this pure mapping from stored ``z_prev`` without mutating it.
    """

    def __init__(
        self,
        base_policy_net: nn.Module,
        evidence_net: OpinionEvidenceNet,
        dynamics: OpinionDynamics,
        residual: OpinionResidual,
        dt: float,
        freeze_base_actor: bool = True,
        freeze_evidence: bool = True,
    ) -> None:
        super().__init__()
        if not freeze_base_actor:
            raise ValueError("M6 requires a frozen Base Actor.")
        if not freeze_evidence:
            raise ValueError(
                "M6 freezes EvidenceNet until sequence PPO is implemented."
            )
        if float(dt) <= 0.0:
            raise ValueError("dt must be positive.")
        self.base_policy_net = base_policy_net
        self.evidence_net = evidence_net
        self.dynamics = dynamics
        self.residual = residual
        self.dt = float(dt)
        for parameter in self.base_policy_net.parameters():
            parameter.requires_grad_(False)
        for parameter in self.evidence_net.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        observation: torch.Tensor,
        pair_features: torch.Tensor,
        urgency: torch.Tensor,
        confidence: torch.Tensor,
        pair_mask: torch.Tensor,
        z_prev: torch.Tensor,
    ):
        base_loc, base_scale = self.base_policy_net(observation)
        evidence_output = self.evidence_net(
            pair_features=pair_features,
            urgency=urgency,
            confidence=confidence,
            pair_mask=pair_mask,
        )
        z_next = self.dynamics(
            z_prev=z_prev,
            evidence=evidence_output.b,
            urgency=urgency,
            pair_mask=pair_mask,
            dt=self.dt,
        )
        residual_output = self.residual(
            z=z_next,
            urgency=urgency,
            pair_mask=pair_mask,
        )
        final_loc = self.residual.apply_to_loc(
            base_loc=base_loc,
            residual=residual_output.residual,
        )
        return (
            final_loc,
            base_scale,
            base_loc,
            evidence_output.raw_b,
            evidence_output.b,
            z_next,
            residual_output.q,
            residual_output.normalized_weights,
            residual_output.aggregate,
            residual_output.residual,
        )


class StatefulOpinionPolicyController(nn.Module):
    """Rollout-only adapter that owns M6 temporal state by global ID."""

    def __init__(self, policy: nn.Module, state_tracker: OpinionStateTracker) -> None:
        super().__init__()
        self.policy = policy
        self.state_tracker = state_tracker
        # TorchRL 0.2.1 recognizes a single-TensorDict module through these
        # attributes. Without them it treats this controller as a regular
        # tensor network and probes the entire observation_spec with rand().
        # That is both unnecessary and invalid for categorical global IDs.
        self.in_keys = list(getattr(policy, "in_keys", ()))
        for key in (
            ("agents", "info", "neighbor_ids"),
            ("agents", "info", "pair_mask"),
            ("agents", "info", "urgency"),
            ("agents", "info", "agent_reset_mask"),
        ):
            if key not in self.in_keys:
                self.in_keys.append(key)
        self.out_keys = list(getattr(policy, "out_keys", ()))

    @property
    def spec(self):
        return getattr(self.policy, "spec", None)

    @staticmethod
    def _optional(tensordict, key):
        try:
            return tensordict.get(key)
        except KeyError:
            return None

    def forward(self, tensordict):
        neighbor_ids = tensordict.get(("agents", "info", "neighbor_ids"))
        pair_mask = tensordict.get(("agents", "info", "pair_mask"))
        urgency = tensordict.get(("agents", "info", "urgency"))
        agent_reset_mask = tensordict.get(
            ("agents", "info", "agent_reset_mask")
        )
        environment_done = self._optional(tensordict, "done")
        z_prev = self.state_tracker.prepare_step(
            neighbor_ids=neighbor_ids,
            pair_mask=pair_mask,
            agent_reset_mask=agent_reset_mask,
            reference=urgency,
            environment_done=environment_done,
        )
        z_dense_prev, edge_active_prev = self.state_tracker.prepared_snapshot()
        tensordict.set(("agents", "opinion", "z_prev"), z_prev)
        tensordict.set(
            ("agents", "opinion", "z_dense_prev"), z_dense_prev
        )
        tensordict.set(
            ("agents", "opinion", "edge_active_prev"), edge_active_prev
        )
        tensordict = self.policy(tensordict)
        self.state_tracker.commit_step(
            neighbor_ids=neighbor_ids,
            z_next=tensordict.get(("agents", "opinion", "z_next")),
        )
        return tensordict

    def reset_state(self) -> None:
        self.state_tracker.reset_all()
