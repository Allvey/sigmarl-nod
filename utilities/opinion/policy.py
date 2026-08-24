"""M5 stateless Direct-Evidence bridge around the frozen Base Actor."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.residual import OpinionResidual


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
