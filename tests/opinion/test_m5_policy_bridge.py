"""Reference checks for the M5 stateless Direct-Evidence bridge.

These tests are provided for manual execution and are not run by this session.
"""

import unittest

import torch
from torch import nn

from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.policy import DirectEvidencePolicyBridge
from utilities.opinion.residual import OpinionResidual


class FakeBasePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, observation):
        parameters = self.linear(observation)
        return parameters[..., :2], parameters[..., 2:].exp()


class DirectEvidencePolicyBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = FakeBasePolicyNet()
        self.bridge = DirectEvidencePolicyBridge(
            base_policy_net=self.base,
            evidence_net=OpinionEvidenceNet(10, (16, 16), 1.0, 1.0),
            residual=OpinionResidual(1.0, 0.1, 0.25),
            freeze_base_actor=True,
        )
        self.observation = torch.randn(2, 4, 3)
        self.features = torch.randn(2, 4, 2, 10)
        self.urgency = torch.ones(2, 4, 2)
        self.confidence = torch.ones(2, 4, 2)
        self.mask = torch.ones(2, 4, 2, dtype=torch.bool)

    def test_only_speed_location_changes_and_scale_is_preserved(self):
        base_loc, base_scale = self.base(self.observation)
        final_loc, scale, returned_base_loc, _, _, _, residual = self.bridge(
            self.observation,
            self.features,
            self.urgency,
            self.confidence,
            self.mask,
        )
        torch.testing.assert_close(returned_base_loc, base_loc)
        torch.testing.assert_close(scale, base_scale)
        torch.testing.assert_close(final_loc[..., 1], base_loc[..., 1])
        torch.testing.assert_close(final_loc[..., :1], base_loc[..., :1] + residual)
        self.assertTrue(torch.all(residual.abs() <= 0.25))

    def test_base_is_frozen_and_evidence_receives_policy_gradient(self):
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.base.parameters())
        )
        final_loc, *_ = self.bridge(
            self.observation,
            self.features,
            self.urgency,
            self.confidence,
            self.mask,
        )
        final_loc[..., 0].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.bridge.evidence_net.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in self.base.parameters())
        )

    def test_inactive_pairs_give_exact_base_action_parameters(self):
        base_loc, base_scale = self.base(self.observation)
        final_loc, scale, _, _, gated_b, direct_z, residual = self.bridge(
            self.observation,
            self.features,
            self.urgency,
            self.confidence,
            torch.zeros_like(self.mask),
        )
        torch.testing.assert_close(final_loc, base_loc)
        torch.testing.assert_close(scale, base_scale)
        self.assertEqual(torch.count_nonzero(gated_b).item(), 0)
        self.assertEqual(torch.count_nonzero(direct_z).item(), 0)
        self.assertEqual(torch.count_nonzero(residual).item(), 0)


if __name__ == "__main__":
    unittest.main()
