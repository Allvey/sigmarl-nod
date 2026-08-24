"""Reference checks for the M3 pure mathematical modules.

These tests use the standard-library unittest runner and are intentionally not
executed automatically by the implementation session.
"""

import unittest

import torch

from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.evidence_net import OpinionEvidenceNet, swap_pair_features
from utilities.opinion.residual import OpinionResidual


def _directed_features(shape):
    features = torch.randn(*shape, 10)
    yaw = torch.randn(*shape)
    features[..., 4:6] = features[..., 4:6].abs()
    features[..., 6] = yaw.sin()
    features[..., 7] = yaw.cos()
    features[..., 8:10] = features[..., 8:10].abs()
    return features


class EvidenceNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = _directed_features((2, 4, 2))
        self.urgency = torch.rand(2, 4, 2)
        self.confidence = torch.rand(2, 4, 2)
        self.mask = torch.ones(2, 4, 2, dtype=torch.bool)
        self.network = OpinionEvidenceNet(10, (32, 32), 1.0, 1.0)

    def test_swap_is_an_involution(self):
        recovered = swap_pair_features(swap_pair_features(self.features))
        torch.testing.assert_close(recovered, self.features, atol=1e-5, rtol=1e-5)

    def test_evidence_is_signed_bounded_and_gated(self):
        output = self.network(
            self.features, self.urgency, self.confidence, self.mask
        )
        bound = self.urgency * self.confidence
        self.assertTrue(torch.all(output.b.abs() <= bound + 1e-6))

        reverse = self.network(
            swap_pair_features(self.features),
            self.urgency,
            self.confidence,
            self.mask,
        )
        torch.testing.assert_close(reverse.raw_b, -output.raw_b, atol=1e-6, rtol=1e-5)

        no_edges = self.network(
            self.features,
            self.urgency,
            self.confidence,
            torch.zeros_like(self.mask),
        )
        self.assertEqual(torch.count_nonzero(no_edges.b).item(), 0)

    def test_policy_gradient_can_reach_evidence_parameters(self):
        output = self.network(
            self.features, self.urgency, self.confidence, self.mask
        )
        output.b.square().mean().backward()
        gradients = [parameter.grad for parameter in self.network.parameters()]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        )


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.dynamics = OpinionDynamics(0.5, 1.0, 0.5, 1.0)

    def test_dynamics_has_no_trainable_parameters(self):
        self.assertEqual(list(self.dynamics.parameters()), [])

    def test_evidence_sign_and_inactive_decay(self):
        zeros = torch.zeros(2, 4, 2)
        mask = torch.ones_like(zeros, dtype=torch.bool)
        positive = self.dynamics(zeros, torch.ones_like(zeros), zeros, mask, 0.05)
        negative = self.dynamics(zeros, -torch.ones_like(zeros), zeros, mask, 0.05)
        self.assertTrue(torch.all(positive > 0))
        torch.testing.assert_close(negative, -positive)

        previous = torch.ones_like(zeros)
        inactive = self.dynamics(
            previous,
            torch.ones_like(previous),
            torch.ones_like(previous),
            torch.zeros_like(mask),
            0.05,
        )
        self.assertTrue(torch.all(inactive.abs() < previous.abs()))

    def test_constant_bounded_input_stays_within_theoretical_bound(self):
        z = torch.zeros(1, 1, 1)
        evidence = torch.ones_like(z)
        urgency = torch.ones_like(z)
        mask = torch.ones_like(z, dtype=torch.bool)
        for _ in range(400):
            z = self.dynamics(z, evidence, urgency, mask, 0.05)
        self.assertLessEqual(z.abs().max().item(), self.dynamics.theoretical_bound(1.0) + 1e-4)


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.residual = OpinionResidual(1.0, 0.1, 0.25, action_index=0)

    def test_residual_is_normalized_bounded_and_masked(self):
        z = torch.tensor([[[10.0, 10.0], [-10.0, 10.0]]])
        urgency = torch.ones_like(z)
        mask = torch.tensor([[[True, True], [False, False]]])
        output = self.residual(z, urgency, mask)
        self.assertTrue(torch.all(output.residual.abs() <= 0.25))
        torch.testing.assert_close(
            output.normalized_weights[0, 0].sum(), torch.tensor(1.0)
        )
        self.assertEqual(output.residual[0, 1, 0].item(), 0.0)

    def test_only_speed_location_is_modified(self):
        base_loc = torch.randn(2, 4, 2)
        delta = torch.full((2, 4, 1), 0.05)
        final_loc = self.residual.apply_to_loc(base_loc, delta)
        torch.testing.assert_close(final_loc[..., 0], base_loc[..., 0] + 0.05)
        torch.testing.assert_close(final_loc[..., 1], base_loc[..., 1])


if __name__ == "__main__":
    unittest.main()
