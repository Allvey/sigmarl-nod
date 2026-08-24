"""Reference checks for the M4 physical pair-information contract.

These tests use unittest and are left for the user to execute manually.
"""

import math
import unittest

import torch

from utilities.opinion.config import ConflictGraphConfig
from utilities.opinion.conflict_graph import ConflictGraph


def _config() -> ConflictGraphConfig:
    return ConflictGraphConfig(
        emit_pair_info=True,
        candidate_count=2,
        pair_feature_dim=10,
        prediction_horizon_seconds=3.0,
        conflict_distance_meters=2.0,
        sensing_distance_meters=20.0,
        cpa_epsilon=1e-6,
        urgency_time_scale_seconds=3.0,
        urgency_distance_scale_meters=2.0,
    )


class ConflictGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = ConflictGraph(_config(), max_speed=8.0)
        self.positions = torch.tensor(
            [[[0.0, 0.0], [4.0, 0.0], [0.0, 15.0]]]
        )
        self.velocities = torch.tensor(
            [[[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]]
        )
        self.yaws = torch.tensor([[0.0, math.pi, math.pi / 2]])
        self.ids = torch.tensor([[[1, 2], [0, 2], [0, 1]]])

    def test_shape_dtype_and_identity_contract(self):
        output = self.graph(
            self.positions, self.velocities, self.yaws, self.ids
        )
        self.assertEqual(output.pair_features.shape, (1, 3, 2, 10))
        self.assertEqual(output.neighbor_ids.dtype, torch.long)
        self.assertEqual(output.pair_mask.dtype, torch.bool)
        torch.testing.assert_close(output.neighbor_ids, self.ids.long())
        self.assertTrue(torch.isfinite(output.pair_features).all())
        self.assertTrue(torch.all(output.pair_features.abs() <= 1.0))
        self.assertTrue(torch.all((0.0 <= output.urgency) & (output.urgency <= 1.0)))
        self.assertTrue(
            torch.all((0.0 <= output.confidence) & (output.confidence <= 1.0))
        )

    def test_head_on_pair_is_active_and_separating_pair_is_not(self):
        approaching = self.graph(
            self.positions, self.velocities, self.yaws, self.ids
        )
        self.assertTrue(approaching.pair_mask[0, 0, 0].item())
        self.assertGreater(approaching.urgency[0, 0, 0].item(), 0.0)

        separating_velocities = self.velocities.clone()
        separating_velocities[0, 0, 0] = -1.0
        separating_velocities[0, 1, 0] = 1.0
        separating = self.graph(
            self.positions, separating_velocities, self.yaws, self.ids
        )
        self.assertFalse(separating.pair_mask[0, 0, 0].item())
        self.assertEqual(separating.urgency[0, 0, 0].item(), 0.0)

    def test_invalid_or_self_ids_are_masked_without_extra_candidates(self):
        ids = self.ids.clone()
        ids[0, 0] = torch.tensor([0, -1])
        output = self.graph(self.positions, self.velocities, self.yaws, ids)
        self.assertFalse(output.pair_mask[0, 0].any().item())
        self.assertEqual(torch.count_nonzero(output.pair_features[0, 0]).item(), 0)
        torch.testing.assert_close(output.neighbor_ids, ids.long())


if __name__ == "__main__":
    unittest.main()
