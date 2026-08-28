"""Tests for the executable-command TTC guard used by A3.2."""

import math
import unittest

import torch

from utilities.avocado.bicycle import BicycleAdapterParameters
from utilities.avocado.road_safety import apply_ttc_braking_shield


class TTCSafetyShieldTests(unittest.TestCase):
    def setUp(self):
        self.adapter = BicycleAdapterParameters(
            front_length=0.08,
            rear_length=0.08,
            maximum_speed=1.0,
            maximum_steering_angle=math.radians(35.0),
        )

    def test_imminent_crossing_brakes_higher_responsibility_vehicle(self):
        positions = torch.tensor([[[-0.2, 0.0], [0.0, -0.2]]])
        actions = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        yaw = torch.tensor([[[0.0], [math.pi / 2]]])
        responsibility = torch.tensor(
            [[[0.5, 0.8], [0.2, 0.5]]]
        )
        result = apply_ttc_braking_shield(
            positions,
            actions,
            yaw,
            torch.tensor([0.08, 0.08]),
            self.adapter,
            minimum_ttc_seconds=0.5,
            responsibility=responsibility,
        )
        self.assertEqual(float(result.action[0, 0, 0]), 0.0)
        self.assertTrue(bool(result.intervention_mask[0, 0]))
        self.assertEqual(int(result.unsafe_pair_count_after[0]), 0)

    def test_diverging_commands_are_not_modified(self):
        positions = torch.tensor([[[-0.2, 0.0], [0.2, 0.0]]])
        actions = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        yaw = torch.tensor([[[math.pi], [0.0]]])
        result = apply_ttc_braking_shield(
            positions,
            actions,
            yaw,
            torch.tensor([0.08, 0.08]),
            self.adapter,
            minimum_ttc_seconds=0.5,
        )
        torch.testing.assert_close(result.action, actions)
        self.assertFalse(bool(result.intervention_mask.any()))


if __name__ == "__main__":
    unittest.main()
