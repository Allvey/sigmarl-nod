"""A3 unit tests for the holonomic-to-bicycle adapter."""

import math
import unittest

import torch

from utilities.avocado.bicycle import (
    BicycleAdapterParameters,
    constrain_velocity_to_path,
    path_velocity_cone_constraints,
    reference_path_preferred_velocity,
    stanley_path_preferred_velocity,
    vector_velocity_to_bicycle_action,
    wrap_to_pi,
)


class BicycleAdapterTests(unittest.TestCase):
    def setUp(self):
        self.parameters = BicycleAdapterParameters(
            front_length=0.08,
            rear_length=0.08,
            maximum_speed=1.0,
            maximum_steering_angle=math.radians(35.0),
            minimum_speed_ratio=0.2,
        )

    def test_aligned_velocity_maps_to_speed_and_zero_steering(self):
        result = vector_velocity_to_bicycle_action(
            torch.tensor([[0.6, 0.0]]),
            torch.tensor([0.0]),
            self.parameters,
        )
        torch.testing.assert_close(result.action, torch.tensor([[0.6, 0.0]]))
        self.assertFalse(bool(result.steering_saturated[0]))

    def test_reachable_direction_inverts_bicycle_slip_equation(self):
        desired_angle = math.radians(10.0)
        desired = 0.5 * torch.tensor(
            [[math.cos(desired_angle), math.sin(desired_angle)]]
        )
        result = vector_velocity_to_bicycle_action(
            desired,
            torch.tensor([0.0]),
            self.parameters,
        )
        steering = result.action[0, 1]
        beta = torch.atan(
            torch.tan(steering)
            * self.parameters.rear_length
            / (self.parameters.front_length + self.parameters.rear_length)
        )
        torch.testing.assert_close(beta, torch.tensor(desired_angle))
        self.assertFalse(bool(result.steering_saturated[0]))

    def test_unreachable_direction_is_bounded_and_reported(self):
        result = vector_velocity_to_bicycle_action(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0.0]),
            self.parameters,
        )
        self.assertTrue(bool(result.steering_saturated[0]))
        self.assertLessEqual(
            abs(float(result.action[0, 1])),
            self.parameters.maximum_steering_angle + 1e-6,
        )
        self.assertGreater(float(result.action[0, 0]), 0.0)

    def test_reference_path_velocity_and_angle_wrapping(self):
        positions = torch.tensor([[[0.0, 0.0]]])
        reference = torch.tensor([[[[0.1, 0.0], [0.5, 0.0]]]])
        velocity = reference_path_preferred_velocity(positions, reference, 0.6)
        torch.testing.assert_close(velocity, torch.tensor([[[0.6, 0.0]]]))
        wrapped = wrap_to_pi(torch.tensor([3.0 * math.pi]))
        torch.testing.assert_close(wrapped.abs(), torch.tensor([math.pi]))

    def test_terminal_reference_continues_along_vehicle_heading(self):
        positions = torch.tensor([[[0.5, 0.5]]])
        reference = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
        fallback = torch.tensor([[[0.0, 1.0]]])
        velocity = reference_path_preferred_velocity(
            positions,
            reference,
            0.6,
            terminal_fallback_directions=fallback,
        )
        torch.testing.assert_close(velocity, torch.tensor([[[0.0, 0.6]]]))

    def test_stanley_tracker_follows_tangent_when_centered(self):
        velocity = stanley_path_preferred_velocity(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]]]),
            0.6,
            cross_track_gain=3.0,
            softening_speed=0.2,
            maximum_correction_angle=math.radians(20.0),
        )
        torch.testing.assert_close(velocity, torch.tensor([[[0.6, 0.0]]]))

    def test_stanley_tracker_corrects_signed_cross_track_error(self):
        reference = torch.tensor(
            [[[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]]]
        )
        above = stanley_path_preferred_velocity(
            torch.tensor([[[0.0, 0.03]]]),
            reference,
            0.6,
            cross_track_gain=3.0,
            softening_speed=0.2,
            maximum_correction_angle=math.radians(20.0),
        )
        below = stanley_path_preferred_velocity(
            torch.tensor([[[0.0, -0.03]]]),
            reference,
            0.6,
            cross_track_gain=3.0,
            softening_speed=0.2,
            maximum_correction_angle=math.radians(20.0),
        )
        self.assertLess(float(above[0, 0, 1]), 0.0)
        self.assertGreater(float(below[0, 0, 1]), 0.0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(above, dim=-1),
            torch.tensor([[0.6]]),
        )

    def test_stanley_tracker_bounds_lateral_correction(self):
        velocity = stanley_path_preferred_velocity(
            torch.tensor([[[0.0, 1.0]]]),
            torch.tensor([[[[0.1, 0.0], [0.2, 0.0]]]]),
            0.6,
            cross_track_gain=10.0,
            softening_speed=0.0,
            maximum_correction_angle=math.radians(15.0),
        )
        heading = torch.atan2(velocity[..., 1], velocity[..., 0]).abs()
        torch.testing.assert_close(
            heading,
            torch.tensor([[math.radians(15.0)]]),
        )

    def test_road_constraint_clamps_heading_but_preserves_speed(self):
        desired = torch.tensor([[0.0, 0.5]])
        path = torch.tensor([[0.6, 0.0]])
        constrained = constrain_velocity_to_path(
            desired,
            path,
            math.radians(12.0),
        )
        angle = torch.atan2(constrained[0, 1], constrained[0, 0])
        torch.testing.assert_close(angle, torch.tensor(math.radians(12.0)))
        torch.testing.assert_close(
            torch.linalg.vector_norm(constrained), torch.tensor(0.5)
        )

    def test_path_velocity_cone_half_planes_accept_only_forward_wedge(self):
        normals, offsets = path_velocity_cone_constraints(
            torch.tensor([[[0.6, 0.0]]]),
            math.radians(20.0),
        )
        inside = torch.tensor([0.5, 0.0])
        outside = 0.5 * torch.tensor(
            [math.cos(math.radians(30.0)), math.sin(math.radians(30.0))]
        )
        backward = torch.tensor([-0.5, 0.0])
        self.assertTrue(bool((normals[0, 0] @ inside >= offsets[0, 0]).all()))
        self.assertFalse(bool((normals[0, 0] @ outside >= offsets[0, 0]).all()))
        self.assertFalse(bool((normals[0, 0] @ backward >= offsets[0, 0]).all()))


if __name__ == "__main__":
    unittest.main()
