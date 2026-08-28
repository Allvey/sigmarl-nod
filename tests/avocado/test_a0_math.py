"""A0 equation-level reference tests against analytic values."""

import unittest

import torch

from utilities.avocado.core import (
    HalfPlane,
    attention_euler_step,
    attention_reference_step,
    build_oca_half_plane,
    collision_time,
    finite_velocity_obstacle_correction,
    opinion_euler_step,
    opinion_to_cooperation,
    projection_estimator,
    solve_closest_admissible_velocity,
)


class CollisionTimeTests(unittest.TestCase):
    def test_first_contact_matches_expanded_equation_13(self):
        value = collision_time(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            0.4,
        )
        torch.testing.assert_close(value, torch.tensor(1.6))

    def test_no_future_contact_and_existing_overlap(self):
        no_contact = collision_time(
            torch.tensor([0.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            0.4,
        )
        self.assertTrue(torch.isinf(no_contact))
        overlap = collision_time(
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.2, 0.0]),
            torch.tensor([0.0, 0.0]),
            0.4,
        )
        self.assertEqual(float(overlap), 0.0)


class OpinionEquationTests(unittest.TestCase):
    def test_pdf_and_official_attention_updates_are_both_explicit(self):
        attention = torch.tensor(0.2)
        time_to_collision = torch.tensor(2.0)
        target = torch.tanh(torch.tensor(14.15 / 2.0))
        pdf_update = attention_euler_step(
            attention,
            time_to_collision,
            dt=0.05,
            delta=0.57,
            kappa=14.15,
        )
        expected_pdf = attention + 0.05 * (-0.57 * attention + 0.43 * target)
        torch.testing.assert_close(pdf_update, expected_pdf)

        reference_update = attention_reference_step(
            attention,
            time_to_collision,
            delta=0.57,
            kappa=14.15,
        )
        expected_reference = 0.43 * attention + 0.57 * target
        torch.testing.assert_close(reference_update, expected_reference)
        self.assertNotAlmostEqual(float(pdf_update), float(reference_update))

    def test_projection_estimator_and_opinion_step(self):
        correction = torch.tensor([2.0, 0.0])
        noncooperative = projection_estimator(
            torch.zeros(2), correction, epsilon=3.22
        )
        cooperative = projection_estimator(
            correction, correction, epsilon=3.22
        )
        torch.testing.assert_close(
            noncooperative, torch.tanh(torch.tensor(-3.22 / 2.0))
        )
        torch.testing.assert_close(
            cooperative, torch.tanh(torch.tensor(3.22 / 2.0))
        )

        opinion = torch.tensor(0.1)
        attention = torch.tensor(0.8)
        estimate = torch.tensor(-0.5)
        updated = opinion_euler_step(
            opinion,
            attention,
            estimate,
            dt=0.05,
            decay=2.0,
            self_weight=0.3,
            estimate_weight=0.7,
            bias=0.0,
        )
        expected = opinion + 0.05 * (
            -2.0 * opinion
            + 2.0 * attention * torch.tanh(0.3 * opinion + 0.7 * estimate)
        )
        torch.testing.assert_close(updated, expected)
        torch.testing.assert_close(
            opinion_to_cooperation(torch.tensor([-1.0, 0.0, 1.0])),
            torch.tensor([0.0, 0.5, 1.0]),
        )


class VelocityObstacleTests(unittest.TestCase):
    def test_inside_and_outside_vo_have_correct_admissible_side(self):
        relative_position = torch.tensor([2.5, 0.0])
        inside = finite_velocity_obstacle_correction(
            relative_position,
            torch.tensor([1.75, 0.0]),
            0.44,
            2.5,
        )
        self.assertTrue(bool(inside.active))
        self.assertGreater(float(torch.dot(inside.correction, inside.normal)), 0)

        current = torch.zeros(2)
        outside = finite_velocity_obstacle_correction(
            relative_position,
            current,
            0.44,
            2.5,
        )
        self.assertFalse(bool(outside.active))
        plane = build_oca_half_plane(
            current,
            outside.correction,
            outside.normal,
            torch.tensor(0.5),
        )
        self.assertGreaterEqual(
            float(torch.dot(plane.normal, current)), float(plane.offset) - 1e-6
        )

    def test_oca_projection_is_the_closest_feasible_velocity(self):
        preferred = torch.zeros(2)
        planes = [
            HalfPlane(
                normal=torch.tensor([1.0, 0.0]),
                offset=torch.tensor(0.2),
                point=torch.tensor([0.2, 0.0]),
            ),
            HalfPlane(
                normal=torch.tensor([0.0, 1.0]),
                offset=torch.tensor(0.3),
                point=torch.tensor([0.0, 0.3]),
            ),
        ]
        result = solve_closest_admissible_velocity(
            preferred, planes, maximum_speed=1.0
        )
        self.assertTrue(result.feasible)
        torch.testing.assert_close(result.velocity, torch.tensor([0.2, 0.3]))

    def test_infeasible_projection_is_reported_and_bounded(self):
        planes = [
            HalfPlane(
                normal=torch.tensor([1.0, 0.0]),
                offset=torch.tensor(0.8),
                point=torch.tensor([0.8, 0.0]),
            ),
            HalfPlane(
                normal=torch.tensor([-1.0, 0.0]),
                offset=torch.tensor(0.8),
                point=torch.tensor([-0.8, 0.0]),
            ),
        ]
        result = solve_closest_admissible_velocity(
            torch.zeros(2), planes, maximum_speed=1.0
        )
        self.assertFalse(result.feasible)
        self.assertTrue(torch.isfinite(result.velocity).all())
        self.assertLessEqual(float(torch.linalg.vector_norm(result.velocity)), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
