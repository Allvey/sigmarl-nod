"""A1 tests for the VMAS holonomic single-integrator environment."""

import unittest

import torch
import vmas
from vmas.simulator.core import Sphere

from scenarios.avocado_holonomic import DirectVelocityDynamics, Scenario
from utilities.avocado.controller import AVOCADOController
from utilities.avocado.bicycle import path_velocity_cone_constraints
from utilities.avocado.core import AVOCADOParameters


class HolonomicEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        self.environment = vmas.make_env(
            scenario=self.scenario,
            num_envs=2,
            device="cpu",
            continuous_actions=True,
            clamp_actions=True,
            layout="head_on_noncooperative",
            n_agents=2,
            controlled_agents=1,
            dt=0.05,
            robot_radius=0.2,
            agent_radius=0.2,
            avoidance_radius_scale=1.1,
        )
        self.environment.reset(seed=17)

    def test_entities_are_noncolliding_discs_with_direct_velocity_dynamics(self):
        for agent in self.scenario.world.agents:
            self.assertIsInstance(agent.shape, Sphere)
            self.assertIsInstance(agent.dynamics, DirectVelocityDynamics)
            self.assertFalse(agent.collide)
            self.assertFalse(agent.rotatable)
        torch.testing.assert_close(
            self.scenario.radii, torch.tensor([0.2, 0.2])
        )
        torch.testing.assert_close(
            self.scenario.security_radii, torch.tensor([0.22, 0.22])
        )

    def test_action_is_exact_single_integrator_velocity(self):
        first = self.scenario.world.agents[0]
        before = first.state.pos.clone()
        command = torch.tensor([[0.3, -0.2], [-0.4, 0.1]])
        zeros = torch.zeros_like(command)
        self.environment.step([command, zeros])
        torch.testing.assert_close(first.state.vel, command)
        torch.testing.assert_close(first.state.pos, before + 0.05 * command)

    def test_controller_uses_every_perceived_neighbor(self):
        parameters = AVOCADOParameters(perception_radius=2.5)
        controller = AVOCADOController(
            parameters,
            batch_size=1,
            entity_count=3,
            controlled_mask=torch.ones(3, dtype=torch.bool),
            security_radii=torch.full((3,), 0.22),
            maximum_speeds=torch.ones(3),
            seed=5,
        )
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        velocities = torch.zeros_like(positions)
        controller.step(positions, velocities, velocities)
        expected = ~torch.eye(3, dtype=torch.bool).unsqueeze(0)
        torch.testing.assert_close(controller.last_neighbor_mask, expected)

    def test_agent_reset_clears_incoming_and_outgoing_pair_state(self):
        controller = AVOCADOController(
            AVOCADOParameters(),
            batch_size=1,
            entity_count=3,
            controlled_mask=torch.ones(3, dtype=torch.bool),
            security_radii=torch.full((3,), 0.22),
            maximum_speeds=torch.ones(3),
            seed=5,
        )
        controller.attention.fill_(1.0)
        controller.opinion.fill_(0.25)
        controller.previous_correction.fill_(1.0)
        controller.reset_agents(torch.tensor([[False, True, False]]))
        self.assertEqual(float(controller.attention[0, 0, 1]), 0.0)
        self.assertEqual(float(controller.attention[0, 1, 2]), 0.0)
        self.assertEqual(float(controller.attention[0, 0, 2]), 1.0)
        self.assertEqual(float(controller.opinion[0, 0, 1]), 0.0)
        self.assertEqual(float(controller.previous_correction[0, 1, 2, 0]), 0.0)

    def test_controller_jointly_enforces_additional_velocity_half_planes(self):
        controller = AVOCADOController(
            AVOCADOParameters(perception_radius=0.1),
            batch_size=1,
            entity_count=2,
            controlled_mask=torch.ones(2, dtype=torch.bool),
            security_radii=torch.full((2,), 0.1),
            maximum_speeds=torch.ones(2),
            seed=5,
        )
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        velocities = torch.zeros_like(positions)
        preferred = torch.tensor([[[0.0, 0.6], [0.0, 0.6]]])
        path = torch.tensor([[[0.6, 0.0], [0.6, 0.0]]])
        normals, offsets = path_velocity_cone_constraints(
            path,
            torch.deg2rad(torch.tensor(20.0)).item(),
        )
        actions = controller.step(
            positions,
            velocities,
            preferred,
            additional_half_plane_normals=normals,
            additional_half_plane_offsets=offsets,
        )
        headings = torch.atan2(actions[..., 1], actions[..., 0]).abs()
        self.assertTrue(bool((headings <= torch.deg2rad(torch.tensor(20.01))).all()))
        self.assertTrue(bool(controller.last_constraint_count.eq(2).all()))

    def test_road_mode_normalizes_pair_responsibility_complementarily(self):
        controller = AVOCADOController(
            AVOCADOParameters(perception_radius=2.0),
            batch_size=1,
            entity_count=2,
            controlled_mask=torch.ones(2, dtype=torch.bool),
            security_radii=torch.full((2,), 0.1),
            maximum_speeds=torch.ones(2),
            seed=5,
            complementary_responsibility=True,
        )
        controller.opinion[0, 0, 1] = 0.8
        controller.opinion[0, 1, 0] = -0.2
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        velocities = torch.zeros_like(positions)
        controller.step(positions, velocities, velocities)
        total = (
            controller.last_responsibility[0, 0, 1]
            + controller.last_responsibility[0, 1, 0]
        )
        torch.testing.assert_close(total, torch.tensor(1.0))
        self.assertNotEqual(
            float(controller.last_responsibility[0, 0, 1]), 0.5
        )


if __name__ == "__main__":
    unittest.main()
