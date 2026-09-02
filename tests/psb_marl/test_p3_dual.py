"""P3.2 dense safety costs and projected dual updates."""

import unittest
from pathlib import Path

import torch

from utilities.psb_marl.p3_dual import (
    P3SafetyCosts,
    ProjectedDualController,
    continuous_safety_costs,
)
from utilities.psb_marl.config import load_psb_experiment
from utilities.psb_marl.p3_dual_evaluation import p32_safety_gate


class P32DualTests(unittest.TestCase):
    def test_locked_p32_config_uses_passed_p31_and_p2_actor_warm_start(self):
        experiment = load_psb_experiment(
            Path("configs/psb_marl/p3_2_primal_dual_ppo.json")
        )
        runtime = experiment.p32_runtime_config()

        self.assertEqual(experiment.stage, "p3_primal_dual_ppo")
        self.assertEqual(runtime["stage"], "p2_frozen_base_bifurcation")
        self.assertEqual(runtime["p3_stage"], experiment.stage)
        self.assertEqual(
            Path(runtime["initial_policy_checkpoint"]),
            experiment.parent_run / "candidate_policy.pth",
        )
        self.assertEqual(
            runtime["branch_adapter"]["conditioning_mode"],
            "supported_sector_q_gate",
        )
        self.assertEqual(
            runtime["branch_adapter"]["action_projection"],
            "longitudinal_only",
        )

    def test_normalized_p32_config_preserves_explicit_scale_mode(self):
        experiment = load_psb_experiment(
            Path(
                "configs/psb_marl/"
                "p3_2_n_normalized_primal_dual_ppo.json"
            )
        )
        runtime = experiment.p32_runtime_config()

        self.assertTrue(runtime["primal_dual"]["normalize_constraints"])
        self.assertEqual(runtime["primal_dual"]["lane_budget"], 0.0075)

    def test_actuation_aligned_config_dualizes_only_vehicle_risk(self):
        experiment = load_psb_experiment(
            Path(
                "configs/psb_marl/"
                "p3_2_c_actuation_aligned_primal_dual_ppo.json"
            )
        )
        dual = experiment.p32_runtime_config()["primal_dual"]

        self.assertEqual(dual["active_constraints"], ["vehicle"])
        self.assertEqual(dual["initial_lane_multiplier"], 0.0)

    def test_costs_match_dense_vehicle_and_lane_definitions(self):
        costs = continuous_safety_costs(
            urgency=torch.tensor([[[0.4, 0.8], [0.2, 0.1]]]),
            confidence=torch.tensor([[[1.0, 0.5], [0.5, 1.0]]]),
            pair_mask=torch.tensor([[[True, True], [True, False]]]),
            distance_left=torch.tensor([[0.035, 0.08]]),
            distance_right=torch.tensor([[0.09, 0.06]]),
            vehicle_collision=torch.tensor([[False, True]]),
            lane_collision=torch.tensor([[False, False]]),
            lane_safety_margin=0.07,
        )

        self.assertTrue(
            torch.allclose(costs.vehicle, torch.tensor([[0.4, 1.0]]))
        )
        self.assertTrue(
            torch.allclose(costs.lane, torch.tensor([[0.5, 1.0 / 7.0]]))
        )

    def test_projected_ascent_and_lagrangian_reward(self):
        controller = ProjectedDualController(
            vehicle_budget=0.2,
            lane_budget=0.1,
            vehicle_learning_rate=0.5,
            lane_learning_rate=0.25,
            maximum_multiplier=0.3,
            initial_vehicle_multiplier=0.1,
            initial_lane_multiplier=0.2,
        )
        costs = P3SafetyCosts(
            vehicle=torch.tensor([[0.6, 0.4]]),
            lane=torch.tensor([[0.0, 0.0]]),
        )
        reward = controller.lagrangian_reward(
            torch.ones(1, 2, 1), costs
        )
        self.assertTrue(
            torch.allclose(reward, torch.tensor([[[0.94], [0.96]]]))
        )

        metrics = controller.update(costs)

        self.assertAlmostEqual(metrics["vehicle_multiplier"], 0.25)
        self.assertAlmostEqual(metrics["lane_multiplier"], 0.175)

    def test_dual_update_never_builds_an_autograd_graph(self):
        controller = ProjectedDualController(
            vehicle_budget=0.0,
            lane_budget=0.0,
            vehicle_learning_rate=1.0,
            lane_learning_rate=1.0,
            maximum_multiplier=1.0,
        )
        costs = P3SafetyCosts(
            vehicle=torch.ones(2, requires_grad=True),
            lane=torch.ones(2, requires_grad=True),
        )

        controller.update(costs)

        self.assertEqual(controller.vehicle_multiplier, 1.0)
        self.assertEqual(controller.lane_multiplier, 1.0)
        self.assertIsNone(costs.vehicle.grad)
        self.assertIsNone(costs.lane.grad)

    def test_budget_normalization_equalizes_constraint_scales(self):
        controller = ProjectedDualController(
            vehicle_budget=0.5,
            lane_budget=0.01,
            vehicle_learning_rate=0.1,
            lane_learning_rate=0.1,
            maximum_multiplier=1.0,
            initial_vehicle_multiplier=0.1,
            initial_lane_multiplier=0.1,
            normalize_constraints=True,
        )
        costs = P3SafetyCosts(
            vehicle=torch.tensor([0.75]),
            lane=torch.tensor([0.015]),
        )

        reward = controller.lagrangian_reward(torch.tensor([1.0]), costs)
        metrics = controller.update(costs)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.7])))
        self.assertAlmostEqual(metrics["vehicle_constraint_residual"], 0.5)
        self.assertAlmostEqual(metrics["lane_constraint_residual"], 0.5)
        self.assertAlmostEqual(metrics["vehicle_multiplier"], 0.15)
        self.assertAlmostEqual(metrics["lane_multiplier"], 0.15)

    def test_inactive_lane_constraint_is_diagnostic_only(self):
        controller = ProjectedDualController(
            vehicle_budget=0.5,
            lane_budget=0.01,
            vehicle_learning_rate=0.1,
            lane_learning_rate=0.1,
            maximum_multiplier=1.0,
            initial_vehicle_multiplier=0.1,
            initial_lane_multiplier=0.0,
            normalize_constraints=True,
            active_constraints=("vehicle",),
        )
        costs = P3SafetyCosts(
            vehicle=torch.tensor([0.5]),
            lane=torch.tensor([1.0]),
        )

        reward = controller.lagrangian_reward(torch.tensor([1.0]), costs)
        metrics = controller.update(costs)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.9])))
        self.assertEqual(metrics["lane_multiplier"], 0.0)
        self.assertFalse(metrics["lane_constraint_dualized"])

    def test_safety_gate_uses_absolute_confidence_bound(self):
        candidate = [
            {"vehicle_cost_mean": 0.2, "lane_cost_mean": 0.002},
            {"vehicle_cost_mean": 0.22, "lane_cost_mean": 0.003},
        ]
        base = [
            {"vehicle_cost_mean": 0.3, "lane_cost_mean": 0.004},
            {"vehicle_cost_mean": 0.31, "lane_cost_mean": 0.004},
        ]
        gate = p32_safety_gate(
            candidate,
            base,
            vehicle_budget=0.25,
            lane_budget=0.005,
            confidence_z=1.0,
        )
        self.assertTrue(gate["passed"])
        self.assertLess(
            gate["vehicle"]["difference_candidate_minus_base_mean"], 0.0
        )

    def test_safety_gate_fails_an_absolute_budget_violation(self):
        candidate = [{"vehicle_cost_mean": 0.4, "lane_cost_mean": 0.002}]
        base = [{"vehicle_cost_mean": 0.5, "lane_cost_mean": 0.003}]
        gate = p32_safety_gate(
            candidate,
            base,
            vehicle_budget=0.35,
            lane_budget=0.005,
            confidence_z=1.645,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["vehicle"]["budget_passed"])

    def test_inactive_safety_channel_does_not_fail_cmdp_gate(self):
        candidate = [{"vehicle_cost_mean": 0.3, "lane_cost_mean": 1.0}]
        base = [{"vehicle_cost_mean": 0.4, "lane_cost_mean": 0.01}]
        gate = p32_safety_gate(
            candidate,
            base,
            vehicle_budget=0.35,
            lane_budget=0.005,
            confidence_z=1.645,
            active_constraints=("vehicle",),
        )
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["lane"]["dualized"])

    def test_absolute_budget_is_diagnostic_across_scenarios(self):
        candidate = [{"vehicle_cost_mean": 1.0, "lane_cost_mean": 1.0}]
        base = [{"vehicle_cost_mean": 1.0, "lane_cost_mean": 1.0}]
        gate = p32_safety_gate(
            candidate,
            base,
            vehicle_budget=0.35,
            lane_budget=0.005,
            confidence_z=1.645,
            budget_applicable=False,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["status"], "not_applicable_cross_scenario")


if __name__ == "__main__":
    unittest.main()
