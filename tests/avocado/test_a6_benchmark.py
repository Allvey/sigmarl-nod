"""A6 paired-comparison configuration and metric tests."""

import math
import unittest
from pathlib import Path

import torch

from utilities.avocado_marl.a6_benchmark import (
    A6ComparisonConfig,
    _paired_comparisons,
    _trace_metrics,
)
from utilities.avocado_marl.bridge import A4BridgeTrace


class A6BenchmarkTests(unittest.TestCase):
    def test_formal_comparison_uses_multiple_paired_seeds(self):
        config = A6ComparisonConfig.from_json(
            Path("configs/avocado_marl/a6_comparison.json")
        )
        self.assertEqual(config.evaluation.environment_seeds, (0, 1, 2, 3, 4))
        self.assertIn("CPM_mixed", config.evaluation.scenarios)
        self.assertIn("intersection_2", config.evaluation.scenarios)

    def test_trace_metrics_respect_resets_and_correction_pair_mask(self):
        steps, environments, agents = 3, 1, 2
        actions = torch.zeros(steps, environments, agents, 2)
        actions[0, ..., 0] = 0.2
        actions[1, ..., 0] = 0.0
        actions[2, ..., 0] = 0.3
        actions[0, ..., 1] = math.radians(2.0)
        actions[1, ..., 1] = math.radians(-2.0)
        actions[2, ..., 1] = math.radians(3.0)
        pair_mask = ~torch.eye(agents, dtype=torch.bool).view(1, 1, agents, agents)
        pair_mask = pair_mask.expand(steps, environments, -1, -1).clone()
        correction = torch.zeros(steps, environments, agents, agents)
        correction[0][pair_mask[0]] = 0.02
        correction[1][pair_mask[1]] = -0.03
        correction[2][pair_mask[2]] = 0.04
        zeros_action = torch.zeros_like(actions)
        zeros_pair = torch.zeros_like(correction)
        trace = A4BridgeTrace(
            nominal_action=actions,
            pre_shield_action=actions,
            executed_action=actions,
            conflict_mask=torch.ones(steps, environments, agents, dtype=torch.bool),
            intervention_mask=torch.zeros(
                steps, environments, agents, dtype=torch.bool
            ),
            shield_mask=torch.zeros(steps, environments, agents, dtype=torch.bool),
            reset_mask=torch.zeros(steps, environments, agents, dtype=torch.bool),
            heuristic_estimate=zeros_pair,
            estimate_correction=correction,
            fused_estimate=correction,
            opinion=zeros_pair,
            attention=zeros_pair,
            pair_mask=pair_mask,
        )
        features = torch.zeros(steps, environments, agents, agents, 14)
        features[..., 8] = 0.2
        features[..., -1] = pair_mask
        metrics, bins = _trace_metrics(
            trace,
            features,
            maximum_correction=0.1,
            velocity_obstacle_horizon=2.5,
            stop_speed_threshold_mps=0.05,
            steering_reversal_threshold_degrees=1.0,
            ttc_bins_seconds=(0.0, 0.6, 1.2, 2.5),
        )
        self.assertEqual(metrics["correction_pair_samples"], 6.0)
        self.assertEqual(metrics["invalid_pair_maximum_absolute_correction"], 0.0)
        self.assertEqual(metrics["correction_sign_switch_rate"], 1.0)
        self.assertEqual(metrics["conflict_steering_reversal_rate"], 1.0)
        self.assertAlmostEqual(metrics["conflict_stopped_action_rate"], 1.0 / 3.0)
        self.assertEqual(bins["[0,0.6)"]["pair_samples"], 6)

    def test_single_seed_is_reported_as_insufficient(self):
        records = [
            {
                "stage": "a5",
                "scenario_type": "CPM_mixed",
                "seed": 0,
                "metrics": {"mean_reward_per_agent_step": 1.0},
            },
            {
                "stage": "a6",
                "scenario_type": "CPM_mixed",
                "seed": 0,
                "metrics": {"mean_reward_per_agent_step": 2.0},
            },
        ]
        result = _paired_comparisons(records, ("CPM_mixed",))
        metric = result["CPM_mixed"]["metrics"]["mean_reward_per_agent_step"]
        self.assertEqual(metric["conclusion"], "insufficient_seeds")


if __name__ == "__main__":
    unittest.main()
