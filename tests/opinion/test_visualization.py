"""Opinion testing-overlay contracts."""

import unittest
from types import SimpleNamespace

import torch

from utilities.opinion.visualization import update_opinion_visualization


class _Scenario:
    def __init__(self):
        self.lines = None

    def set_opinion_visualization(self, lines):
        self.lines = lines


class OpinionVisualizationTests(unittest.TestCase):
    def test_panel_compares_base_residual_final_and_executed_speed(self):
        scenario = _Scenario()
        env = SimpleNamespace(
            base_env=SimpleNamespace(scenario_name=scenario)
        )
        tensordict = {
            ("agents", "info", "neighbor_ids"): torch.tensor([[[1]]]),
            ("agents", "info", "pair_features"): torch.zeros(1, 1, 1, 10),
            ("agents", "info", "pair_mask"): torch.tensor([[[True]]]),
            ("agents", "info", "urgency"): torch.ones(1, 1, 1),
            ("agents", "info", "confidence"): torch.ones(1, 1, 1),
            ("agents", "opinion", "base_loc"): torch.tensor([[[0.25, 0.1]]]),
            ("agents", "opinion", "residual"): torch.tensor([[[-0.05]]]),
            ("agents", "loc"): torch.tensor([[[0.20, 0.1]]]),
            ("agents", "action"): torch.tensor([[[0.18, 0.05]]]),
        }

        update_opinion_visualization(
            env,
            tensordict,
            {
                "agent_id": 0,
                "prediction_horizon_seconds": 3.0,
                "sensing_distance_meters": 20.0,
            },
        )

        self.assertEqual(scenario.lines[0], "Opinion | ego=0")
        self.assertEqual(
            scenario.lines[1],
            "speed loc | base=+0.2500 residual=-0.0500 final=+0.2000",
        )
        self.assertEqual(
            scenario.lines[2], "executed speed=+0.1800 m/s"
        )


if __name__ == "__main__":
    unittest.main()
