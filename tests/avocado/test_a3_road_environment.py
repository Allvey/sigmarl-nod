"""A3 integration tests against SigmaRL's actual road environment."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from utilities.avocado.road_benchmark import (
    _build_road_environment,
    run_live_road_case,
    run_road_benchmark,
    run_road_case,
)
from utilities.avocado.road_config import (
    A3RoadExperimentConfig,
    RoadCaseConfig,
)
from utilities.kinematic_bicycle import KinematicBicycle


CONFIG_PATH = Path("configs/avocado/a3_road_environment.json")


class RoadEnvironmentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = A3RoadExperimentConfig.from_json(CONFIG_PATH)

    def test_environment_uses_project_road_scenario_and_bicycle_dynamics(self):
        case = self.config.cases[0]
        environment, scenario = _build_road_environment(
            self.config,
            case,
            episodes=1,
            max_steps=5,
        )
        self.assertEqual(scenario.parameters.scenario_type, "CPM_mixed")
        for agent in scenario.world.agents:
            self.assertIsInstance(agent.dynamics, KinematicBicycle)
        before = scenario.world.agents[0].state.pos.clone()
        actions = [torch.tensor([[0.3, 0.0]]) for _ in range(case.n_agents)]
        environment.step(actions)
        after = scenario.world.agents[0].state.pos
        self.assertFalse(torch.equal(before, after))
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    def test_avocado_kb_rollout_emits_finite_road_metrics(self):
        metrics, _ = run_road_case(
            self.config,
            self.config.cases[0],
            "avocado_kb",
            episodes_override=1,
            max_steps_override=5,
        )
        self.assertEqual(metrics.executed_steps, 5)
        self.assertEqual(metrics.nonfinite_action_count, 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.mean_tracking_error_mps)))
        self.assertGreater(metrics.maximum_attention, 0.0)
        self.assertGreaterEqual(
            metrics.maximum_reference_distance_meters,
            metrics.p95_reference_distance_meters,
        )
        self.assertTrue(
            torch.isfinite(torch.tensor(metrics.minimum_lane_clearance_meters))
        )

    def test_intersection_2_joint_constraints_prevent_step_181_regression(self):
        case = RoadCaseConfig(
            name="intersection_2_6",
            scenario_type="intersection_2",
            n_agents=6,
        )
        metrics, _ = run_road_case(
            self.config,
            case,
            "avocado_kb",
            episodes_override=1,
            max_steps_override=200,
        )
        self.assertEqual(metrics.agent_collision_events_per_1000_steps, 0.0)
        self.assertEqual(
            metrics.post_shield_unsafe_pair_events_per_1000_steps, 0.0
        )
        self.assertLess(metrics.p95_reference_distance_meters, 0.05)

    @patch("utilities.avocado.road_benchmark._render_road_environment")
    def test_live_rollout_renders_every_environment_step(self, render_mock):
        metrics, _ = run_road_case(
            self.config,
            self.config.cases[0],
            "avocado_kb",
            episodes_override=1,
            max_steps_override=3,
            render_live=True,
        )
        self.assertEqual(metrics.executed_steps, 3)
        self.assertEqual(render_mock.call_count, 3)

    def test_live_rollout_rejects_multiple_environments(self):
        with self.assertRaisesRegex(ValueError, "exactly one environment"):
            run_road_case(
                self.config,
                self.config.cases[0],
                "avocado_kb",
                episodes_override=2,
                max_steps_override=1,
                render_live=True,
            )

    @patch("utilities.avocado.road_benchmark._render_road_environment")
    def test_live_scenario_selector_builds_requested_environment(
        self, render_mock
    ):
        metrics = run_live_road_case(
            self.config,
            scenario_type="on_ramp_1",
            max_steps_override=2,
        )
        self.assertEqual(metrics.case, "on_ramp_1_8")
        self.assertEqual(metrics.executed_steps, 2)
        self.assertEqual(render_mock.call_count, 2)

    def test_live_scenario_selector_rejects_unknown_scenario(self):
        with self.assertRaisesRegex(ValueError, "Unknown scenario type"):
            run_live_road_case(
                self.config,
                scenario_type="not_a_scenario",
                max_steps_override=1,
            )

    def test_short_benchmark_persists_summary_with_relaxed_test_gate(self):
        validation = replace(
            self.config.validation,
            maximum_agent_collision_events_per_1000_steps=1000.0,
            maximum_lane_collision_events_per_1000_steps=1000.0,
            maximum_mean_tracking_error_mps=1.0,
            maximum_steering_saturation_rate=1.0,
            minimum_route_completion_events_per_1000_steps=0.0,
            minimum_maximum_attention=0.0,
            minimum_agent_collision_improvement=0.0,
        )
        reduced = replace(
            self.config,
            planners=("path_following", "avocado_kb"),
            cases=(self.config.cases[0],),
            validation=validation,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_road_benchmark(
                reduced,
                output_directory=Path(directory),
                save_plots=False,
                episodes_override=1,
                max_steps_override=5,
            )
            self.assertTrue(result.validation.passed)
            self.assertTrue((Path(directory) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
