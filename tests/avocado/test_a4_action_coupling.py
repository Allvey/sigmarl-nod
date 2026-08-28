"""A4 integration tests for Base-MAPPO + fixed AVOCADO coupling."""

import unittest
from pathlib import Path

from utilities.avocado.road_config import A3RoadExperimentConfig, RoadCaseConfig
from utilities.avocado_marl.benchmark import (
    resolve_a4_policy_source,
    run_a4_rollout,
)
from utilities.avocado_marl.config import A4ExperimentConfig


CONFIG_PATH = Path("configs/avocado_marl/a4_base_avocado.json")


class A4ActionCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = A4ExperimentConfig.from_json(CONFIG_PATH)
        cls.a3_config = A3RoadExperimentConfig.from_json(cls.config.a3_config)

    def test_configuration_keeps_a4_learning_disabled(self):
        self.assertEqual(self.a3_config.parameters.opinion_bias, 0.0)
        self.assertEqual(self.config.coupling.velocity_continuity_weight, 0.0)
        self.assertTrue(self.config.base_policy.deterministic)

    def test_default_base_policy_source_is_resolvable(self):
        run_directory, checkpoint = resolve_a4_policy_source(self.config)
        self.assertTrue((run_directory / "config_resolved.json").is_file())
        self.assertTrue(checkpoint.is_file())

    def test_raw_base_action_passes_through_exactly(self):
        metrics = run_a4_rollout(
            self.config,
            self.a3_config.cases[0],
            "base_mappo",
            episodes_override=1,
            max_steps_override=3,
        )
        self.assertEqual(metrics.bridge.action_intervention_rate, 0.0)
        self.assertEqual(metrics.bridge.no_conflict_passthrough_rate, 1.0)
        self.assertEqual(metrics.bridge.nonfinite_action_count, 0)
        self.assertEqual(metrics.bridge.nominal_executed_speed_correlation, 1.0)

    def test_hybrid_executes_frozen_actor_and_avocado_on_intersection(self):
        case = RoadCaseConfig(
            name="intersection_2_6",
            scenario_type="intersection_2",
            n_agents=6,
        )
        metrics = run_a4_rollout(
            self.config,
            case,
            "base_mappo_avocado",
            episodes_override=1,
            max_steps_override=3,
        )
        self.assertEqual(metrics.executed_steps, 3)
        self.assertEqual(metrics.bridge.action_samples, 18)
        self.assertEqual(metrics.bridge.nonfinite_action_count, 0)
        self.assertGreater(metrics.bridge.mean_nominal_speed_mps, 0.0)
        self.assertGreater(metrics.bridge.mean_executed_speed_mps, 0.0)
        self.assertGreater(metrics.bridge.maximum_attention, 0.0)


if __name__ == "__main__":
    unittest.main()
