"""A5 tests for bounded y-fusion and strict-zero A4 equivalence."""

import tempfile
import unittest
from pathlib import Path

import torch

from utilities.avocado.controller import AVOCADOController
from utilities.avocado.core import AVOCADOParameters
from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.avocado_marl.a5_benchmark import run_a5_equivalence
from utilities.avocado_marl.a5_config import A5ExperimentConfig
from utilities.avocado_marl.y_correction import (
    YCorrectionNet,
    build_y_correction_features,
)


A5_CONFIG_PATH = Path("configs/avocado_marl/a5_zero_correction.json")


def _controller(entity_count: int = 3) -> AVOCADOController:
    return AVOCADOController(
        AVOCADOParameters(noise_sigma=0.0),
        batch_size=1,
        entity_count=entity_count,
        controlled_mask=torch.ones(entity_count, dtype=torch.bool),
        security_radii=torch.full((entity_count,), 0.1),
        maximum_speeds=torch.ones(entity_count),
        seed=7,
        complementary_responsibility=True,
    )


class A5ZeroCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = A5ExperimentConfig.from_json(A5_CONFIG_PATH)

    def test_feature_contract_is_dense_local_and_excludes_self_pairs(self):
        controller = _controller()
        positions = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.8]]])
        velocities = torch.tensor([[[0.2, 0.0], [0.0, 0.1], [0.0, 0.0]]])
        yaws = torch.zeros(1, 3, 1)
        features = build_y_correction_features(
            controller,
            positions,
            velocities,
            yaws,
            candidate_count=2,
        )
        self.assertEqual(features.values.shape, (1, 3, 3, 14))
        self.assertFalse(bool(torch.diagonal(features.pair_mask, dim1=1, dim2=2).any()))
        self.assertTrue(bool((features.pair_mask.sum(dim=-1) <= 2).all()))
        torch.testing.assert_close(
            features.values[..., -1], features.pair_mask.float()
        )

    def test_strict_zero_network_is_bounded_exact_and_frozen(self):
        network = YCorrectionNet(
            feature_dim=14,
            hidden_sizes=(32, 32),
            maximum_correction=0.1,
            temperature=1.0,
            strict_zero=True,
            freeze=True,
        )
        features = torch.randn(2, 4, 4, 14)
        mask = ~torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
        output = network(features, torch.ones(2, 4, 4), mask)
        self.assertTrue(bool((output.correction == 0).all()))
        self.assertTrue(bool((output.correction.abs() <= 0.1).all()))
        self.assertFalse(any(parameter.requires_grad for parameter in network.parameters()))

    def test_controller_none_and_zero_correction_are_stepwise_identical(self):
        baseline = _controller(entity_count=2)
        corrected = _controller(entity_count=2)
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        velocities = torch.tensor([[[0.4, 0.0], [-0.4, 0.0]]])
        preferred = velocities.clone()
        zero = torch.zeros(1, 2, 2)
        for _ in range(4):
            baseline_action = baseline.step(positions, velocities, preferred)
            corrected_action = corrected.step(
                positions,
                velocities,
                preferred,
                estimated_opinion_correction=zero,
            )
            torch.testing.assert_close(baseline_action, corrected_action, rtol=0, atol=0)
            torch.testing.assert_close(
                baseline.last_estimated_opinion,
                corrected.last_fused_estimated_opinion,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                baseline.opinion, corrected.opinion, rtol=0, atol=0
            )
            positions = positions + baseline_action * baseline.parameters.dt
            velocities = baseline_action
            preferred = baseline_action

    def test_a4_a5_environment_rollout_and_trace_are_exact(self):
        a4_config = A3RoadExperimentConfig.from_json(
            Path("configs/avocado/a3_road_environment.json")
        )
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.pt"
            result = run_a5_equivalence(
                self.config,
                a4_config.cases[0],
                episodes_override=1,
                max_steps_override=4,
                trace_output_path=trace_path,
            )
            self.assertTrue(result.passed)
            self.assertTrue(trace_path.is_file())
            trace = torch.load(trace_path, map_location="cpu")
            self.assertEqual(
                set(
                    (
                        "attention",
                        "y_heuristic",
                        "delta_y",
                        "y_fused",
                        "opinion_z",
                        "nominal_action",
                        "executed_action",
                    )
                ).difference(trace),
                set(),
            )
            self.assertTrue(bool((trace["delta_y"] == 0).all()))
            torch.testing.assert_close(
                trace["y_heuristic"], trace["y_fused"], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
