"""P2.2-R independent seeds, lane gate, and locked holdout aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from utilities.psb_marl.config import load_psb_experiment
from utilities.psb_marl.evaluator import _p2_noninferiority_gate
from utilities.psb_marl.p2_robustness import (
    P2RobustnessError,
    aggregate_robustness,
    load_robustness_protocol,
)


PROTOCOL = Path("configs/psb_marl/p2_2_r_holdout_protocol.json")


class P22RobustnessTests(unittest.TestCase):
    def test_training_seed_is_independent_and_runtime_isolated(self):
        experiments = [
            load_psb_experiment(
                Path(f"configs/psb_marl/p2_2_r_seed{seed}.json")
            )
            for seed in range(3)
        ]

        self.assertEqual([item.seed for item in experiments], [0, 0, 0])
        self.assertEqual(
            [item.effective_training_seed for item in experiments],
            [0, 1, 2],
        )
        self.assertEqual(
            [item.p2_runtime_config()["training_seed"] for item in experiments],
            [0, 1, 2],
        )
        self.assertEqual(
            len(
                {
                    json.dumps(item.p2_runtime_config(), sort_keys=True)
                    for item in experiments
                }
            ),
            3,
        )

    def test_lane_gate_can_reject_when_total_collision_gate_passes(self):
        comparisons = [
            {
                "reward_difference_candidate_minus_base": 0.01,
                "collision_difference_candidate_minus_base": 0.0,
                "lane_collision_difference_candidate_minus_base": 0.002,
            }
            for _ in range(10)
        ]
        result = _p2_noninferiority_gate(
            comparisons,
            {
                "minimum_paired_seeds": 10,
                "confidence_z": 1.645,
                "reward_margin": 0.002,
                "collision_margin": 0.002,
                "lane_collision_margin": 0.001,
            },
        )

        self.assertTrue(result["reward_passed"])
        self.assertTrue(result["collision_passed"])
        self.assertFalse(result["lane_collision_passed"])
        self.assertFalse(result["passed"])

    def _write_run(self, root: Path, seed: int) -> Path:
        experiment = load_psb_experiment(
            Path(f"configs/psb_marl/p2_2_r_seed{seed}.json")
        )
        run = root / f"seed-{seed}"
        run.mkdir()
        (run / "training_status.json").write_text(
            json.dumps({"status": "completed", "iteration": 30}),
            encoding="utf-8",
        )
        (run / "psb_config_resolved.json").write_text(
            json.dumps({"runtime_config": experiment.p2_runtime_config()}),
            encoding="utf-8",
        )
        torch.save({"seed": seed}, run / "candidate_policy.pth")
        protocol = load_robustness_protocol(PROTOCOL)
        for evaluation in protocol["evaluations"]:
            comparisons = [
                {
                    "seed": evaluation_seed,
                    "reward_difference_candidate_minus_base": 0.001 + seed
                    * 0.0001,
                    "vehicle_collision_difference_candidate_minus_base": -0.001,
                    "lane_collision_difference_candidate_minus_base": 0.0002,
                    "collision_difference_candidate_minus_base": -0.0008,
                }
                for evaluation_seed in evaluation["seeds"]
            ]
            gate = {
                "passed": True,
                "reward_passed": True,
                "collision_passed": True,
                "lane_collision_passed": True,
                "structural_passed": True,
                "reward_lower_bound": 0.0005,
                "collision_upper_bound": -0.0002,
                "lane_collision_upper_bound": 0.0005,
            }
            report = {
                "report_label": evaluation["name"],
                "training_seed": seed,
                "evaluation_protocol": {
                    "scenario_type": evaluation["scenario"],
                    "max_steps": evaluation["max_steps"],
                    "episodes": evaluation["episodes"],
                    "seeds": evaluation["seeds"],
                    "compare_base": True,
                },
                "paired_comparisons": comparisons,
                "noninferiority_gate": gate,
            }
            (run / f"p2_manual_validation_{evaluation['name']}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
        return run

    def test_locked_protocol_aggregates_three_training_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self._write_run(root, seed) for seed in range(3)]
            summary = aggregate_robustness(PROTOCOL, runs)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["training_seed_count"], 3)
        self.assertEqual(len(summary["run_results"]), 3)
        self.assertTrue(
            summary["scenario_summaries"]["holdout_cpm_mixed"][
                "all_training_seeds_passed"
            ]
        )
        reward = summary["scenario_summaries"]["holdout_cpm_mixed"][
            "across_training_seeds"
        ]["reward_difference"]
        self.assertAlmostEqual(reward["mean"], 0.0011)
        self.assertGreater(reward["sample_std"], 0.0)

    def test_aggregator_rejects_a_report_with_wrong_holdout_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self._write_run(root, seed) for seed in range(3)]
            report_path = (
                runs[0] / "p2_manual_validation_holdout_cpm_mixed.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["evaluation_protocol"]["seeds"][0] = 999
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaises(P2RobustnessError):
                aggregate_robustness(PROTOCOL, runs)

    def test_aggregator_rejects_identical_candidate_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self._write_run(root, seed) for seed in range(3)]
            duplicate = (runs[0] / "candidate_policy.pth").read_bytes()
            (runs[1] / "candidate_policy.pth").write_bytes(duplicate)

            with self.assertRaisesRegex(
                P2RobustnessError, "byte-identical across training seeds"
            ):
                aggregate_robustness(PROTOCOL, runs)


if __name__ == "__main__":
    unittest.main()
