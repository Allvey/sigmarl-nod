"""A2 integration checks for the independent validation entry point."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from utilities.avocado.benchmark import run_benchmark, run_case
from utilities.avocado.config import AVOCADOExperimentConfig


CONFIG_PATH = Path("configs/avocado/a2_strict_benchmark.json")


class StrictBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AVOCADOExperimentConfig.from_json(CONFIG_PATH)

    def test_head_on_conflict_is_exposed_and_avocado_resolves_it(self):
        case = self.config.cases[0]
        preferred, _ = run_case(
            self.config,
            case,
            "preferred",
            episodes_override=2,
            max_steps_override=200,
        )
        avocado, _ = run_case(
            self.config,
            case,
            "avocado",
            episodes_override=2,
            max_steps_override=200,
        )
        self.assertEqual(preferred.collision_rate, 1.0)
        self.assertEqual(avocado.collision_rate, 0.0)
        self.assertEqual(avocado.success_rate, 1.0)
        self.assertGreater(avocado.maximum_attention, 0.9)
        self.assertGreater(avocado.mean_absolute_opinion, 0.05)

    def test_benchmark_persists_summary_and_passes_gate(self):
        reduced = replace(
            self.config,
            planners=("preferred", "avocado"),
            cases=(self.config.cases[0],),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_benchmark(
                reduced,
                output_directory=Path(directory),
                save_plots=False,
                episodes_override=2,
                max_steps_override=200,
            )
            self.assertTrue(result.validation.passed)
            summary_path = Path(directory) / "summary.json"
            self.assertTrue(summary_path.is_file())
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
            self.assertTrue(summary["validation"]["passed"])
            self.assertEqual(len(summary["metrics"]), 2)


if __name__ == "__main__":
    unittest.main()
