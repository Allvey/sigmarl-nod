"""P0 Base passthrough, artifact, and unified-entrypoint contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import main_testing
import main_training
from utilities.psb_marl.checkpoint import sha256_file
from utilities.psb_marl.config import PSBConfigError, load_psb_experiment
from utilities.psb_marl.evaluator import test_psb as run_psb_test
from utilities.psb_marl.trainer import train_psb


class P0PassthroughTests(unittest.TestCase):
    def _experiment(self, root: Path) -> tuple[Path, Path, Path]:
        repository = Path(__file__).resolve().parents[2]
        with (repository / "config.json").open("r", encoding="utf-8") as stream:
            base_config = json.load(stream)
        base_config_path = root / "base.json"
        base_config_path.write_text(json.dumps(base_config), encoding="utf-8")

        base_run = root / "base-run"
        base_run.mkdir()
        (base_run / "config_resolved.json").write_text(
            json.dumps(base_config), encoding="utf-8"
        )
        policy = base_run / "final_policy.pth"
        critic = base_run / "final_critic.pth"
        torch.save({"weight": torch.arange(3)}, policy)
        torch.save({"weight": torch.arange(5)}, critic)
        config = {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": "p0_base_passthrough",
            "base_config": str(base_config_path),
            "output_root": str(root / "psb-output"),
            "base": {
                "run_directory": str(base_run),
                "policy_checkpoint": str(policy),
                "critic_checkpoint": str(critic),
            },
        }
        config_path = root / "p0.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, policy, critic

    def test_config_loads_only_exact_p0_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, policy, critic = self._experiment(Path(directory))
            experiment = load_psb_experiment(config_path)
            self.assertEqual(experiment.stage, "p0_base_passthrough")
            self.assertEqual(experiment.base.policy_checkpoint, policy.resolve())
            self.assertEqual(experiment.base.critic_checkpoint, critic.resolve())

            source = json.loads(config_path.read_text(encoding="utf-8"))
            source["unexpected"] = True
            config_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(PSBConfigError):
                load_psb_experiment(config_path)

    def test_training_packages_policy_and_critic_byte_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, source_policy, source_critic = self._experiment(
                Path(directory)
            )
            run_directory = train_psb(config_path)
            packaged_policy = run_directory / "final_policy.pth"
            packaged_critic = run_directory / "final_critic.pth"
            self.assertEqual(
                sha256_file(packaged_policy), sha256_file(source_policy)
            )
            self.assertEqual(
                sha256_file(packaged_critic), sha256_file(source_critic)
            )
            proof = json.loads(
                (run_directory / "p0_equivalence.json").read_text(encoding="utf-8")
            )
            deployment = json.loads(
                (run_directory / "deployment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            status = json.loads(
                (run_directory / "training_status.json").read_text(encoding="utf-8")
            )
            payload = torch.load(run_directory / "final_checkpoint.pt")
            self.assertTrue(proof["policy_bytes_identical"])
            self.assertTrue(proof["critic_bytes_identical"])
            self.assertEqual(deployment["selected"], "base_passthrough")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(payload["stage"], "p0_base_passthrough")

    def test_p0_rejects_resume_and_iteration_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._experiment(Path(directory))
            with self.assertRaisesRegex(PSBConfigError, "resume"):
                train_psb(config_path, resume_checkpoint=Path("checkpoint.pt"))
            with self.assertRaisesRegex(PSBConfigError, "iterations"):
                train_psb(config_path, iterations_override=1)

    def test_p0_evaluator_proves_identity_before_rollout(self):
        class FakeRollout:
            def __init__(self):
                self.values = {
                    ("agents", "action"): torch.zeros(2, 3, 4, 2),
                    ("next", "agents", "reward"): torch.ones(2, 3, 4, 1),
                }

            def get(self, key):
                return self.values[key]

        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._experiment(Path(directory))
            run_directory = train_psb(config_path)
            with patch("main_testing.test_base", return_value=FakeRollout()) as mocked:
                report = run_psb_test(
                    config_path,
                    run_directory=run_directory,
                    scenario_type="CPM_mixed",
                    max_steps=4,
                    episodes=2,
                    seeds=(7, 8),
                    render=False,
                    compare_base=True,
                    promote_if_noninferior=True,
                )
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(
                report["noninferiority_result"],
                "proven_by_identical_policy_checkpoint",
            )
            self.assertTrue(report["equivalence"]["policy_bytes_identical"])
            self.assertTrue(report["equivalence"]["critic_bytes_identical"])
            self.assertTrue(
                all(item["nonfinite_action_count"] == 0 for item in report["rollouts"])
            )

    def test_main_training_dispatches_psb_without_constructing_base(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._experiment(Path(directory))
            expected = Path(directory) / "dispatched-run"
            with patch(
                "utilities.psb_marl.trainer.train_psb", return_value=expected
            ) as mocked_psb, patch(
                "main_training.train_base",
                side_effect=AssertionError("PSB must not enter Base training"),
            ):
                result = main_training.main(config_path)
            self.assertEqual(result, expected)
            mocked_psb.assert_called_once_with(
                config_path,
                resume_checkpoint=None,
                iterations_override=None,
            )

    def test_main_testing_forwards_unified_psb_options(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._experiment(Path(directory))
            run_directory = Path(directory) / "run"
            checkpoint = run_directory / "final_policy.pth"
            with patch(
                "utilities.psb_marl.evaluator.test_psb",
                return_value={"passed": True},
            ) as mocked_test:
                result = main_testing.main(
                    config_path,
                    run_directory,
                    checkpoint,
                    scenario_type="CPM_mixed",
                    max_steps=64,
                    episodes=2,
                    seeds=(3, 4),
                    render=False,
                    compare_base=True,
                    promote_if_noninferior=True,
                )
            self.assertEqual(result, {"passed": True})
            kwargs = mocked_test.call_args.kwargs
            self.assertEqual(kwargs["scenario_type"], "CPM_mixed")
            self.assertEqual(kwargs["max_steps"], 64)
            self.assertEqual(kwargs["episodes"], 2)
            self.assertEqual(kwargs["seeds"], (3, 4))
            self.assertFalse(kwargs["render"])
            self.assertTrue(kwargs["compare_base"])
            self.assertTrue(kwargs["promote_if_noninferior"])


if __name__ == "__main__":
    unittest.main()
