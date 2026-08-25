"""Pure filesystem checks for final/intermediate testing checkpoint selection."""

import json
import tempfile
import unittest
from pathlib import Path

from utilities.experiment_artifacts import (
    resolve_evidence_critic_pair,
    resolve_latest_testable_run,
    resolve_policy_checkpoint,
    resolve_policy_critic_pair,
)


class TestingCheckpointResolutionTests(unittest.TestCase):
    def test_final_policy_is_preferred_over_intermediate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            (run_directory / "reward9.00_policy.pth").touch()
            final_policy = run_directory / "final_policy.pth"
            final_policy.touch()

            self.assertEqual(
                resolve_policy_checkpoint(run_directory),
                final_policy.resolve(),
            )

    def test_highest_reward_intermediate_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            lower = run_directory / "reward-0.52_policy.pth"
            higher = run_directory / "reward1.25_policy.pth"
            lower.touch()
            higher.touch()

            self.assertEqual(
                resolve_policy_checkpoint(run_directory),
                higher.resolve(),
            )

    def test_newest_in_progress_run_is_testable_without_final_policy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "runs" / "base-seed0-current"
            run_directory.mkdir(parents=True)
            (run_directory / "config_resolved.json").write_text(
                json.dumps({"seed": 0}),
                encoding="utf-8",
            )
            (run_directory / "reward-0.52_policy.pth").touch()

            self.assertEqual(
                resolve_latest_testable_run(str(root)),
                run_directory.resolve(),
            )

    def test_non_policy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            critic = run_directory / "reward1.00_critic.pth"
            critic.touch()

            with self.assertRaisesRegex(ValueError, "testable policy"):
                resolve_policy_checkpoint(run_directory, critic)

    def test_intermediate_policy_and_critic_are_resolved_as_a_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            policy = run_directory / "reward-0.52_policy.pth"
            critic = run_directory / "reward-0.52_critic.pth"
            policy.touch()
            critic.touch()

            self.assertEqual(
                resolve_policy_critic_pair(run_directory),
                (policy.resolve(), critic.resolve()),
            )

    def test_intermediate_base_requires_matching_critic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            (run_directory / "reward-0.52_policy.pth").touch()

            with self.assertRaisesRegex(FileNotFoundError, "matching critic"):
                resolve_policy_critic_pair(run_directory)

    def test_m5_evidence_requires_same_reward_critic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            evidence = run_directory / "reward2.50_evidence_net.pth"
            critic = run_directory / "reward2.50_critic.pth"
            evidence.touch()
            critic.touch()

            self.assertEqual(
                resolve_evidence_critic_pair(run_directory),
                (evidence.resolve(), critic.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
