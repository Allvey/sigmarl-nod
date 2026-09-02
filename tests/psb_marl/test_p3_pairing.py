"""P3.0 immutable source bridge and paired rollout data contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from tensordict import TensorDict

from utilities.psb_marl.checkpoint import sha256_file
from utilities.psb_marl.config import PSBConfigError, load_psb_experiment
from utilities.psb_marl.evaluator import test_psb
from utilities.psb_marl.p3_pairing import build_paired_batch
from utilities.psb_marl.trainer import train_psb


CONFIG = Path("configs/psb_marl/p3_0_paired_rollout_equivalence.json")


def _rollout(action_value: float, reward_value: float) -> TensorDict:
    environments, steps, agents = 2, 4, 4
    candidate_count = 2
    neighbor_ids = torch.empty(
        environments, steps, agents, candidate_count, dtype=torch.long
    )
    for ego in range(agents):
        candidates = [index for index in range(agents) if index != ego][
            :candidate_count
        ]
        neighbor_ids[..., ego, :] = torch.tensor(candidates)
    return TensorDict(
        {
            ("agents", "observation"): torch.full(
                (environments, steps, agents, 6), reward_value
            ),
            ("agents", "action"): torch.full(
                (environments, steps, agents, 2), action_value
            ),
            ("agents", "psb", "z_next_dense"): torch.zeros(
                environments, steps, agents, agents
            ),
            ("agents", "psb", "b"): torch.zeros(
                environments, steps, agents, agents
            ),
            ("agents", "info", "neighbor_ids"): neighbor_ids,
            ("agents", "info", "pair_mask"): torch.ones(
                environments,
                steps,
                agents,
                candidate_count,
                dtype=torch.bool,
            ),
            ("next", "agents", "info", "urgency"): torch.full(
                (
                    environments,
                    steps,
                    agents,
                    candidate_count,
                ),
                0.25,
            ),
            ("next", "agents", "info", "confidence"): torch.ones(
                environments, steps, agents, candidate_count
            ),
            ("next", "agents", "info", "pair_mask"): torch.ones(
                environments,
                steps,
                agents,
                candidate_count,
                dtype=torch.bool,
            ),
            ("next", "agents", "info", "distance_left_b"): torch.full(
                (environments, steps, agents), 0.1
            ),
            ("next", "agents", "info", "distance_right_b"): torch.full(
                (environments, steps, agents), 0.1
            ),
            ("next", "agents", "reward"): torch.full(
                (environments, steps, agents, 1), reward_value
            ),
            ("next", "done"): torch.zeros(
                environments, steps, 1, dtype=torch.bool
            ),
            ("next", "terminated"): torch.zeros(
                environments, steps, 1, dtype=torch.bool
            ),
            (
                "next",
                "agents",
                "info",
                "is_collision_with_agents",
            ): torch.zeros(environments, steps, agents, dtype=torch.bool),
            (
                "next",
                "agents",
                "info",
                "is_collision_with_lanelets",
            ): torch.zeros(environments, steps, agents, dtype=torch.bool),
        },
        batch_size=[environments, steps],
    )


def _summary(rollout, seed, **_kwargs):
    reward = rollout.get(("next", "agents", "reward"))
    return {
        "seed": int(seed),
        "mean_reward_per_agent_step": float(reward.mean()),
        "collision_with_agents_rate": 0.0,
        "collision_with_lanelets_rate": 0.0,
        "total_collision_rate": 0.0,
    }


class P30PairingTests(unittest.TestCase):
    def test_locked_repository_config_selects_certified_p21u_parent(self):
        experiment = load_psb_experiment(CONFIG)

        self.assertEqual(experiment.stage, "p3_paired_rollout_equivalence")
        self.assertEqual(experiment.effective_training_seed, 0)
        self.assertFalse(experiment.paired_rollout.learning_enabled)
        self.assertEqual(
            experiment.source_p2_runtime_config(),
            json.loads(
                (experiment.parent_run / "psb_config_resolved.json").read_text(
                    encoding="utf-8"
                )
            )["runtime_config"],
        )

    def test_main_training_dispatches_to_byte_exact_p3_packaging(self):
        experiment = load_psb_experiment(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            isolated = replace(experiment, output_root=directory)
            with patch(
                "utilities.psb_marl.trainer.load_psb_experiment",
                return_value=isolated,
            ):
                run = train_psb(CONFIG)

            self.assertEqual(
                sha256_file(run / "candidate_policy.pth"),
                sha256_file(experiment.parent_run / "candidate_policy.pth"),
            )
            self.assertEqual(
                sha256_file(run / "candidate_critic.pth"),
                sha256_file(experiment.parent_run / "candidate_critic.pth"),
            )
            self.assertEqual(
                sha256_file(run / "final_policy.pth"),
                sha256_file(experiment.base.policy_checkpoint),
            )
            manifest = json.loads(
                (run / "deployment_manifest.json").read_text(encoding="utf-8")
            )
            status = json.loads(
                (run / "training_status.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["learning_enabled"])
            self.assertEqual(manifest["selected"], "base_fallback_p3_pairing_only")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["iteration"], 0)

    def test_pair_contract_contains_differential_critic_inputs(self):
        candidate = _rollout(action_value=0.25, reward_value=2.0)
        base = _rollout(action_value=-0.25, reward_value=1.5)

        batch = build_paired_batch(candidate, base)
        summary = batch.summary(seed=13)

        self.assertTrue(
            torch.equal(
                batch.delta_reward,
                torch.full_like(batch.delta_reward, 0.5),
            )
        )
        self.assertEqual(
            batch.candidate_observation.shape, batch.base_observation.shape
        )
        self.assertEqual(summary["candidate_branch_state_shape"], [2, 4, 4, 4])
        self.assertEqual(summary["candidate_edge_mask_shape"], [2, 4, 4, 4])
        self.assertTrue(summary["finite"])

    def test_testing_runs_p3_source_and_base_with_the_same_seed(self):
        experiment = load_psb_experiment(CONFIG)
        candidate = _rollout(action_value=0.2, reward_value=1.1)
        base = _rollout(action_value=0.0, reward_value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            isolated = replace(experiment, output_root=directory)
            with patch(
                "utilities.psb_marl.trainer.load_psb_experiment",
                return_value=isolated,
            ):
                run = train_psb(CONFIG)
            with patch(
                "main_testing.test_base",
                side_effect=[candidate, candidate.clone(), base],
            ) as mocked, patch(
                "utilities.psb_marl.evaluator._rollout_summary",
                side_effect=_summary,
            ), patch("builtins.print"):
                report = test_psb(
                    CONFIG,
                    run_directory=run,
                    checkpoint_path=run / "candidate_policy.pth",
                    scenario_type="intersection_2",
                    max_steps=4,
                    episodes=2,
                    seeds=(601,),
                    render=False,
                    compare_base=True,
                )

        self.assertEqual(mocked.call_count, 3)
        self.assertTrue(report["passed"])
        self.assertTrue(report["source_equivalence_passed"])
        self.assertTrue(report["paired_contract_passed"])
        self.assertEqual(report["evaluation_protocol"]["seeds"], [601])

    def test_p3_rejects_learning(self):
        source = json.loads(CONFIG.read_text(encoding="utf-8"))
        source["paired_rollout"]["learning_enabled"] = True
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid-p3.json"
            # Relative repository paths are resolved from the config's parent,
            # so retain absolute paths while testing only the stage guard.
            repository = CONFIG.resolve().parents[2]
            for key in ("base_config", "parent_run", "robustness_summary"):
                source[key] = str((repository / source[key]).resolve())
            for key in (
                "run_directory",
                "policy_checkpoint",
                "critic_checkpoint",
            ):
                source["base"][key] = str(
                    (repository / source["base"][key]).resolve()
                )
            config.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(PSBConfigError, "learning_enabled=false"):
                load_psb_experiment(config)

    def test_p3_testing_requires_base_comparison(self):
        with self.assertRaisesRegex(ValueError, "requires --compare-base"):
            test_psb(CONFIG, render=False, compare_base=False)


if __name__ == "__main__":
    unittest.main()
