"""M9 unified trainer and checkpoint contracts."""

import random
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from utilities.opinion.checkpoint import (
    load_m9_checkpoint,
    restore_rng_state,
    save_m9_checkpoint,
)
from utilities.opinion.config import (
    OpinionConfigError,
    load_opinion_experiment,
    require_m9_supported_mode,
)
from utilities.opinion.trainer import (
    OpinionTrainingSchedule,
    clip_m9_gradients,
)
from main_testing_opinion import main as test_opinion
from main_training_opinion import main as train_opinion


class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_policy_net = nn.Linear(3, 2)
        self.evidence_net = nn.Linear(4, 1)


def make_optimizer(bridge, critic):
    return torch.optim.Adam(
        [
            {
                "params": bridge.base_policy_net.parameters(),
                "lr": 2e-5,
                "lr_scale": 0.02,
                "group_name": "base_actor",
            },
            {
                "params": bridge.evidence_net.parameters(),
                "lr": 1e-4,
                "lr_scale": 0.1,
                "group_name": "evidence",
            },
            {
                "params": critic.parameters(),
                "lr": 1e-3,
                "lr_scale": 1.0,
                "group_name": "critic",
            },
        ]
    )


class M9TrainerTests(unittest.TestCase):
    def test_all_committed_m9_modes_are_valid(self):
        root = Path(__file__).resolve().parents[2]
        expected = {
            "m9_evidence_only.json": "evidence_only",
            "m9_joint_from_base.json": "joint",
            "m9_joint_from_scratch.json": "joint",
            "m9_joint_from_scratch_pilot.json": "joint",
            "m9_joint_from_m8.json": "joint",
            "m9_warmup_then_joint.json": "warmup_then_joint",
        }
        for filename, mode in expected.items():
            experiment = load_opinion_experiment(
                root / "configs" / "opinion" / filename
            )
            require_m9_supported_mode(experiment)
            self.assertEqual(experiment.config.stage, "joint")
            self.assertEqual(experiment.config.opinion.trainer.mode, mode)

    def test_scratch_config_matches_original_sigmarl_budget(self):
        root = Path(__file__).resolve().parents[2]
        experiment = load_opinion_experiment(
            root / "configs" / "opinion" / "m9_joint_from_scratch.json"
        )
        trainer = experiment.config.opinion.trainer
        bridge = experiment.config.opinion.policy_bridge
        sequence = experiment.config.opinion.sequence_ppo

        self.assertEqual(trainer.initialization, "none")
        self.assertEqual(trainer.mode, "joint")
        self.assertEqual(trainer.evidence_warmup_iterations, 0)
        self.assertEqual(trainer.base_anchor_coefficient, 0.0)
        self.assertTrue(experiment.config.opinion.residual.apply_to_action)
        self.assertTrue(sequence.use_base_ppo_update)
        self.assertIsNone(trainer.source_output_root)
        self.assertIsNone(bridge.base_output_root)
        self.assertIsNone(sequence.source_output_root)
        self.assertFalse(bridge.freeze_base_actor)
        self.assertEqual(sequence.evidence_learning_rate_scale, 0.1)
        self.assertEqual(experiment.parameters.n_iters, 250)
        self.assertEqual(experiment.parameters.frames_per_batch, 4096)
        self.assertEqual(experiment.parameters.num_epochs, 60)
        self.assertEqual(experiment.parameters.minibatch_size, 512)
        self.assertEqual(experiment.parameters.lr, 2e-4)
        self.assertEqual(experiment.parameters.lr_min, 1e-5)

    def test_scratch_entrypoint_never_resolves_or_loads_a_base_run(self):
        root = Path(__file__).resolve().parents[2]
        config = root / "configs" / "opinion" / "m9_joint_from_scratch.json"

        with patch(
            "main_training_opinion.resolve_latest_run",
            side_effect=AssertionError("scratch mode must not resolve Base runs"),
        ), patch(
            "main_training_opinion.resolve_latest_testable_run",
            side_effect=AssertionError("scratch mode must not resolve Base runs"),
        ), patch("main_training_opinion.train_base") as mocked_train:
            mocked_train.return_value = Path("scratch-run")
            result = train_opinion(config)

        self.assertEqual(result, Path("scratch-run"))
        runtime = mocked_train.call_args.kwargs["opinion_policy_config"]
        snapshots = mocked_train.call_args.kwargs["supplementary_snapshots"]
        resolved = snapshots["opinion_config_resolved.json"]
        self.assertTrue(runtime["initialize_from_scratch"])
        self.assertTrue(runtime["use_base_ppo_update"])
        self.assertFalse(runtime["sequence_buffer_enabled"])
        self.assertFalse(runtime["sequence_evidence_training"])
        self.assertNotIn("base_actor_checkpoint", runtime)
        self.assertNotIn("base_critic_checkpoint", runtime)
        self.assertNotIn("initial_policy_checkpoint", runtime)
        self.assertNotIn("resolved_base_run_directory", resolved)
        self.assertEqual(
            mocked_train.call_args.kwargs["artifact_stage"],
            "trainer_joint_from_scratch",
        )

    def test_scratch_config_rejects_any_external_base_or_anchor(self):
        root = Path(__file__).resolve().parents[2]
        source_path = (
            root / "configs" / "opinion" / "m9_joint_from_scratch.json"
        )
        with source_path.open("r", encoding="utf-8") as file:
            source = json.load(file)

        invalid_cases = (
            ("base_output_root", "outputs/base/"),
            ("base_anchor_coefficient", 0.001),
        )
        for field, value in invalid_cases:
            candidate = json.loads(json.dumps(source))
            if field == "base_output_root":
                candidate["opinion"]["policy_bridge"][field] = value
            else:
                candidate["opinion"]["trainer"][field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                candidate["base_config"] = str(root / "config.json")
                with path.open("w", encoding="utf-8") as file:
                    json.dump(candidate, file)
                with self.assertRaises(OpinionConfigError):
                    load_opinion_experiment(path)

    def test_scratch_run_is_accepted_by_testing_entrypoint(self):
        root = Path(__file__).resolve().parents[2]
        config = root / "configs" / "opinion" / "m9_joint_from_scratch.json"
        run_directory = root / "outputs" / "opinion" / "scratch-run"
        checkpoint = run_directory / "reward1.00_policy.pth"

        with patch("main_testing_opinion.test_base") as mocked_test:
            test_opinion(config, run_directory, checkpoint)

        args = mocked_test.call_args.args
        kwargs = mocked_test.call_args.kwargs
        self.assertEqual(args[1], run_directory)
        self.assertEqual(args[2], checkpoint)
        self.assertEqual(kwargs["opinion_policy_config"]["mode"], "stateful_opinion")

    def test_opinion_testing_forwards_video_request(self):
        root = Path(__file__).resolve().parents[2]
        config = root / "configs" / "opinion" / "m9_joint_from_scratch.json"
        run_directory = root / "outputs" / "opinion" / "scratch-run"
        checkpoint = run_directory / "reward1.00_policy.pth"

        with patch("main_testing_opinion.test_base") as mocked_test:
            test_opinion(
                config,
                run_directory,
                checkpoint,
                save_simulation_video=True,
            )

        self.assertTrue(
            mocked_test.call_args.kwargs["save_simulation_video"]
        )

    def test_schedule_freezes_then_activates_base_without_resetting_groups(self):
        bridge = Bridge()
        critic = nn.Linear(3, 1)
        optimizer = make_optimizer(bridge, critic)
        schedule = OpinionTrainingSchedule(
            {
                "mode": "warmup_then_joint",
                "evidence_warmup_iterations": 2,
                "base_actor_learning_rate_scale": 1.0,
                "evidence_learning_rate_scale": 0.1,
                "critic_learning_rate_scale": 1.0,
            }
        )
        warmup = schedule.phase_for_iteration(2)
        schedule.apply(warmup, bridge, optimizer)
        self.assertFalse(next(bridge.base_policy_net.parameters()).requires_grad)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)

        joint = schedule.phase_for_iteration(3)
        schedule.apply(joint, bridge, optimizer)
        self.assertTrue(next(bridge.base_policy_net.parameters()).requires_grad)
        self.assertGreater(optimizer.param_groups[0]["lr"], 0.0)
        self.assertEqual([g["group_name"] for g in optimizer.param_groups], [
            "base_actor", "evidence", "critic"
        ])

    def test_gradient_clipping_separates_evidence_from_base_and_critic(self):
        bridge = Bridge()
        critic = nn.Linear(3, 1)
        optimizer = make_optimizer(bridge, critic)
        groups = {
            group["group_name"]: group for group in optimizer.param_groups
        }

        base_parameter = groups["base_actor"]["params"][0]
        evidence_parameter = groups["evidence"]["params"][0]
        critic_parameter = groups["critic"]["params"][0]
        base_parameter.grad = torch.full_like(base_parameter, 3.0)
        critic_parameter.grad = torch.full_like(critic_parameter, 4.0)
        evidence_parameter.grad = torch.full_like(evidence_parameter, 100.0)

        base_critic_norm_before = torch.linalg.vector_norm(
            torch.cat(
                [
                    base_parameter.grad.flatten(),
                    critic_parameter.grad.flatten(),
                ]
            )
        )
        expected_scale = 1.0 / float(base_critic_norm_before)

        norms = clip_m9_gradients(optimizer, max_grad_norm=1.0)

        self.assertGreater(norms["evidence"], norms["base_critic"])
        self.assertTrue(
            torch.allclose(
                base_parameter.grad,
                torch.full_like(base_parameter, 3.0 * expected_scale),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                critic_parameter.grad,
                torch.full_like(critic_parameter, 4.0 * expected_scale),
                atol=1e-6,
            )
        )
        self.assertLessEqual(
            float(evidence_parameter.grad.norm()), 1.0 + 1e-6
        )

    def test_checkpoint_roundtrip_restores_models_optimizer_metrics_and_rng(self):
        torch.manual_seed(3)
        random.seed(3)
        np.random.seed(3)
        bridge = Bridge()
        critic = nn.Linear(3, 1)
        policy = nn.Sequential(bridge.base_policy_net, nn.Tanh())
        optimizer = make_optimizer(bridge, critic)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_checkpoint.pt"
            save_m9_checkpoint(
                path,
                iteration=7,
                training_mode="joint",
                training_phase="joint",
                policy=policy,
                critic=critic,
                optimizer=optimizer,
                artifact_iterations=[{"iteration": 7, "episode_reward_mean": 1.0}],
                opinion_runtime_config={"trainer": {"mode": "joint"}},
            )
            expected_random = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = float(torch.rand(()))
            payload = load_m9_checkpoint(path, "cpu")
            self.assertEqual(payload["iteration"], 7)
            self.assertEqual(payload["training_mode"], "joint")
            restore_rng_state(payload)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.rand()), expected_numpy)
            self.assertEqual(float(torch.rand(())), expected_torch)


if __name__ == "__main__":
    unittest.main()
