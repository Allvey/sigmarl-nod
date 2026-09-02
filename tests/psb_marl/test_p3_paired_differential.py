"""P3.3 paired differential advantage and configuration contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tensordict.tensordict import TensorDict

from utilities.psb_marl.config import (
    PSBP33PairedDifferentialConfig,
    PSBConfigError,
    load_psb_experiment,
)
from utilities.psb_marl.p3_critic import BaseRelativeDifferentialCritic
from utilities.psb_marl.p3_differential import (
    differential_advantage,
    differential_critic_loss,
    normalize_differential_advantage,
    paired_iteration_seed,
    paired_reset_seed,
    paired_transition_seed,
    save_online_differential_critic,
)
from utilities.psb_marl.p3_dual_evaluation import p33_efficacy_gate
from utilities.psb_marl.p3_pairing import (
    synchronize_paired_transition_boundaries,
)
from utilities.psb_marl.trainer import train_psb


CONFIG = Path(
    "configs/psb_marl/p3_3_paired_differential_primal_dual_ppo.json"
)


class P33PairedDifferentialTests(unittest.TestCase):
    def test_locked_config_reuses_p31_source_and_vehicle_only_dual(self):
        experiment = load_psb_experiment(CONFIG)
        self.assertEqual(
            experiment.stage,
            "p3_paired_differential_primal_dual_ppo",
        )
        runtime = experiment.p33_runtime_config()
        self.assertTrue(runtime["paired_differential"]["common_random_numbers"])
        self.assertTrue(
            runtime["paired_differential"]["reset_at_each_iteration"]
        )
        self.assertTrue(
            runtime["paired_differential"][
                "synchronize_episode_boundaries"
            ]
        )
        self.assertEqual(runtime["primal_dual"]["active_constraints"], ["vehicle"])
        self.assertTrue(runtime["primal_dual"]["normalize_constraints"])
        self.assertTrue(
            runtime["p3_differential_critic_checkpoint"].endswith(
                "candidate_critic.pth"
            )
        )

    def test_config_requires_exact_crn_online_learning_contract(self):
        valid = {
            "common_random_numbers": True,
            "reset_at_each_iteration": True,
            "synchronize_episode_boundaries": True,
            "online_critic_learning_enabled": True,
            "critic_learning_rate_scale": 2.0,
            "huber_delta": 1.0,
            "gradient_clip_norm": 1.0,
            "normalize_advantage": True,
            "advantage_scale_floor": 1e-4,
        }
        for name in (
            "common_random_numbers",
            "reset_at_each_iteration",
            "synchronize_episode_boundaries",
            "online_critic_learning_enabled",
        ):
            invalid = dict(valid)
            invalid[name] = False
            with self.assertRaises(PSBConfigError):
                PSBP33PairedDifferentialConfig.from_dict(invalid)

    def test_vehicle_normalized_differential_advantage_ignores_lane(self):
        target = torch.tensor([[[[2.0, 4.0, 80.0]]]])
        prediction = torch.tensor([[[[0.5, 1.0, -30.0]]]])
        advantage, terms = differential_advantage(
            target,
            prediction,
            vehicle_multiplier=0.2,
            lane_multiplier=10.0,
            vehicle_budget=0.5,
            lane_budget=0.01,
            normalize_constraints=True,
            active_constraints=("vehicle",),
            normalize_advantage=False,
            advantage_scale_floor=1e-4,
        )
        # (2 - .2/.5*4) - (.5 - .2/.5*1) = .3
        torch.testing.assert_close(advantage, torch.tensor([[[[0.3]]]]))
        torch.testing.assert_close(
            terms["target_lagrangian"], torch.tensor([[[[0.4]]]])
        )

    def test_per_agent_advantage_normalization_preserves_agent_axis(self):
        advantage = torch.tensor(
            [
                [[[1.0], [10.0]], [[3.0], [14.0]]],
                [[[5.0], [18.0]], [[7.0], [22.0]]],
            ]
        )
        normalized, _, _ = normalize_differential_advantage(
            advantage, scale_floor=1e-4
        )
        torch.testing.assert_close(
            normalized.mean(dim=(0, 1)), torch.zeros(2, 1), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            normalized.std(dim=(0, 1), unbiased=False),
            torch.ones(2, 1),
            atol=1e-6,
            rtol=0,
        )

    def test_online_critic_loss_updates_only_critic_parameters(self):
        model = BaseRelativeDifferentialCritic(
            observation_dim=3,
            embedding_dim=8,
            hidden_sizes=(8,),
        )
        candidate = torch.randn(2, 4, 3, requires_grad=True)
        base = torch.randn(2, 4, 3, requires_grad=True)
        z = torch.randn(2, 4, 4, requires_grad=True)
        mask = ~torch.eye(4, dtype=torch.bool).expand(2, 4, 4)
        target = torch.randn(2, 4, 3, requires_grad=True)
        loss, prediction = differential_critic_loss(
            model,
            candidate_observation=candidate,
            base_observation=base,
            candidate_z=z,
            edge_mask=mask,
            target_channels=target,
            huber_delta=1.0,
        )
        self.assertEqual(prediction.shape, target.shape)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertIsNone(candidate.grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(z.grad)
        self.assertIsNone(target.grad)

    def test_pair_seed_is_reproducible_and_iteration_specific(self):
        self.assertEqual(paired_iteration_seed(7, 3), paired_iteration_seed(7, 3))
        self.assertNotEqual(paired_iteration_seed(7, 3), paired_iteration_seed(7, 4))
        self.assertNotEqual(paired_iteration_seed(7, 3), paired_iteration_seed(8, 3))
        self.assertNotEqual(paired_transition_seed(7, 3), paired_reset_seed(7, 3))
        self.assertNotEqual(paired_transition_seed(7, 3), paired_transition_seed(7, 4))

    def test_episode_boundary_union_truncates_only_running_counterpart(self):
        candidate = TensorDict(
            {
                "next": TensorDict(
                    {
                        "done": torch.tensor([[True], [False]]),
                        "terminated": torch.tensor([[True], [False]]),
                        "truncated": torch.zeros(2, 1, dtype=torch.bool),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[2],
        )
        base = TensorDict(
            {
                "next": TensorDict(
                    {
                        "done": torch.tensor([[False], [True]]),
                        "terminated": torch.tensor([[False], [False]]),
                        "truncated": torch.tensor([[False], [True]]),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[2],
        )
        boundary = synchronize_paired_transition_boundaries(candidate, base)
        expected_done = torch.ones(2, 1, dtype=torch.bool)
        torch.testing.assert_close(candidate.get(("next", "done")), expected_done)
        torch.testing.assert_close(base.get(("next", "done")), expected_done)
        torch.testing.assert_close(
            candidate.get(("next", "truncated")),
            torch.tensor([[False], [True]]),
        )
        torch.testing.assert_close(
            base.get(("next", "truncated")),
            torch.tensor([[True], [True]]),
        )
        torch.testing.assert_close(
            candidate.get(("next", "terminated")),
            torch.tensor([[True], [False]]),
        )
        torch.testing.assert_close(
            base.get(("next", "terminated")),
            torch.zeros(2, 1, dtype=torch.bool),
        )
        self.assertEqual(boundary.count, 2)

    def test_online_critic_checkpoint_is_self_describing(self):
        model = BaseRelativeDifferentialCritic(
            observation_dim=3,
            embedding_dim=8,
            hidden_sizes=(8,),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            destination = Path(directory) / "candidate.pth"
            torch.save({"source": True}, source)
            save_online_differential_critic(
                destination,
                model=model,
                runtime_config={"p3_stage": "test"},
                source_checkpoint=source,
            )
            payload = torch.load(destination, map_location="cpu")
            self.assertEqual(
                payload["stage"],
                "p3_paired_differential_primal_dual_ppo",
            )
            self.assertEqual(payload["model_config"], model.model_config())

    def test_efficacy_requires_one_strict_base_relative_advantage(self):
        neither = p33_efficacy_gate(
            {"reward_lower_bound": -0.001, "reward_passed": True},
            {"vehicle": {"difference_upper_bound": 0.001}},
        )
        reward = p33_efficacy_gate(
            {"reward_lower_bound": 0.0001, "reward_passed": True},
            {"vehicle": {"difference_upper_bound": 0.001}},
        )
        risk = p33_efficacy_gate(
            {"reward_lower_bound": -0.001, "reward_passed": True},
            {"vehicle": {"difference_upper_bound": -0.0001}},
        )
        unsafe_tradeoff = p33_efficacy_gate(
            {"reward_lower_bound": -0.003, "reward_passed": False},
            {"vehicle": {"difference_upper_bound": -0.0001}},
        )
        self.assertFalse(neither["passed"])
        self.assertTrue(reward["passed"])
        self.assertTrue(risk["passed"])
        self.assertFalse(unsafe_tradeoff["passed"])

    def test_training_dispatches_to_p33(self):
        expected = Path("/tmp/p33-dispatch")
        with patch(
            "utilities.psb_marl.trainer._train_p33", return_value=expected
        ) as train:
            result = train_psb(CONFIG)
        self.assertEqual(result, expected)
        train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
