"""P3.1 graph critic, paired targets, and Actor-free training contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import torch
from tensordict import TensorDict

from utilities.psb_marl.checkpoint import sha256_file
from utilities.psb_marl.config import (
    PSBConfigError,
    PSBP31DifferentialCriticConfig,
    load_psb_experiment,
)
from utilities.psb_marl.evaluator import test_psb
from utilities.psb_marl.p3_critic import BaseRelativeDifferentialCritic
from utilities.psb_marl.p3_critic_training import (
    critic_channel_quality,
    load_differential_critic,
    paired_critic_samples,
)
from utilities.psb_marl.p3_pairing import build_paired_batch
from utilities.psb_marl.trainer import train_psb


CONFIG = Path("configs/psb_marl/p3_1_differential_critic.json")


def _rollout(
    reward: float,
    *,
    observation_offset: float = 0.0,
    done_time: Optional[int] = None,
    vehicle_risk: float = 0.0,
    lane_clearance: float = 0.1,
) -> TensorDict:
    environments, steps, agents, observation_dim = 2, 5, 4, 32
    candidate_count = 2
    neighbor_ids = torch.empty(
        environments, steps, agents, candidate_count, dtype=torch.long
    )
    for ego in range(agents):
        candidates = [index for index in range(agents) if index != ego][
            :candidate_count
        ]
        neighbor_ids[..., ego, :] = torch.tensor(candidates)
    done = torch.zeros(environments, steps, 1, dtype=torch.bool)
    if done_time is not None:
        done[:, done_time] = True
    return TensorDict(
        {
            ("agents", "observation"): torch.full(
                (environments, steps, agents, observation_dim),
                observation_offset,
            ),
            ("agents", "action"): torch.zeros(
                environments, steps, agents, 2
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
                vehicle_risk,
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
                (environments, steps, agents), lane_clearance
            ),
            ("next", "agents", "info", "distance_right_b"): torch.full(
                (environments, steps, agents), lane_clearance
            ),
            ("next", "agents", "reward"): torch.full(
                (environments, steps, agents, 1), reward
            ),
            ("next", "done"): done,
            ("next", "terminated"): done.clone(),
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


def _paired_zero():
    candidate = _rollout(
        1.0,
        observation_offset=0.25,
        vehicle_risk=0.3,
        lane_clearance=0.03,
    )
    base = _rollout(
        1.0,
        observation_offset=-0.25,
        vehicle_risk=0.1,
        lane_clearance=0.08,
    )
    return build_paired_batch(candidate, base), candidate, base


def _summary(rollout, seed, **kwargs):
    reward = rollout.get(("next", "agents", "reward"))
    result = {
        "seed": seed,
        "nonfinite_action_count": 0,
        "nonfinite_reward_count": 0,
        "mean_reward_per_agent_step": float(reward.mean()),
        "collision_with_agents_rate": 0.0,
        "collision_with_lanelets_rate": 0.0,
        "total_collision_rate": 0.0,
    }
    if kwargs.get("p2_runtime_config") is not None:
        result.update(
            {
                "nonfinite_z_count": 0,
                "max_antisymmetry_error": 0.0,
                "max_root_residual": 0.0,
                "min_root_denominator": 3.0,
                "rollout_sector_bound_max_violation": 0.0,
                "rollout_delta_steering_abs_max": 0.0,
                "rollout_delta_log_scale_abs_max": 0.0,
                "rollout_scale_matches_base_exactly": True,
            }
        )
    return result


class P31DifferentialCriticTests(unittest.TestCase):
    def test_locked_config_uses_passed_p30_parent_and_disjoint_seeds(self):
        experiment = load_psb_experiment(CONFIG)

        self.assertEqual(experiment.stage, "p3_differential_critic")
        self.assertFalse(experiment.differential_critic.actor_learning_enabled)
        self.assertFalse(experiment.differential_critic.dual_learning_enabled)
        self.assertFalse(
            set(experiment.differential_critic.training_seeds)
            & set(experiment.differential_critic.validation_seeds)
        )
        self.assertEqual(
            experiment.source_p2_runtime["branch_adapter"]["conditioning_mode"],
            "supported_sector_q_gate",
        )

    def test_config_rejects_actor_or_dual_learning(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "differential_critic"
        ]
        raw["actor_learning_enabled"] = True
        with self.assertRaisesRegex(PSBConfigError, "actor_learning_enabled=false"):
            PSBP31DifferentialCriticConfig.from_dict(raw)

    def test_graph_critic_is_agent_permutation_equivariant(self):
        torch.manual_seed(3)
        model = BaseRelativeDifferentialCritic(
            observation_dim=5,
            embedding_dim=16,
            hidden_sizes=(16,),
        )
        candidate = torch.randn(2, 4, 5)
        base = torch.randn(2, 4, 5)
        raw_z = torch.randn(2, 4, 4)
        z = 0.5 * (raw_z - raw_z.transpose(-1, -2))
        mask = ~torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
        permutation = torch.tensor([2, 0, 3, 1])

        expected = model(candidate, base, z, mask)[:, permutation]
        actual = model(
            candidate[:, permutation],
            base[:, permutation],
            z[:, permutation][:, :, permutation],
            mask[:, permutation][:, :, permutation],
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_critic_regression_stops_input_state_gradients(self):
        model = BaseRelativeDifferentialCritic(
            observation_dim=5,
            embedding_dim=8,
            hidden_sizes=(8,),
        )
        candidate = torch.randn(2, 4, 5, requires_grad=True)
        base = torch.randn(2, 4, 5, requires_grad=True)
        z = torch.randn(2, 4, 4, requires_grad=True)
        mask = ~torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)

        model(candidate, base, z, mask).sum().backward()

        self.assertIsNone(candidate.grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(z.grad)

    def test_return_targets_respect_candidate_and_base_boundaries_separately(self):
        candidate = _rollout(1.0, done_time=1)
        base = _rollout(0.0)
        samples = paired_critic_samples(
            build_paired_batch(candidate, base),
            gamma=0.5,
            energy_coefficient=0.0,
            lane_safety_margin=0.07,
        )

        target = samples.target.reshape(2, 5, 4, 3)
        self.assertAlmostEqual(float(target[0, 0, 0, 0]), 1.5)
        self.assertAlmostEqual(float(target[0, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(float(target[0, 2, 0, 0]), 1.75)
        self.assertTrue(
            torch.equal(target[..., 1:], torch.zeros_like(target[..., 1:]))
        )

    def test_dense_safety_targets_respond_before_discrete_collisions(self):
        candidate = _rollout(
            0.0,
            vehicle_risk=0.4,
            lane_clearance=0.035,
        )
        base = _rollout(
            0.0,
            vehicle_risk=0.1,
            lane_clearance=0.07,
        )

        samples = paired_critic_samples(
            build_paired_batch(candidate, base),
            gamma=0.0,
            energy_coefficient=0.0,
            lane_safety_margin=0.07,
        )

        self.assertTrue(
            torch.allclose(
                samples.target[..., 0],
                torch.zeros_like(samples.target[..., 0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                samples.target[..., 1],
                torch.full_like(samples.target[..., 1], 0.3),
            )
        )
        self.assertTrue(
            torch.allclose(
                samples.target[..., 2],
                torch.full_like(samples.target[..., 2], 0.5),
            )
        )

    def test_channel_quality_rejects_an_uninformative_head(self):
        metrics = {
            "channels": {
                name: {
                    "target_std": 0.2,
                    "explained_variance": 0.1,
                    "normalized_huber": 0.4,
                }
                for name in (
                    "augmented_reward_return_delta",
                    "vehicle_conflict_risk_return_delta",
                    "lane_margin_violation_return_delta",
                )
            }
        }
        metrics["channels"]["lane_margin_violation_return_delta"][
            "target_std"
        ] = 0.0

        quality = critic_channel_quality(
            metrics,
            baseline_metrics={
                "channels": {
                    name: {"normalized_huber": 0.5}
                    for name in metrics["channels"]
                }
            },
            minimum_target_std=1e-4,
            minimum_explained_variance=0.0,
        )

        self.assertFalse(quality["passed"])
        self.assertFalse(
            quality["channels"]["lane_margin_violation_return_delta"][
                "target_informative"
            ]
        )

    def test_channel_quality_uses_the_fitted_loss_for_noninferiority(self):
        channels = {
            name: {
                "target_std": 0.2,
                "explained_variance": -0.002,
                "normalized_huber": 0.49,
            }
            for name in (
                "augmented_reward_return_delta",
                "vehicle_conflict_risk_return_delta",
                "lane_margin_violation_return_delta",
            )
        }

        quality = critic_channel_quality(
            {"channels": channels},
            baseline_metrics={
                "channels": {
                    name: {"normalized_huber": 0.5} for name in channels
                }
            },
            minimum_target_std=1e-4,
            minimum_explained_variance=0.0,
        )

        self.assertTrue(quality["passed"])
        self.assertFalse(
            quality["channels"]["lane_margin_violation_return_delta"][
                "explained_variance_passed"
            ]
        )
        self.assertTrue(
            quality["channels"]["lane_margin_violation_return_delta"][
                "loss_noninferiority_passed"
            ]
        )

    def test_paired_targets_detach_actor_and_proximal_graphs(self):
        paired, _, _ = _paired_zero()
        paired = replace(
            paired,
            candidate_observation=paired.candidate_observation.requires_grad_(),
            candidate_branch_state=paired.candidate_branch_state.requires_grad_(),
            candidate_control=paired.candidate_control.requires_grad_(),
            candidate_reward=paired.candidate_reward.requires_grad_(),
        )

        samples = paired_critic_samples(
            paired,
            gamma=0.99,
            energy_coefficient=0.001,
            lane_safety_margin=0.07,
        )

        self.assertTrue(
            all(
                not getattr(samples, name).requires_grad
                for name in samples.__dataclass_fields__
            )
        )

    def test_training_changes_only_critic_and_keeps_base_deployed(self):
        experiment = load_psb_experiment(CONFIG)
        tiny_config = replace(
            experiment.differential_critic,
            training_seeds=(701, 702),
            validation_seeds=(705,),
            epochs=2,
            early_stopping_patience=1,
            minibatch_size=8,
            embedding_dim=8,
            hidden_sizes=(8,),
            required_relative_improvement=0.0,
            minimum_target_std=0.0,
            minimum_channel_explained_variance=-1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            isolated = replace(
                experiment,
                output_root=directory,
                differential_critic=tiny_config,
            )
            with patch(
                "utilities.psb_marl.trainer.load_psb_experiment",
                return_value=isolated,
            ), patch(
                "utilities.psb_marl.p3_critic_training._collect_pair",
                side_effect=lambda *_args, **_kwargs: _paired_zero(),
            ), patch("builtins.print"):
                run = train_psb(CONFIG)

            certification = json.loads(
                (run / "p3_1_certification.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (run / "deployment_manifest.json").read_text(encoding="utf-8")
            )
            model, payload = load_differential_critic(
                run / "candidate_critic.pth"
            )
            self.assertTrue(certification["passed"])
            self.assertFalse(certification["actor_learning_enabled"])
            self.assertEqual(
                sha256_file(run / "candidate_policy.pth"),
                sha256_file(experiment.parent_run / "candidate_policy.pth"),
            )
            self.assertEqual(
                sha256_file(run / "final_policy.pth"),
                sha256_file(experiment.base.policy_checkpoint),
            )
            self.assertEqual(manifest["selected"], "base_fallback_p3_critic_only")
            self.assertEqual(payload["model_config"], model.model_config())

    def test_loader_can_reuse_architecture_without_source_weights(self):
        source = BaseRelativeDifferentialCritic(
            observation_dim=5,
            embedding_dim=4,
            hidden_sizes=(8,),
        )
        source.set_target_normalization(
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0]),
        )
        with torch.no_grad():
            source.head[-1].weight.fill_(1.0)
            source.head[-1].bias.fill_(1.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "differential.pth"
            torch.save(
                {
                    "method": "psb_marl",
                    "stage": "p3_differential_critic",
                    "model_config": source.model_config(),
                    "critic_state": source.state_dict(),
                },
                checkpoint,
            )

            fresh, _ = load_differential_critic(
                checkpoint, load_weights=False
            )
            loaded, _ = load_differential_critic(checkpoint)

        self.assertTrue(torch.equal(fresh.target_center, torch.zeros(3)))
        self.assertTrue(torch.equal(fresh.target_scale, torch.ones(3)))
        self.assertTrue(
            torch.equal(
                fresh.head[-1].weight,
                torch.zeros_like(fresh.head[-1].weight),
            )
        )
        self.assertTrue(torch.equal(loaded.target_center, source.target_center))
        self.assertTrue(
            torch.equal(loaded.head[-1].weight, source.head[-1].weight)
        )

    def test_manual_validation_uses_unseen_seeds(self):
        experiment = load_psb_experiment(CONFIG)
        tiny_config = replace(
            experiment.differential_critic,
            training_seeds=(701, 702),
            validation_seeds=(705,),
            epochs=2,
            early_stopping_patience=1,
            minibatch_size=8,
            embedding_dim=8,
            hidden_sizes=(8,),
            required_relative_improvement=0.0,
            minimum_target_std=0.0,
            minimum_channel_explained_variance=-1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            isolated = replace(
                experiment,
                output_root=directory,
                differential_critic=tiny_config,
            )
            with patch(
                "utilities.psb_marl.trainer.load_psb_experiment",
                return_value=isolated,
            ), patch(
                "utilities.psb_marl.p3_critic_training._collect_pair",
                side_effect=lambda *_args, **_kwargs: _paired_zero(),
            ), patch("builtins.print"):
                run = train_psb(CONFIG)
            with patch(
                "utilities.psb_marl.evaluator.load_psb_experiment",
                return_value=isolated,
            ), patch(
                "utilities.psb_marl.p3_critic_training._collect_pair",
                side_effect=lambda *_args, **_kwargs: _paired_zero(),
            ) as collected, patch(
                "utilities.psb_marl.evaluator._rollout_summary",
                side_effect=_summary,
            ), patch("builtins.print"):
                report = test_psb(
                    CONFIG,
                    run_directory=run,
                    checkpoint_path=run / "candidate_policy.pth",
                    scenario_type="CPM_mixed",
                    max_steps=8,
                    episodes=2,
                    seeds=(711, 712),
                    render=False,
                    compare_base=True,
                )

        self.assertEqual(collected.call_count, 2)
        self.assertTrue(report["passed"])
        self.assertTrue(report["critic_passed"])
        self.assertEqual(report["evaluation_protocol"]["seeds"], [711, 712])


if __name__ == "__main__":
    unittest.main()
