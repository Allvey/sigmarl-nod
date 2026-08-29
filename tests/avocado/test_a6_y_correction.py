"""A6 one-step y-correction policy tests."""

import unittest
from pathlib import Path

import torch
from torch import nn

from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.avocado_marl.a6_config import A6ExperimentConfig
from utilities.avocado_marl.a6_policy import A6OneStepPolicy
from utilities.avocado_marl.a6_trainer import (
    A6OneStepTrainer,
    _state_hash,
    resolve_latest_a6_checkpoint,
)
from utilities.avocado_marl.y_correction import YCorrectionNet
from utilities.opinion.residual import OpinionResidual


class _BasePolicyNet(nn.Module):
    def __init__(self, observation_dim: int = 7):
        super().__init__()
        self.linear = nn.Linear(observation_dim, 4)

    def forward(self, observation):
        values = self.linear(observation)
        loc, raw_scale = values.chunk(2, dim=-1)
        return loc, torch.nn.functional.softplus(raw_scale) + 0.05


class A6YCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = A6ExperimentConfig.from_json(
            Path("configs/avocado_marl/a6_y_correction.json")
        )
        cls.a3_config = A3RoadExperimentConfig.from_json(
            Path("configs/avocado/a3_road_environment.json")
        )

    def _policy(self):
        correction = self.config.y_correction
        base = _BasePolicyNet()
        network = YCorrectionNet(
            feature_dim=correction.feature_dim,
            hidden_sizes=correction.hidden_sizes,
            maximum_correction=correction.maximum_correction,
            temperature=correction.temperature,
            strict_zero=False,
            freeze=False,
        )
        residual_config = self.config.opinion_residual
        residual = OpinionResidual(
            opinion_scale=residual_config.opinion_scale,
            gain=residual_config.gain,
            max_abs=residual_config.maximum_absolute_residual,
        )
        return A6OneStepPolicy(base, network, residual, self.a3_config)

    def test_config_requires_explicit_existing_base_checkpoints(self):
        self.assertTrue(self.config.base_policy.policy_checkpoint.is_file())
        self.assertTrue(self.config.base_policy.critic_checkpoint.is_file())
        self.assertFalse(self.config.y_correction.strict_zero)
        self.assertFalse(self.config.y_correction.freeze)

    def test_latest_a6_checkpoint_uses_a6_artifact_names(self):
        checkpoint = resolve_latest_a6_checkpoint(self.config.output_root)
        self.assertIn(
            checkpoint.name, {"final_checkpoint.pt", "latest_checkpoint.pt"}
        )
        self.assertTrue(checkpoint.is_file())

    def test_one_step_policy_is_bounded_masked_and_freezes_base(self):
        policy = self._policy()
        batch, agents = 2, 4
        observation = torch.randn(batch, agents, 7)
        features = torch.randn(batch, agents, agents, 14)
        mask = ~torch.eye(agents, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
        confidence = torch.rand(batch, agents, agents) * mask
        attention = torch.rand(batch, agents, agents) * mask
        heuristic = torch.rand(batch, agents, agents) * 2.0 - 1.0
        z_prev = torch.zeros(batch, agents, agents)
        output = policy(
            observation,
            features,
            confidence,
            mask,
            attention,
            heuristic,
            z_prev,
        )
        self.assertTrue(bool((output.correction.abs() <= 0.1).all()))
        self.assertTrue(bool((output.correction[~mask] == 0).all()))
        self.assertEqual(output.residual.shape, (batch, agents, 1))
        self.assertTrue(bool((output.residual.abs() <= 0.1).all()))
        self.assertFalse(any(p.requires_grad for p in policy.base_policy_net.parameters()))

    def test_single_step_actor_path_reaches_only_y_correction(self):
        policy = self._policy()
        batch, agents = 3, 4
        mask = ~torch.eye(agents, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
        output = policy(
            torch.randn(batch, agents, 7),
            torch.randn(batch, agents, agents, 14),
            torch.ones(batch, agents, agents),
            mask,
            torch.ones(batch, agents, agents),
            torch.zeros(batch, agents, agents),
            torch.zeros(batch, agents, agents).detach(),
        )
        output.loc[..., 0].sum().backward()
        y_gradients = [
            parameter.grad
            for parameter in policy.y_correction_net.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(y_gradients)
        self.assertGreater(sum(float(grad.abs().sum()) for grad in y_gradients), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in policy.base_policy_net.parameters())
        )

    def test_pilot_rollout_updates_y_but_not_base_actor(self):
        pilot = A6ExperimentConfig.from_json(
            Path("configs/avocado_marl/a6_y_correction_pilot.json")
        )
        trainer = A6OneStepTrainer(pilot)
        try:
            base_before = _state_hash(trainer.base_policy_net)
            y_before = _state_hash(trainer.y_correction_net)
            rollout = trainer.collect()
            update = trainer.update(rollout)
            self.assertEqual(base_before, _state_hash(trainer.base_policy_net))
            self.assertNotEqual(y_before, _state_hash(trainer.y_correction_net))
            self.assertGreater(update["y_grad"], 0.0)
            self.assertTrue(bool((rollout["correction"].abs() <= 0.1).all()))
            self.assertTrue(
                bool((rollout["correction"][~rollout["pair_mask"]] == 0).all())
            )
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
