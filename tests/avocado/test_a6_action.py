"""A6-Action preferred-action learning and A5-equivalence tests."""

import unittest
from pathlib import Path

import torch
from torch import nn

from utilities.avocado_marl.a6_action_config import A6ActionExperimentConfig
from utilities.avocado_marl.a6_action_policy import (
    A6ActionPolicy,
    InteractionActionNet,
)
from utilities.avocado_marl.a6_action_trainer import (
    A6ActionOneStepTrainer,
    _state_hash,
    verify_a6_action_zero_equivalence,
)


CONFIG_PATH = Path("configs/avocado_marl/a6_action.json")
PILOT_CONFIG_PATH = Path("configs/avocado_marl/a6_action_pilot.json")


class _BasePolicyNet(nn.Module):
    def __init__(self, observation_dim: int = 7):
        super().__init__()
        self.linear = nn.Linear(observation_dim, 4)

    def forward(self, observation):
        values = self.linear(observation)
        loc, raw_scale = values.chunk(2, dim=-1)
        return loc, torch.nn.functional.softplus(raw_scale) + 0.05


class A6ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = A6ActionExperimentConfig.from_json(CONFIG_PATH)

    def _policy(self):
        action = self.config.action_policy
        network = InteractionActionNet(
            feature_dim=action.feature_dim,
            hidden_sizes=action.hidden_sizes,
            maximum_loc_correction=action.maximum_loc_correction,
            zero_initialization=action.zero_initialization,
            freeze=action.freeze,
        )
        return A6ActionPolicy(_BasePolicyNet(), network)

    @staticmethod
    def _inputs(batch=2, agents=4):
        observation = torch.randn(batch, agents, 7)
        features = torch.randn(batch, agents, agents, 14)
        mask = ~torch.eye(agents, dtype=torch.bool).unsqueeze(0).expand(
            batch, -1, -1
        )
        confidence = torch.rand(batch, agents, agents) * mask
        return observation, features, confidence, mask

    def test_configuration_preserves_a5_safety_and_trains_action_head(self):
        self.assertTrue(self.config.a5_config.is_file())
        self.assertTrue(self.config.base_policy.policy_checkpoint.is_file())
        self.assertTrue(self.config.base_policy.critic_checkpoint.is_file())
        self.assertTrue(self.config.action_policy.zero_initialization)
        self.assertFalse(self.config.action_policy.freeze)

    def test_zero_initialized_policy_exactly_recovers_base_distribution(self):
        policy = self._policy()
        observation, features, confidence, mask = self._inputs()
        output = policy(observation, features, confidence, mask)
        base_loc, base_scale = policy.base_policy_net(observation)
        torch.testing.assert_close(output.loc, base_loc, rtol=0, atol=0)
        torch.testing.assert_close(output.scale, base_scale, rtol=0, atol=0)
        self.assertTrue(bool((output.loc_correction == 0).all()))
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in policy.base_policy_net.parameters()
            )
        )

    def test_no_interaction_always_has_zero_action_correction(self):
        policy = self._policy()
        observation, features, confidence, mask = self._inputs()
        with torch.no_grad():
            final = policy.action_net.action_head[-1]
            final.weight.fill_(0.2)
            final.bias.fill_(0.3)
        mask.zero_()
        output = policy(observation, features, confidence, mask)
        self.assertTrue(bool((output.loc_correction == 0).all()))

    def test_pair_aggregation_is_permutation_invariant(self):
        action = self.config.action_policy
        network = InteractionActionNet(
            feature_dim=action.feature_dim,
            hidden_sizes=action.hidden_sizes,
            maximum_loc_correction=action.maximum_loc_correction,
            zero_initialization=False,
            freeze=False,
        )
        _, features, confidence, mask = self._inputs()
        base_loc = torch.randn(2, 4, 2)
        permutation = torch.tensor([2, 0, 3, 1])
        first = network(features, confidence, mask, base_loc)
        second = network(
            features[:, :, permutation],
            confidence[:, :, permutation],
            mask[:, :, permutation],
            base_loc,
        )
        torch.testing.assert_close(
            first.loc_correction,
            second.loc_correction,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_actor_gradient_reaches_only_interaction_action_network(self):
        policy = self._policy()
        observation, features, confidence, mask = self._inputs(batch=3)
        output = policy(observation, features, confidence, mask)
        output.loc[..., 0].sum().backward()
        action_gradients = [
            parameter.grad
            for parameter in policy.action_net.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(action_gradients)
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in action_gradients),
            0.0,
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in policy.base_policy_net.parameters()
            )
        )

    def test_zero_initialized_environment_rollout_is_exactly_a5(self):
        result = verify_a6_action_zero_equivalence(CONFIG_PATH, steps=4)
        self.assertTrue(result.passed, result.maximum_differences)
        self.assertEqual(result.maximum_absolute_loc_correction, 0.0)
        self.assertTrue(
            all(value == 0.0 for value in result.maximum_differences.values())
        )

    def test_pilot_rollout_updates_action_head_but_not_base_actor(self):
        pilot = A6ActionExperimentConfig.from_json(PILOT_CONFIG_PATH)
        trainer = A6ActionOneStepTrainer(pilot)
        try:
            base_before = _state_hash(trainer.base_policy_net)
            action_before = _state_hash(trainer.action_net)
            rollout = trainer.collect()
            update = trainer.update(rollout)
            self.assertEqual(base_before, _state_hash(trainer.base_policy_net))
            self.assertNotEqual(action_before, _state_hash(trainer.action_net))
            self.assertGreater(update["action_grad"], 0.0)
            maximum = trainer.action_net.maximum_loc_correction
            self.assertTrue(
                bool((rollout["loc_correction"].abs() <= maximum).all())
            )
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
