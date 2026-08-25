"""M8 contracts for differentiable truncated Opinion sequence PPO."""

import unittest
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules.distributions import TanhNormal

from utilities.opinion.config import (
    load_opinion_experiment,
    require_m8_supported_mode,
)
from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.policy import StatefulOpinionPolicyBridge
from utilities.opinion.residual import OpinionResidual
from utilities.opinion.sequence_buffer import OpinionSequenceBuffer
from utilities.opinion.sequence_ppo import OpinionSequencePPOLoss
from utilities.opinion.state import OpinionStateTracker


class TinyBaseActor(nn.Module):
    def __init__(self, observation_size=4, action_size=2):
        super().__init__()
        self.linear = nn.Linear(observation_size, action_size * 2)

    def forward(self, observation):
        loc, raw_scale = self.linear(observation).chunk(2, dim=-1)
        return loc, raw_scale.sigmoid() + 0.2


class DistributionBuilder:
    def build_dist_from_params(self, tensordict):
        return TanhNormal(
            tensordict.get(("agents", "loc")),
            tensordict.get(("agents", "scale")),
            min=-1.0,
            max=1.0,
        )


def make_rollout(environments=2, steps=4, agents=3, candidates=2):
    observation = torch.randn(environments, steps, agents, 4)
    neighbor_ids = torch.empty(
        environments, steps, agents, candidates, dtype=torch.long
    )
    for ego in range(agents):
        others = [index for index in range(agents) if index != ego]
        neighbor_ids[:, :, ego] = torch.tensor(others[:candidates])
    pair_mask = torch.ones_like(neighbor_ids, dtype=torch.bool)
    pair_features = torch.randn(
        environments, steps, agents, candidates, 10
    )
    urgency = torch.rand(environments, steps, agents, candidates)
    confidence = torch.rand(environments, steps, agents, candidates)
    reset = torch.zeros(environments, steps, agents, dtype=torch.bool)
    z_dense = torch.zeros(environments, steps, agents, agents)
    edge_active = torch.zeros_like(z_dense, dtype=torch.bool)
    z_next = torch.zeros(environments, steps, agents, candidates)
    action = torch.zeros(environments, steps, agents, 2)
    log_prob = torch.zeros(environments, steps, agents)
    advantage = torch.ones(environments, steps, agents, 1)
    done = torch.zeros(environments, steps, 1, dtype=torch.bool)
    trajectory_ids = torch.zeros(environments, steps, dtype=torch.long)
    rollout = TensorDict({}, batch_size=[environments, steps])
    for key, value in (
        (("agents", "observation"), observation),
        (("agents", "action"), action),
        (("agents", "sample_log_prob"), log_prob),
        (("agents", "advantage"), advantage),
        (("agents", "info", "neighbor_ids"), neighbor_ids),
        (("agents", "info", "pair_mask"), pair_mask),
        (("agents", "info", "pair_features"), pair_features),
        (("agents", "info", "urgency"), urgency),
        (("agents", "info", "confidence"), confidence),
        (("agents", "info", "agent_reset_mask"), reset),
        (("agents", "opinion", "z_dense_prev"), z_dense),
        (("agents", "opinion", "edge_active_prev"), edge_active),
        (("agents", "opinion", "z_next"), z_next),
        (("next", "done"), done),
        (("collector", "traj_ids"), trajectory_ids),
        ("done", done),
    ):
        rollout.set(key, value)
    return rollout


def make_loss():
    bridge = StatefulOpinionPolicyBridge(
        base_policy_net=TinyBaseActor(),
        evidence_net=OpinionEvidenceNet(
            pair_feature_dim=10,
            hidden_sizes=[16],
            b_max=1.0,
            temperature=1.0,
        ),
        dynamics=OpinionDynamics(
            response_rate=0.5,
            decay_rate=1.0,
            self_reinforcement=0.5,
            nonlinear_sensitivity=1.0,
        ),
        residual=OpinionResidual(
            opinion_scale=1.0,
            gain=0.1,
            max_abs=0.25,
        ),
        dt=0.05,
        freeze_base_actor=True,
        freeze_evidence=False,
    )
    loss = OpinionSequencePPOLoss(
        actor=DistributionBuilder(),
        bridge=bridge,
        observation_key=("agents", "observation"),
        action_key=("agents", "action"),
        advantage_key=("agents", "advantage"),
        n_agents=3,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        neutral_loss_coefficient=0.01,
        magnitude_loss_coefficient=0.001,
        decay_factor=0.975,
        zero_threshold=1e-6,
    )
    return bridge, loss


class M8SequencePPOTests(unittest.TestCase):
    def test_committed_config_selects_trainable_sequence_stage(self):
        root = Path(__file__).resolve().parents[2]
        experiment = load_opinion_experiment(
            root / "configs/opinion/m8_sequence_ppo.json"
        )
        require_m8_supported_mode(experiment)
        self.assertEqual(experiment.config.stage, "sequence_ppo")
        self.assertTrue(experiment.config.opinion.sequence_ppo.train_evidence)
        self.assertFalse(experiment.config.opinion.stateful.freeze_evidence)

    def test_equal_length_sequence_batches_keep_time_axis(self):
        rollout = make_rollout(environments=1, steps=5)
        buffer = OpinionSequenceBuffer(rollout, chunk_length=4)
        batches = list(buffer.iter_sequence_minibatches(minibatch_size=4))
        self.assertEqual(
            sorted(tuple(batch.tensordict.batch_size) for batch in batches),
            [(1, 1), (1, 4)],
        )
        self.assertEqual(sum(batch.valid_steps for batch in batches), 5)

    def test_actor_loss_backpropagates_only_into_evidence(self):
        torch.manual_seed(7)
        rollout = make_rollout()
        bridge, loss = make_loss()
        initial_buffer = OpinionSequenceBuffer(rollout, chunk_length=4)
        initial_batch = next(
            initial_buffer.iter_sequence_minibatches(minibatch_size=8)
        )
        with torch.no_grad():
            recomputed = loss.unroll(initial_batch)
            dist_td = initial_batch.tensordict.clone(False)
            dist_td.set(("agents", "loc"), recomputed["loc"])
            dist_td.set(("agents", "scale"), recomputed["scale"])
            distribution = loss.actor.build_dist_from_params(dist_td)
            action = distribution.rsample()
            old_log_prob = distribution.log_prob(action)
        rollout.set(("agents", "action"), action)
        rollout.set(("agents", "sample_log_prob"), old_log_prob)
        rollout.set(("agents", "opinion", "z_next"), recomputed["z_next"])

        buffer = OpinionSequenceBuffer(rollout, chunk_length=4)
        batch = next(buffer.iter_sequence_minibatches(minibatch_size=8))
        values = loss(batch)
        total = (
            values["loss_objective"]
            + values["loss_entropy"]
            + values["loss_regularization"]
        )
        total.backward()

        evidence_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in bridge.evidence_net.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(evidence_gradient, 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in bridge.base_policy_net.parameters())
        )
        self.assertLess(float(values["log_prob_abs_error"]), 1e-5)
        self.assertLess(float(values["state_replay_abs_error"]), 1e-6)

    def test_unroll_matches_rollout_tracker_across_mask_and_reset(self):
        torch.manual_seed(11)
        rollout = make_rollout(environments=1, steps=4)
        bridge, loss = make_loss()
        rollout.get(("agents", "info", "neighbor_ids"))[:, 1] = rollout.get(
            ("agents", "info", "neighbor_ids")
        )[:, 1].flip(-1)
        rollout.get(("agents", "info", "pair_mask"))[:, 2, 0, 0] = False
        rollout.get(("agents", "info", "agent_reset_mask"))[:, 3, 1] = True
        tracker = OpinionStateTracker(
            n_agents=3, decay_factor=0.975, zero_threshold=1e-6
        )
        expected_z_prev = []
        expected_z_next = []
        for time_index in range(4):
            ids = rollout.get(("agents", "info", "neighbor_ids"))[:, time_index]
            mask = rollout.get(("agents", "info", "pair_mask"))[:, time_index]
            urgency = rollout.get(("agents", "info", "urgency"))[:, time_index]
            confidence = rollout.get(("agents", "info", "confidence"))[:, time_index]
            features = rollout.get(("agents", "info", "pair_features"))[:, time_index]
            z_prev = tracker.prepare_step(
                neighbor_ids=ids,
                pair_mask=mask,
                agent_reset_mask=rollout.get(
                    ("agents", "info", "agent_reset_mask")
                )[:, time_index],
                reference=urgency,
                environment_done=rollout.get("done")[:, time_index],
            )
            dense, active = tracker.prepared_snapshot()
            rollout.get(("agents", "opinion", "z_dense_prev"))[
                :, time_index
            ] = dense
            rollout.get(("agents", "opinion", "edge_active_prev"))[
                :, time_index
            ] = active
            with torch.no_grad():
                output = bridge(
                    rollout.get(("agents", "observation"))[:, time_index],
                    features,
                    urgency,
                    confidence,
                    mask,
                    z_prev,
                )
                z_next = output[5]
            tracker.commit_step(ids, z_next)
            expected_z_prev.append(z_prev)
            expected_z_next.append(z_next)
            rollout.get(("agents", "opinion", "z_next"))[:, time_index] = z_next

        buffer = OpinionSequenceBuffer(rollout, chunk_length=4)
        batch = next(buffer.iter_sequence_minibatches(minibatch_size=4))
        recomputed = loss.unroll(batch)
        torch.testing.assert_close(
            recomputed["z_prev"], torch.stack(expected_z_prev, dim=1)
        )
        torch.testing.assert_close(
            recomputed["z_next"], torch.stack(expected_z_next, dim=1)
        )

    def test_later_opinion_state_backpropagates_to_early_evidence(self):
        torch.manual_seed(19)
        rollout = make_rollout(environments=1, steps=4)
        _, loss = make_loss()
        buffer = OpinionSequenceBuffer(rollout, chunk_length=4)
        batch = next(buffer.iter_sequence_minibatches(minibatch_size=4))
        recomputed = loss.unroll(batch)
        recomputed["b"].retain_grad()
        recomputed["z_next"][:, -1].sum().backward()
        self.assertIsNotNone(recomputed["b"].grad)
        self.assertGreater(
            float(recomputed["b"].grad[:, 0].abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
