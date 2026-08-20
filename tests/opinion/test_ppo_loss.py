import json
from pathlib import Path

import torch

from utilities.opinion.collector import OpinionStatefulCollector
from utilities.opinion.config import OpinionConfig
from utilities.opinion.policy import OpinionAugmentedPolicyCore, OpinionTanhNormalPolicy
from utilities.opinion.ppo_loss import OpinionCentralizedCritic, OpinionSequencePPOLoss
from utilities.opinion.sequence_buffer import OpinionSequenceBuffer


def _components():
    raw = json.loads(Path("config_opinion.json").read_text(encoding="utf-8"))[
        "opinion_config"
    ]
    raw["stage"] = "joint"
    config = OpinionConfig.from_dict(raw)
    core = OpinionAugmentedPolicyCore.from_config(
        observation_dim=5,
        action_dim=2,
        config=config,
        dt=0.05,
        actor_hidden_dim=16,
        actor_depth=1,
    )
    policy = OpinionTanhNormalPolicy(
        core=core,
        action_low=-torch.ones(3, 2),
        action_high=torch.ones(3, 2),
    )
    critic = OpinionCentralizedCritic(
        observation_dim=5, n_agents=3, hidden_dim=16, include_z=False
    )
    return config, policy, critic


def _chunk():
    torch.manual_seed(42)
    _, policy, _ = _components()
    collector = OpinionStatefulCollector(policy=policy, n_envs=1, n_agents=3)
    buffer = OpinionSequenceBuffer(n_envs=1, n_agents=3)
    ids = torch.tensor([[[1], [0], [0]]], dtype=torch.float32)
    for step in range(3):
        observation = torch.randn(1, 3, 5)
        pair_features = torch.randn(1, 3, 1, 12)
        urgency = torch.full((1, 3, 1), 0.8)
        confidence = torch.ones(1, 3, 1)
        mask = torch.ones(1, 3, 1, dtype=torch.bool)
        output = collector.step(
            step_id=step,
            observation=observation,
            pair_features=pair_features,
            neighbor_ids=ids,
            pair_mask=mask,
            urgency=urgency,
            confidence=confidence,
            agent_reset_mask=torch.zeros(1, 3, dtype=torch.bool),
            environment_done=torch.zeros(1, dtype=torch.bool),
            residual_scale=0.1,
        )
        buffer.append(
            observation=observation,
            action=output.action,
            old_log_prob=output.log_prob,
            reward=torch.ones(1, 3),
            done=torch.tensor([step == 2]),
            pair_features=pair_features,
            neighbor_ids=output.neighbor_ids,
            pair_mask=mask,
            urgency=urgency,
            confidence=confidence,
            agent_reset_mask=torch.zeros(1, 3, dtype=torch.bool),
            z_dense_prev=output.z_dense_prev,
            value=torch.zeros(1, 3),
            advantage=torch.ones(1, 3),
            returns=torch.ones(1, 3),
        )
    return policy, next(buffer.iter_chunks(chunk_length=3))


def test_sequence_replay_recomputes_rollout_log_prob_before_updates():
    policy, chunk = _chunk()
    _, _, critic = _components()
    loss = OpinionSequencePPOLoss(
        policy=policy,
        critic=critic,
        clip_epsilon=0.2,
        entropy_eps=1e-4,
        neutral_loss_weight=1e-3,
        magnitude_loss_weight=1e-4,
    )

    output = loss(chunk, residual_scale=0.1)

    assert torch.allclose(
        output.new_log_prob, chunk.data["old_log_prob"], atol=1e-6, rtol=1e-6
    )
    assert torch.isfinite(output.total_loss)


def test_actor_loss_backpropagates_to_evidence_but_not_fixed_dynamics():
    policy, chunk = _chunk()
    _, _, critic = _components()
    loss = OpinionSequencePPOLoss(
        policy=policy,
        critic=critic,
        clip_epsilon=0.2,
        entropy_eps=0.0,
        neutral_loss_weight=0.0,
        magnitude_loss_weight=0.0,
    )
    output = loss(chunk, residual_scale=0.1)

    output.actor_loss.backward()

    gradients = [p.grad for p in policy.core.evidence_net.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert list(policy.core.dynamics.parameters()) == []


def test_critic_loss_cannot_update_evidence_net():
    policy, chunk = _chunk()
    _, _, critic = _components()
    loss = OpinionSequencePPOLoss(
        policy=policy,
        critic=critic,
        clip_epsilon=0.2,
        entropy_eps=0.0,
        neutral_loss_weight=0.0,
        magnitude_loss_weight=0.0,
    )
    output = loss(chunk, residual_scale=0.1)

    output.critic_loss.backward()

    assert all(p.grad is None for p in policy.core.evidence_net.parameters())
    assert any(p.grad is not None for p in critic.parameters())


def test_early_evidence_remains_differentiable_through_later_opinion_state():
    policy, chunk = _chunk()
    _, _, critic = _components()
    loss = OpinionSequencePPOLoss(
        policy=policy,
        critic=critic,
        clip_epsilon=0.2,
        entropy_eps=0.0,
        neutral_loss_weight=0.0,
        magnitude_loss_weight=0.0,
    )
    pair_features = chunk.data["pair_features"].clone().requires_grad_(True)
    chunk.data["pair_features"] = pair_features

    output = loss(chunk, residual_scale=0.1)
    output.final_z_dense.square().sum().backward()

    assert pair_features.grad is not None
    assert pair_features.grad[0].abs().sum() > 0
