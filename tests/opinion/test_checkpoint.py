import json
from pathlib import Path

import torch
import pytest

from utilities.opinion.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_opinion_checkpoint,
    save_opinion_checkpoint,
)
from utilities.opinion.config import OpinionConfig
from utilities.opinion.policy import OpinionAugmentedPolicyCore, OpinionTanhNormalPolicy
from utilities.opinion.ppo_loss import OpinionCentralizedCritic
from utilities.opinion.trainer import build_stage_optimizers


def _runtime(stage="joint"):
    raw = json.loads(Path("config_opinion.json").read_text(encoding="utf-8"))[
        "opinion_config"
    ]
    raw["stage"] = stage
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
        core=core, action_low=-torch.ones(3, 2), action_high=torch.ones(3, 2)
    )
    critic = OpinionCentralizedCritic(
        observation_dim=5, n_agents=3, hidden_dim=16
    )
    optimizers = build_stage_optimizers(core=core, critic=critic, config=config)
    return config, policy, critic, optimizers


def test_optimizer_groups_are_disjoint_and_exclude_fixed_dynamics():
    config, policy, critic, optimizers = _runtime("joint")
    groups = [
        {id(p) for group in optimizer.param_groups for p in group["params"]}
        for optimizer in optimizers.values()
    ]

    assert set(optimizers) == {"actor", "evidence", "critic"}
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3))
    optimized = set().union(*groups)
    assert not any(id(p) in optimized for p in policy.core.dynamics.parameters())


@pytest.mark.parametrize(
    ("stage", "optimizer_names", "actor_trainable", "evidence_trainable"),
    [
        ("base", {"actor", "critic"}, True, False),
        ("evidence", {"evidence", "critic"}, False, True),
        ("joint", {"actor", "evidence", "critic"}, True, True),
    ],
)
def test_stage_optimizer_and_freezing_contract(
    stage, optimizer_names, actor_trainable, evidence_trainable
):
    _, policy, _, optimizers = _runtime(stage)

    assert set(optimizers) == optimizer_names
    assert all(
        parameter.requires_grad is actor_trainable
        for parameter in policy.core.base_actor.parameters()
    )
    assert all(
        parameter.requires_grad is evidence_trainable
        for parameter in policy.core.evidence_net.parameters()
    )


def test_checkpoint_roundtrip_restores_identical_distribution_parameters(tmp_path):
    config, policy, critic, optimizers = _runtime("joint")
    checkpoint = tmp_path / "opinion.pt"
    save_opinion_checkpoint(
        checkpoint,
        policy=policy,
        critic=critic,
        optimizers=optimizers,
        resolved_config={"opinion_config": config.to_dict()},
        stage="joint",
        iteration=3,
    )
    torch.manual_seed(9)
    observation = torch.randn(1, 3, 5)
    pair_features = torch.randn(1, 3, 1, 12)
    urgency = torch.rand(1, 3, 1)
    confidence = torch.ones(1, 3, 1)
    pair_mask = torch.ones(1, 3, 1, dtype=torch.bool)
    z_prev = torch.randn(1, 3, 1) * 0.1
    before = policy.core(
        observation,
        pair_features,
        urgency,
        confidence,
        pair_mask,
        z_prev,
        residual_scale=0.1,
    )
    original = [parameter.detach().clone() for parameter in policy.parameters()]
    for parameter in policy.parameters():
        parameter.data.zero_()

    metadata = load_opinion_checkpoint(
        checkpoint, policy=policy, critic=critic, optimizers=optimizers
    )

    assert metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert metadata["stage"] == "joint"
    assert metadata["iteration"] == 3
    assert all(torch.equal(a, b) for a, b in zip(original, policy.parameters()))
    after = policy.core(
        observation,
        pair_features,
        urgency,
        confidence,
        pair_mask,
        z_prev,
        residual_scale=0.1,
    )
    assert torch.equal(before.final_loc, after.final_loc)
    assert torch.equal(before.scale, after.scale)


def test_checkpoint_stage_is_validated_before_weights_are_mutated(tmp_path):
    config, source_policy, source_critic, source_optimizers = _runtime("base")
    checkpoint = tmp_path / "base.pt"
    save_opinion_checkpoint(
        checkpoint,
        policy=source_policy,
        critic=source_critic,
        optimizers=source_optimizers,
        resolved_config={"opinion_config": config.to_dict()},
        stage="base",
        iteration=1,
    )
    _, target_policy, target_critic, _ = _runtime("joint")
    before = [parameter.detach().clone() for parameter in target_policy.parameters()]

    with pytest.raises(ValueError, match="checkpoint stage"):
        load_opinion_checkpoint(
            checkpoint,
            policy=target_policy,
            critic=target_critic,
            expected_stages={"joint"},
        )

    assert all(torch.equal(a, b) for a, b in zip(before, target_policy.parameters()))
