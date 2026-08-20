import json
from pathlib import Path

import pytest
import torch

from utilities.opinion.collector import (
    OpinionStatefulCollector,
    apply_opinion_resets,
    decay_dense_opinions,
)
from utilities.opinion.config import OpinionConfig
from utilities.opinion.policy import OpinionAugmentedPolicyCore, OpinionTanhNormalPolicy


def _policy(stage="joint"):
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
    return OpinionTanhNormalPolicy(
        core=core,
        action_low=-torch.ones(3, 2),
        action_high=torch.ones(3, 2),
    )


def _step_inputs(ids):
    ids = torch.tensor(ids, dtype=torch.float32).unsqueeze(0)
    mask = ids.ge(0)
    shape = ids.shape
    return dict(
        observation=torch.randn(1, 3, 5),
        pair_features=torch.randn(*shape, 12),
        neighbor_ids=ids,
        pair_mask=mask,
        urgency=torch.ones(shape),
        confidence=torch.ones(shape),
        agent_reset_mask=torch.zeros(1, 3, dtype=torch.bool),
        environment_done=torch.zeros(1, dtype=torch.bool),
        residual_scale=0.1,
    )


def test_partial_and_environment_resets_clear_exact_dense_state_regions():
    dense = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    dense[:, torch.arange(3), torch.arange(3)] = 0
    agent_reset = torch.tensor([[False, True, False], [False, False, False]])
    environment_done = torch.tensor([False, True])

    reset = apply_opinion_resets(dense, agent_reset, environment_done)

    assert not reset[0, 1].any()
    assert not reset[0, :, 1].any()
    assert not reset[1].any()
    assert reset[0, 0, 2] == dense[0, 0, 2]


def test_non_candidate_dense_edges_decay_instead_of_freezing():
    collector = OpinionStatefulCollector(
        policy=_policy(), n_envs=1, n_agents=3
    )
    dense = torch.zeros(1, 3, 3)
    dense[0, 0, 2] = 1.0

    decayed = decay_dense_opinions(dense, collector.policy.core.dynamics, dt=0.05)

    assert 0 < decayed[0, 0, 2] < 1
    assert not decayed.diagonal(dim1=-2, dim2=-1).any()


def test_collector_tracks_global_ids_when_candidate_slot_changes():
    torch.manual_seed(4)
    collector = OpinionStatefulCollector(policy=_policy(), n_envs=1, n_agents=3)
    first = _step_inputs([[1], [0], [0]])
    out1 = collector.step(step_id=0, **first)
    saved_01 = collector.z_dense[0, 0, 1].clone()

    second = _step_inputs([[2], [0], [0]])
    out2 = collector.step(step_id=1, **second)

    assert out1.z_prev.shape == (1, 3, 1)
    assert out2.z_prev[0, 0, 0] == 0
    assert collector.z_dense[0, 0, 1].abs() < saved_01.abs()
    assert torch.isfinite(collector.z_dense).all()


def test_collector_rejects_duplicate_physical_step_updates():
    collector = OpinionStatefulCollector(policy=_policy(), n_envs=1, n_agents=3)
    inputs = _step_inputs([[1], [0], [0]])
    collector.step(step_id=7, **inputs)

    with pytest.raises(RuntimeError, match="exactly once"):
        collector.step(step_id=7, **inputs)


def test_single_agent_reset_is_applied_before_gathering_current_candidates():
    collector = OpinionStatefulCollector(policy=_policy(), n_envs=1, n_agents=3)
    collector.z_dense[0, 0, 1] = 0.8
    collector.z_dense[0, 1, 0] = -0.4
    inputs = _step_inputs([[1], [0], [0]])
    inputs["agent_reset_mask"][0, 1] = True

    output = collector.step(step_id=0, **inputs)

    assert output.z_dense_prev[0, 0, 1] == 0
    assert output.z_dense_prev[0, 1, 0] == 0
    assert output.z_prev[0, 0, 0] == 0
