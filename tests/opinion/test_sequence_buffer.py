import pytest
import torch

from utilities.opinion.sequence_buffer import OpinionSequenceBuffer


def _append(buffer, step, done):
    buffer.append(
        observation=torch.full((2, 3, 2), float(step)),
        action=torch.full((2, 3, 2), float(step)),
        old_log_prob=torch.full((2, 3), float(step)),
        reward=torch.ones(2, 3),
        done=torch.tensor(done, dtype=torch.bool),
        pair_features=torch.zeros(2, 3, 1, 12),
        neighbor_ids=torch.tensor([[[1], [0], [0]], [[1], [0], [0]]]),
        pair_mask=torch.ones(2, 3, 1, dtype=torch.bool),
        urgency=torch.ones(2, 3, 1),
        confidence=torch.ones(2, 3, 1),
        agent_reset_mask=torch.zeros(2, 3, dtype=torch.bool),
        z_dense_prev=torch.full((2, 3, 3), float(step)),
        value=torch.zeros(2, 3),
    )


def test_chunks_preserve_order_and_never_cross_done_boundary():
    buffer = OpinionSequenceBuffer(n_envs=2, n_agents=3)
    _append(buffer, 0, [False, False])
    _append(buffer, 1, [True, False])
    _append(buffer, 2, [False, False])
    _append(buffer, 3, [False, True])

    chunks = list(buffer.iter_chunks(chunk_length=3))

    env0 = [chunk for chunk in chunks if chunk.env_index == 0]
    env1 = [chunk for chunk in chunks if chunk.env_index == 1]
    assert [chunk.data["observation"][:, 0, 0].tolist() for chunk in env0] == [
        [0.0, 1.0],
        [2.0, 3.0],
    ]
    assert [chunk.data["observation"][:, 0, 0].tolist() for chunk in env1] == [
        [0.0, 1.0, 2.0],
        [3.0],
    ]
    assert all(not chunk.data["done"][:-1].any() for chunk in chunks)


def test_chunk_z_init_is_detached_at_the_exact_start_time():
    buffer = OpinionSequenceBuffer(n_envs=2, n_agents=3)
    _append(buffer, 0, [False, False])
    _append(buffer, 1, [False, False])
    _append(buffer, 2, [False, False])

    chunks = list(buffer.iter_chunks(chunk_length=2))
    second_env0 = next(
        chunk for chunk in chunks if chunk.env_index == 0 and chunk.start == 2
    )

    assert torch.equal(second_env0.z_init, torch.full((3, 3), 2.0))
    assert not second_env0.z_init.requires_grad


def test_stacked_rollout_recovers_original_time_major_data():
    buffer = OpinionSequenceBuffer(n_envs=2, n_agents=3)
    for step in range(3):
        _append(buffer, step, [False, False])

    rollout = buffer.as_rollout()

    assert rollout["observation"].shape == (3, 2, 3, 2)
    assert rollout["observation"][:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_schema_and_chunk_length_are_strict():
    buffer = OpinionSequenceBuffer(n_envs=2, n_agents=3)
    _append(buffer, 0, [False, False])
    with pytest.raises(ValueError, match="same fields"):
        buffer.append(observation=torch.zeros(2, 3, 2))
    with pytest.raises(ValueError, match="chunk_length"):
        list(buffer.iter_chunks(chunk_length=0))
