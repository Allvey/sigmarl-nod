import torch

from utilities.opinion.trainer import compute_gae


def test_gae_stops_bootstrap_at_terminal_boundaries():
    reward = torch.tensor([[[1.0]], [[2.0]]])
    value = torch.tensor([[[0.5]], [[0.25]]])
    done = torch.tensor([[False], [True]])

    advantage, returns = compute_gae(
        reward,
        value,
        done,
        next_value=torch.tensor([[99.0]]),
        gamma=0.9,
        lmbda=0.8,
    )

    expected_last = torch.tensor(1.75)
    expected_first = torch.tensor(1.0 + 0.9 * 0.25 - 0.5 + 0.9 * 0.8 * 1.75)
    assert torch.allclose(advantage[:, 0, 0], torch.stack((expected_first, expected_last)))
    assert torch.allclose(returns, advantage + value)


def test_gae_bootstraps_a_truncated_nonterminal_rollout():
    reward = torch.zeros(1, 1, 1)
    value = torch.zeros(1, 1, 1)
    done = torch.tensor([[False]])

    advantage, returns = compute_gae(
        reward,
        value,
        done,
        next_value=torch.tensor([[2.0]]),
        gamma=0.5,
        lmbda=0.7,
    )

    assert torch.equal(advantage, torch.ones_like(advantage))
    assert torch.equal(returns, torch.ones_like(returns))
