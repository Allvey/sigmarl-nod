import pytest
import torch

from utilities.opinion.residual import OpinionResidual


def test_residual_is_bounded_and_returns_normalized_diagnostics():
    module = OpinionResidual(z0=1.0, eps=1e-8)
    z = torch.tensor([[2.0, -1.0, 0.5], [-2.0, -1.0, 1.5]])
    urgency = torch.tensor([[1.0, 0.5, 0.2], [0.8, 0.4, 0.3]])
    direction = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    mask = torch.ones_like(z, dtype=torch.bool)

    output = module(z, urgency, direction, mask, residual_scale=0.1)

    assert output.q.shape == z.shape
    assert output.normalized_weights.shape == z.shape
    assert output.residual.shape == (2,)
    assert torch.isfinite(output.residual).all()
    assert output.q.abs().max() <= 1
    assert output.residual.abs().max() <= 0.1
    assert torch.all(output.normalized_weights.sum(dim=-1) <= 1.0 + 1e-6)


def test_no_valid_conflict_edge_produces_exact_zero():
    module = OpinionResidual(z0=1.0)
    z = torch.tensor([[float("nan"), float("nan")]])
    urgency = torch.tensor([[float("nan"), float("nan")]])
    direction = torch.tensor([[float("nan"), float("nan")]])
    mask = torch.zeros_like(z, dtype=torch.bool)

    output = module(z, urgency, direction, mask, residual_scale=0.1)

    assert torch.equal(output.q, torch.zeros_like(z))
    assert torch.equal(output.normalized_weights, torch.zeros_like(z))
    assert torch.equal(output.residual, torch.zeros(1))


def test_masked_edges_make_no_contribution():
    module = OpinionResidual(z0=1.0)
    z = torch.tensor([[1.0, -1000.0]])
    urgency = torch.tensor([[0.7, 1.0]])
    direction = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[True, False]])

    with_masked_extreme = module(z, urgency, direction, mask, residual_scale=0.2)
    reference = module(
        z[:, :1],
        urgency[:, :1],
        direction[:, :1],
        torch.ones(1, 1, dtype=torch.bool),
        residual_scale=0.2,
    )

    assert torch.allclose(with_masked_extreme.residual, reference.residual)


def test_opposite_opinions_produce_opposite_residuals():
    module = OpinionResidual(z0=1.0)
    urgency = torch.ones(1, 2)
    direction = torch.tensor([[1.0, -1.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    z = torch.tensor([[1.0, -0.5]])

    positive = module(z, urgency, direction, mask, residual_scale=0.1)
    negative = module(-z, urgency, direction, mask, residual_scale=0.1)

    assert torch.allclose(negative.residual, -positive.residual, atol=1e-7)


def test_candidate_count_does_not_increase_absolute_bound():
    module = OpinionResidual(z0=0.1)
    for n_candidates in (1, 3, 32):
        z = torch.full((2, n_candidates), 100.0)
        urgency = torch.ones_like(z)
        direction = torch.ones_like(z)
        mask = torch.ones_like(z, dtype=torch.bool)

        output = module(z, urgency, direction, mask, residual_scale=0.15)

        assert output.residual.abs().max() <= 0.15


@pytest.mark.parametrize(
    ("field", "invalid"),
    (("urgency", 1.1), ("direction", -1.1), ("residual_scale", -0.1)),
)
def test_invalid_active_inputs_are_rejected(field, invalid):
    module = OpinionResidual(z0=1.0)
    z = torch.tensor([[0.2]])
    urgency = torch.tensor([[0.5]])
    direction = torch.tensor([[1.0]])
    mask = torch.tensor([[True]])
    scale = 0.1
    if field == "urgency":
        urgency[0, 0] = invalid
    elif field == "direction":
        direction[0, 0] = invalid
    else:
        scale = invalid

    with pytest.raises(ValueError, match=field):
        module(z, urgency, direction, mask, residual_scale=scale)
