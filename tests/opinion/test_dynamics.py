import pytest
import torch

from utilities.opinion.dynamics import (
    OpinionDynamics,
    gather_candidate_opinions,
    scatter_candidate_opinions,
)


def _dynamics(n_substeps=1, z_clip=2.0):
    return OpinionDynamics(
        kappa=1.0,
        nu=1.0,
        alpha=2.0,
        eta=1.0,
        z_clip=z_clip,
        n_substeps=n_substeps,
    )


def _evolve(dynamics, z, b, urgency, mask, steps=200, dt=0.05):
    for _ in range(steps):
        z = dynamics(z, b, urgency, mask, dt=dt)
    return z


def test_dynamics_has_no_trainable_parameters_and_exposes_rho_c():
    dynamics = _dynamics()

    assert list(dynamics.parameters()) == []
    assert dynamics.rho_c == pytest.approx(0.5)


def test_no_conflict_or_invalid_edge_naturally_decays_to_zero():
    dynamics = _dynamics(n_substeps=2)
    z_initial = torch.tensor([1.0, -1.0])
    b = torch.tensor([0.4, float("nan")])
    urgency = torch.tensor([0.0, float("nan")])
    mask = torch.tensor([True, False])

    z_final = _evolve(dynamics, z_initial, b * 0, urgency, mask, steps=200)

    assert torch.isfinite(z_final).all()
    assert z_final.abs().max() < 1e-4


def test_subcritical_small_perturbations_return_to_neutral():
    dynamics = _dynamics(n_substeps=2)
    z_initial = torch.tensor([0.01, -0.01])
    zeros = torch.zeros_like(z_initial)
    urgency = torch.full_like(z_initial, 0.25)
    mask = torch.ones_like(z_initial, dtype=torch.bool)

    z_final = _evolve(dynamics, z_initial, zeros, urgency, mask, steps=300)

    assert z_final.abs().max() < 1e-4


def test_supercritical_small_perturbations_form_opposite_branches():
    dynamics = _dynamics(n_substeps=2)
    z_initial = torch.tensor([0.01, -0.01])
    zeros = torch.zeros_like(z_initial)
    urgency = torch.full_like(z_initial, 0.75)
    mask = torch.ones_like(z_initial, dtype=torch.bool)

    z_final = _evolve(dynamics, z_initial, zeros, urgency, mask, steps=500)

    assert z_final[0] > 0.4
    assert z_final[1] < -0.4
    assert torch.allclose(z_final[0], -z_final[1], atol=1e-5)


def test_strong_reverse_evidence_can_flip_an_established_opinion():
    dynamics = _dynamics(n_substeps=4)
    z = torch.tensor([0.8])
    b = torch.tensor([-0.5])
    urgency = torch.tensor([1.0])
    mask = torch.tensor([True])

    z_final = _evolve(dynamics, z, b, urgency, mask, steps=300)

    assert z_final.item() < -0.2


def test_clipping_and_substeps_keep_extreme_finite_inputs_bounded():
    dynamics = _dynamics(n_substeps=8, z_clip=1.25)
    z = torch.tensor([1.0, -1.0])
    b = torch.tensor([1e6, -1e6])
    urgency = torch.ones(2)
    mask = torch.ones(2, dtype=torch.bool)

    z_next = dynamics(z, b, urgency, mask, dt=0.05)

    assert torch.isfinite(z_next).all()
    assert z_next.abs().max() <= 1.25


def test_gradient_flows_through_dynamics_to_evidence():
    dynamics = _dynamics(n_substeps=3)
    z = torch.tensor([0.1, -0.2])
    b = torch.tensor([0.2, -0.1], requires_grad=True)
    urgency = torch.tensor([0.8, 0.6])
    mask = torch.ones(2, dtype=torch.bool)

    z_next = dynamics(z, b, urgency, mask, dt=0.05)
    z_next.sum().backward()

    assert b.grad is not None
    assert torch.isfinite(b.grad).all()
    assert b.grad.abs().sum() > 0


def test_candidate_gather_and_scatter_use_global_agent_ids():
    z_dense = torch.zeros(2, 4, 4)
    z_dense[0, 0, 2] = 0.3
    z_dense[0, 0, 3] = -0.4
    z_dense[1, 0, 2] = 0.8
    candidate_ids = torch.tensor(
        [
            [[2, 3], [0, -1], [3, -1], [1, -1]],
            [[2, 3], [0, -1], [3, -1], [1, -1]],
        ],
        dtype=torch.long,
    )
    mask = candidate_ids >= 0

    gathered = gather_candidate_opinions(z_dense, candidate_ids, mask)

    assert gathered.shape == candidate_ids.shape
    assert torch.allclose(gathered[0, 0], torch.tensor([0.3, -0.4]))
    assert torch.allclose(gathered[1, 0], torch.tensor([0.8, 0.0]))
    updates = gathered + torch.where(mask, 0.1, 0.0)
    scattered = scatter_candidate_opinions(z_dense, candidate_ids, updates, mask)
    assert scattered[0, 0, 2] == pytest.approx(0.4)
    assert scattered[0, 0, 3] == pytest.approx(-0.3)
    assert scattered[1, 0, 2] == pytest.approx(0.9)
    assert torch.equal(
        scattered.diagonal(dim1=-2, dim2=-1),
        torch.zeros(2, 4),
    )


@pytest.mark.parametrize("bad_id", (4, -2))
def test_gather_rejects_out_of_range_active_candidate_ids(bad_id):
    z_dense = torch.zeros(1, 3, 3)
    candidate_ids = torch.tensor([[[1], [2], [bad_id]]])
    mask = torch.ones_like(candidate_ids, dtype=torch.bool)

    with pytest.raises(ValueError, match="candidate_ids"):
        gather_candidate_opinions(z_dense, candidate_ids, mask)


def test_gather_rejects_self_edges_and_duplicate_active_candidates():
    z_dense = torch.zeros(1, 3, 3)
    self_edge = torch.tensor([[[0, 1], [0, 2], [0, 1]]])
    duplicates = torch.tensor([[[1, 1], [0, 2], [0, 1]]])
    mask = torch.ones_like(self_edge, dtype=torch.bool)

    with pytest.raises(ValueError, match="self edge"):
        gather_candidate_opinions(z_dense, self_edge, mask)
    with pytest.raises(ValueError, match="duplicate"):
        gather_candidate_opinions(z_dense, duplicates, mask)


def test_scatter_does_not_write_masked_padding_or_mutate_input():
    z_dense = torch.zeros(1, 3, 3)
    candidate_ids = torch.tensor([[[1, -1], [2, -1], [0, -1]]])
    mask = candidate_ids >= 0
    values = torch.tensor([[[0.2, 99.0], [0.3, 99.0], [0.4, 99.0]]])

    result = scatter_candidate_opinions(z_dense, candidate_ids, values, mask)

    assert torch.equal(z_dense, torch.zeros_like(z_dense))
    assert result[0, 0, 1] == pytest.approx(0.2)
    assert result[0, 1, 2] == pytest.approx(0.3)
    assert result[0, 2, 0] == pytest.approx(0.4)
    assert torch.count_nonzero(result) == 3
