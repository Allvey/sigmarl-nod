import pytest
import torch

from utilities.opinion.evidence_net import OpinionEvidenceNet, swap_roles


def _inputs():
    torch.manual_seed(11)
    shape = (2, 3, 4)
    ego = torch.randn(*shape, 5)
    neighbor = torch.randn(*shape, 5)
    symmetric = torch.randn(*shape, 2)
    antisymmetric = torch.randn(*shape, 3)
    urgency = torch.rand(*shape)
    confidence = torch.rand(*shape)
    mask = torch.rand(*shape) > 0.25
    return ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask


def _network():
    torch.manual_seed(7)
    return OpinionEvidenceNet(
        individual_feature_dim=5,
        symmetric_context_dim=2,
        antisymmetric_context_dim=3,
        hidden_dim=16,
        num_layers=2,
        b_max=0.5,
        b_temperature=1.0,
    )


def test_evidence_shapes_bounds_and_physical_gating():
    network = _network()
    ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask = _inputs()

    output = network(
        ego,
        neighbor,
        symmetric,
        antisymmetric,
        urgency,
        confidence,
        mask,
    )

    assert output.raw_b.shape == urgency.shape
    assert output.b.shape == urgency.shape
    assert torch.isfinite(output.raw_b).all()
    assert torch.isfinite(output.b).all()
    assert output.raw_b.abs().max() <= 0.5
    assert torch.all(
        output.b.abs() <= 0.5 * urgency * confidence + 1e-7
    )
    assert torch.equal(output.b[~mask], torch.zeros_like(output.b[~mask]))
    assert torch.allclose(
        output.b[mask],
        output.raw_b[mask] * urgency[mask] * confidence[mask],
    )


def test_role_swap_negates_raw_evidence_exactly():
    network = _network()
    ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask = _inputs()
    original = network(
        ego,
        neighbor,
        symmetric,
        antisymmetric,
        urgency,
        confidence,
        mask,
    )
    swapped = swap_roles(ego, neighbor, symmetric, antisymmetric)

    reversed_output = network(
        *swapped,
        urgency,
        confidence,
        mask,
    )

    assert torch.allclose(reversed_output.raw_b, -original.raw_b, atol=1e-7)
    assert torch.allclose(reversed_output.b, -original.b, atol=1e-7)


def test_masked_padding_can_contain_nan_without_contaminating_evidence():
    network = _network()
    ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask = _inputs()
    mask[:] = True
    mask[..., -1] = False
    ego[..., -1, :] = float("nan")
    neighbor[..., -1, :] = float("nan")
    symmetric[..., -1, :] = float("nan")
    antisymmetric[..., -1, :] = float("nan")
    urgency[..., -1] = float("nan")
    confidence[..., -1] = float("nan")

    output = network(
        ego,
        neighbor,
        symmetric,
        antisymmetric,
        urgency,
        confidence,
        mask,
    )

    assert torch.isfinite(output.raw_b).all()
    assert torch.isfinite(output.b).all()
    assert torch.equal(output.raw_b[..., -1], torch.zeros_like(output.raw_b[..., -1]))
    assert torch.equal(output.b[..., -1], torch.zeros_like(output.b[..., -1]))


@pytest.mark.parametrize(("field", "value"), (("urgency", 1.1), ("confidence", -0.1)))
def test_invalid_gate_values_on_active_edges_are_rejected(field, value):
    network = _network()
    ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask = _inputs()
    mask[:] = True
    if field == "urgency":
        urgency[0, 0, 0] = value
    else:
        confidence[0, 0, 0] = value

    with pytest.raises(ValueError, match=field):
        network(
            ego,
            neighbor,
            symmetric,
            antisymmetric,
            urgency,
            confidence,
            mask,
        )


def test_evidence_parameters_receive_finite_nonzero_gradients():
    network = _network()
    ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask = _inputs()
    mask[:] = True

    output = network(
        ego,
        neighbor,
        symmetric,
        antisymmetric,
        urgency,
        confidence,
        mask,
    )
    loss = output.b.square().mean() + output.b.mean()
    loss.backward()

    gradients = [parameter.grad for parameter in network.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_evidence_forward_does_not_accept_opinion_state_inputs():
    network = _network()
    ego, neighbor, symmetric, antisymmetric, urgency, confidence, mask = _inputs()

    with pytest.raises(TypeError):
        network(
            ego,
            neighbor,
            symmetric,
            antisymmetric,
            urgency,
            confidence,
            mask,
            z=torch.zeros_like(urgency),
        )
