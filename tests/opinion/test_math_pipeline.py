import torch

from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.residual import OpinionResidual


def test_full_math_pipeline_is_finite_and_trains_evidence_only():
    torch.manual_seed(19)
    evidence_net = OpinionEvidenceNet(
        individual_feature_dim=4,
        symmetric_context_dim=2,
        antisymmetric_context_dim=2,
        hidden_dim=16,
        num_layers=2,
        b_max=0.5,
        b_temperature=1.0,
    )
    dynamics = OpinionDynamics(
        kappa=1.0,
        nu=1.0,
        alpha=2.0,
        eta=1.0,
        z_clip=2.0,
        n_substeps=2,
    )
    residual_module = OpinionResidual(z0=1.0)
    shape = (2, 4, 3)
    ego = torch.randn(*shape, 4)
    neighbor = torch.randn(*shape, 4)
    symmetric = torch.randn(*shape, 2)
    antisymmetric = torch.randn(*shape, 2)
    urgency = torch.rand(*shape)
    confidence = torch.rand(*shape)
    mask = torch.ones(*shape, dtype=torch.bool)
    z_prev = torch.zeros(*shape)
    direction = torch.tanh(torch.randn(*shape))

    evidence = evidence_net(
        ego,
        neighbor,
        symmetric,
        antisymmetric,
        urgency,
        confidence,
        mask,
    )
    z_next = dynamics(z_prev, evidence.b, urgency, mask, dt=0.05)
    residual = residual_module(
        z_next,
        urgency,
        direction,
        mask,
        residual_scale=0.1,
    )
    loss = residual.residual.square().mean() + residual.residual.mean()
    loss.backward()

    assert torch.isfinite(evidence.b).all()
    assert torch.isfinite(z_next).all()
    assert torch.isfinite(residual.residual).all()
    assert list(dynamics.parameters()) == []
    assert list(residual_module.parameters()) == []
    gradients = [parameter.grad for parameter in evidence_net.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
