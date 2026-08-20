import json
from pathlib import Path

import pytest
import torch
from torch import nn

from utilities.opinion.config import OpinionConfig
from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.evidence_net import EvidenceOutput
from utilities.opinion.policy import (
    BaseActorOutput,
    BaseGaussianActor,
    OpinionAugmentedPolicyCore,
    OpinionTanhNormalPolicy,
    PairInteractionEncoder,
)
from utilities.opinion.residual import OpinionResidual


def _opinion_config(stage="joint"):
    raw = json.loads(Path("config_opinion.json").read_text(encoding="utf-8"))[
        "opinion_config"
    ]
    raw["stage"] = stage
    return OpinionConfig.from_dict(raw)


def _inputs(batch=2, n_agents=4, n_candidates=3, observation_dim=7):
    torch.manual_seed(31)
    return {
        "observation": torch.randn(batch, n_agents, observation_dim),
        "pair_features": torch.randn(batch, n_agents, n_candidates, 12),
        "urgency": torch.rand(batch, n_agents, n_candidates),
        "confidence": torch.rand(batch, n_agents, n_candidates),
        "pair_mask": torch.ones(batch, n_agents, n_candidates, dtype=torch.bool),
        "z_prev": torch.zeros(batch, n_agents, n_candidates),
    }


class _FixedBaseActor(nn.Module):
    def forward(self, observation):
        prefix = observation.shape[:-1]
        loc = torch.zeros(*prefix, 2, device=observation.device)
        loc[..., 0] = 0.1
        loc[..., 1] = -0.2
        scale = torch.full_like(loc, 0.4)
        return BaseActorOutput(loc=loc, scale=scale)


class _PositiveEvidence(nn.Module):
    def forward(
        self,
        ego_features,
        neighbor_features,
        symmetric_context,
        antisymmetric_context,
        urgency,
        confidence,
        mask,
    ):
        raw_b = torch.where(mask, torch.full_like(urgency, 0.5), torch.zeros_like(urgency))
        return EvidenceOutput(raw_b=raw_b, b=raw_b * urgency * confidence)


class _MustNotRunEvidence(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("base stage must not execute EvidenceNet")


def _core(stage="joint", *, evidence_net=None):
    config = _opinion_config(stage)
    return OpinionAugmentedPolicyCore(
        base_actor=_FixedBaseActor(),
        interaction_encoder=PairInteractionEncoder(),
        evidence_net=evidence_net or _PositiveEvidence(),
        dynamics=OpinionDynamics(
            kappa=config.kappa,
            nu=config.nu,
            alpha=config.alpha,
            eta=config.eta,
            z_clip=config.z_clip,
            n_substeps=config.n_substeps,
        ),
        residual_module=OpinionResidual(z0=config.z0),
        stage=stage,
        dt=0.05,
    )


def test_interaction_encoder_uses_all_m4_features_with_explicit_roles():
    pair = torch.arange(12, dtype=torch.float32).reshape(1, 1, 1, 12)

    encoded = PairInteractionEncoder()(pair)

    assert encoded.ego_features.tolist() == [[[[10.0]]]]
    assert encoded.neighbor_features.tolist() == [[[[11.0]]]]
    assert encoded.symmetric_context.tolist() == [[[[4.0, 5.0, 6.0, 7.0, 9.0]]]]
    assert encoded.antisymmetric_context.tolist() == [[[[0.0, 1.0, 2.0, 3.0, 8.0]]]]


def test_base_gaussian_actor_has_shared_last_dim_contract():
    actor = BaseGaussianActor(
        observation_dim=7,
        action_dim=2,
        hidden_dim=32,
        depth=2,
    )
    observation = torch.randn(3, 4, 7)

    output = actor(observation)

    assert output.loc.shape == (3, 4, 2)
    assert output.scale.shape == (3, 4, 2)
    assert torch.isfinite(output.loc).all()
    assert torch.isfinite(output.scale).all()
    assert (output.scale > 0).all()


def test_base_stage_is_an_exact_passthrough_and_skips_opinion_modules():
    inputs = _inputs()
    core = _core(stage="base", evidence_net=_MustNotRunEvidence())

    output = core(**inputs, residual_scale=0.1)

    assert torch.equal(output.final_loc, output.base_loc)
    assert not output.residual.any()
    assert not output.raw_b.any()
    assert not output.b.any()
    assert not output.z_next.any()


def test_joint_stage_changes_only_speed_loc_and_preserves_scale():
    inputs = _inputs()
    core = _core()

    output = core(**inputs, residual_scale=0.1)

    assert (output.residual > 0).all()
    assert torch.allclose(
        output.final_loc[..., 0], output.base_loc[..., 0] + output.residual
    )
    assert torch.equal(output.final_loc[..., 1], output.base_loc[..., 1])
    assert torch.equal(output.scale, torch.full_like(output.scale, 0.4))
    assert output.residual.abs().max() <= 0.1
    assert torch.isfinite(output.z_next).all()


def test_zero_residual_scale_is_an_exact_base_policy_ablation():
    output = _core()(**_inputs(), residual_scale=0.0)

    assert torch.equal(output.final_loc, output.base_loc)
    assert not output.residual.any()


@pytest.mark.parametrize("invalid", (-0.1, float("nan"), "0.1", True))
def test_residual_scale_contract_applies_even_in_base_stage(invalid):
    with pytest.raises(ValueError, match="residual_scale"):
        _core(stage="base")(**_inputs(), residual_scale=invalid)


def test_masked_nan_pair_is_sanitized_and_cannot_change_the_action():
    inputs = _inputs(batch=1, n_agents=2, n_candidates=1)
    inputs["pair_features"].fill_(float("nan"))
    inputs["urgency"].fill_(float("nan"))
    inputs["confidence"].fill_(float("nan"))
    inputs["pair_mask"].fill_(False)

    output = _core()(**inputs, residual_scale=0.1)

    assert torch.equal(output.final_loc, output.base_loc)
    assert not output.b.any()
    assert not output.z_next.any()
    assert torch.isfinite(output.final_loc).all()


def test_tanh_normal_wrapper_samples_finite_bounded_action_and_log_prob():
    policy = OpinionTanhNormalPolicy(
        core=_core(),
        action_low=torch.tensor([-1.0, -1.0]),
        action_high=torch.tensor([1.0, 1.0]),
    )

    output = policy(**_inputs(), residual_scale=0.1)

    assert output.action.shape == (2, 4, 2)
    assert output.log_prob.shape == (2, 4)
    assert torch.isfinite(output.action).all()
    assert torch.isfinite(output.log_prob).all()
    assert (output.action >= -1).all() and (output.action <= 1).all()


def test_tanh_normal_wrapper_accepts_vmas_per_agent_action_bounds():
    policy = OpinionTanhNormalPolicy(
        core=_core(),
        action_low=-torch.ones(4, 2),
        action_high=torch.ones(4, 2),
    )

    output = policy(**_inputs(), residual_scale=0.1)

    assert output.action.shape == (2, 4, 2)
    assert output.log_prob.shape == (2, 4)


def test_supplied_action_recomputes_identical_log_prob_and_parameters():
    inputs = _inputs()
    policy = OpinionTanhNormalPolicy(
        core=_core(),
        action_low=-torch.ones(2),
        action_high=torch.ones(2),
    )
    rollout = policy(**inputs, residual_scale=0.1)

    recomputed = policy(
        **inputs,
        residual_scale=0.1,
        action=rollout.action.detach(),
    )

    assert torch.equal(recomputed.core.final_loc, rollout.core.final_loc)
    assert torch.equal(recomputed.core.scale, rollout.core.scale)
    assert torch.allclose(recomputed.log_prob, rollout.log_prob, atol=1e-6, rtol=1e-6)


def test_standard_factory_connects_config_and_backpropagates_to_evidence_net():
    config = _opinion_config("joint")
    core = OpinionAugmentedPolicyCore.from_config(
        observation_dim=7,
        action_dim=2,
        config=config,
        dt=0.05,
        actor_hidden_dim=32,
        actor_depth=2,
    )
    inputs = _inputs()
    inputs["z_prev"] = torch.randn_like(inputs["z_prev"]) * 0.1

    output = core(**inputs, residual_scale=0.1)
    loss = output.final_loc[..., 0].mean()
    loss.backward()

    evidence_gradients = [
        parameter.grad for parameter in core.evidence_net.parameters()
    ]
    assert evidence_gradients
    assert all(gradient is not None for gradient in evidence_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in evidence_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in evidence_gradients)
    assert list(core.dynamics.parameters()) == []
    assert list(core.residual_module.parameters()) == []


@pytest.mark.parametrize("stage", ("", "train", "tsc"))
def test_core_rejects_unknown_stage(stage):
    with pytest.raises(ValueError, match="stage"):
        OpinionAugmentedPolicyCore(
            base_actor=_FixedBaseActor(),
            interaction_encoder=PairInteractionEncoder(),
            evidence_net=_PositiveEvidence(),
            dynamics=OpinionDynamics(
                kappa=1.0,
                nu=1.0,
                alpha=2.0,
                eta=1.0,
                z_clip=2.0,
                n_substeps=1,
            ),
            residual_module=OpinionResidual(z0=1.0),
            stage=stage,
            dt=0.05,
        )
