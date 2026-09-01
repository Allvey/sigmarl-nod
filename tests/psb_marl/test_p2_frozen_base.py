"""P2 frozen-Base bifurcation actor, sequence PPO, and promotion contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules.distributions import TanhNormal

from utilities.psb_marl.evaluator import _p2_noninferiority_gate
from utilities.psb_marl.config import (
    PSBBranchAdapterConfig,
    PSBConfigError,
    load_psb_experiment,
)
from utilities.psb_marl.p2_buffer import P2SequenceBuffer
from utilities.psb_marl.p2_critic import AugmentedCentralCritic
from utilities.psb_marl.p2_diagnostics import (
    p2_state_diagnostics,
    p2_zero_branch_counterfactual_diagnostics,
)
from utilities.psb_marl.p2_loss import P2SequencePPOLoss
from utilities.psb_marl.p2_network import (
    AntisymmetricBifurcationControl,
    BranchContextEncoder,
    BranchDistributionAdapter,
    swap_pair_features,
)
from utilities.psb_marl.p2_policy import FrozenBaseBifurcationPolicyBridge
from utilities.psb_marl.p2_state import P2EdgeStateTracker
from utilities.psb_marl.proximal import ProximalSaturatingBifurcation


class TinyBaseActor(nn.Module):
    def __init__(self, observation_dim: int = 4, action_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(observation_dim, action_dim * 2)

    def forward(self, observation):
        loc, raw_scale = self.linear(observation).chunk(2, dim=-1)
        return loc, raw_scale.sigmoid() + 0.2


class DistributionBuilder:
    def build_dist_from_params(self, tensordict):
        return TanhNormal(
            tensordict.get(("agents", "loc")),
            tensordict.get(("agents", "scale")),
            min=-1.0,
            max=1.0,
        )


def make_bridge(
    n_agents: int = 3,
    conditioning_mode: str = "general",
    max_delta_log_scale: float = 0.2,
    mean_action_mask=None,
) -> FrozenBaseBifurcationPolicyBridge:
    proximal = ProximalSaturatingBifurcation(
        kappa=1.0,
        nu=1.0,
        alpha=1.0,
        rho_max=2.0,
        h_z=0.25,
        b_max=0.5,
        residual_tolerance=1e-6,
        max_iterations=64,
    )
    return FrozenBaseBifurcationPolicyBridge(
        base_policy_net=TinyBaseActor(),
        control_net=AntisymmetricBifurcationControl(
            pair_feature_dim=10,
            hidden_sizes=(16,),
            b_max=0.5,
            temperature=1.0,
            support_power=1.0,
            critical_gate_enabled=False,
            critical_width=0.25,
            critical_floor=0.25,
            final_layer_gain=0.1,
            rho_c=1.0,
            rho_max=2.0,
        ),
        proximal=proximal,
        branch_encoder=BranchContextEncoder(
            pair_feature_dim=10,
            hidden_sizes=(16,),
            context_dim=8,
            z_scale=0.5,
            rho_max=2.0,
            conditioning_mode=conditioning_mode,
        ),
        adapter=BranchDistributionAdapter(
            observation_dim=4,
            context_dim=8,
            action_dim=2,
            hidden_sizes=(16,),
            max_delta_loc=0.3,
            max_delta_log_scale=max_delta_log_scale,
            conditioning_mode=conditioning_mode,
            mean_action_mask=mean_action_mask,
        ),
        n_agents=n_agents,
    )


def pair_inputs(environments=2, n_agents=3):
    candidates = n_agents - 1
    neighbor_ids = torch.empty(
        environments, n_agents, candidates, dtype=torch.long
    )
    for ego in range(n_agents):
        neighbor_ids[:, ego] = torch.tensor(
            [index for index in range(n_agents) if index != ego]
        )
    return {
        "observation": torch.randn(environments, n_agents, 4),
        "pair_features": torch.randn(
            environments, n_agents, candidates, 10
        ),
        "neighbor_ids": neighbor_ids,
        "urgency": torch.full(
            (environments, n_agents, candidates), 0.75
        ),
        "confidence": torch.ones(environments, n_agents, candidates),
        "pair_mask": torch.ones(
            environments, n_agents, candidates, dtype=torch.bool
        ),
        "z_prev_dense": torch.zeros(environments, n_agents, n_agents),
    }


class P2NetworkTests(unittest.TestCase):
    def test_swap_is_involution_and_control_is_exchange_antisymmetric(self):
        torch.manual_seed(2)
        control = make_bridge().control_net
        features = torch.randn(2, 3, 2, 10)
        # A physically valid relative yaw satisfies sin^2+cos^2=1.
        yaw = torch.randn(2, 3, 2)
        features[..., 6] = yaw.sin()
        features[..., 7] = yaw.cos()
        self.assertTrue(
            torch.allclose(
                swap_pair_features(swap_pair_features(features)),
                features,
                atol=1e-6,
            )
        )
        z = torch.randn(2, 3, 2)
        rho = torch.full_like(z, 1.5)
        confidence = torch.ones_like(z)
        mask = torch.ones_like(z, dtype=torch.bool)
        forward = control(features, z, rho, confidence, mask)
        reverse = control(
            swap_pair_features(features), -z, rho, confidence, mask
        )
        self.assertTrue(torch.allclose(forward.b, -reverse.b, atol=1e-7))

    def test_zero_initialized_adapter_exactly_matches_frozen_base(self):
        torch.manual_seed(4)
        bridge = make_bridge()
        inputs = pair_inputs()
        output = bridge(**inputs)
        base_loc, base_scale = bridge.base_policy_net(inputs["observation"])
        self.assertTrue(torch.equal(output.loc, base_loc))
        self.assertTrue(torch.equal(output.scale, base_scale))
        self.assertEqual(float(output.delta_loc.abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(
                output.z_next_dense,
                -output.z_next_dense.transpose(-1, -2),
            )
        )
        self.assertLessEqual(
            float(output.root_residual.abs().max()),
            bridge.proximal.residual_tolerance,
        )

    def test_gradient_reaches_control_and_adapter_but_not_base_actor(self):
        torch.manual_seed(7)
        bridge = make_bridge()
        output = bridge(**pair_inputs())
        upper = torch.triu(
            torch.ones(3, 3, dtype=torch.bool), diagonal=1
        )
        loss = (
            output.z_next_dense[..., upper].square().mean()
            + output.loc.square().mean()
        )
        loss.backward()
        control_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in bridge.control_net.parameters()
            if parameter.grad is not None
        )
        adapter_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in bridge.adapter.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(control_gradient, 0.0)
        self.assertGreater(adapter_gradient, 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in bridge.base_policy_net.parameters()
            )
        )

    def test_reset_supports_rollout_leading_dimensions(self):
        z = torch.ones(2, 4, 3, 3)
        reset = torch.zeros(2, 4, 3, dtype=torch.bool)
        reset[0, 2, 1] = True
        done = torch.zeros(2, 4, 1, dtype=torch.bool)
        done[1, 3] = True
        cleared = P2EdgeStateTracker.apply_resets(z, reset, done)
        self.assertEqual(float(cleared[0, 2, 1].abs().max()), 0.0)
        self.assertEqual(float(cleared[0, 2, :, 1].abs().max()), 0.0)
        self.assertEqual(float(cleared[1, 3].abs().max()), 0.0)

    def test_actor_checkpoint_is_independent_of_agent_count(self):
        source = make_bridge(n_agents=3)
        destination = make_bridge(n_agents=5)
        destination.load_state_dict(source.state_dict(), strict=True)


class P21DiagnosticsTests(unittest.TestCase):
    def test_state_diagnostics_use_unique_active_and_critical_edges(self):
        b = torch.zeros(1, 4, 3, 3)
        rho = torch.zeros_like(b)
        z_prev = torch.zeros_like(b)
        z_next = torch.zeros_like(b)
        active_b = (0.1, 0.2, -0.3)
        previous_z = (0.0, 0.5, 0.5)
        next_z = (0.5, 0.5, -0.5)
        for time in range(3):
            b[0, time, 0, 1] = active_b[time]
            b[0, time, 1, 0] = -active_b[time]
            rho[0, time, 0, 1] = 1.5
            rho[0, time, 1, 0] = 1.5
            z_prev[0, time, 0, 1] = previous_z[time]
            z_prev[0, time, 1, 0] = -previous_z[time]
            z_next[0, time, 0, 1] = next_z[time]
            z_next[0, time, 1, 0] = -next_z[time]
        rollout = TensorDict(
            {
                ("agents", "psb", "b"): b,
                ("agents", "psb", "rho"): rho,
                ("agents", "psb", "z_prev_dense"): z_prev,
                ("agents", "psb", "z_next_dense"): z_next,
            },
            batch_size=[1, 4],
        )

        metrics = p2_state_diagnostics(
            rollout,
            rho_c=1.0,
            z_scale=0.5,
            commitment_threshold=0.5,
        )

        self.assertEqual(metrics["rollout_active_edge_samples"], 3)
        self.assertEqual(metrics["rollout_critical_edge_samples"], 3)
        self.assertEqual(metrics["rollout_committed_edge_samples"], 3)
        self.assertAlmostEqual(metrics["rollout_active_edge_fraction"], 0.25)
        self.assertAlmostEqual(
            metrics["rollout_critical_given_active_fraction"], 1.0
        )
        self.assertAlmostEqual(metrics["rollout_active_b_abs_mean"], 0.2)
        self.assertAlmostEqual(metrics["rollout_active_z_abs_mean"], 0.5)
        self.assertEqual(metrics["rollout_branch_switch_eligible_samples"], 2)
        self.assertEqual(metrics["rollout_branch_switch_count"], 1)
        self.assertAlmostEqual(metrics["rollout_branch_switch_rate"], 0.5)
        self.assertAlmostEqual(metrics["rollout_branch_dwell_mean_steps"], 1.5)
        self.assertEqual(metrics["rollout_branch_dwell_max_steps"], 2)

    def test_zero_branch_counterfactual_is_read_only_and_detects_bypass(self):
        torch.manual_seed(19)
        bridge = make_bridge()
        with torch.no_grad():
            bridge.adapter.network[-1].bias[:2].fill_(0.2)
            bridge.adapter.network[-1].bias[2:].fill_(0.1)
        inputs = pair_inputs(environments=2)
        with torch.no_grad():
            output = bridge(**inputs)
        rollout = TensorDict(
            {
                ("agents", "observation"): inputs["observation"],
                ("agents", "info", "pair_features"): inputs["pair_features"],
                ("agents", "info", "neighbor_ids"): inputs["neighbor_ids"],
                ("agents", "info", "urgency"): inputs["urgency"],
                ("agents", "info", "confidence"): inputs["confidence"],
                ("agents", "info", "pair_mask"): inputs["pair_mask"],
                ("agents", "loc"): output.loc,
                ("agents", "scale"): output.scale,
                ("agents", "psb", "base_loc"): output.base_loc,
                ("agents", "psb", "base_scale"): output.base_scale,
            },
            batch_size=[2],
        )
        state_before = {
            name: value.detach().clone()
            for name, value in bridge.state_dict().items()
        }
        loc_before = rollout.get(("agents", "loc")).clone()
        scale_before = rollout.get(("agents", "scale")).clone()

        metrics = p2_zero_branch_counterfactual_diagnostics(
            rollout,
            bridge=bridge,
        )

        self.assertGreater(
            metrics["rollout_zero_branch_bypass_loc_abs_mean"], 0.0
        )
        self.assertGreater(
            metrics["rollout_zero_branch_bypass_log_scale_abs_mean"], 0.0
        )
        self.assertAlmostEqual(
            metrics["rollout_branch_loc_effect_abs_mean"], 0.0
        )
        self.assertAlmostEqual(
            metrics["rollout_branch_log_scale_effect_abs_mean"], 0.0
        )
        self.assertAlmostEqual(
            metrics["rollout_branch_loc_dependency_ratio"], 0.0
        )
        self.assertAlmostEqual(
            metrics["rollout_branch_log_scale_dependency_ratio"], 0.0
        )
        self.assertTrue(
            torch.equal(loc_before, rollout.get(("agents", "loc")))
        )
        self.assertTrue(
            torch.equal(scale_before, rollout.get(("agents", "scale")))
        )
        self.assertTrue(
            all(
                torch.equal(state_before[name], value)
                for name, value in bridge.state_dict().items()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in bridge.parameters())
        )


class P21CausalBranchTests(unittest.TestCase):
    def test_legacy_and_causal_configs_have_distinct_runtime_contracts(self):
        legacy = load_psb_experiment(
            Path("configs/psb_marl/p2_frozen_base_bifurcation.json")
        )
        causal = load_psb_experiment(
            Path("configs/psb_marl/p2_1_c_causal_branch.json")
        )
        legacy_runtime = legacy.p2_runtime_config()
        causal_runtime = causal.p2_runtime_config()
        self.assertNotIn(
            "conditioning_mode", legacy_runtime["branch_adapter"]
        )
        self.assertEqual(
            causal_runtime["branch_adapter"]["conditioning_mode"],
            "causal_q_gate",
        )
        self.assertNotEqual(legacy_runtime, causal_runtime)
        self.assertNotEqual(legacy.output_root, causal.output_root)

    def test_causal_gate_exactly_recovers_base_for_zero_branch(self):
        torch.manual_seed(23)
        bridge = make_bridge(conditioning_mode="causal_q_gate")
        with torch.no_grad():
            for parameter in bridge.control_net.parameters():
                parameter.zero_()
            for parameter in bridge.adapter.parameters():
                parameter.normal_(mean=0.2, std=0.3)
        inputs = pair_inputs()
        output = bridge(**inputs)
        base_loc, base_scale = bridge.base_policy_net(inputs["observation"])

        self.assertEqual(float(output.q.abs().max()), 0.0)
        self.assertEqual(float(output.branch_context.abs().max()), 0.0)
        self.assertEqual(float(output.delta_loc.abs().max()), 0.0)
        self.assertEqual(float(output.delta_log_scale.abs().max()), 0.0)
        self.assertTrue(torch.equal(output.loc, base_loc))
        self.assertTrue(torch.equal(output.scale, base_scale))

        rollout = TensorDict(
            {
                ("agents", "observation"): inputs["observation"],
                ("agents", "info", "pair_features"): inputs["pair_features"],
                ("agents", "info", "neighbor_ids"): inputs["neighbor_ids"],
                ("agents", "info", "urgency"): inputs["urgency"],
                ("agents", "info", "confidence"): inputs["confidence"],
                ("agents", "info", "pair_mask"): inputs["pair_mask"],
                ("agents", "loc"): output.loc,
                ("agents", "scale"): output.scale,
                ("agents", "psb", "base_loc"): output.base_loc,
                ("agents", "psb", "base_scale"): output.base_scale,
            },
            batch_size=[2],
        )
        diagnostics = p2_zero_branch_counterfactual_diagnostics(
            rollout,
            bridge=bridge,
        )
        self.assertEqual(
            diagnostics["rollout_zero_branch_bypass_loc_abs_mean"], 0.0
        )
        self.assertEqual(
            diagnostics[
                "rollout_zero_branch_bypass_log_scale_abs_mean"
            ],
            0.0,
        )

    def test_nonzero_branch_can_learn_complementary_action_changes(self):
        bridge = make_bridge(conditioning_mode="causal_q_gate")
        with torch.no_grad():
            for parameter in bridge.control_net.parameters():
                parameter.zero_()
            for parameter in bridge.branch_encoder.parameters():
                parameter.zero_()
            bridge.branch_encoder.encoder[-1].bias.fill_(1.0)
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        inputs = pair_inputs(environments=1)
        inputs["z_prev_dense"][0, 0, 1] = 0.5
        inputs["z_prev_dense"][0, 1, 0] = -0.5

        output = bridge(**inputs)

        self.assertGreater(float(output.q[0, 0].abs().max()), 0.0)
        self.assertGreater(float(output.delta_loc[0, 0].abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(
                output.delta_loc[0, 0],
                -output.delta_loc[0, 1],
                atol=1e-7,
            )
        )
        self.assertEqual(float(output.delta_loc[0, 2].abs().max()), 0.0)

    def test_action_gradient_crosses_causal_gate_and_proximal_layer(self):
        torch.manual_seed(29)
        bridge = make_bridge(conditioning_mode="causal_q_gate")
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())
        self.assertGreater(float(output.delta_loc.abs().max()), 0.0)

        output.delta_loc.square().mean().backward()

        control_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in bridge.control_net.parameters()
            if parameter.grad is not None
        )
        branch_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in bridge.branch_encoder.parameters()
            if parameter.grad is not None
        )
        adapter_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in bridge.adapter.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(control_gradient, 0.0)
        self.assertGreater(branch_gradient, 0.0)
        self.assertGreater(adapter_gradient, 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in bridge.base_policy_net.parameters()
            )
        )


class P21MeanOnlyTests(unittest.TestCase):
    def test_mean_only_config_is_isolated_from_distribution_adapter(self):
        distribution = load_psb_experiment(
            Path("configs/psb_marl/p2_1_c_causal_branch.json")
        )
        mean_only = load_psb_experiment(
            Path("configs/psb_marl/p2_1_s_mean_only.json")
        )
        distribution_runtime = distribution.p2_runtime_config()
        mean_only_runtime = mean_only.p2_runtime_config()

        self.assertEqual(
            distribution_runtime["branch_adapter"]["max_delta_log_scale"],
            0.25,
        )
        self.assertEqual(
            mean_only_runtime["branch_adapter"]["max_delta_log_scale"],
            0.0,
        )
        self.assertEqual(
            mean_only_runtime["branch_adapter"]["conditioning_mode"],
            "causal_q_gate",
        )
        self.assertNotEqual(distribution_runtime, mean_only_runtime)
        self.assertNotEqual(distribution.output_root, mean_only.output_root)

    def test_negative_log_scale_bound_is_rejected(self):
        with self.assertRaises(PSBConfigError):
            PSBBranchAdapterConfig.from_dict(
                {
                    "pair_hidden_sizes": [8],
                    "context_dim": 4,
                    "adapter_hidden_sizes": [8],
                    "z_scale": 0.5,
                    "max_delta_loc": 0.3,
                    "max_delta_log_scale": -0.1,
                    "conditioning_mode": "causal_q_gate",
                }
            )

    def test_mean_only_adapter_has_no_scale_head_and_preserves_base(self):
        bridge = make_bridge(
            conditioning_mode="causal_q_gate",
            max_delta_log_scale=0.0,
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())

        self.assertFalse(bridge.adapter.adapts_log_scale)
        self.assertEqual(bridge.adapter.network[-1].out_features, 2)
        self.assertEqual(bridge.adapter.causal_gate.out_features, 2)
        self.assertGreater(float(output.delta_loc.abs().max()), 0.0)
        self.assertEqual(float(output.delta_log_scale.abs().max()), 0.0)
        self.assertTrue(torch.equal(output.scale, output.base_scale))

    def test_mean_only_action_gradient_crosses_proximal_layer(self):
        torch.manual_seed(31)
        bridge = make_bridge(
            conditioning_mode="causal_q_gate",
            max_delta_log_scale=0.0,
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())

        output.delta_loc.square().mean().backward()

        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.control_net.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.branch_encoder.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.adapter.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertTrue(torch.equal(output.scale, output.base_scale))
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in bridge.base_policy_net.parameters()
            )
        )

    def test_promotion_gate_enforces_mean_only_scale_contract(self):
        comparisons = [
            {
                "reward_difference_candidate_minus_base": 0.01,
                "collision_difference_candidate_minus_base": -0.01,
            }
            for _ in range(5)
        ]
        rollout = {
            "nonfinite_action_count": 0,
            "nonfinite_reward_count": 0,
            "nonfinite_z_count": 0,
            "max_antisymmetry_error": 0.0,
            "max_root_residual": 0.0,
            "min_root_denominator": 1.0,
            "max_abs_b": 0.1,
            "rollout_scale_matches_base_exactly": True,
            "rollout_delta_log_scale_abs_max": 0.0,
        }
        kwargs = {
            "candidate_rollouts": [dict(rollout) for _ in range(5)],
            "proximal": {
                "residual_tolerance": 1e-6,
                "b_max": 0.5,
            },
            "branch_adapter": {"max_delta_log_scale": 0.0},
        }
        result = _p2_noninferiority_gate(
            comparisons,
            {
                "minimum_paired_seeds": 5,
                "confidence_z": 1.645,
                "reward_margin": 0.0,
                "collision_margin": 0.0,
            },
            **kwargs,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(
            result["structural_checks"]["base_scale_exactly_preserved"]
        )

        kwargs["candidate_rollouts"][0][
            "rollout_scale_matches_base_exactly"
        ] = False
        result = _p2_noninferiority_gate(
            comparisons,
            {
                "minimum_paired_seeds": 5,
                "confidence_z": 1.645,
                "reward_margin": 0.0,
                "collision_margin": 0.0,
            },
            **kwargs,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["structural_checks"]["base_scale_exactly_preserved"]
        )


class P21ProjectedCounterfactualTests(unittest.TestCase):
    def test_projected_training_config_has_distinct_runtime_contract(self):
        mean_only = load_psb_experiment(
            Path("configs/psb_marl/p2_1_s_mean_only.json")
        )
        projected = load_psb_experiment(
            Path("configs/psb_marl/p2_1_p_projected_mean_only.json")
        )

        self.assertEqual(mean_only.branch_adapter.action_projection, "full")
        self.assertEqual(
            projected.branch_adapter.action_projection,
            "longitudinal_only",
        )
        self.assertNotIn(
            "action_projection",
            mean_only.p2_runtime_config()["branch_adapter"],
        )
        self.assertEqual(
            projected.p2_runtime_config()["branch_adapter"][
                "action_projection"
            ],
            "longitudinal_only",
        )
        self.assertNotEqual(
            mean_only.p2_runtime_config(), projected.p2_runtime_config()
        )
        self.assertNotEqual(mean_only.output_root, projected.output_root)

    def test_projected_training_requires_causal_mean_only_policy(self):
        base = {
            "pair_hidden_sizes": [8],
            "context_dim": 4,
            "adapter_hidden_sizes": [8],
            "z_scale": 0.5,
            "max_delta_loc": 0.3,
            "max_delta_log_scale": 0.2,
            "conditioning_mode": "causal_q_gate",
            "action_projection": "longitudinal_only",
        }
        with self.assertRaises(PSBConfigError):
            PSBBranchAdapterConfig.from_dict(base)
        base["max_delta_log_scale"] = 0.0
        base["conditioning_mode"] = "general"
        with self.assertRaises(PSBConfigError):
            PSBBranchAdapterConfig.from_dict(base)

    def test_longitudinal_projection_preserves_base_steering_mean(self):
        bridge = make_bridge(
            conditioning_mode="causal_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())

        self.assertGreater(float(output.delta_loc[..., 0].abs().max()), 0.0)
        self.assertEqual(float(output.delta_loc[..., 1].abs().max()), 0.0)
        self.assertTrue(
            torch.equal(output.loc[..., 1], output.base_loc[..., 1])
        )
        self.assertTrue(torch.equal(output.scale, output.base_scale))

    def test_projection_is_compatible_with_existing_mean_only_checkpoint(self):
        source = make_bridge(
            conditioning_mode="causal_q_gate",
            max_delta_log_scale=0.0,
        )
        projected = make_bridge(
            conditioning_mode="causal_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )

        projected.load_state_dict(source.state_dict(), strict=True)

        self.assertNotIn("adapter.mean_action_mask", source.state_dict())
        self.assertNotIn("adapter.mean_action_mask", projected.state_dict())
        self.assertTrue(
            torch.equal(
                projected.adapter.mean_action_mask,
                torch.tensor([1.0, 0.0]),
            )
        )

    def test_projection_keeps_speed_gradient_through_proximal_layer(self):
        torch.manual_seed(37)
        bridge = make_bridge(
            conditioning_mode="causal_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())

        output.delta_loc[..., 0].square().mean().backward()

        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.control_net.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertEqual(float(output.delta_loc[..., 1].abs().max()), 0.0)

    def test_projection_gate_rejects_any_steering_mean_drift(self):
        comparisons = [
            {
                "reward_difference_candidate_minus_base": 0.01,
                "collision_difference_candidate_minus_base": -0.01,
            }
            for _ in range(5)
        ]
        rollout = {
            "nonfinite_action_count": 0,
            "nonfinite_reward_count": 0,
            "nonfinite_z_count": 0,
            "max_antisymmetry_error": 0.0,
            "max_root_residual": 0.0,
            "min_root_denominator": 1.0,
            "max_abs_b": 0.1,
            "rollout_scale_matches_base_exactly": True,
            "rollout_delta_log_scale_abs_max": 0.0,
            "rollout_delta_steering_abs_max": 0.0,
        }
        candidate_rollouts = [dict(rollout) for _ in range(5)]
        promotion = {
            "minimum_paired_seeds": 5,
            "confidence_z": 1.645,
            "reward_margin": 0.0,
            "collision_margin": 0.0,
        }
        kwargs = {
            "candidate_rollouts": candidate_rollouts,
            "proximal": {
                "residual_tolerance": 1e-6,
                "b_max": 0.5,
            },
            "branch_adapter": {"max_delta_log_scale": 0.0},
            "action_projection": "longitudinal_only",
        }

        result = _p2_noninferiority_gate(comparisons, promotion, **kwargs)
        self.assertTrue(result["passed"])
        self.assertTrue(
            result["structural_checks"][
                "base_steering_mean_exactly_preserved"
            ]
        )

        candidate_rollouts[0]["rollout_delta_steering_abs_max"] = 1e-8
        result = _p2_noninferiority_gate(comparisons, promotion, **kwargs)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["structural_checks"][
                "base_steering_mean_exactly_preserved"
            ]
        )


class P21GainBoundedSectorTests(unittest.TestCase):
    def test_sector_config_has_a_distinct_runtime_contract(self):
        projected = load_psb_experiment(
            Path("configs/psb_marl/p2_1_p_projected_mean_only.json")
        )
        sector = load_psb_experiment(
            Path("configs/psb_marl/p2_1_g_sector_projected_mean_only.json")
        )

        self.assertEqual(
            sector.branch_adapter.conditioning_mode,
            "sector_q_gate",
        )
        self.assertEqual(
            sector.branch_adapter.action_projection,
            "longitudinal_only",
        )
        self.assertEqual(sector.branch_adapter.max_delta_log_scale, 0.0)
        self.assertNotEqual(
            projected.p2_runtime_config(), sector.p2_runtime_config()
        )
        self.assertNotEqual(projected.output_root, sector.output_root)

    def test_sector_envelope_bounds_action_under_large_learned_gain(self):
        torch.manual_seed(41)
        bridge = make_bridge(
            conditioning_mode="sector_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(50.0)
            bridge.adapter.causal_gate.weight.fill_(50.0)
        inputs = pair_inputs(environments=1)
        inputs["z_prev_dense"][0, 0, 1] = 0.1
        inputs["z_prev_dense"][0, 1, 0] = -0.1

        output = bridge(**inputs)
        sector_bound = bridge.adapter.max_delta_loc * output.branch_activity

        self.assertGreater(float(output.branch_activity.max()), 0.0)
        self.assertLessEqual(float(output.branch_activity.max()), 1.0)
        self.assertTrue(
            torch.all(output.delta_loc.abs() <= sector_bound + 1e-7)
        )
        self.assertEqual(float(output.delta_loc[..., 1].abs().max()), 0.0)
        self.assertTrue(torch.equal(output.scale, output.base_scale))

    def test_zero_sector_activity_exactly_recovers_base(self):
        torch.manual_seed(43)
        bridge = make_bridge(
            conditioning_mode="sector_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.control_net.parameters():
                parameter.zero_()
            for parameter in bridge.adapter.parameters():
                parameter.normal_(mean=1.0, std=0.5)
        output = bridge(**pair_inputs())

        self.assertEqual(float(output.branch_activity.abs().max()), 0.0)
        self.assertEqual(float(output.delta_loc.abs().max()), 0.0)
        self.assertTrue(torch.equal(output.loc, output.base_loc))
        self.assertTrue(torch.equal(output.scale, output.base_scale))

    def test_sector_speed_gradient_crosses_proximal_layer(self):
        torch.manual_seed(47)
        bridge = make_bridge(
            conditioning_mode="sector_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())
        self.assertGreater(float(output.delta_loc[..., 0].abs().max()), 0.0)

        output.delta_loc[..., 0].square().mean().backward()

        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.control_net.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in bridge.base_policy_net.parameters()
            )
        )

    def test_promotion_gate_rejects_sector_bound_violation(self):
        comparisons = [
            {
                "reward_difference_candidate_minus_base": 0.01,
                "collision_difference_candidate_minus_base": -0.01,
            }
            for _ in range(5)
        ]
        rollout = {
            "nonfinite_action_count": 0,
            "nonfinite_reward_count": 0,
            "nonfinite_z_count": 0,
            "max_antisymmetry_error": 0.0,
            "max_root_residual": 0.0,
            "min_root_denominator": 1.0,
            "max_abs_b": 0.1,
            "rollout_scale_matches_base_exactly": True,
            "rollout_delta_log_scale_abs_max": 0.0,
            "rollout_delta_steering_abs_max": 0.0,
            "rollout_sector_bound_max_violation": 0.0,
        }
        candidate_rollouts = [dict(rollout) for _ in range(5)]
        kwargs = {
            "candidate_rollouts": candidate_rollouts,
            "proximal": {
                "residual_tolerance": 1e-6,
                "b_max": 0.5,
            },
            "branch_adapter": {
                "max_delta_log_scale": 0.0,
                "conditioning_mode": "sector_q_gate",
            },
            "action_projection": "longitudinal_only",
        }
        promotion = {
            "minimum_paired_seeds": 5,
            "confidence_z": 1.645,
            "reward_margin": 0.0,
            "collision_margin": 0.0,
        }

        result = _p2_noninferiority_gate(comparisons, promotion, **kwargs)
        self.assertTrue(result["passed"])
        self.assertTrue(result["structural_checks"]["sector_bound_satisfied"])

        candidate_rollouts[0]["rollout_sector_bound_max_violation"] = 1e-6
        result = _p2_noninferiority_gate(comparisons, promotion, **kwargs)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["structural_checks"]["sector_bound_satisfied"]
        )


class P21UrgencySupportedSectorTests(unittest.TestCase):
    def test_supported_config_is_isolated_from_gain_bounded_config(self):
        gain_bounded = load_psb_experiment(
            Path("configs/psb_marl/p2_1_g_sector_projected_mean_only.json")
        )
        supported = load_psb_experiment(
            Path("configs/psb_marl/p2_1_u_urgency_supported_sector.json")
        )

        self.assertEqual(
            supported.branch_adapter.conditioning_mode,
            "supported_sector_q_gate",
        )
        self.assertEqual(
            supported.branch_adapter.action_projection,
            "longitudinal_only",
        )
        self.assertEqual(supported.branch_adapter.max_delta_log_scale, 0.0)
        self.assertNotEqual(
            gain_bounded.p2_runtime_config(), supported.p2_runtime_config()
        )
        self.assertNotEqual(gain_bounded.output_root, supported.output_root)

    def test_zero_urgency_suppresses_action_but_preserves_opinion_memory(self):
        torch.manual_seed(53)
        bridge = make_bridge(
            conditioning_mode="supported_sector_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.control_net.parameters():
                parameter.zero_()
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(50.0)
            bridge.adapter.causal_gate.weight.fill_(50.0)
        inputs = pair_inputs(environments=1)
        inputs["urgency"].zero_()
        inputs["z_prev_dense"][0, 0, 1] = 0.25
        inputs["z_prev_dense"][0, 1, 0] = -0.25

        output = bridge(**inputs)

        self.assertGreater(float(output.q.abs().max()), 0.0)
        self.assertGreater(float(output.z_next_dense.abs().max()), 0.0)
        self.assertEqual(float(output.branch_activity.abs().max()), 0.0)
        self.assertEqual(float(output.delta_loc.abs().max()), 0.0)
        self.assertTrue(torch.equal(output.loc, output.base_loc))

    def test_supported_sector_is_bounded_and_keeps_proximal_gradient(self):
        torch.manual_seed(59)
        bridge = make_bridge(
            conditioning_mode="supported_sector_q_gate",
            max_delta_log_scale=0.0,
            mean_action_mask=(1.0, 0.0),
        )
        with torch.no_grad():
            for parameter in bridge.adapter.network.parameters():
                parameter.zero_()
            bridge.adapter.network[-1].bias.fill_(0.5)
            bridge.adapter.causal_gate.weight.fill_(0.5)
        output = bridge(**pair_inputs())
        sector_bound = bridge.adapter.max_delta_loc * output.branch_activity

        self.assertGreater(float(output.branch_activity.max()), 0.0)
        self.assertTrue(
            torch.all(output.delta_loc.abs() <= sector_bound + 1e-7)
        )
        output.delta_loc[..., 0].square().mean().backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.control_net.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertEqual(float(output.delta_loc[..., 1].abs().max()), 0.0)


class P2SequenceTests(unittest.TestCase):
    def _rollout(self, bridge, steps=4):
        torch.manual_seed(11)
        inputs = pair_inputs(environments=1)
        td = TensorDict({}, batch_size=[1, steps])
        repeated = {
            "observation": torch.stack(
                [
                    # The stored Base parameters do not depend on this original
                    # placeholder; replace it below with matching observations.
                    torch.zeros_like(inputs["observation"])
                    for _ in range(steps)
                ],
                dim=1,
            ),
            "pair_features": inputs["pair_features"].unsqueeze(1).expand(
                -1, steps, -1, -1, -1
            ),
            "neighbor_ids": inputs["neighbor_ids"].unsqueeze(1).expand(
                -1, steps, -1, -1
            ),
            "urgency": inputs["urgency"].unsqueeze(1).expand(
                -1, steps, -1, -1
            ),
            "confidence": inputs["confidence"].unsqueeze(1).expand(
                -1, steps, -1, -1
            ),
            "pair_mask": inputs["pair_mask"].unsqueeze(1).expand(
                -1, steps, -1, -1
            ),
        }
        # Recreate and store one consistent rollout because each step above used
        # a distinct observation.
        z_dense = inputs["z_prev_dense"]
        observations = []
        actions = []
        log_probs = []
        stored = {name: [] for name in bridge(**inputs)._fields}
        z_prev_values = []
        for _ in range(steps):
            observation = torch.randn_like(inputs["observation"])
            with torch.no_grad():
                output = bridge(
                    **{**inputs, "observation": observation, "z_prev_dense": z_dense}
                )
                distribution = TanhNormal(
                    output.loc, output.scale, min=-1.0, max=1.0
                )
                action = distribution.rsample()
                log_prob = distribution.log_prob(action)
            observations.append(observation)
            z_prev_values.append(z_dense)
            actions.append(action)
            log_probs.append(log_prob)
            for name in output._fields:
                stored[name].append(getattr(output, name))
            z_dense = output.z_next_dense
        repeated["observation"] = torch.stack(observations, dim=1)

        values = (
            (("agents", "observation"), repeated["observation"]),
            (("agents", "action"), torch.stack(actions, dim=1)),
            (("agents", "sample_log_prob"), torch.stack(log_probs, dim=1)),
            (("agents", "advantage"), torch.ones(1, steps, 3, 1)),
            (("agents", "info", "pair_features"), repeated["pair_features"]),
            (("agents", "info", "neighbor_ids"), repeated["neighbor_ids"]),
            (("agents", "info", "urgency"), repeated["urgency"]),
            (("agents", "info", "confidence"), repeated["confidence"]),
            (("agents", "info", "pair_mask"), repeated["pair_mask"]),
            (
                ("agents", "info", "agent_reset_mask"),
                torch.zeros(1, steps, 3, dtype=torch.bool),
            ),
            (
                ("agents", "psb", "z_prev_dense"),
                torch.stack(z_prev_values, dim=1),
            ),
            (("agents", "psb", "b"), torch.stack(stored["b_dense"], dim=1)),
            (
                ("agents", "psb", "z_next_dense"),
                torch.stack(stored["z_next_dense"], dim=1),
            ),
            (("next", "done"), torch.zeros(1, steps, 1, dtype=torch.bool)),
            (("collector", "traj_ids"), torch.zeros(1, steps, dtype=torch.long)),
            ("done", torch.zeros(1, steps, 1, dtype=torch.bool)),
        )
        for key, value in values:
            td.set(key, value)
        return td

    def test_sequence_recompute_matches_rollout_and_backpropagates(self):
        bridge = make_bridge()
        rollout = self._rollout(bridge)
        buffer = P2SequenceBuffer(rollout, chunk_length=4)
        batch = next(buffer.iter_minibatches(minibatch_size=8))
        loss = P2SequencePPOLoss(
            actor=DistributionBuilder(),
            bridge=bridge,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            advantage_key=("agents", "advantage"),
            n_agents=3,
            clip_epsilon=0.2,
            entropy_coefficient=1e-4,
            energy_coefficient=1e-3,
            control_trust_region_coefficient=1e-2,
            saturation_coefficient=1e-3,
            saturation_fraction=0.8,
        )
        result = loss(batch)
        total = (
            result["loss_objective"]
            + result["loss_entropy"]
            + result["loss_regularization"]
        )
        total.backward()
        self.assertLess(float(result["log_prob_abs_error"]), 1e-5)
        self.assertLess(float(result["state_replay_abs_error"]), 1e-6)
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in bridge.adapter.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )


class P2CriticAndPromotionTests(unittest.TestCase):
    def test_augmented_critic_starts_at_base_and_stops_z_gradient(self):
        class BaseCritic(nn.Module):
            def forward(self, observation):
                return observation.sum(dim=-1, keepdim=True)

        critic = AugmentedCentralCritic(
            base_critic_net=BaseCritic(),
            n_agents=3,
            observation_dim=4,
            candidate_count=2,
            hidden_sizes=(8,),
        )
        observation = torch.randn(2, 3, 4)
        z_dense = torch.randn(2, 3, 3, requires_grad=True)
        pair_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        value = critic(observation, z_dense, pair_mask)
        self.assertTrue(torch.equal(value, BaseCritic()(observation)))
        value.sum().backward()
        self.assertIsNone(z_dense.grad)

    def test_promotion_gate_uses_reward_lower_and_collision_upper_bounds(self):
        comparisons = [
            {
                "reward_difference_candidate_minus_base": 0.01,
                "collision_difference_candidate_minus_base": -0.01,
            }
            for _ in range(5)
        ]
        result = _p2_noninferiority_gate(
            comparisons,
            {
                "minimum_paired_seeds": 5,
                "confidence_z": 1.645,
                "reward_margin": 0.0,
                "collision_margin": 0.0,
            },
        )
        self.assertTrue(result["passed"])
        comparisons[0]["reward_difference_candidate_minus_base"] = -1.0
        self.assertFalse(
            _p2_noninferiority_gate(
                comparisons,
                {
                    "minimum_paired_seeds": 5,
                    "confidence_z": 1.645,
                    "reward_margin": 0.0,
                    "collision_margin": 0.0,
                },
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
