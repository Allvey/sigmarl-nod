"""A6-Action policy: learn preferred actions, retain AVOCADO as safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torchrl.modules import TanhNormal

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.avocado_marl.a6_action_config import A6ActionExperimentConfig
from utilities.avocado_marl.bridge import A4ActionBridge
from utilities.avocado_marl.y_correction import (
    YCorrectionFeatures,
    build_y_correction_features,
)


class ActionCorrectionOutput(NamedTuple):
    logit: Tensor
    raw_correction: Tensor
    loc_correction: Tensor
    interaction_context: Tensor


class A6ActionPolicyOutput(NamedTuple):
    loc: Tensor
    scale: Tensor
    base_loc: Tensor
    loc_correction: Tensor
    interaction_context: Tensor


@dataclass(frozen=True)
class A6ActionRolloutStep:
    tensordict: object
    features: Tensor
    confidence: Tensor
    pair_mask: Tensor
    heuristic_estimate: Tensor
    nominal_action: Tensor
    old_log_prob: Tensor
    policy_output: A6ActionPolicyOutput


class InteractionActionNet(nn.Module):
    """Permutation-invariant interaction network for actor-loc corrections."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_sizes: Sequence[int],
        maximum_loc_correction: Sequence[float],
        zero_initialization: bool,
        freeze: bool,
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(
            type(size) is not int or size <= 0 for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers.")
        if len(maximum_loc_correction) != 2 or any(
            float(value) <= 0.0 for value in maximum_loc_correction
        ):
            raise ValueError("maximum_loc_correction must contain two positives.")
        pair_layers = []
        input_dim = int(feature_dim)
        for hidden_dim in hidden_sizes:
            pair_layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        self.pair_encoder = nn.Sequential(*pair_layers)
        context_dim = 2 * input_dim + 2
        self.action_head = nn.Sequential(
            nn.Linear(context_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 2),
        )
        final_layer = self.action_head[-1]
        if zero_initialization:
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        else:
            nn.init.xavier_uniform_(final_layer.weight, gain=1e-2)
            nn.init.zeros_(final_layer.bias)
        self.feature_dim = int(feature_dim)
        self.register_buffer(
            "maximum_loc_correction",
            torch.tensor(tuple(maximum_loc_correction), dtype=torch.float32),
        )
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        features: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
        base_loc: Tensor,
    ) -> ActionCorrectionOutput:
        if features.shape[:-1] != pair_mask.shape:
            raise ValueError("features and pair_mask axes must match.")
        if features.shape[-1] != self.feature_dim:
            raise ValueError("Invalid InteractionActionNet feature dimension.")
        if confidence.shape != pair_mask.shape or pair_mask.dtype != torch.bool:
            raise ValueError("confidence and bool pair_mask must have equal shape.")
        if base_loc.shape != pair_mask.shape[:-1] + (2,):
            raise ValueError("base_loc must have shape [batch, agents, 2].")

        encoded = self.pair_encoder(features)
        mask = pair_mask.unsqueeze(-1)
        has_pair = pair_mask.any(dim=-1, keepdim=True)
        weights = confidence.clamp(0.0, 1.0) * pair_mask.to(confidence.dtype)
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weighted_context = (encoded * normalized.unsqueeze(-1)).sum(dim=-2)
        maximum_context = encoded.masked_fill(~mask, -torch.inf).max(dim=-2).values
        maximum_context = torch.where(
            has_pair,
            maximum_context,
            torch.zeros_like(maximum_context),
        )
        context = torch.cat(
            (weighted_context, maximum_context, base_loc.detach()), dim=-1
        )
        logit = self.action_head(context)
        raw = torch.tanh(logit)
        correction = raw * self.maximum_loc_correction.to(raw)
        correction = correction * has_pair.to(correction.dtype)
        return ActionCorrectionOutput(logit, raw, correction, context)


class A6ActionPolicy(nn.Module):
    """Frozen Base Actor plus a trainable interaction-conditioned action head."""

    def __init__(
        self,
        base_policy_net: nn.Module,
        action_net: InteractionActionNet,
    ) -> None:
        super().__init__()
        self.base_policy_net = base_policy_net
        self.action_net = action_net
        self.base_policy_net.eval()
        for parameter in self.base_policy_net.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        observation: Tensor,
        pair_features: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
    ) -> A6ActionPolicyOutput:
        base_loc, base_scale = self.base_policy_net(observation)
        correction = self.action_net(
            pair_features,
            confidence,
            pair_mask,
            base_loc,
        )
        return A6ActionPolicyOutput(
            loc=base_loc + correction.loc_correction,
            scale=base_scale,
            base_loc=base_loc,
            loc_correction=correction.loc_correction,
            interaction_context=correction.interaction_context,
        )

    def trainable_parameters(self):
        return self.action_net.parameters()


class _IdentityActionPolicy(nn.Module):
    def forward(self, tensordict):
        return tensordict


class A6ActionExecutionBridge(A4ActionBridge):
    """Apply unchanged heuristic AVOCADO/OCA to a learned preferred action."""

    stage_label = "A6-Action learned preferred action + fixed AVOCADO safety"

    def __init__(
        self,
        scenario: ScenarioRoadTraffic,
        a3_config: A3RoadExperimentConfig,
        config: A6ActionExperimentConfig,
        *,
        velocity_continuity_weight: float,
        speed_tolerance: float,
        steering_tolerance_degrees: float,
    ) -> None:
        super().__init__(
            _IdentityActionPolicy(),
            scenario,
            a3_config,
            use_avocado=True,
            deterministic=False,
            velocity_continuity_weight=velocity_continuity_weight,
            speed_intervention_tolerance_mps=speed_tolerance,
            steering_intervention_tolerance_degrees=steering_tolerance_degrees,
        )
        self.a6_action_config = config


class A6ActionRolloutController:
    """Build local interaction features and execute preferred actions safely."""

    nominal_action_key = ("agents", "a6_action", "nominal_action")

    def __init__(
        self,
        policy: A6ActionPolicy,
        execution_bridge: A6ActionExecutionBridge,
        observation_key,
        action_low: Tensor,
        action_high: Tensor,
        candidate_count: int,
    ) -> None:
        self.policy = policy
        self.execution_bridge = execution_bridge
        self.observation_key = observation_key
        self.action_low = action_low
        self.action_high = action_high
        self.candidate_count = int(candidate_count)

    @property
    def controller(self):
        controller = self.execution_bridge.controller
        if controller is None:
            raise RuntimeError("A6-Action requires an AVOCADO controller.")
        return controller

    def distribution(self, output: A6ActionPolicyOutput) -> TanhNormal:
        return TanhNormal(
            output.loc,
            output.scale,
            min=self.action_low,
            max=self.action_high,
        )

    def _features(self) -> YCorrectionFeatures:
        bridge = self.execution_bridge
        positions, velocities, yaws = bridge._road_state()
        return build_y_correction_features(
            self.controller,
            positions,
            velocities,
            yaws,
            candidate_count=self.candidate_count,
        )

    @torch.no_grad()
    def step(
        self, tensordict, *, deterministic: bool = False
    ) -> A6ActionRolloutStep:
        features = self._features()
        output = self.policy(
            tensordict.get(self.observation_key),
            features.values,
            features.confidence,
            features.pair_mask,
        )
        distribution = self.distribution(output)
        nominal_action = (
            distribution.mode if deterministic else distribution.rsample()
        )
        old_log_prob = distribution.log_prob(nominal_action)
        tensordict.set(self.execution_bridge.action_key, nominal_action)
        tensordict.set(self.nominal_action_key, nominal_action.clone())
        tensordict = self.execution_bridge(tensordict)
        return A6ActionRolloutStep(
            tensordict=tensordict,
            features=features.values.detach().clone(),
            confidence=features.confidence.detach().clone(),
            pair_mask=features.pair_mask.detach().clone(),
            heuristic_estimate=features.heuristic_estimate.detach().clone(),
            nominal_action=nominal_action.detach().clone(),
            old_log_prob=old_log_prob.detach().clone(),
            policy_output=A6ActionPolicyOutput(
                *(value.detach().clone() for value in output)
            ),
        )

    def reset_agents(self, reset_mask: Tensor) -> None:
        self.execution_bridge.reset_agents(reset_mask)

    def reset_all(self) -> None:
        self.controller.reset()
