"""A6 one-step differentiable policy and rollout-only safety controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torchrl.modules import TanhNormal

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.avocado.core import opinion_euler_step
from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.avocado_marl.a6_config import A6ExperimentConfig
from utilities.avocado_marl.bridge import A4ActionBridge
from utilities.avocado_marl.y_correction import (
    YCorrectionFeatures,
    YCorrectionNet,
    build_y_correction_features,
)
from utilities.opinion.residual import OpinionResidual


class A6PolicyOutput(NamedTuple):
    loc: Tensor
    scale: Tensor
    base_loc: Tensor
    correction: Tensor
    fused_estimate: Tensor
    z_next: Tensor
    q: Tensor
    normalized_weights: Tensor
    aggregate: Tensor
    residual: Tensor


@dataclass(frozen=True)
class A6RolloutStep:
    tensordict: object
    features: Tensor
    confidence: Tensor
    pair_mask: Tensor
    prospective_attention: Tensor
    heuristic_estimate: Tensor
    z_prev: Tensor
    nominal_action: Tensor
    old_log_prob: Tensor
    policy_output: A6PolicyOutput


class A6OneStepPolicy(nn.Module):
    """Pure current-step mapping used both during rollout and PPO replay.

    ``z_prev`` is deliberately supplied as detached rollout data.  The module
    differentiates through exactly one AVOCADO Euler update and never owns
    temporal state.
    """

    def __init__(
        self,
        base_policy_net: nn.Module,
        y_correction_net: YCorrectionNet,
        residual: OpinionResidual,
        a3_config: A3RoadExperimentConfig,
    ) -> None:
        super().__init__()
        self.base_policy_net = base_policy_net
        self.y_correction_net = y_correction_net
        self.residual = residual
        self.avocado_parameters = a3_config.parameters
        self.base_policy_net.eval()
        for parameter in self.base_policy_net.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        observation: Tensor,
        pair_features: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
        prospective_attention: Tensor,
        heuristic_estimate: Tensor,
        z_prev: Tensor,
    ) -> A6PolicyOutput:
        if pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool.")
        pair_shape = pair_mask.shape
        for value, label in (
            (confidence, "confidence"),
            (prospective_attention, "prospective_attention"),
            (heuristic_estimate, "heuristic_estimate"),
            (z_prev, "z_prev"),
        ):
            if value.shape != pair_shape:
                raise ValueError(f"{label} must match pair_mask shape.")

        base_loc, base_scale = self.base_policy_net(observation)
        correction_output = self.y_correction_net(
            pair_features, confidence, pair_mask
        )
        correction = correction_output.correction
        fused = (heuristic_estimate + correction).clamp(-1.0, 1.0)
        parameters = self.avocado_parameters
        proposed_z = opinion_euler_step(
            z_prev,
            prospective_attention,
            fused,
            dt=parameters.dt,
            decay=parameters.opinion_decay,
            self_weight=parameters.opinion_self_weight,
            estimate_weight=parameters.opinion_estimate_weight,
            bias=parameters.opinion_bias,
        )
        z_next = torch.where(pair_mask, proposed_z, z_prev)
        residual_output = self.residual(
            z_next, prospective_attention, pair_mask
        )
        loc = self.residual.apply_to_loc(base_loc, residual_output.residual)
        return A6PolicyOutput(
            loc=loc,
            scale=base_scale,
            base_loc=base_loc,
            correction=correction,
            fused_estimate=fused,
            z_next=z_next,
            q=residual_output.q,
            normalized_weights=residual_output.normalized_weights,
            aggregate=residual_output.aggregate,
            residual=residual_output.residual,
        )

    def trainable_parameters(self):
        return self.y_correction_net.parameters()


class _IdentityActionPolicy(nn.Module):
    def forward(self, tensordict):
        return tensordict


class A6ExecutionBridge(A4ActionBridge):
    """Apply the frozen A3.4 safety chain to an already sampled action."""

    stage_label = "A6 learned y-correction"

    def __init__(
        self,
        scenario: ScenarioRoadTraffic,
        a3_config: A3RoadExperimentConfig,
        config: A6ExperimentConfig,
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
        self.a6_config = config
        self._staged_correction = None

    def stage_correction(self, correction: Tensor) -> None:
        self._staged_correction = correction.detach()

    def opinion_estimate_correction(
        self, positions: Tensor, velocities: Tensor, yaws: Tensor
    ) -> Tensor:
        del positions, velocities, yaws
        if self._staged_correction is None:
            raise RuntimeError("A6 correction must be staged before execution.")
        correction = self._staged_correction
        self._staged_correction = None
        return correction


class A6RolloutController:
    """Own AVOCADO state during collection while keeping PPO replay pure."""

    nominal_action_key = ("agents", "a6", "nominal_action")

    def __init__(
        self,
        policy: A6OneStepPolicy,
        execution_bridge: A6ExecutionBridge,
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
            raise RuntimeError("A6 requires an AVOCADO controller.")
        return controller

    def distribution(self, output: A6PolicyOutput) -> TanhNormal:
        return TanhNormal(
            output.loc,
            output.scale,
            min=self.action_low,
            max=self.action_high,
        )

    def _features(self) -> tuple[Tensor, Tensor, Tensor, YCorrectionFeatures]:
        bridge = self.execution_bridge
        positions, velocities, yaws = bridge._road_state()
        features = build_y_correction_features(
            self.controller,
            positions,
            velocities,
            yaws,
            candidate_count=self.candidate_count,
        )
        return positions, velocities, yaws, features

    @torch.no_grad()
    def step(self, tensordict, *, deterministic: bool = False) -> A6RolloutStep:
        _, _, _, features = self._features()
        z_prev = self.controller.opinion.detach().clone()
        output = self.policy(
            tensordict.get(self.observation_key),
            features.values,
            features.confidence,
            features.pair_mask,
            features.prospective_attention,
            features.heuristic_estimate,
            z_prev,
        )
        distribution = self.distribution(output)
        nominal_action = distribution.mode if deterministic else distribution.rsample()
        old_log_prob = distribution.log_prob(nominal_action)
        tensordict.set(self.execution_bridge.action_key, nominal_action)
        tensordict.set(self.nominal_action_key, nominal_action.clone())
        self.execution_bridge.stage_correction(output.correction)
        tensordict = self.execution_bridge(tensordict)

        # The differentiable policy intentionally carries only the configured
        # nearest candidates.  The execution controller still updates every
        # perceived AVOCADO edge for geometric responsibility.
        active = features.pair_mask
        if bool(active.any()):
            torch.testing.assert_close(
                output.z_next[active],
                self.controller.opinion[active],
                rtol=1e-5,
                atol=1e-6,
            )
        return A6RolloutStep(
            tensordict=tensordict,
            features=features.values.detach().clone(),
            confidence=features.confidence.detach().clone(),
            pair_mask=features.pair_mask.detach().clone(),
            prospective_attention=features.prospective_attention.detach().clone(),
            heuristic_estimate=features.heuristic_estimate.detach().clone(),
            z_prev=z_prev,
            nominal_action=nominal_action.detach().clone(),
            old_log_prob=old_log_prob.detach().clone(),
            policy_output=A6PolicyOutput(*(value.detach().clone() for value in output)),
        )

    def reset_agents(self, reset_mask: Tensor) -> None:
        self.execution_bridge.reset_agents(reset_mask)

    def reset_all(self) -> None:
        self.controller.reset()
