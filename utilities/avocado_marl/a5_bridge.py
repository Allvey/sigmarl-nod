"""A5 bridge with a frozen strict-zero YCorrectionNet."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.avocado.road_config import A3RoadExperimentConfig
from utilities.avocado_marl.a5_config import A5YCorrectionConfig
from utilities.avocado_marl.bridge import A4ActionBridge
from utilities.avocado_marl.y_correction import (
    YCorrectionFeatures,
    YCorrectionNet,
    YCorrectionOutput,
    build_y_correction_features,
)


class A5ActionBridge(A4ActionBridge):
    """Inject an exactly zero learned correction into the A4 controller."""

    stage_label = "A5 frozen zero correction"

    def __init__(
        self,
        base_policy: torch.nn.Module,
        scenario: ScenarioRoadTraffic,
        a3_config: A3RoadExperimentConfig,
        correction_config: A5YCorrectionConfig,
        *,
        deterministic: bool,
        velocity_continuity_weight: float,
        speed_intervention_tolerance_mps: float,
        steering_intervention_tolerance_degrees: float,
    ) -> None:
        super().__init__(
            base_policy,
            scenario,
            a3_config,
            use_avocado=True,
            deterministic=deterministic,
            velocity_continuity_weight=velocity_continuity_weight,
            speed_intervention_tolerance_mps=speed_intervention_tolerance_mps,
            steering_intervention_tolerance_degrees=(
                steering_intervention_tolerance_degrees
            ),
        )
        self.correction_config = correction_config
        # Network construction must not advance the environment RNG: A4/A5
        # equality must survive partial vehicle resets as well as short runs.
        with torch.random.fork_rng():
            torch.manual_seed(a3_config.simulation.seed)
            self.y_correction_net = YCorrectionNet(
                feature_dim=correction_config.feature_dim,
                hidden_sizes=correction_config.hidden_sizes,
                maximum_correction=correction_config.maximum_correction,
                temperature=correction_config.temperature,
                strict_zero=correction_config.strict_zero,
                freeze=correction_config.freeze,
            ).to(scenario.world.device)
        self.y_correction_net.eval()
        self.last_correction_features: Optional[YCorrectionFeatures] = None
        self.last_correction_output: Optional[YCorrectionOutput] = None

    def opinion_estimate_correction(
        self,
        positions: Tensor,
        velocities: Tensor,
        yaws: Tensor,
    ) -> Tensor:
        if self.controller is None:
            raise RuntimeError("A5 requires an AVOCADO controller.")
        features = build_y_correction_features(
            self.controller,
            positions,
            velocities,
            yaws,
            candidate_count=self.correction_config.candidate_count,
        )
        output = self.y_correction_net(
            features.values,
            features.confidence,
            features.pair_mask,
        )
        self.last_correction_features = features
        self.last_correction_output = output
        return output.correction

    @torch.no_grad()
    def __call__(self, tensordict):
        tensordict = super().__call__(tensordict)
        if (
            self.controller is None
            or self.last_correction_features is None
            or self.last_correction_output is None
        ):
            raise RuntimeError("A5 correction diagnostics were not produced.")
        active = self.controller.last_neighbor_mask
        if not torch.equal(
            self.last_correction_features.heuristic_estimate[active],
            self.controller.last_estimated_opinion[active],
        ):
            raise RuntimeError(
                "A5 prospective y^H does not match AVOCADO's current estimate."
            )
        if not torch.equal(
            self.last_correction_features.prospective_attention[active],
            self.controller.attention[active],
        ):
            raise RuntimeError(
                "A5 prospective attention does not match AVOCADO's update."
            )
        return tensordict
