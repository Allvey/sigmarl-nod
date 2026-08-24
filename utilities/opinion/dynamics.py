"""Fixed nonlinear opinion dynamics with no trainable parameters."""

from __future__ import annotations

import torch
from torch import nn

from utilities.opinion.config import DynamicsConfig


class OpinionDynamics(nn.Module):
    """Integrate one physical step of the fixed opinion ODE.

    ``pair_mask`` gates conflict evidence and self-reinforcement. When an edge
    becomes inactive, the remaining opinion follows the configured decay term
    toward zero. Stateful storage, identity mapping, and reset handling remain
    the responsibility of the M6 collector.
    """

    def __init__(
        self,
        response_rate: float,
        decay_rate: float,
        self_reinforcement: float,
        nonlinear_sensitivity: float,
    ) -> None:
        super().__init__()
        values = {
            "response_rate": response_rate,
            "decay_rate": decay_rate,
            "self_reinforcement": self_reinforcement,
            "nonlinear_sensitivity": nonlinear_sensitivity,
        }
        if response_rate <= 0 or decay_rate <= 0 or nonlinear_sensitivity <= 0:
            raise ValueError(
                "response_rate, decay_rate, and nonlinear_sensitivity must be positive."
            )
        if self_reinforcement < 0:
            raise ValueError("self_reinforcement must be non-negative.")
        for name, value in values.items():
            self.register_buffer(name, torch.tensor(float(value)))
        self._response_rate_value = float(response_rate)
        self._decay_rate_value = float(decay_rate)
        self._self_reinforcement_value = float(self_reinforcement)

    @classmethod
    def from_config(cls, config: DynamicsConfig) -> "OpinionDynamics":
        return cls(
            response_rate=config.response_rate,
            decay_rate=config.decay_rate,
            self_reinforcement=config.self_reinforcement,
            nonlinear_sensitivity=config.nonlinear_sensitivity,
        )

    def forward(
        self,
        z_prev: torch.Tensor,
        evidence: torch.Tensor,
        urgency: torch.Tensor,
        pair_mask: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        if z_prev.shape != evidence.shape or z_prev.shape != urgency.shape:
            raise ValueError("z_prev, evidence, and urgency must have identical shapes.")
        if pair_mask.shape != z_prev.shape or pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool and have the same shape as z_prev.")
        if type(dt) not in (int, float) or dt <= 0:
            raise ValueError("dt must be a positive scalar.")

        step_size = self.response_rate.to(z_prev) * float(dt)
        if float(dt) * self._response_rate_value * self._decay_rate_value > 1.0:
            raise ValueError(
                "dt * response_rate * decay_rate must be <= 1 for the "
                "configured explicit-Euler decay step."
            )

        active = pair_mask.to(dtype=z_prev.dtype)
        rho = urgency.clamp(0.0, 1.0) * active
        gated_evidence = evidence * active
        derivative = (
            -self.decay_rate.to(z_prev) * z_prev
            + rho
            * self.self_reinforcement.to(z_prev)
            * torch.tanh(self.nonlinear_sensitivity.to(z_prev) * z_prev)
            + gated_evidence
        )
        return z_prev + step_size * derivative

    def theoretical_bound(self, b_max: float) -> float:
        """Continuous-time ultimate bound used by diagnostics."""

        if b_max <= 0:
            raise ValueError("b_max must be positive.")
        numerator = self._self_reinforcement_value + float(b_max)
        return numerator / self._decay_rate_value
