"""Bounded mapping from pairwise opinions to the Actor speed-location residual."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from utilities.opinion.config import ResidualConfig


class ResidualOutput(NamedTuple):
    q: torch.Tensor
    normalized_weights: torch.Tensor
    aggregate: torch.Tensor
    residual: torch.Tensor


class OpinionResidual(nn.Module):
    """Aggregate opinions without letting neighbor count amplify the action."""

    def __init__(
        self,
        opinion_scale: float,
        gain: float,
        max_abs: float,
        action_index: int = 0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if opinion_scale <= 0 or gain <= 0 or max_abs <= 0 or epsilon <= 0:
            raise ValueError(
                "opinion_scale, gain, max_abs, and epsilon must be positive."
            )
        if gain > max_abs or max_abs > 1.0:
            raise ValueError("gain must be <= max_abs, and max_abs must be <= 1.")
        if type(action_index) is not int or action_index != 0:
            raise ValueError("The first method version only modifies action_index=0.")

        self.opinion_scale = float(opinion_scale)
        self.gain = float(gain)
        self.max_abs = float(max_abs)
        self.action_index = action_index
        self.epsilon = float(epsilon)

    @classmethod
    def from_config(cls, config: ResidualConfig) -> "OpinionResidual":
        return cls(
            opinion_scale=config.opinion_scale,
            gain=config.gain,
            max_abs=config.max_abs,
            action_index=config.action_index,
        )

    def forward(
        self,
        z: torch.Tensor,
        urgency: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> ResidualOutput:
        if z.ndim < 1 or z.shape != urgency.shape:
            raise ValueError("z and urgency must have identical [..., K] shapes.")
        if pair_mask.shape != z.shape or pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool and have the same shape as z.")

        q = torch.tanh(z / self.opinion_scale)
        weights = urgency.clamp(0.0, 1.0) * pair_mask.to(dtype=z.dtype)
        weight_sum = weights.sum(dim=-1, keepdim=True)
        normalized_weights = weights / (self.epsilon + weight_sum)
        aggregate = (normalized_weights * q).sum(dim=-1, keepdim=True)
        residual = (self.gain * aggregate).clamp(-self.max_abs, self.max_abs)
        return ResidualOutput(
            q=q,
            normalized_weights=normalized_weights,
            aggregate=aggregate,
            residual=residual,
        )

    def apply_to_loc(
        self, base_loc: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Return a new loc tensor with only the speed component corrected."""

        if base_loc.ndim < 1 or base_loc.shape[-1] <= self.action_index:
            raise ValueError("base_loc does not contain the configured action index.")
        expected_residual_shape = base_loc.shape[:-1] + (1,)
        if residual.shape != expected_residual_shape:
            raise ValueError(
                "residual must have shape "
                f"{tuple(expected_residual_shape)}, got {tuple(residual.shape)}."
            )
        before = base_loc[..., : self.action_index]
        corrected = (
            base_loc[..., self.action_index : self.action_index + 1] + residual
        )
        after = base_loc[..., self.action_index + 1 :]
        return torch.cat((before, corrected, after), dim=-1)
