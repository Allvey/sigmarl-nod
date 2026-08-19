"""Bounded aggregation from pairwise opinion state to scalar speed residual."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn


class ResidualOutput(NamedTuple):
    q: torch.Tensor
    normalized_weights: torch.Tensor
    aggregate: torch.Tensor
    residual: torch.Tensor


class OpinionResidual(nn.Module):
    """Aggregate candidate opinions without growing with candidate count."""

    def __init__(self, *, z0: float, eps: float = 1e-8) -> None:
        super().__init__()
        for name, value in {"z0": z0, "eps": eps}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.z0 = float(z0)
        self.eps = float(eps)

    def forward(
        self,
        z: torch.Tensor,
        urgency: torch.Tensor,
        direction: torch.Tensor,
        mask: torch.Tensor,
        *,
        residual_scale: float,
    ) -> ResidualOutput:
        if z.shape != urgency.shape or z.shape != direction.shape:
            raise ValueError("z, urgency, and direction must have equal shapes")
        if z.ndim < 1:
            raise ValueError("opinion tensors must include a candidate dimension")
        if mask.dtype is not torch.bool or mask.shape != z.shape:
            raise ValueError("mask must be bool and match the opinion tensors")
        if isinstance(residual_scale, bool) or not isinstance(
            residual_scale, (int, float)
        ):
            raise ValueError("residual_scale must be numeric")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and non-negative")

        for name, tensor in (("z", z), ("urgency", urgency), ("direction", direction)):
            active = tensor[mask]
            if active.numel() and not torch.isfinite(active).all():
                raise ValueError(f"{name} must be finite on active edges")
        active_urgency = urgency[mask]
        if active_urgency.numel() and (
            (active_urgency < 0).any() or (active_urgency > 1).any()
        ):
            raise ValueError("urgency must be in [0, 1] on active edges")
        active_direction = direction[mask]
        if active_direction.numel() and (
            (active_direction < -1).any() or (active_direction > 1).any()
        ):
            raise ValueError("direction must be in [-1, 1] on active edges")

        z_safe = torch.where(mask, z, torch.zeros_like(z))
        urgency_safe = torch.where(mask, urgency, torch.zeros_like(urgency))
        direction_safe = torch.where(mask, direction, torch.zeros_like(direction))
        q = torch.where(mask, torch.tanh(z_safe / self.z0), torch.zeros_like(z))
        denominator = urgency_safe.sum(dim=-1, keepdim=True) + self.eps
        normalized_weights = urgency_safe / denominator
        aggregate = (normalized_weights * q * direction_safe).sum(dim=-1)
        residual = float(residual_scale) * torch.tanh(aggregate)
        return ResidualOutput(
            q=q,
            normalized_weights=normalized_weights,
            aggregate=aggregate,
            residual=residual,
        )
