"""Learned, bounded instantaneous evidence for directed vehicle pairs."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import nn

from utilities.opinion.config import EvidenceConfig


class EvidenceOutput(NamedTuple):
    """Evidence tensors with shape ``[..., K]``."""

    antisymmetric_logit: torch.Tensor
    raw_b: torch.Tensor
    b: torch.Tensor


def swap_pair_features(pair_features: torch.Tensor) -> torch.Tensor:
    """Express the fixed 10-D directed pair features from the other vehicle.

    The input layout is ``r_x, r_y, u_x, u_y, v_ego, v_neighbor,
    sin(d_yaw), cos(d_yaw), t_cpa, d_cpa``. Relative vectors are rotated into
    the neighbor frame, vehicle speeds are exchanged, and relative yaw changes
    sign. Applying this operation twice recovers the original features.
    """

    if pair_features.ndim < 1 or pair_features.shape[-1] != 10:
        raise ValueError(
            "pair_features must have final dimension 10, got "
            f"{tuple(pair_features.shape)}."
        )

    rx, ry, ux, uy = pair_features[..., 0:4].unbind(dim=-1)
    ego_speed = pair_features[..., 4]
    neighbor_speed = pair_features[..., 5]
    sin_yaw = pair_features[..., 6]
    cos_yaw = pair_features[..., 7]

    # R(-d_yaw) @ (-vector_in_ego_frame)
    swapped_rx = -cos_yaw * rx - sin_yaw * ry
    swapped_ry = sin_yaw * rx - cos_yaw * ry
    swapped_ux = -cos_yaw * ux - sin_yaw * uy
    swapped_uy = sin_yaw * ux - cos_yaw * uy

    return torch.stack(
        (
            swapped_rx,
            swapped_ry,
            swapped_ux,
            swapped_uy,
            neighbor_speed,
            ego_speed,
            -sin_yaw,
            cos_yaw,
            pair_features[..., 8],
            pair_features[..., 9],
        ),
        dim=-1,
    )


class OpinionEvidenceNet(nn.Module):
    """Shared relative scorer implementing bounded signed evidence.

    ``G(chi_ij) - G(chi_ji)`` gives the sign an explicit relative meaning.
    Urgency, confidence, and the physical edge mask gate the bounded raw
    evidence. The network never receives the historical opinion state ``z``.
    """

    def __init__(
        self,
        pair_feature_dim: int,
        hidden_sizes: Sequence[int],
        b_max: float,
        temperature: float,
    ) -> None:
        super().__init__()
        if pair_feature_dim != 10:
            raise ValueError("The first method version requires pair_feature_dim=10.")
        if not hidden_sizes or any(type(size) is not int or size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")
        if b_max <= 0 or temperature <= 0:
            raise ValueError("b_max and temperature must be positive.")

        layers = []
        input_size = pair_feature_dim
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.Tanh()))
            input_size = hidden_size
        final_layer = nn.Linear(input_size, 1)
        # Start close to neutral while retaining a non-zero gradient path.
        nn.init.xavier_uniform_(final_layer.weight, gain=1e-2)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.scorer = nn.Sequential(*layers)
        self.pair_feature_dim = pair_feature_dim
        self.b_max = float(b_max)
        self.temperature = float(temperature)

    @classmethod
    def from_config(
        cls, pair_feature_dim: int, config: EvidenceConfig
    ) -> "OpinionEvidenceNet":
        return cls(
            pair_feature_dim=pair_feature_dim,
            hidden_sizes=config.hidden_sizes,
            b_max=config.b_max,
            temperature=config.temperature,
        )

    def forward(
        self,
        pair_features: torch.Tensor,
        urgency: torch.Tensor,
        confidence: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> EvidenceOutput:
        if pair_features.ndim < 2 or pair_features.shape[-1] != self.pair_feature_dim:
            raise ValueError(
                "pair_features must have shape [..., K, 10], got "
                f"{tuple(pair_features.shape)}."
            )
        expected_shape = pair_features.shape[:-1]
        for name, tensor in (("urgency", urgency), ("confidence", confidence)):
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_shape)}, got "
                    f"{tuple(tensor.shape)}."
                )
        if pair_mask.shape != expected_shape or pair_mask.dtype != torch.bool:
            raise ValueError(
                "pair_mask must be a bool tensor with shape "
                f"{tuple(expected_shape)}."
            )

        forward_score = self.scorer(pair_features).squeeze(-1)
        reverse_score = self.scorer(swap_pair_features(pair_features)).squeeze(-1)
        antisymmetric_logit = forward_score - reverse_score
        raw_b = self.b_max * torch.tanh(
            antisymmetric_logit / self.temperature
        )
        gate = (
            urgency.clamp(0.0, 1.0)
            * confidence.clamp(0.0, 1.0)
            * pair_mask.to(dtype=pair_features.dtype)
        )
        b = raw_b * gate
        return EvidenceOutput(
            antisymmetric_logit=antisymmetric_logit,
            raw_b=raw_b,
            b=b,
        )
