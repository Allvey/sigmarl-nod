"""Trainable P2 bifurcation control and branch-conditioned distribution path."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch
from torch import nn


def swap_pair_features(pair_features: torch.Tensor) -> torch.Tensor:
    """Apply the involutive 10-D directed-pair exchange transform."""

    if pair_features.shape[-1] != 10:
        raise ValueError("PSB pair features must have final dimension 10.")
    rx, ry, ux, uy = pair_features[..., :4].unbind(dim=-1)
    ego_speed = pair_features[..., 4]
    neighbor_speed = pair_features[..., 5]
    sin_yaw = pair_features[..., 6]
    cos_yaw = pair_features[..., 7]
    return torch.stack(
        (
            -cos_yaw * rx - sin_yaw * ry,
            sin_yaw * rx - cos_yaw * ry,
            -cos_yaw * ux - sin_yaw * uy,
            sin_yaw * ux - cos_yaw * uy,
            neighbor_speed,
            ego_speed,
            -sin_yaw,
            cos_yaw,
            pair_features[..., 8],
            pair_features[..., 9],
        ),
        dim=-1,
    )


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ControlOutput(NamedTuple):
    antisymmetric_logit: torch.Tensor
    raw_b: torch.Tensor
    b: torch.Tensor
    support_gate: torch.Tensor
    critical_gate: torch.Tensor


class AntisymmetricBifurcationControl(nn.Module):
    """Shared PairScorer implementing bounded exchange-antisymmetric control."""

    def __init__(
        self,
        *,
        pair_feature_dim: int,
        hidden_sizes: Sequence[int],
        b_max: float,
        temperature: float,
        support_power: float,
        critical_gate_enabled: bool,
        critical_width: float,
        critical_floor: float,
        final_layer_gain: float,
        rho_c: float,
        rho_max: float,
    ) -> None:
        super().__init__()
        if pair_feature_dim != 10:
            raise ValueError("P2 requires pair_feature_dim=10.")
        if b_max <= 0.0 or temperature <= 0.0 or support_power <= 0.0:
            raise ValueError("Control bounds, temperature, and support power must be positive.")
        if critical_width <= 0.0 or not 0.0 <= critical_floor <= 1.0:
            raise ValueError("Invalid critical gate configuration.")
        if rho_c <= 0.0 or rho_max <= rho_c:
            raise ValueError("Control requires 0 < rho_c < rho_max.")
        self.scorer = _mlp(pair_feature_dim + 1, hidden_sizes, 1)
        final_layer = self.scorer[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=final_layer_gain)
        nn.init.zeros_(final_layer.bias)
        self.pair_feature_dim = pair_feature_dim
        self.b_max = float(b_max)
        self.temperature = float(temperature)
        self.support_power = float(support_power)
        self.critical_gate_enabled = bool(critical_gate_enabled)
        self.critical_width = float(critical_width)
        self.critical_floor = float(critical_floor)
        self.rho_c = float(rho_c)
        self.rho_max = float(rho_max)

    def forward(
        self,
        pair_features: torch.Tensor,
        z_candidates: torch.Tensor,
        rho_candidates: torch.Tensor,
        confidence: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> ControlOutput:
        expected = pair_features.shape[:-1]
        if pair_features.shape[-1] != self.pair_feature_dim:
            raise ValueError("Invalid P2 pair feature dimension.")
        for name, tensor in (
            ("z_candidates", z_candidates),
            ("rho_candidates", rho_candidates),
            ("confidence", confidence),
            ("pair_mask", pair_mask),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} must have shape {tuple(expected)}.")
        if pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be boolean.")

        forward_input = torch.cat(
            (pair_features, z_candidates.unsqueeze(-1)), dim=-1
        )
        reverse_input = torch.cat(
            (swap_pair_features(pair_features), -z_candidates.unsqueeze(-1)),
            dim=-1,
        )
        forward_score = self.scorer(forward_input).squeeze(-1)
        reverse_score = self.scorer(reverse_input).squeeze(-1)
        antisymmetric_logit = forward_score - reverse_score
        raw_b = self.b_max * torch.tanh(
            antisymmetric_logit / self.temperature
        )
        normalized_rho = (rho_candidates / self.rho_max).clamp(0.0, 1.0)
        support_gate = normalized_rho.pow(self.support_power)
        if self.critical_gate_enabled:
            critical_gate = self.critical_floor + (
                1.0 - self.critical_floor
            ) * torch.exp(
                -0.5
                * ((rho_candidates - self.rho_c) / self.critical_width).square()
            )
        else:
            critical_gate = torch.ones_like(raw_b)
        gate = (
            support_gate
            * critical_gate
            * confidence.clamp(0.0, 1.0)
            * pair_mask.to(dtype=raw_b.dtype)
        )
        return ControlOutput(
            antisymmetric_logit=antisymmetric_logit,
            raw_b=raw_b,
            b=raw_b * gate,
            support_gate=support_gate,
            critical_gate=critical_gate,
        )


class BranchContextOutput(NamedTuple):
    context: torch.Tensor
    attention: torch.Tensor
    q: torch.Tensor
    activity: torch.Tensor


class BranchContextEncoder(nn.Module):
    """Masked local pair aggregation conditioned on the continuous branch."""

    def __init__(
        self,
        *,
        pair_feature_dim: int,
        hidden_sizes: Sequence[int],
        context_dim: int,
        z_scale: float,
        rho_max: float,
        conditioning_mode: str = "general",
    ) -> None:
        super().__init__()
        if pair_feature_dim != 10 or context_dim <= 0:
            raise ValueError("Invalid branch context dimensions.")
        if z_scale <= 0.0 or rho_max <= 0.0:
            raise ValueError("z_scale and rho_max must be positive.")
        if conditioning_mode not in {
            "general",
            "causal_q_gate",
            "sector_q_gate",
            "supported_sector_q_gate",
        }:
            raise ValueError("Unsupported branch conditioning mode.")
        self.encoder = _mlp(pair_feature_dim + 2, hidden_sizes, context_dim)
        self.attention = nn.Linear(context_dim, 1)
        self.z_scale = float(z_scale)
        self.rho_max = float(rho_max)
        self.conditioning_mode = conditioning_mode

    def forward(
        self,
        pair_features: torch.Tensor,
        z_candidates: torch.Tensor,
        rho_candidates: torch.Tensor,
        confidence: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> BranchContextOutput:
        q = torch.tanh(z_candidates / self.z_scale)
        normalized_rho = (rho_candidates / self.rho_max).clamp(0.0, 1.0)
        causal_conditioning = self.conditioning_mode in {
            "causal_q_gate",
            "sector_q_gate",
            "supported_sector_q_gate",
        }
        branch_coordinate = q.abs() if causal_conditioning else q
        encoded = self.encoder(
            torch.cat(
                (
                    pair_features,
                    branch_coordinate.unsqueeze(-1),
                    normalized_rho.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        valid = pair_mask.to(dtype=torch.bool)
        logits = self.attention(encoded).squeeze(-1) + torch.log(
            confidence.clamp_min(1e-6)
        )
        valid_any = valid.any(dim=-1, keepdim=True)
        safe_logits = torch.where(
            valid_any,
            logits.masked_fill(~valid, torch.finfo(logits.dtype).min),
            torch.zeros_like(logits),
        )
        attention = torch.softmax(safe_logits, dim=-1)
        attention = attention * valid.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        message = q.unsqueeze(-1) * encoded if causal_conditioning else encoded
        context = (attention.unsqueeze(-1) * message).sum(dim=-2)
        activity = (attention * q.abs()).sum(dim=-1, keepdim=True)
        if self.conditioning_mode == "supported_sector_q_gate":
            activity = (
                attention * normalized_rho * q.abs()
            ).sum(dim=-1, keepdim=True)
        return BranchContextOutput(
            context=context,
            attention=attention,
            q=q,
            activity=activity,
        )


class BranchDistributionAdapter(nn.Module):
    """Base-anchored adapter with an optional learned log-scale head.

    ``max_delta_log_scale == 0`` is a structural mean-only mode: no log-scale
    output parameters are constructed and the Base scale is returned directly.
    """

    def __init__(
        self,
        *,
        observation_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        max_delta_loc: float,
        max_delta_log_scale: float,
        conditioning_mode: str = "general",
        mean_action_mask: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if min(observation_dim, context_dim, action_dim) <= 0:
            raise ValueError("Adapter dimensions must be positive.")
        if max_delta_loc <= 0.0 or max_delta_log_scale < 0.0:
            raise ValueError(
                "max_delta_loc must be positive and max_delta_log_scale "
                "must be non-negative."
            )
        if conditioning_mode not in {
            "general",
            "causal_q_gate",
            "sector_q_gate",
            "supported_sector_q_gate",
        }:
            raise ValueError("Unsupported branch conditioning mode.")
        self.adapts_log_scale = float(max_delta_log_scale) > 0.0
        output_dim = action_dim * (2 if self.adapts_log_scale else 1)
        self.network = _mlp(
            observation_dim + context_dim + action_dim,
            hidden_sizes,
            output_dim,
        )
        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        self.action_dim = action_dim
        self.max_delta_loc = float(max_delta_loc)
        self.max_delta_log_scale = float(max_delta_log_scale)
        self.conditioning_mode = conditioning_mode
        if mean_action_mask is None:
            mask_values = [1.0] * action_dim
        else:
            mask_values = [float(value) for value in mean_action_mask]
            if len(mask_values) != action_dim or any(
                value not in {0.0, 1.0} for value in mask_values
            ):
                raise ValueError(
                    "mean_action_mask must contain one binary value per "
                    "action dimension."
                )
        # The inference-only projection must remain compatible with existing
        # checkpoints, so it is intentionally excluded from state_dict.
        self.register_buffer(
            "mean_action_mask",
            torch.tensor(mask_values, dtype=torch.float32),
            persistent=False,
        )
        self.causal_gate = (
            nn.Linear(context_dim, output_dim, bias=False)
            if conditioning_mode
            in {
                "causal_q_gate",
                "sector_q_gate",
                "supported_sector_q_gate",
            }
            else None
        )

    def forward(
        self,
        observation: torch.Tensor,
        context: torch.Tensor,
        base_loc: torch.Tensor,
        base_scale: torch.Tensor,
        branch_activity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conditioning_mode in {
            "sector_q_gate",
            "supported_sector_q_gate",
        }:
            expected_shape = base_loc.shape[:-1] + (1,)
            if branch_activity is None or branch_activity.shape != expected_shape:
                raise ValueError(
                    "A sector q gate requires branch_activity with shape "
                    "base_loc.shape[:-1] + (1,)."
                )
            if not bool(torch.isfinite(branch_activity).all()):
                raise ValueError("branch_activity must be finite.")
            if bool((branch_activity < 0.0).any()) or bool(
                (branch_activity > 1.0).any()
            ):
                raise ValueError("branch_activity must lie in [0, 1].")
        raw = self.network(torch.cat((observation, context, base_loc), dim=-1))
        if self.causal_gate is not None:
            raw = raw * torch.tanh(self.causal_gate(context))
        if self.adapts_log_scale:
            raw_loc, raw_log_scale = raw.split(self.action_dim, dim=-1)
        else:
            raw_loc = raw
        delta_loc = (
            self.max_delta_loc
            * torch.tanh(raw_loc)
            * self.mean_action_mask.to(dtype=raw_loc.dtype)
        )
        if self.conditioning_mode in {
            "sector_q_gate",
            "supported_sector_q_gate",
        }:
            delta_loc = delta_loc * branch_activity.to(
                dtype=delta_loc.dtype,
                device=delta_loc.device,
            )
        loc = base_loc + delta_loc
        if self.adapts_log_scale:
            delta_log_scale = self.max_delta_log_scale * torch.tanh(
                raw_log_scale
            )
            if self.conditioning_mode in {
                "sector_q_gate",
                "supported_sector_q_gate",
            }:
                delta_log_scale = delta_log_scale * branch_activity.to(
                    dtype=delta_log_scale.dtype,
                    device=delta_log_scale.device,
                )
            scale = base_scale * torch.exp(delta_log_scale)
        else:
            delta_log_scale = torch.zeros_like(base_scale)
            scale = base_scale
        return loc, scale, delta_loc, delta_log_scale
