"""Trainable instantaneous evidence with exact role antisymmetry."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
from torch import nn


class EvidenceOutput(NamedTuple):
    raw_b: torch.Tensor
    b: torch.Tensor


def swap_roles(
    ego_features: torch.Tensor,
    neighbor_features: torch.Tensor,
    symmetric_context: torch.Tensor,
    antisymmetric_context: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Swap focal/other roles and reverse antisymmetric pair quantities."""
    if ego_features.shape != neighbor_features.shape:
        raise ValueError("ego_features and neighbor_features must have equal shapes")
    prefix = ego_features.shape[:-1]
    if symmetric_context.shape[:-1] != prefix:
        raise ValueError("symmetric_context must share the pair batch shape")
    if antisymmetric_context.shape[:-1] != prefix:
        raise ValueError("antisymmetric_context must share the pair batch shape")
    return (
        neighbor_features,
        ego_features,
        symmetric_context,
        -antisymmetric_context,
    )


def _masked_finite(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> None:
    values = tensor[mask]
    if values.numel() and not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite on active edges")


def _masked_unit_interval(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> None:
    _masked_finite(name, tensor, mask)
    values = tensor[mask]
    if values.numel() and ((values < 0).any() or (values > 1).any()):
        raise ValueError(f"{name} must be in [0, 1] on active edges")


class OpinionEvidenceNet(nn.Module):
    """Shared relative scorer producing physically gated bounded evidence."""

    def __init__(
        self,
        *,
        individual_feature_dim: int,
        symmetric_context_dim: int,
        antisymmetric_context_dim: int,
        hidden_dim: int,
        num_layers: int,
        b_max: float,
        b_temperature: float,
    ) -> None:
        super().__init__()
        integer_values = {
            "individual_feature_dim": individual_feature_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        for name, value in integer_values.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        for name, value in {
            "symmetric_context_dim": symmetric_context_dim,
            "antisymmetric_context_dim": antisymmetric_context_dim,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        for name, value in {"b_max": b_max, "b_temperature": b_temperature}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not torch.isfinite(torch.tensor(float(value))) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.individual_feature_dim = individual_feature_dim
        self.symmetric_context_dim = symmetric_context_dim
        self.antisymmetric_context_dim = antisymmetric_context_dim
        self.b_max = float(b_max)
        self.b_temperature = float(b_temperature)

        input_dim = (
            2 * individual_feature_dim
            + symmetric_context_dim
            + antisymmetric_context_dim
        )
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.Tanh()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.shared_scorer = nn.Sequential(*layers)

    def _validate_inputs(
        self,
        ego_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        symmetric_context: torch.Tensor,
        antisymmetric_context: torch.Tensor,
        urgency: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if ego_features.shape != neighbor_features.shape:
            raise ValueError("ego_features and neighbor_features must have equal shapes")
        prefix = ego_features.shape[:-1]
        expected_dims = (
            ("ego_features", ego_features, self.individual_feature_dim),
            ("neighbor_features", neighbor_features, self.individual_feature_dim),
            ("symmetric_context", symmetric_context, self.symmetric_context_dim),
            (
                "antisymmetric_context",
                antisymmetric_context,
                self.antisymmetric_context_dim,
            ),
        )
        for name, tensor, expected_dim in expected_dims:
            if tensor.shape[:-1] != prefix or tensor.shape[-1] != expected_dim:
                raise ValueError(
                    f"{name} must have shape {prefix + (expected_dim,)}; "
                    f"got {tuple(tensor.shape)}"
                )
        if urgency.shape != prefix or confidence.shape != prefix:
            raise ValueError("urgency and confidence must match the pair batch shape")
        if mask.shape != prefix or mask.dtype is not torch.bool:
            raise ValueError("mask must be a bool tensor matching the pair batch shape")
        for name, tensor, _ in expected_dims:
            feature_mask = mask.unsqueeze(-1).expand_as(tensor)
            _masked_finite(name, tensor, feature_mask)
        _masked_unit_interval("urgency", urgency, mask)
        _masked_unit_interval("confidence", confidence, mask)

    @staticmethod
    def _sanitize_features(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask.unsqueeze(-1), tensor, torch.zeros_like(tensor))

    def _score(
        self,
        focal: torch.Tensor,
        other: torch.Tensor,
        symmetric: torch.Tensor,
        antisymmetric: torch.Tensor,
    ) -> torch.Tensor:
        scorer_input = torch.cat((focal, other, symmetric, antisymmetric), dim=-1)
        return self.shared_scorer(scorer_input).squeeze(-1)

    def forward(
        self,
        ego_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        symmetric_context: torch.Tensor,
        antisymmetric_context: torch.Tensor,
        urgency: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> EvidenceOutput:
        self._validate_inputs(
            ego_features,
            neighbor_features,
            symmetric_context,
            antisymmetric_context,
            urgency,
            confidence,
            mask,
        )
        ego = self._sanitize_features(ego_features, mask)
        neighbor = self._sanitize_features(neighbor_features, mask)
        symmetric = self._sanitize_features(symmetric_context, mask)
        antisymmetric = self._sanitize_features(antisymmetric_context, mask)
        urgency_safe = torch.where(mask, urgency, torch.zeros_like(urgency))
        confidence_safe = torch.where(mask, confidence, torch.zeros_like(confidence))

        score_ego = self._score(ego, neighbor, symmetric, antisymmetric)
        score_neighbor = self._score(neighbor, ego, symmetric, -antisymmetric)
        raw_b = self.b_max * torch.tanh(
            (score_ego - score_neighbor) / self.b_temperature
        )
        raw_b = torch.where(mask, raw_b, torch.zeros_like(raw_b))
        b = raw_b * urgency_safe * confidence_safe
        return EvidenceOutput(raw_b=raw_b, b=b)
