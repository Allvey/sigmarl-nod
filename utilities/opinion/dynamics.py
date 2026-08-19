"""Fixed nonlinear opinion dynamics and global-ID state indexing helpers."""

from __future__ import annotations

import math

import torch
from torch import nn


def _validate_candidate_layout(
    z_dense: torch.Tensor,
    candidate_ids: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if z_dense.ndim < 2 or z_dense.shape[-2] != z_dense.shape[-1]:
        raise ValueError("z_dense must end with a square [n_agents, n_agents] matrix")
    if candidate_ids.dtype is not torch.long:
        raise ValueError("candidate_ids must have dtype torch.long")
    if mask.dtype is not torch.bool or mask.shape != candidate_ids.shape:
        raise ValueError("mask must be bool and match candidate_ids")
    if candidate_ids.ndim < 2 or candidate_ids.shape[:-1] != z_dense.shape[:-1]:
        raise ValueError(
            "candidate_ids must have shape [..., n_agents, n_candidates] "
            "matching z_dense"
        )

    n_agents = z_dense.shape[-1]
    active_ids = candidate_ids[mask]
    if active_ids.numel() and (
        (active_ids < 0).any() or (active_ids >= n_agents).any()
    ):
        raise ValueError("candidate_ids contain an out-of-range active ID")

    ego_shape = (1,) * (candidate_ids.ndim - 2) + (n_agents, 1)
    ego_ids = torch.arange(n_agents, device=candidate_ids.device).view(ego_shape)
    if ((candidate_ids == ego_ids) & mask).any():
        raise ValueError("candidate_ids must not contain an active self edge")

    n_candidates = candidate_ids.shape[-1]
    if n_candidates > 1:
        equal_pairs = candidate_ids.unsqueeze(-1) == candidate_ids.unsqueeze(-2)
        valid_pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        upper = torch.triu(
            torch.ones(
                n_candidates,
                n_candidates,
                dtype=torch.bool,
                device=candidate_ids.device,
            ),
            diagonal=1,
        )
        if (equal_pairs & valid_pairs & upper).any():
            raise ValueError("candidate_ids contain duplicate active candidates")


def gather_candidate_opinions(
    z_dense: torch.Tensor,
    candidate_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gather z_ij by global neighbor IDs, returning zero for padding slots."""
    _validate_candidate_layout(z_dense, candidate_ids, mask)
    if not torch.isfinite(z_dense).all():
        raise ValueError("z_dense must contain finite values")
    safe_ids = torch.where(mask, candidate_ids, torch.zeros_like(candidate_ids))
    gathered = torch.gather(z_dense, dim=-1, index=safe_ids)
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def scatter_candidate_opinions(
    z_dense: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return a new dense state with active candidate updates scattered by ID."""
    _validate_candidate_layout(z_dense, candidate_ids, mask)
    if candidate_values.shape != candidate_ids.shape:
        raise ValueError("candidate_values must match candidate_ids")
    if not torch.isfinite(z_dense).all():
        raise ValueError("z_dense must contain finite values")
    active_values = candidate_values[mask]
    if active_values.numel() and not torch.isfinite(active_values).all():
        raise ValueError("candidate_values must be finite on active edges")

    safe_ids = torch.where(mask, candidate_ids, torch.zeros_like(candidate_ids))
    current = torch.gather(z_dense, dim=-1, index=safe_ids)
    delta = torch.where(mask, candidate_values - current, torch.zeros_like(current))
    result = z_dense.scatter_add(dim=-1, index=safe_ids, src=delta)
    off_diagonal = ~torch.eye(
        z_dense.shape[-1], dtype=torch.bool, device=z_dense.device
    )
    return torch.where(off_diagonal, result, torch.zeros_like(result))


class OpinionDynamics(nn.Module):
    """Stateless Euler integration of fixed, interpretable opinion dynamics."""

    def __init__(
        self,
        *,
        kappa: float,
        nu: float,
        alpha: float,
        eta: float,
        z_clip: float,
        n_substeps: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "kappa": kappa,
            "nu": nu,
            "alpha": alpha,
            "eta": eta,
            "z_clip": z_clip,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(n_substeps) is not int or n_substeps <= 0:
            raise ValueError("n_substeps must be a positive int")
        if nu * alpha <= kappa:
            raise ValueError("OpinionDynamics requires nu * alpha > kappa")

        self.kappa = float(kappa)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.z_clip = float(z_clip)
        self.n_substeps = n_substeps

    @property
    def rho_c(self) -> float:
        return self.kappa / (self.nu * self.alpha)

    def forward(
        self,
        z_prev: torch.Tensor,
        b: torch.Tensor,
        urgency: torch.Tensor,
        mask: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        if z_prev.shape != b.shape or z_prev.shape != urgency.shape:
            raise ValueError("z_prev, b, and urgency must have equal shapes")
        if mask.dtype is not torch.bool or mask.shape != z_prev.shape:
            raise ValueError("mask must be bool and match the opinion state shape")
        if not torch.isfinite(z_prev).all():
            raise ValueError("z_prev must contain finite values")
        for name, tensor in (("b", b), ("urgency", urgency)):
            active = tensor[mask]
            if active.numel() and not torch.isfinite(active).all():
                raise ValueError(f"{name} must be finite on active edges")
        active_urgency = urgency[mask]
        if active_urgency.numel() and (
            (active_urgency < 0).any() or (active_urgency > 1).any()
        ):
            raise ValueError("urgency must be in [0, 1] on active edges")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise ValueError("dt must be numeric")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not 0 < dt * self.eta * self.kappa < 2:
            raise ValueError("dynamics require 0 < dt * eta * kappa < 2")

        evidence = torch.where(mask, b, torch.zeros_like(b))
        rho = torch.where(mask, urgency, torch.zeros_like(urgency))
        dt_sub = float(dt) / self.n_substeps
        z = z_prev
        for _ in range(self.n_substeps):
            derivative = self.eta * (
                -self.kappa * z
                + rho * self.nu * torch.tanh(self.alpha * z)
                + evidence
            )
            z = torch.clamp(z + dt_sub * derivative, -self.z_clip, self.z_clip)
        return z
