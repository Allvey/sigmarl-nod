"""Differentiable proximal realization of the saturating bifurcation state."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


def _root_residual(
    z: torch.Tensor,
    z_prev: torch.Tensor,
    rho: torch.Tensor,
    b: torch.Tensor,
    *,
    h_z: float,
    kappa: float,
    nu: float,
    alpha: float,
) -> torch.Tensor:
    return (
        (z - z_prev) / h_z
        + kappa * z
        - rho * nu * torch.tanh(alpha * z)
        - b
    )


def _solve_bracketed_root(
    z_prev: torch.Tensor,
    rho: torch.Tensor,
    b: torch.Tensor,
    *,
    h_z: float,
    kappa: float,
    nu: float,
    alpha: float,
    residual_tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    """Vectorized safeguarded Newton solve on the theorem's explicit bracket."""

    inverse_h = 1.0 / h_z
    linear_coefficient = inverse_h + kappa
    lower = (inverse_h * z_prev + b - rho * nu) / linear_coefficient
    upper = (inverse_h * z_prev + b + rho * nu) / linear_coefficient
    z = z_prev.clamp(min=lower, max=upper)
    converged = torch.zeros_like(z, dtype=torch.bool)

    for _ in range(max_iterations):
        residual = _root_residual(
            z,
            z_prev,
            rho,
            b,
            h_z=h_z,
            kappa=kappa,
            nu=nu,
            alpha=alpha,
        )
        converged = converged | (residual.abs() <= residual_tolerance)
        if bool(converged.all()):
            break

        tanh_value = torch.tanh(alpha * z)
        derivative = (
            inverse_h
            + kappa
            - rho * nu * alpha * (1.0 - tanh_value.square())
        )
        newton = z - residual / derivative
        midpoint = 0.5 * (lower + upper)
        use_newton = (
            torch.isfinite(newton)
            & (newton > lower)
            & (newton < upper)
        )
        candidate = torch.where(use_newton, newton, midpoint)
        candidate_residual = _root_residual(
            candidate,
            z_prev,
            rho,
            b,
            h_z=h_z,
            kappa=kappa,
            nu=nu,
            alpha=alpha,
        )
        lower = torch.where(candidate_residual <= 0.0, candidate, lower)
        upper = torch.where(candidate_residual > 0.0, candidate, upper)
        z = torch.where(converged, z, candidate)

    midpoint = 0.5 * (lower + upper)
    residual = _root_residual(
        z,
        z_prev,
        rho,
        b,
        h_z=h_z,
        kappa=kappa,
        nu=nu,
        alpha=alpha,
    )
    midpoint_residual = _root_residual(
        midpoint,
        z_prev,
        rho,
        b,
        h_z=h_z,
        kappa=kappa,
        nu=nu,
        alpha=alpha,
    )
    z = torch.where(midpoint_residual.abs() < residual.abs(), midpoint, z)
    final_residual = _root_residual(
        z,
        z_prev,
        rho,
        b,
        h_z=h_z,
        kappa=kappa,
        nu=nu,
        alpha=alpha,
    )
    if not bool(torch.isfinite(final_residual).all()):
        raise RuntimeError("The proximal root solver produced a non-finite residual.")
    if float(final_residual.abs().max().item()) > residual_tolerance:
        raise RuntimeError(
            "The proximal root solver did not reach its configured residual "
            f"tolerance {residual_tolerance:.3e}; maximum residual is "
            f"{float(final_residual.abs().max().item()):.3e}."
        )
    return z


class _ImplicitProximalRoot(torch.autograd.Function):
    """Exact implicit backward evaluated at the converged numerical root."""

    @staticmethod
    def forward(
        ctx,
        z_prev: torch.Tensor,
        rho: torch.Tensor,
        b: torch.Tensor,
        h_z: float,
        kappa: float,
        nu: float,
        alpha: float,
        residual_tolerance: float,
        max_iterations: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_next = _solve_bracketed_root(
                z_prev,
                rho,
                b,
                h_z=h_z,
                kappa=kappa,
                nu=nu,
                alpha=alpha,
                residual_tolerance=residual_tolerance,
                max_iterations=max_iterations,
            )
        ctx.save_for_backward(z_next, rho)
        ctx.h_z = h_z
        ctx.kappa = kappa
        ctx.nu = nu
        ctx.alpha = alpha
        return z_next

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z_next, rho = ctx.saved_tensors
        inverse_h = 1.0 / ctx.h_z
        tanh_value = torch.tanh(ctx.alpha * z_next)
        denominator = (
            inverse_h
            + ctx.kappa
            - rho
            * ctx.nu
            * ctx.alpha
            * (1.0 - tanh_value.square())
        )
        grad_z_prev = grad_output * inverse_h / denominator
        grad_rho = grad_output * ctx.nu * tanh_value / denominator
        grad_b = grad_output / denominator
        return grad_z_prev, grad_rho, grad_b, None, None, None, None, None, None


class ProximalResult(NamedTuple):
    z_next: torch.Tensor
    residual: torch.Tensor
    denominator: torch.Tensor


class ProximalSaturatingBifurcation(nn.Module):
    """Single-valued differentiable map ``P_h(z_prev; rho, b)``."""

    def __init__(
        self,
        *,
        kappa: float,
        nu: float,
        alpha: float,
        rho_max: float,
        h_z: float,
        b_max: float,
        residual_tolerance: float,
        max_iterations: int,
    ) -> None:
        super().__init__()
        values = (kappa, nu, alpha, rho_max, h_z, residual_tolerance)
        if any(
            not torch.isfinite(torch.tensor(value)) or value <= 0
            for value in values
        ):
            raise ValueError("Positive proximal parameters must be finite.")
        if not torch.isfinite(torch.tensor(b_max)) or b_max < 0:
            raise ValueError("b_max must be finite and non-negative.")
        if type(max_iterations) is not int or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        margin = 1.0 / h_z + kappa - rho_max * nu * alpha
        if margin <= 0.0:
            raise ValueError(
                "Proximal uniqueness requires 1/h_z + kappa > rho_max*nu*alpha."
            )
        self.kappa = float(kappa)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.rho_max = float(rho_max)
        self.h_z = float(h_z)
        self.b_max = float(b_max)
        self.residual_tolerance = float(residual_tolerance)
        self.max_iterations = max_iterations
        self.convexity_margin = float(margin)

    @classmethod
    def from_runtime_config(cls, raw) -> "ProximalSaturatingBifurcation":
        required = {
            "kappa",
            "nu",
            "alpha",
            "rho_max",
            "h_z",
            "b_max",
            "residual_tolerance",
            "max_iterations",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Missing proximal runtime keys: {missing}.")
        return cls(**{name: raw[name] for name in required})

    def _validate_inputs(
        self,
        z_prev: torch.Tensor,
        rho: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        if not (z_prev.shape == rho.shape == b.shape):
            raise ValueError("z_prev, rho, and b must have identical shapes.")
        if not (
            z_prev.is_floating_point()
            and rho.is_floating_point()
            and b.is_floating_point()
        ):
            raise ValueError("z_prev, rho, and b must be floating tensors.")
        if not (
            z_prev.dtype == rho.dtype == b.dtype
            and z_prev.device == rho.device == b.device
        ):
            raise ValueError("z_prev, rho, and b must share dtype and device.")
        if not bool(torch.isfinite(z_prev).all()):
            raise ValueError("z_prev must be finite.")
        if not bool(torch.isfinite(rho).all()) or bool((rho < 0.0).any()):
            raise ValueError("rho must be finite and non-negative.")
        if bool((rho > self.rho_max + 1e-7).any()):
            raise ValueError("rho exceeds rho_max.")
        if not bool(torch.isfinite(b).all()) or bool(
            (b.abs() > self.b_max + 1e-7).any()
        ):
            raise ValueError("b must be finite and satisfy |b| <= b_max.")

    def forward(
        self,
        z_prev: torch.Tensor,
        rho: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(z_prev, rho, b)
        return _ImplicitProximalRoot.apply(
            z_prev,
            rho,
            b,
            self.h_z,
            self.kappa,
            self.nu,
            self.alpha,
            self.residual_tolerance,
            self.max_iterations,
        )

    def solve_with_diagnostics(
        self,
        z_prev: torch.Tensor,
        rho: torch.Tensor,
        b: torch.Tensor,
    ) -> ProximalResult:
        z_next = self(z_prev, rho, b)
        residual = _root_residual(
            z_next,
            z_prev,
            rho,
            b,
            h_z=self.h_z,
            kappa=self.kappa,
            nu=self.nu,
            alpha=self.alpha,
        )
        tanh_value = torch.tanh(self.alpha * z_next)
        denominator = (
            1.0 / self.h_z
            + self.kappa
            - rho * self.nu * self.alpha * (1.0 - tanh_value.square())
        )
        return ProximalResult(z_next, residual, denominator)
