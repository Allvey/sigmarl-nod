"""Deterministic mathematical certification for the P1 proximal layer."""

from __future__ import annotations

from typing import Dict

import torch

from utilities.psb_marl.proximal import ProximalSaturatingBifurcation


def _potential(
    z: torch.Tensor,
    rho: torch.Tensor,
    b: torch.Tensor,
    layer: ProximalSaturatingBifurcation,
) -> torch.Tensor:
    return (
        0.5 * layer.kappa * z.square()
        - rho
        * layer.nu
        / layer.alpha
        * torch.log(torch.cosh(layer.alpha * z))
        - b * z
    )


def certify_p1_layer(
    layer: ProximalSaturatingBifurcation,
) -> Dict[str, object]:
    """Check residual, uniqueness, symmetry, dissipation, and implicit gradient."""

    dtype = torch.float64
    z_axis = torch.linspace(-2.0, 2.0, 17, dtype=dtype)
    rho_axis = torch.linspace(0.0, layer.rho_max, 13, dtype=dtype)
    z_prev, rho = torch.meshgrid(z_axis, rho_axis, indexing="ij")
    b = torch.zeros_like(z_prev)
    result = layer.solve_with_diagnostics(z_prev, rho, b)
    odd = layer(-z_prev, rho, b)
    inactive = layer(z_prev, torch.zeros_like(rho), b)
    inactive_expected = z_prev / (1.0 + layer.h_z * layer.kappa)

    energy_before = _potential(z_prev, rho, b, layer)
    energy_after = _potential(result.z_next, rho, b, layer)
    proximal_dissipation = (
        result.z_next - z_prev
    ).square() / (2.0 * layer.h_z)
    energy_violation = energy_after + proximal_dissipation - energy_before

    gradient_z = torch.tensor([0.31], dtype=dtype, requires_grad=True)
    gradient_rho = torch.tensor(
        [0.75 * layer.rho_max], dtype=dtype, requires_grad=True
    )
    gradient_b = torch.zeros(1, dtype=dtype, requires_grad=True)
    gradient_next = layer(gradient_z, gradient_rho, gradient_b)
    gradient_next.sum().backward()
    diagnostic = layer.solve_with_diagnostics(
        gradient_z.detach(), gradient_rho.detach(), gradient_b.detach()
    )
    denominator = diagnostic.denominator.item()
    analytic_z = (1.0 / layer.h_z) / denominator
    analytic_rho = (
        layer.nu * torch.tanh(layer.alpha * diagnostic.z_next).item() / denominator
    )
    analytic_b = 1.0 / denominator
    gradient_error = max(
        abs(float(gradient_z.grad.item()) - analytic_z),
        abs(float(gradient_rho.grad.item()) - analytic_rho),
        abs(float(gradient_b.grad.item()) - analytic_b),
    )

    metrics = {
        "grid_points": int(z_prev.numel()),
        "rho_c": layer.kappa / (layer.nu * layer.alpha),
        "rho_max": layer.rho_max,
        "convexity_margin": layer.convexity_margin,
        "max_root_residual": float(result.residual.abs().max().item()),
        "min_root_denominator": float(result.denominator.min().item()),
        "max_odd_symmetry_error": float(
            (result.z_next + odd).abs().max().item()
        ),
        "max_inactive_contraction_error": float(
            (inactive - inactive_expected).abs().max().item()
        ),
        "max_energy_inequality_violation": float(
            energy_violation.clamp_min(0.0).max().item()
        ),
        "max_implicit_gradient_error": float(gradient_error),
    }
    # Two independently converged odd-symmetric roots can differ by at most
    # twice the residual-to-state error bound |F|/m_P.
    odd_symmetry_tolerance = max(
        1e-10,
        2.0 * layer.residual_tolerance / layer.convexity_margin,
    )
    thresholds = {
        "root_residual": layer.residual_tolerance,
        "odd_symmetry": odd_symmetry_tolerance,
        "inactive_contraction": 1e-10,
        "energy_inequality": 1e-10,
        "implicit_gradient": 1e-10,
    }
    checks = {
        "strong_convexity": metrics["convexity_margin"] > 0.0,
        "root_residual": metrics["max_root_residual"] <= thresholds["root_residual"],
        "odd_symmetry": metrics["max_odd_symmetry_error"] <= thresholds["odd_symmetry"],
        "inactive_contraction": (
            metrics["max_inactive_contraction_error"]
            <= thresholds["inactive_contraction"]
        ),
        "energy_inequality": (
            metrics["max_energy_inequality_violation"]
            <= thresholds["energy_inequality"]
        ),
        "implicit_gradient": (
            metrics["max_implicit_gradient_error"]
            <= thresholds["implicit_gradient"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P1 proximal certification failed: {failed}.")
    return {
        "schema_version": 1,
        "stage": "p1_zero_control_equivalence",
        "solver": "safeguarded_newton_bisection",
        "backward": "implicit_jacobian_at_converged_root",
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": True,
    }
