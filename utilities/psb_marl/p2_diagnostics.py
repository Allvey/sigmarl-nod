"""Read-only P2.1-D diagnostics for bifurcation usage and policy bypass."""

from __future__ import annotations

from typing import Dict

import torch


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().mean().item())


def _fraction(mask: torch.Tensor, denominator: torch.Tensor) -> float:
    count = int(denominator.sum().item())
    if count == 0:
        return 0.0
    return float(mask.sum().item() / count)


def _branch_dwell_lengths(
    q: torch.Tensor,
    committed: torch.Tensor,
) -> list[int]:
    """Return same-sign committed dwell lengths for ``[env,time,edge]`` data."""

    if q.ndim != 3 or committed.shape != q.shape:
        raise ValueError("P2 dwell diagnostics require [env,time,edge] tensors.")
    lengths: list[int] = []
    for environment in range(q.shape[0]):
        for edge in range(q.shape[2]):
            current_sign = 0
            current_length = 0
            for time in range(q.shape[1]):
                if bool(committed[environment, time, edge]):
                    sign = 1 if float(q[environment, time, edge]) > 0.0 else -1
                    if sign == current_sign:
                        current_length += 1
                    else:
                        if current_length:
                            lengths.append(current_length)
                        current_sign = sign
                        current_length = 1
                else:
                    if current_length:
                        lengths.append(current_length)
                    current_sign = 0
                    current_length = 0
            if current_length:
                lengths.append(current_length)
    return lengths


@torch.no_grad()
def p2_state_diagnostics(
    rollout,
    *,
    rho_c: float,
    z_scale: float,
    commitment_threshold: float = 0.5,
) -> Dict[str, float | int]:
    """Summarize unique active edges without changing rollout or policy state.

    Fractions use unordered dense edges. ``critical_edge_fraction`` is over all
    possible edge-time samples, while ``critical_given_active_fraction`` is
    conditioned on a currently active conflict edge.
    """

    if rho_c <= 0.0 or z_scale <= 0.0:
        raise ValueError("rho_c and z_scale must be positive.")
    if not 0.0 < commitment_threshold < 1.0:
        raise ValueError("commitment_threshold must lie in (0,1).")

    b_dense = rollout.get(("agents", "psb", "b"))
    rho_dense = rollout.get(("agents", "psb", "rho"))
    z_prev_dense = rollout.get(("agents", "psb", "z_prev_dense"))
    z_next_dense = rollout.get(("agents", "psb", "z_next_dense"))
    if not (
        b_dense.shape
        == rho_dense.shape
        == z_prev_dense.shape
        == z_next_dense.shape
    ):
        raise ValueError("Dense P2 diagnostic tensors must have equal shapes.")
    if b_dense.ndim != 4 or b_dense.shape[-1] != b_dense.shape[-2]:
        raise ValueError(
            "P2 rollout diagnostics require [environment,time,agent,agent]."
        )

    n_agents = int(b_dense.shape[-1])
    upper = torch.triu(
        torch.ones(
            n_agents,
            n_agents,
            dtype=torch.bool,
            device=b_dense.device,
        ),
        diagonal=1,
    )
    b = b_dense[..., upper]
    rho = rho_dense[..., upper]
    z_prev = z_prev_dense[..., upper]
    z_next = z_next_dense[..., upper]
    q_prev = torch.tanh(z_prev / z_scale)
    q_next = torch.tanh(z_next / z_scale)

    universe = torch.ones_like(rho, dtype=torch.bool)
    active = rho > 0.0
    critical = active & (rho > rho_c)
    committed = active & (q_next.abs() >= commitment_threshold)
    switch_eligible = (
        active
        & (q_prev.abs() >= commitment_threshold)
        & (q_next.abs() >= commitment_threshold)
    )
    switches = switch_eligible & (q_prev * q_next < 0.0)
    dwell_lengths = _branch_dwell_lengths(q_next, committed)

    return {
        "rollout_active_edge_samples": int(active.sum().item()),
        "rollout_critical_edge_samples": int(critical.sum().item()),
        "rollout_committed_edge_samples": int(committed.sum().item()),
        "rollout_active_edge_fraction": _fraction(active, universe),
        "rollout_critical_edge_fraction": _fraction(critical, universe),
        "rollout_critical_given_active_fraction": _fraction(critical, active),
        "rollout_committed_given_active_fraction": _fraction(committed, active),
        "rollout_active_b_abs_mean": _masked_mean(b.abs(), active),
        "rollout_active_z_abs_mean": _masked_mean(z_next.abs(), active),
        "rollout_active_q_abs_mean": _masked_mean(q_next.abs(), active),
        "rollout_critical_b_abs_mean": _masked_mean(b.abs(), critical),
        "rollout_critical_z_abs_mean": _masked_mean(z_next.abs(), critical),
        "rollout_critical_q_abs_mean": _masked_mean(q_next.abs(), critical),
        "rollout_branch_switch_eligible_samples": int(
            switch_eligible.sum().item()
        ),
        "rollout_branch_switch_count": int(switches.sum().item()),
        "rollout_branch_switch_rate": _fraction(switches, switch_eligible),
        "rollout_branch_dwell_mean_steps": (
            float(sum(dwell_lengths) / len(dwell_lengths))
            if dwell_lengths
            else 0.0
        ),
        "rollout_branch_dwell_max_steps": max(dwell_lengths, default=0),
    }


@torch.no_grad()
def p2_zero_branch_counterfactual_diagnostics(
    rollout,
    *,
    bridge,
) -> Dict[str, float]:
    """Measure the current adapter's q=0 bypass on a collected rollout.

    This performs a deterministic, no-gradient diagnostic pass through the
    existing branch encoder and adapter. It neither recomputes the proximal
    state nor changes the collected actions or optimization objective.
    """

    observation = rollout.get(("agents", "observation"))
    pair_features = rollout.get(("agents", "info", "pair_features"))
    neighbor_ids = rollout.get(("agents", "info", "neighbor_ids")).to(
        torch.long
    )
    urgency = rollout.get(("agents", "info", "urgency"))
    confidence = rollout.get(("agents", "info", "confidence"))
    pair_mask = rollout.get(("agents", "info", "pair_mask")).to(torch.bool)
    actual_loc = rollout.get(("agents", "loc"))
    actual_scale = rollout.get(("agents", "scale"))
    base_loc = rollout.get(("agents", "psb", "base_loc"))
    base_scale = rollout.get(("agents", "psb", "base_scale"))

    n_agents = int(bridge.n_agents)
    if neighbor_ids.shape[-2] != n_agents:
        raise ValueError("P2 diagnostic neighbor axis does not match the bridge.")
    ego_shape = [1] * (neighbor_ids.ndim - 2) + [n_agents, 1]
    ego_ids = torch.arange(
        n_agents,
        dtype=neighbor_ids.dtype,
        device=neighbor_ids.device,
    ).view(*ego_shape)
    valid_ids = (neighbor_ids >= 0) & (neighbor_ids < n_agents)
    safe_ids = torch.where(valid_ids, neighbor_ids, ego_ids)
    valid = valid_ids & (safe_ids != ego_ids) & pair_mask
    rho_candidates = (
        bridge.proximal.rho_max
        * urgency.clamp(0.0, 1.0)
        * valid.to(dtype=urgency.dtype)
    )
    zero_branch = bridge.branch_encoder(
        pair_features=pair_features,
        z_candidates=torch.zeros_like(urgency),
        rho_candidates=rho_candidates,
        confidence=confidence,
        pair_mask=valid,
    )
    zero_loc, zero_scale, _, _ = bridge.adapter(
        observation=observation,
        context=zero_branch.context,
        base_loc=base_loc,
        base_scale=base_scale,
        branch_activity=zero_branch.activity,
    )
    tiny = torch.finfo(base_scale.dtype).tiny
    zero_log_scale = torch.log(zero_scale.clamp_min(tiny))
    base_log_scale = torch.log(base_scale.clamp_min(tiny))
    actual_log_scale = torch.log(actual_scale.clamp_min(tiny))
    bypass_loc = float((zero_loc - base_loc).abs().mean().item())
    bypass_log_scale = float(
        (zero_log_scale - base_log_scale).abs().mean().item()
    )
    branch_loc_effect = float((actual_loc - zero_loc).abs().mean().item())
    branch_log_scale_effect = float(
        (actual_log_scale - zero_log_scale).abs().mean().item()
    )
    loc_total = bypass_loc + branch_loc_effect
    log_scale_total = bypass_log_scale + branch_log_scale_effect

    return {
        "rollout_zero_branch_context_abs_mean": float(
            zero_branch.context.abs().mean().item()
        ),
        "rollout_zero_branch_bypass_loc_abs_mean": bypass_loc,
        "rollout_zero_branch_bypass_log_scale_abs_mean": bypass_log_scale,
        "rollout_branch_loc_effect_abs_mean": branch_loc_effect,
        "rollout_branch_log_scale_effect_abs_mean": branch_log_scale_effect,
        "rollout_branch_loc_dependency_ratio": (
            branch_loc_effect / loc_total if loc_total > 0.0 else 0.0
        ),
        "rollout_branch_log_scale_dependency_ratio": (
            branch_log_scale_effect / log_scale_total
            if log_scale_total > 0.0
            else 0.0
        ),
    }
