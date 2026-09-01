"""Frozen-Base P2 policy bridge and stateful rollout controller."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from utilities.psb_marl.p2_network import (
    AntisymmetricBifurcationControl,
    BranchContextEncoder,
    BranchDistributionAdapter,
)
from utilities.psb_marl.p2_state import P2EdgeStateTracker
from utilities.psb_marl.proximal import ProximalSaturatingBifurcation


class P2BridgeOutput(NamedTuple):
    loc: torch.Tensor
    scale: torch.Tensor
    base_loc: torch.Tensor
    base_scale: torch.Tensor
    raw_b: torch.Tensor
    b_candidates: torch.Tensor
    b_dense: torch.Tensor
    rho_dense: torch.Tensor
    z_next_dense: torch.Tensor
    z_next_candidates: torch.Tensor
    q: torch.Tensor
    attention: torch.Tensor
    branch_context: torch.Tensor
    branch_activity: torch.Tensor
    delta_loc: torch.Tensor
    delta_log_scale: torch.Tensor
    root_residual: torch.Tensor
    root_denominator: torch.Tensor


def validate_p2_runtime_contract(runtime_config, environment_n_agents: int) -> None:
    if runtime_config.get("stage") != "p2_frozen_base_bifurcation":
        raise ValueError("Unsupported P2 runtime stage.")
    if runtime_config.get("control_mode") != "learned_antisymmetric":
        raise ValueError("P2 requires learned antisymmetric control.")
    if runtime_config.get("freeze_base_actor") is not True:
        raise ValueError("P2 requires a frozen Base Actor.")
    training_seed = runtime_config.get("training_seed")
    if training_seed is not None and (
        type(training_seed) is not int or training_seed < 0
    ):
        raise ValueError("P2 training_seed must be a non-negative integer.")
    source_n_agents = runtime_config.get("n_agents")
    if type(source_n_agents) is not int or source_n_agents < 2:
        raise ValueError("P2 source n_agents metadata must be an integer >= 2.")
    if type(environment_n_agents) is not int or environment_n_agents < 2:
        raise ValueError("P2 environment n_agents must be an integer >= 2.")
    proximal = runtime_config.get("proximal")
    if not isinstance(proximal, dict) or float(proximal.get("b_max", 0.0)) <= 0.0:
        raise ValueError("P2 requires a positive proximal b_max.")
    branch_adapter = runtime_config.get("branch_adapter")
    if not isinstance(branch_adapter, dict) or branch_adapter.get(
        "conditioning_mode", "general"
    ) not in {
        "general",
        "causal_q_gate",
        "sector_q_gate",
        "supported_sector_q_gate",
    }:
        raise ValueError("P2 has an invalid branch conditioning mode.")
    try:
        max_delta_log_scale = float(branch_adapter["max_delta_log_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "P2 requires a numeric branch max_delta_log_scale."
        ) from error
    if max_delta_log_scale < 0.0:
        raise ValueError("P2 branch max_delta_log_scale must be non-negative.")
    action_projection = branch_adapter.get("action_projection", "full")
    if action_projection not in {"full", "longitudinal_only"}:
        raise ValueError("P2 has an invalid action projection.")
    if action_projection == "longitudinal_only" and (
        branch_adapter.get("conditioning_mode")
        not in {
            "causal_q_gate",
            "sector_q_gate",
            "supported_sector_q_gate",
        }
        or max_delta_log_scale != 0.0
    ):
        raise ValueError(
            "P2 longitudinal projection requires causal or sector "
            "mean-only control."
        )


class FrozenBaseBifurcationPolicyBridge(nn.Module):
    """Pure P2 mapping from explicit dense state to an action distribution."""

    def __init__(
        self,
        *,
        base_policy_net: nn.Module,
        control_net: AntisymmetricBifurcationControl,
        proximal: ProximalSaturatingBifurcation,
        branch_encoder: BranchContextEncoder,
        adapter: BranchDistributionAdapter,
        n_agents: int,
    ) -> None:
        super().__init__()
        self.base_policy_net = base_policy_net
        self.control_net = control_net
        self.proximal = proximal
        self.branch_encoder = branch_encoder
        self.adapter = adapter
        self.n_agents = int(n_agents)
        for parameter in self.base_policy_net.parameters():
            parameter.requires_grad_(False)

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "control": list(self.control_net.parameters()),
            "adapter": list(self.branch_encoder.parameters())
            + list(self.adapter.parameters()),
        }

    def _candidate_mapping(
        self,
        z_dense: torch.Tensor,
        neighbor_ids: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if neighbor_ids.ndim != 3 or neighbor_ids.shape[1] != self.n_agents:
            raise ValueError("neighbor_ids must have shape [B, N, K].")
        if pair_mask.shape != neighbor_ids.shape or pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool with shape [B, N, K].")
        ego_ids = torch.arange(
            self.n_agents,
            device=neighbor_ids.device,
            dtype=neighbor_ids.dtype,
        ).view(1, self.n_agents, 1)
        valid_ids = (neighbor_ids >= 0) & (neighbor_ids < self.n_agents)
        safe_ids = torch.where(valid_ids, neighbor_ids, ego_ids)
        valid = valid_ids & (safe_ids != ego_ids) & pair_mask
        z_candidates = torch.gather(z_dense, 2, safe_ids)
        z_candidates = torch.where(
            valid, z_candidates, torch.zeros_like(z_candidates)
        )
        return z_candidates, safe_ids, valid

    def forward(
        self,
        observation: torch.Tensor,
        pair_features: torch.Tensor,
        neighbor_ids: torch.Tensor,
        urgency: torch.Tensor,
        confidence: torch.Tensor,
        pair_mask: torch.Tensor,
        z_prev_dense: torch.Tensor,
    ) -> P2BridgeOutput:
        if z_prev_dense.shape[-2:] != (self.n_agents, self.n_agents):
            raise ValueError("z_prev_dense must have shape [B, N, N].")
        base_loc, base_scale = self.base_policy_net(observation)
        z_candidates, safe_ids, valid = self._candidate_mapping(
            z_prev_dense, neighbor_ids.to(torch.long), pair_mask.to(torch.bool)
        )
        rho_candidates = (
            self.proximal.rho_max
            * urgency.clamp(0.0, 1.0)
            * valid.to(dtype=urgency.dtype)
        )
        control = self.control_net(
            pair_features=pair_features,
            z_candidates=z_candidates,
            rho_candidates=rho_candidates,
            confidence=confidence,
            pair_mask=valid,
        )
        directed_b = torch.zeros_like(z_prev_dense).scatter(
            2,
            safe_ids,
            torch.where(valid, control.b, torch.zeros_like(control.b)),
        )
        directed_rho = torch.zeros_like(z_prev_dense).scatter(
            2,
            safe_ids,
            rho_candidates,
        )
        rho_dense = torch.maximum(directed_rho, directed_rho.transpose(-1, -2))
        b_dense = 0.5 * (directed_b - directed_b.transpose(-1, -2))
        diagonal_mask = torch.eye(
            self.n_agents,
            dtype=torch.bool,
            device=z_prev_dense.device,
        ).unsqueeze(0)
        rho_dense = rho_dense.masked_fill(diagonal_mask, 0.0)
        b_dense = b_dense.masked_fill(diagonal_mask, 0.0)

        proximal_result = self.proximal.solve_with_diagnostics(
            z_prev_dense, rho_dense, b_dense
        )
        z_next_dense = 0.5 * (
            proximal_result.z_next - proximal_result.z_next.transpose(-1, -2)
        )
        z_next_dense = z_next_dense.masked_fill(diagonal_mask, 0.0)
        z_next_candidates = torch.gather(z_next_dense, 2, safe_ids)
        z_next_candidates = torch.where(
            valid, z_next_candidates, torch.zeros_like(z_next_candidates)
        )
        branch = self.branch_encoder(
            pair_features=pair_features,
            z_candidates=z_next_candidates,
            rho_candidates=rho_candidates,
            confidence=confidence,
            pair_mask=valid,
        )
        loc, scale, delta_loc, delta_log_scale = self.adapter(
            observation=observation,
            context=branch.context,
            base_loc=base_loc,
            base_scale=base_scale,
            branch_activity=branch.activity,
        )
        tanh_value = torch.tanh(self.proximal.alpha * z_next_dense)
        root_residual = (
            (z_next_dense - z_prev_dense) / self.proximal.h_z
            + self.proximal.kappa * z_next_dense
            - rho_dense * self.proximal.nu * tanh_value
            - b_dense
        )
        denominator = (
            1.0 / self.proximal.h_z
            + self.proximal.kappa
            - rho_dense
            * self.proximal.nu
            * self.proximal.alpha
            * (1.0 - tanh_value.square())
        )
        return P2BridgeOutput(
            loc=loc,
            scale=scale,
            base_loc=base_loc,
            base_scale=base_scale,
            raw_b=control.raw_b,
            b_candidates=control.b,
            b_dense=b_dense,
            rho_dense=rho_dense,
            z_next_dense=z_next_dense,
            z_next_candidates=z_next_candidates,
            q=branch.q,
            attention=branch.attention,
            branch_context=branch.context,
            branch_activity=branch.activity,
            delta_loc=delta_loc,
            delta_log_scale=delta_log_scale,
            root_residual=root_residual,
            root_denominator=denominator,
        )


class P2PolicyController(nn.Module):
    """Rollout-only owner of the recurrent P2 edge state."""

    Z_PREV_KEY = ("agents", "psb", "z_prev_dense")
    Z_NEXT_KEY = ("agents", "psb", "z_next_dense")

    def __init__(self, policy: nn.Module, tracker: P2EdgeStateTracker) -> None:
        super().__init__()
        self.policy = policy
        self.tracker = tracker
        self.in_keys = list(getattr(policy, "in_keys", ()))
        for key in (
            ("agents", "info", "urgency"),
            ("agents", "info", "agent_reset_mask"),
        ):
            if key not in self.in_keys:
                self.in_keys.append(key)
        self.out_keys = list(getattr(policy, "out_keys", ()))

    @property
    def spec(self):
        return getattr(self.policy, "spec", None)

    @staticmethod
    def _optional(tensordict, key):
        try:
            return tensordict.get(key)
        except KeyError:
            return None

    def forward(self, tensordict):
        z_prev = self.tracker.prepare_step(
            reference=tensordict.get(("agents", "info", "urgency")),
            agent_reset_mask=tensordict.get(
                ("agents", "info", "agent_reset_mask")
            ),
            environment_done=self._optional(tensordict, "done"),
        )
        tensordict.set(self.Z_PREV_KEY, z_prev)
        tensordict = self.policy(tensordict)
        self.tracker.commit_step(tensordict.get(self.Z_NEXT_KEY))
        return tensordict

    def reset_state(self) -> None:
        self.tracker.reset_all()
