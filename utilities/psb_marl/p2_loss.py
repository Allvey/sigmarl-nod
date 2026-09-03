"""Truncated sequence PPO objective for frozen-Base P2 training."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from utilities.psb_marl.p2_buffer import P2SequenceMiniBatch
from utilities.psb_marl.p2_state import P2EdgeStateTracker


def ppo_approx_kl_from_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """Return PPO's non-negative ``k3`` KL approximation.

    A signed mean of ``-log_ratio`` can be negative even for a large policy
    change, which makes it unsafe as an early-stop guard.
    """

    if not bool(torch.isfinite(log_ratio).all()):
        raise ValueError("PPO log ratios must be finite.")
    bounded_log_ratio = log_ratio.clamp(max=20.0)
    return (bounded_log_ratio.expm1() - bounded_log_ratio).mean()


class P2SequencePPOLoss(nn.Module):
    def __init__(
        self,
        *,
        actor,
        bridge: nn.Module,
        observation_key,
        action_key,
        advantage_key,
        n_agents: int,
        clip_epsilon: float,
        entropy_coefficient: float,
        energy_coefficient: float,
        control_trust_region_coefficient: float,
        saturation_coefficient: float,
        saturation_fraction: float,
        base_anchor_net: Optional[nn.Module] = None,
        base_anchor_coefficient: float = 0.0,
    ) -> None:
        super().__init__()
        if n_agents < 2 or not 0.0 < clip_epsilon < 1.0:
            raise ValueError("Invalid P2 PPO dimensions or clipping.")
        if min(
            entropy_coefficient,
            energy_coefficient,
            control_trust_region_coefficient,
            saturation_coefficient,
            base_anchor_coefficient,
        ) < 0.0:
            raise ValueError("P2 loss coefficients must be non-negative.")
        if not 0.0 < saturation_fraction <= 1.0:
            raise ValueError("saturation_fraction must lie in (0,1].")
        self.actor = actor
        self.bridge = bridge
        self.observation_key = observation_key
        self.action_key = action_key
        self.advantage_key = advantage_key
        self.n_agents = n_agents
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coefficient = float(entropy_coefficient)
        self.energy_coefficient = float(energy_coefficient)
        self.control_trust_region_coefficient = float(
            control_trust_region_coefficient
        )
        self.saturation_coefficient = float(saturation_coefficient)
        self.saturation_fraction = float(saturation_fraction)
        self.base_anchor_net = base_anchor_net
        self.base_anchor_coefficient = float(base_anchor_coefficient)

    @staticmethod
    def _optional(tensordict, key) -> Optional[torch.Tensor]:
        try:
            return tensordict.get(key)
        except KeyError:
            return None

    @staticmethod
    def _align_log_prob(
        value: torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if value.shape == reference.shape:
            return value
        if value.shape == reference.shape + (1,):
            return value.squeeze(-1)
        raise ValueError(
            f"{name} shape {tuple(value.shape)} does not match "
            f"{tuple(reference.shape)}."
        )

    def unroll(self, mini_batch: P2SequenceMiniBatch) -> Dict[str, torch.Tensor]:
        td = mini_batch.tensordict
        if len(td.batch_size) != 2:
            raise ValueError("P2 sequence mini-batch must have [chunk,time] shape.")
        observations = td.get(self.observation_key)
        pair_features = td.get(("agents", "info", "pair_features"))
        neighbor_ids = td.get(("agents", "info", "neighbor_ids")).to(torch.long)
        urgency = td.get(("agents", "info", "urgency"))
        confidence = td.get(("agents", "info", "confidence"))
        pair_mask = td.get(("agents", "info", "pair_mask")).to(torch.bool)
        reset_mask = td.get(("agents", "info", "agent_reset_mask"))
        environment_done = self._optional(td, "done")
        z_dense = mini_batch.z_init.detach()

        outputs = []
        for time_index in range(int(td.batch_size[1])):
            if time_index > 0:
                done_t = (
                    None
                    if environment_done is None
                    else environment_done[:, time_index]
                )
                z_dense = P2EdgeStateTracker.apply_resets(
                    z_dense,
                    reset_mask[:, time_index],
                    done_t,
                )
            output = self.bridge(
                observation=observations[:, time_index],
                pair_features=pair_features[:, time_index],
                neighbor_ids=neighbor_ids[:, time_index],
                urgency=urgency[:, time_index],
                confidence=confidence[:, time_index],
                pair_mask=pair_mask[:, time_index],
                z_prev_dense=z_dense,
            )
            outputs.append(output)
            z_dense = output.z_next_dense

        fields = {}
        for name in outputs[0]._fields:
            fields[name] = torch.stack(
                [getattr(output, name) for output in outputs], dim=1
            )
        return fields

    def _loss_from_recomputed(
        self,
        td,
        recomputed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        distribution_td = td.clone(False)
        distribution_td.set(("agents", "loc"), recomputed["loc"])
        distribution_td.set(("agents", "scale"), recomputed["scale"])
        distribution = self.actor.build_dist_from_params(distribution_td)
        action = td.get(self.action_key)
        if action.requires_grad:
            raise RuntimeError("Stored P2 action must be detached.")
        current_log_prob = distribution.log_prob(action)
        old_log_prob = self._align_log_prob(
            td.get(("agents", "sample_log_prob")),
            current_log_prob,
            "old log-prob",
        )
        if old_log_prob.requires_grad:
            raise RuntimeError("Stored P2 log-prob must be detached.")
        advantage = self._align_log_prob(
            td.get(self.advantage_key), current_log_prob, "advantage"
        ).unsqueeze(-1)

        log_ratio = (current_log_prob - old_log_prob).unsqueeze(-1)
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(
            1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        loss_objective = -torch.minimum(
            ratio * advantage, clipped_ratio * advantage
        ).mean()
        try:
            entropy = distribution.entropy()
        except NotImplementedError:
            entropy_sample = distribution.rsample()
            entropy = -distribution.log_prob(entropy_sample)
        loss_entropy = -self.entropy_coefficient * entropy.mean()

        upper_mask = torch.triu(
            torch.ones(
                self.n_agents,
                self.n_agents,
                dtype=torch.bool,
                device=recomputed["b_dense"].device,
            ),
            diagonal=1,
        )
        edge_normalizer = float(self.n_agents * (self.n_agents - 1) // 2)
        control_energy = (
            recomputed["b_dense"].square()[..., upper_mask].sum(dim=-1)
            / edge_normalizer
        ).mean()
        collected_b = td.get(("agents", "psb", "b")).detach()
        control_trust = (
            recomputed["b_dense"] - collected_b
        ).square()[..., upper_mask].mean()
        b_max = float(self.bridge.proximal.b_max)
        saturation = torch.relu(
            recomputed["b_dense"].abs() / b_max - self.saturation_fraction
        ).square()[..., upper_mask].mean()
        loss_regularization = (
            self.energy_coefficient * control_energy
            + self.control_trust_region_coefficient * control_trust
            + self.saturation_coefficient * saturation
        )

        loss_base_anchor = torch.zeros(
            (), dtype=recomputed["base_loc"].dtype,
            device=recomputed["base_loc"].device,
        )
        base_output_anchor = loss_base_anchor.detach()
        if self.base_anchor_net is not None and self.base_anchor_coefficient > 0.0:
            with torch.no_grad():
                source_loc, source_scale = self.base_anchor_net(
                    td.get(self.observation_key)
                )
            candidate_scale = recomputed["base_scale"].clamp_min(1e-8)
            source_scale = source_scale.clamp_min(1e-8)
            base_output_anchor = (
                (source_scale / candidate_scale).log()
                + (
                    candidate_scale.square()
                    + (recomputed["base_loc"] - source_loc).square()
                )
                / (2.0 * source_scale.square())
                - 0.5
            ).mean()
            loss_base_anchor = (
                self.base_anchor_coefficient * base_output_anchor
            )

        with torch.no_grad():
            collected_z = td.get(("agents", "psb", "z_next_dense"))
            state_error = (
                recomputed["z_next_dense"] - collected_z
            ).abs().mean()
            approx_kl = ppo_approx_kl_from_log_ratio(log_ratio)
            clip_fraction = (
                (ratio < 1.0 - self.clip_epsilon)
                | (ratio > 1.0 + self.clip_epsilon)
            ).to(torch.float32).mean()

        return {
            "loss_objective": loss_objective,
            "loss_entropy": loss_entropy,
            "loss_regularization": loss_regularization,
            "loss_base_anchor": loss_base_anchor,
            "base_output_anchor": base_output_anchor.detach(),
            "control_energy": control_energy.detach(),
            "control_trust": control_trust.detach(),
            "saturation_penalty": saturation.detach(),
            "entropy": entropy.mean().detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "log_prob_abs_error": (
                current_log_prob - old_log_prob
            ).abs().mean().detach(),
            "state_replay_abs_error": state_error.detach(),
            "max_root_residual": recomputed["root_residual"].abs().max().detach(),
            "min_root_denominator": recomputed["root_denominator"].min().detach(),
            "mean_abs_b": recomputed["b_dense"].abs().mean().detach(),
            "mean_abs_z": recomputed["z_next_dense"].abs().mean().detach(),
            "mean_abs_delta_loc": recomputed["delta_loc"].abs().mean().detach(),
        }

    def forward(self, mini_batch: P2SequenceMiniBatch) -> Dict[str, torch.Tensor]:
        td = mini_batch.tensordict
        return self._loss_from_recomputed(td, self.unroll(mini_batch))


class P2TransitionPPOLoss(P2SequencePPOLoss):
    """One-step PPO objective with rollout state treated as fixed input.

    The collector still evolves the physical PSB state through time. During
    optimization, however, every transition starts from its stored ``z_prev``
    and gradients stop at that boundary. This preserves the differentiable
    current-step proximal update while removing sequence replay and truncated
    backpropagation through time.
    """

    def forward(self, td) -> Dict[str, torch.Tensor]:
        if len(td.batch_size) != 1:
            raise ValueError(
                "P2 transition mini-batch must have a flat [sample] batch shape."
            )
        output = self.bridge(
            observation=td.get(self.observation_key),
            pair_features=td.get(("agents", "info", "pair_features")),
            neighbor_ids=td.get(("agents", "info", "neighbor_ids")).to(
                torch.long
            ),
            urgency=td.get(("agents", "info", "urgency")),
            confidence=td.get(("agents", "info", "confidence")),
            pair_mask=td.get(("agents", "info", "pair_mask")).to(torch.bool),
            z_prev_dense=td.get(("agents", "psb", "z_prev_dense")).detach(),
        )
        return self._loss_from_recomputed(td, output._asdict())
