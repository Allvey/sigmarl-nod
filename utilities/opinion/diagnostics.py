"""Online, finite diagnostic summaries for Opinion-MARL."""

from __future__ import annotations

import math

import torch


class _Moments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0

    def add(self, tensor):
        values = tensor.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        values = values[torch.isfinite(values)]
        self.count += values.numel()
        self.total += float(values.sum())
        self.total_square += float(values.square().sum())

    @property
    def mean(self):
        return self.total / max(1, self.count)

    @property
    def variance(self):
        return max(0.0, self.total_square / max(1, self.count) - self.mean**2)


class OpinionDiagnostics:
    def __init__(self, *, b_max=0.5, z_clip=2.0):
        self.b_max = float(b_max)
        self.z_clip = float(z_clip)
        self.reward = _Moments()
        self.raw_b = _Moments()
        self.b = _Moments()
        self.z = _Moments()
        self.z_abs = _Moments()
        self.residual_abs = _Moments()
        self.edge_count = _Moments()
        self.agent_collision_count = 0
        self.lane_collision_count = 0
        self.total_collision_count = 0
        self.collision_denominator = 0
        self.mask_active = 0
        self.mask_total = 0
        self.reset_count = 0
        self.flip_count = 0
        self.flip_denominator = 0
        self.raw_saturated = 0
        self.b_saturated = 0
        self.z_saturated = 0
        self.residual_saturated = 0
        self.active_count = 0
        self.residual_count = 0
        self.last_residual_scale = 0.0
        self.gradient_norms = {"evidence": 0.0, "actor": 0.0, "critic": 0.0}

    def update(
        self,
        *,
        reward,
        collision_agents,
        collision_lanelets,
        raw_b,
        b,
        z_prev,
        z_next,
        residual,
        pair_mask,
        agent_reset_mask,
        residual_scale,
        gradient_norms=None,
    ):
        mask = pair_mask.bool()
        self.reward.add(reward)
        self.raw_b.add(raw_b[mask])
        self.b.add(b[mask])
        self.z.add(z_next[mask])
        self.z_abs.add(z_next[mask].abs())
        self.residual_abs.add(residual.abs())
        self.edge_count.add(mask.sum(dim=-1))
        agent_collision = collision_agents.bool()
        lane_collision = collision_lanelets.bool()
        self.agent_collision_count += int(agent_collision.sum())
        self.lane_collision_count += int(lane_collision.sum())
        self.total_collision_count += int((agent_collision | lane_collision).sum())
        self.collision_denominator += agent_collision.numel()
        self.mask_active += int(mask.sum())
        self.mask_total += mask.numel()
        self.reset_count += int(agent_reset_mask.bool().sum())
        valid_flip = mask & (z_prev.abs() > 1e-6) & (z_next.abs() > 1e-6)
        self.flip_count += int(((z_prev * z_next < 0) & valid_flip).sum())
        self.flip_denominator += int(valid_flip.sum())
        self.active_count += int(mask.sum())
        self.raw_saturated += int((raw_b[mask].abs() >= 0.95 * self.b_max).sum())
        self.b_saturated += int((b[mask].abs() >= 0.95 * self.b_max).sum())
        self.z_saturated += int((z_next[mask].abs() >= 0.95 * self.z_clip).sum())
        self.residual_count += residual.numel()
        if residual_scale > 0:
            self.residual_saturated += int(
                (residual.abs() >= 0.95 * residual_scale).sum()
            )
        self.last_residual_scale = float(residual_scale)
        if gradient_norms:
            for name in self.gradient_norms:
                if name in gradient_norms:
                    self.gradient_norms[name] = float(gradient_norms[name])

    def summary(self):
        denominator = max(1, self.collision_denominator)
        result = {
            "reward_mean": self.reward.mean,
            "collision_agents_rate": self.agent_collision_count / denominator,
            "collision_lanelets_rate": self.lane_collision_count / denominator,
            "collision_total_rate": self.total_collision_count / denominator,
            "raw_b_mean": self.raw_b.mean,
            "raw_b_variance": self.raw_b.variance,
            "raw_b_saturation_rate": self.raw_saturated / max(1, self.active_count),
            "b_mean": self.b.mean,
            "b_variance": self.b.variance,
            "b_saturation_rate": self.b_saturated / max(1, self.active_count),
            "z_mean": self.z.mean,
            "z_variance": self.z.variance,
            "z_abs_mean": self.z_abs.mean,
            "z_flip_rate": self.flip_count / max(1, self.flip_denominator),
            "z_saturation_rate": self.z_saturated / max(1, self.active_count),
            "residual_abs_mean": self.residual_abs.mean,
            "residual_saturation_rate": self.residual_saturated
            / max(1, self.residual_count),
            "edge_count_mean": self.edge_count.mean,
            "mask_ratio": self.mask_active / max(1, self.mask_total),
            "reset_count": float(self.reset_count),
            "residual_scale": self.last_residual_scale,
            "evidence_gradient_norm": self.gradient_norms["evidence"],
            "actor_gradient_norm": self.gradient_norms["actor"],
            "critic_gradient_norm": self.gradient_norms["critic"],
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise RuntimeError("Opinion diagnostics produced non-finite metrics")
        return result
