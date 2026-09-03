"""Projected primal-dual state for PSB-MARL P3.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass(frozen=True)
class P3SafetyCosts:
    vehicle: torch.Tensor
    lane: torch.Tensor


def continuous_safety_costs(
    *,
    urgency: torch.Tensor,
    confidence: torch.Tensor,
    pair_mask: torch.Tensor,
    distance_left: torch.Tensor,
    distance_right: torch.Tensor,
    vehicle_collision: torch.Tensor,
    lane_collision: torch.Tensor,
    lane_safety_margin: float,
) -> P3SafetyCosts:
    """Compute the same dense costs used to certify the P3.1 critic."""

    if lane_safety_margin <= 0.0:
        raise ValueError("lane_safety_margin must be positive.")
    if urgency.shape != confidence.shape or urgency.shape != pair_mask.shape:
        raise ValueError("Vehicle risk tensors must have identical shapes.")
    if urgency.ndim < 2:
        raise ValueError("Vehicle risk tensors require a candidate dimension.")
    expected = urgency.shape[:-1]

    def scalar(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.shape == expected + (1,):
            value = value.squeeze(-1)
        if value.shape != expected:
            raise ValueError(f"{name} does not align with agent costs.")
        return value.to(dtype=urgency.dtype)

    vehicle_risk = (
        urgency.clamp(0.0, 1.0)
        * confidence.clamp(0.0, 1.0)
        * pair_mask.to(urgency.dtype)
    ).amax(dim=-1)
    vehicle = torch.maximum(
        vehicle_risk,
        scalar(vehicle_collision, "vehicle_collision"),
    )
    clearance = torch.minimum(
        scalar(distance_left, "distance_left"),
        scalar(distance_right, "distance_right"),
    ).clamp_min(0.0)
    lane_margin_cost = (
        (float(lane_safety_margin) - clearance)
        / float(lane_safety_margin)
    ).clamp(0.0, 1.0)
    lane = torch.maximum(
        lane_margin_cost,
        scalar(lane_collision, "lane_collision"),
    )
    if not bool(torch.isfinite(vehicle).all() and torch.isfinite(lane).all()):
        raise ValueError("P3.2 safety costs must be finite.")
    return P3SafetyCosts(vehicle=vehicle, lane=lane)


class ProjectedDualController:
    """Two scalar multipliers updated outside the policy-gradient graph."""

    def __init__(
        self,
        *,
        vehicle_budget: float,
        lane_budget: float,
        vehicle_learning_rate: float,
        lane_learning_rate: float,
        maximum_multiplier: float,
        initial_vehicle_multiplier: float = 0.0,
        initial_lane_multiplier: float = 0.0,
        normalize_constraints: bool = False,
        active_constraints: Sequence[str] = ("vehicle", "lane"),
    ) -> None:
        values = (
            vehicle_budget,
            lane_budget,
            vehicle_learning_rate,
            lane_learning_rate,
            maximum_multiplier,
            initial_vehicle_multiplier,
            initial_lane_multiplier,
        )
        if any(not torch.isfinite(torch.tensor(value)) for value in values):
            raise ValueError("P3.2 dual parameters must be finite.")
        if min(vehicle_budget, lane_budget) < 0.0:
            raise ValueError("P3.2 safety budgets must be non-negative.")
        if normalize_constraints and min(vehicle_budget, lane_budget) <= 0.0:
            raise ValueError("Normalized P3.2 budgets must be positive.")
        if min(vehicle_learning_rate, lane_learning_rate) <= 0.0:
            raise ValueError("P3.2 dual learning rates must be positive.")
        if maximum_multiplier <= 0.0:
            raise ValueError("P3.2 maximum_multiplier must be positive.")
        if not (
            0.0 <= initial_vehicle_multiplier <= maximum_multiplier
            and 0.0 <= initial_lane_multiplier <= maximum_multiplier
        ):
            raise ValueError("Initial dual multipliers violate projection bounds.")
        active = tuple(active_constraints)
        if (
            not active
            or len(set(active)) != len(active)
            or not set(active).issubset({"vehicle", "lane"})
        ):
            raise ValueError("P3.2 active constraints are invalid.")
        if "vehicle" not in active and initial_vehicle_multiplier != 0.0:
            raise ValueError("Inactive vehicle multiplier must start at zero.")
        if "lane" not in active and initial_lane_multiplier != 0.0:
            raise ValueError("Inactive lane multiplier must start at zero.")
        self.vehicle_budget = float(vehicle_budget)
        self.lane_budget = float(lane_budget)
        self.vehicle_learning_rate = float(vehicle_learning_rate)
        self.lane_learning_rate = float(lane_learning_rate)
        self.maximum_multiplier = float(maximum_multiplier)
        self.vehicle_multiplier = float(initial_vehicle_multiplier)
        self.lane_multiplier = float(initial_lane_multiplier)
        self.normalize_constraints = bool(normalize_constraints)
        self.active_constraints = active

    def _constraint_costs(self, costs: P3SafetyCosts) -> P3SafetyCosts:
        if not self.normalize_constraints:
            return costs
        return P3SafetyCosts(
            vehicle=costs.vehicle / self.vehicle_budget,
            lane=costs.lane / self.lane_budget,
        )

    def lagrangian_reward(
        self,
        augmented_reward: torch.Tensor,
        costs: P3SafetyCosts,
    ) -> torch.Tensor:
        scaled_costs = self._constraint_costs(costs)
        vehicle = scaled_costs.vehicle
        lane = scaled_costs.lane
        if augmented_reward.shape == vehicle.shape + (1,):
            vehicle = vehicle.unsqueeze(-1)
            lane = lane.unsqueeze(-1)
        if augmented_reward.shape != vehicle.shape:
            raise ValueError("P3.2 reward and safety costs do not align.")
        result = augmented_reward
        if "vehicle" in self.active_constraints:
            result = result - self.vehicle_multiplier * vehicle
        if "lane" in self.active_constraints:
            result = result - self.lane_multiplier * lane
        return result

    @torch.no_grad()
    def update(
        self,
        costs: P3SafetyCosts,
        *,
        enabled: bool = True,
    ) -> Dict[str, object]:
        vehicle_mean = float(costs.vehicle.detach().mean().item())
        lane_mean = float(costs.lane.detach().mean().item())
        vehicle_raw_residual = vehicle_mean - self.vehicle_budget
        lane_raw_residual = lane_mean - self.lane_budget
        if self.normalize_constraints:
            vehicle_residual = vehicle_raw_residual / self.vehicle_budget
            lane_residual = lane_raw_residual / self.lane_budget
        else:
            vehicle_residual = vehicle_raw_residual
            lane_residual = lane_raw_residual
        if enabled and "vehicle" in self.active_constraints:
            self.vehicle_multiplier = min(
                self.maximum_multiplier,
                max(
                    0.0,
                    self.vehicle_multiplier
                    + self.vehicle_learning_rate * vehicle_residual,
                ),
            )
        if enabled and "lane" in self.active_constraints:
            self.lane_multiplier = min(
                self.maximum_multiplier,
                max(
                    0.0,
                    self.lane_multiplier
                    + self.lane_learning_rate * lane_residual,
                ),
            )
        return {
            "vehicle_cost_mean": vehicle_mean,
            "lane_cost_mean": lane_mean,
            "vehicle_constraint_residual": vehicle_residual,
            "lane_constraint_residual": lane_residual,
            "vehicle_constraint_raw_residual": vehicle_raw_residual,
            "lane_constraint_raw_residual": lane_raw_residual,
            "constraints_budget_normalized": self.normalize_constraints,
            "vehicle_constraint_dualized": (
                "vehicle" in self.active_constraints
            ),
            "lane_constraint_dualized": "lane" in self.active_constraints,
            "dual_update_enabled": bool(enabled),
            "vehicle_multiplier": self.vehicle_multiplier,
            "lane_multiplier": self.lane_multiplier,
        }

    def state_dict(self) -> Dict[str, float]:
        result = {
            "vehicle_budget": self.vehicle_budget,
            "lane_budget": self.lane_budget,
            "vehicle_learning_rate": self.vehicle_learning_rate,
            "lane_learning_rate": self.lane_learning_rate,
            "maximum_multiplier": self.maximum_multiplier,
            "vehicle_multiplier": self.vehicle_multiplier,
            "lane_multiplier": self.lane_multiplier,
        }
        if self.normalize_constraints:
            result["normalize_constraints"] = True
        if self.active_constraints != ("vehicle", "lane"):
            result["active_constraints"] = list(self.active_constraints)
        return result
