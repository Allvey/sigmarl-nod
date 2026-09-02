"""Paired Base-relative learning utilities for PSB-MARL P3.3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from utilities.psb_marl.p3_critic import TARGET_CHANNELS


def paired_iteration_seed(training_seed: int, iteration: int) -> int:
    """Return a deterministic, stage-local CRN seed for one paired batch."""

    if type(training_seed) is not int or training_seed < 0:
        raise ValueError("P3.3 training_seed must be a non-negative integer.")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("P3.3 iteration must be a non-negative integer.")
    return int((training_seed * 1_000_003 + iteration) % (2**31 - 1))


def paired_transition_seed(pair_seed: int, time_index: int) -> int:
    """Return the CRN seed shared by both sides of one physical step."""

    if type(pair_seed) is not int or pair_seed < 0:
        raise ValueError("P3.3 pair_seed must be a non-negative integer.")
    if type(time_index) is not int or time_index < 0:
        raise ValueError("P3.3 time_index must be a non-negative integer.")
    return int((pair_seed * 65_537 + time_index * 2 + 1) % (2**31 - 1))


def paired_reset_seed(pair_seed: int, time_index: int) -> int:
    """Return the seed used to synchronously reset a paired episode boundary."""

    if type(pair_seed) is not int or pair_seed < 0:
        raise ValueError("P3.3 pair_seed must be a non-negative integer.")
    if type(time_index) is not int or time_index < 0:
        raise ValueError("P3.3 time_index must be a non-negative integer.")
    return int((pair_seed * 65_537 + time_index * 2 + 2) % (2**31 - 1))


def lagrangian_channel_weights(
    *,
    vehicle_multiplier: float,
    lane_multiplier: float,
    vehicle_budget: float,
    lane_budget: float,
    normalize_constraints: bool,
    active_constraints: Iterable[str],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Build weights for reward, vehicle-cost, and lane-cost differences."""

    if reference.shape[-1] != len(TARGET_CHANNELS):
        raise ValueError("P3.3 differential channels must have width three.")
    active = tuple(active_constraints)
    if len(set(active)) != len(active) or not set(active).issubset(
        {"vehicle", "lane"}
    ):
        raise ValueError("P3.3 active constraints are invalid.")
    values = [1.0, 0.0, 0.0]
    for index, (name, multiplier, budget) in enumerate(
        (
            ("vehicle", vehicle_multiplier, vehicle_budget),
            ("lane", lane_multiplier, lane_budget),
        ),
        start=1,
    ):
        multiplier = float(multiplier)
        budget = float(budget)
        if not torch.isfinite(torch.tensor(multiplier)):
            raise ValueError("P3.3 multipliers must be finite.")
        if name not in active:
            continue
        if normalize_constraints and budget <= 0.0:
            raise ValueError("Normalized P3.3 budgets must be positive.")
        values[index] = -multiplier / budget if normalize_constraints else -multiplier
    return reference.new_tensor(values)


def combine_lagrangian_channels(
    channels: torch.Tensor,
    *,
    vehicle_multiplier: float,
    lane_multiplier: float,
    vehicle_budget: float,
    lane_budget: float,
    normalize_constraints: bool,
    active_constraints: Iterable[str],
) -> torch.Tensor:
    """Combine vector return differences into one scalar Lagrangian value."""

    weights = lagrangian_channel_weights(
        vehicle_multiplier=vehicle_multiplier,
        lane_multiplier=lane_multiplier,
        vehicle_budget=vehicle_budget,
        lane_budget=lane_budget,
        normalize_constraints=normalize_constraints,
        active_constraints=active_constraints,
        reference=channels,
    )
    return (channels * weights).sum(dim=-1, keepdim=True)


def normalize_differential_advantage(
    advantage: torch.Tensor,
    *,
    scale_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize across environment/time while retaining each agent axis."""

    if advantage.ndim != 4 or advantage.shape[-1] != 1:
        raise ValueError("P3.3 advantage must have shape [E,T,N,1].")
    if scale_floor <= 0.0:
        raise ValueError("P3.3 advantage scale floor must be positive.")
    center = advantage.mean(dim=(0, 1), keepdim=True)
    scale = advantage.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(
        float(scale_floor)
    )
    return (advantage - center) / scale, center, scale


def differential_advantage(
    target_channels: torch.Tensor,
    predicted_channels: torch.Tensor,
    *,
    vehicle_multiplier: float,
    lane_multiplier: float,
    vehicle_budget: float,
    lane_budget: float,
    normalize_constraints: bool,
    active_constraints: Iterable[str],
    normalize_advantage: bool,
    advantage_scale_floor: float,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Return detached Base-relative advantage and its diagnostic terms."""

    if target_channels.shape != predicted_channels.shape:
        raise ValueError("P3.3 target and predicted channels must align.")
    target_value = combine_lagrangian_channels(
        target_channels,
        vehicle_multiplier=vehicle_multiplier,
        lane_multiplier=lane_multiplier,
        vehicle_budget=vehicle_budget,
        lane_budget=lane_budget,
        normalize_constraints=normalize_constraints,
        active_constraints=active_constraints,
    )
    predicted_value = combine_lagrangian_channels(
        predicted_channels,
        vehicle_multiplier=vehicle_multiplier,
        lane_multiplier=lane_multiplier,
        vehicle_budget=vehicle_budget,
        lane_budget=lane_budget,
        normalize_constraints=normalize_constraints,
        active_constraints=active_constraints,
    )
    raw = (target_value - predicted_value).detach()
    center = torch.zeros_like(raw[:, :1, :1])
    scale = torch.ones_like(center)
    result = raw
    if normalize_advantage:
        result, center, scale = normalize_differential_advantage(
            raw, scale_floor=advantage_scale_floor
        )
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("P3.3 differential advantage is non-finite.")
    return result.detach(), {
        "target_lagrangian": target_value.detach(),
        "predicted_lagrangian": predicted_value.detach(),
        "raw_advantage": raw,
        "advantage_center": center.detach(),
        "advantage_scale": scale.detach(),
    }


def differential_critic_loss(
    model,
    *,
    candidate_observation: torch.Tensor,
    base_observation: torch.Tensor,
    candidate_z: torch.Tensor,
    edge_mask: torch.Tensor,
    target_channels: torch.Tensor,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online normalized Huber regression for the P3.1 vector critic."""

    if huber_delta <= 0.0:
        raise ValueError("P3.3 huber_delta must be positive.")
    prediction = model(
        candidate_observation,
        base_observation,
        candidate_z,
        edge_mask,
    )
    if prediction.shape != target_channels.shape:
        raise ValueError("P3.3 critic prediction and target must align.")
    shape = [1] * (prediction.ndim - 1) + [len(TARGET_CHANNELS)]
    normalized_error = (
        prediction - target_channels.detach()
    ) / model.target_scale.view(*shape)
    loss = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=float(huber_delta),
    )
    return loss, prediction


def save_online_differential_critic(
    path: Path,
    *,
    model,
    runtime_config: Mapping[str, object],
    source_checkpoint: Path,
) -> None:
    """Atomically save the trainable P3.3 critic separately from scalar MAPPO."""

    payload = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": runtime_config.get(
            "p5_stage", "p3_paired_differential_primal_dual_ppo"
        ),
        "model_config": model.model_config(),
        "critic_state": model.state_dict(),
        "runtime_config": dict(runtime_config),
        "source_checkpoint": str(Path(source_checkpoint).expanduser().resolve()),
    }
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.saving")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
