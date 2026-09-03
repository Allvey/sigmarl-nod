"""Single-run curriculum for P5 joint training from random initialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utilities.psb_marl.p3_differential import (
    normalize_differential_advantage,
)


@dataclass(frozen=True)
class P5ScratchPhase:
    differential_weight: float
    absolute_weight: float
    dual_update_enabled: bool
    paired_learning_enabled: bool
    psb_learning_enabled: bool
    branch_activity_offset: float
    name: str


def scratch_phase(
    iteration: int,
    *,
    base_pretrain_iterations: int = 0,
    absolute_warmup_iterations: int,
    advantage_blend_iterations: int,
    dual_warmup_iterations: int,
    branch_bootstrap_iterations: int = 0,
    branch_activity_bootstrap_offset: float = 0.0,
) -> P5ScratchPhase:
    """Return the locked curriculum state for one one-indexed iteration."""

    if type(iteration) is not int or iteration < 1:
        raise ValueError("P5 scratch iteration must be a positive integer.")
    if min(
        absolute_warmup_iterations,
        advantage_blend_iterations,
        dual_warmup_iterations,
        base_pretrain_iterations,
        branch_bootstrap_iterations,
    ) < 0:
        raise ValueError("P5 scratch schedule lengths must be non-negative.")
    if not 0.0 <= branch_activity_bootstrap_offset <= 1.0:
        raise ValueError("P5 branch bootstrap offset must lie in [0, 1].")
    if iteration <= base_pretrain_iterations:
        return P5ScratchPhase(
            differential_weight=0.0,
            absolute_weight=1.0,
            dual_update_enabled=False,
            paired_learning_enabled=False,
            psb_learning_enabled=False,
            branch_activity_offset=0.0,
            name="base_actor_pretrain",
        )

    post_pretrain_iteration = iteration - base_pretrain_iterations
    if post_pretrain_iteration <= absolute_warmup_iterations:
        differential_weight = 0.0
        name = "absolute_warmup"
    elif advantage_blend_iterations == 0:
        differential_weight = 1.0
        name = "differential"
    else:
        differential_weight = min(
            1.0,
            float(post_pretrain_iteration - absolute_warmup_iterations)
            / float(advantage_blend_iterations),
        )
        name = (
            "absolute_differential_blend"
            if differential_weight < 1.0
            else "differential"
        )
    branch_activity_offset = 0.0
    if 0 < post_pretrain_iteration <= branch_bootstrap_iterations:
        branch_activity_offset = (
            float(branch_activity_bootstrap_offset)
            * float(branch_bootstrap_iterations - post_pretrain_iteration + 1)
            / float(branch_bootstrap_iterations)
        )
    return P5ScratchPhase(
        differential_weight=differential_weight,
        absolute_weight=1.0 - differential_weight,
        dual_update_enabled=iteration > dual_warmup_iterations,
        paired_learning_enabled=True,
        psb_learning_enabled=True,
        branch_activity_offset=branch_activity_offset,
        name=name,
    )


def blend_actor_advantages(
    absolute_advantage: torch.Tensor,
    differential_advantage: torch.Tensor,
    *,
    differential_weight: float,
    scale_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize, blend, and renormalize absolute and differential signals."""

    if absolute_advantage.shape != differential_advantage.shape:
        raise ValueError("P5 absolute and differential advantages must align.")
    if not 0.0 <= differential_weight <= 1.0:
        raise ValueError("P5 differential advantage weight must lie in [0,1].")
    absolute_normalized, absolute_center, absolute_scale = (
        normalize_differential_advantage(
            absolute_advantage.detach(), scale_floor=scale_floor
        )
    )
    mixed = (
        (1.0 - float(differential_weight)) * absolute_normalized
        + float(differential_weight) * differential_advantage.detach()
    )
    result, mixed_center, mixed_scale = normalize_differential_advantage(
        mixed, scale_floor=scale_floor
    )
    return result.detach(), {
        "absolute_advantage_center": absolute_center,
        "absolute_advantage_scale": absolute_scale,
        "mixed_advantage_center": mixed_center,
        "mixed_advantage_scale": mixed_scale,
    }
