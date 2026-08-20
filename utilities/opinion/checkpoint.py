"""Versioned Opinion-MARL checkpoint save/load contract."""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Dict, Mapping, Optional

import torch


CHECKPOINT_SCHEMA_VERSION = 1


def save_opinion_checkpoint(
    path,
    *,
    policy,
    critic,
    optimizers: Mapping[str, torch.optim.Optimizer],
    resolved_config: dict,
    stage: str,
    iteration: int,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "iteration": int(iteration),
        "resolved_config": resolved_config,
        "policy_state": policy.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer_states": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "episode_boundary_resume": True,
    }
    torch.save(payload, target)
    return target


def load_opinion_checkpoint(
    path,
    *,
    policy,
    critic,
    optimizers: Mapping[str, torch.optim.Optimizer] = None,
    map_location="cpu",
    expected_stages: Optional[Collection[str]] = None,
) -> Dict[str, object]:
    source = Path(path)
    payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Opinion checkpoint must contain a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported Opinion checkpoint schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    required = {
        "stage",
        "iteration",
        "resolved_config",
        "policy_state",
        "critic_state",
        "optimizer_states",
        "episode_boundary_resume",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Opinion checkpoint missing fields: {missing}")
    if payload["episode_boundary_resume"] is not True:
        raise ValueError("Opinion checkpoints must resume at an episode boundary")
    if expected_stages is not None and payload["stage"] not in expected_stages:
        raise ValueError(
            f"checkpoint stage {payload['stage']!r} is not allowed; "
            f"expected one of {sorted(expected_stages)!r}"
        )
    policy.load_state_dict(payload["policy_state"], strict=True)
    critic.load_state_dict(payload["critic_state"], strict=True)
    if optimizers is not None:
        stored = payload["optimizer_states"]
        if set(stored) != set(optimizers):
            raise ValueError("checkpoint optimizer groups do not match runtime stage")
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(stored[name])
    return {
        "schema_version": payload["schema_version"],
        "stage": payload["stage"],
        "iteration": payload["iteration"],
        "resolved_config": payload["resolved_config"],
        "episode_boundary_resume": payload["episode_boundary_resume"],
    }
