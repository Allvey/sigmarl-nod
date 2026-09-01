"""Resumable P2 checkpoint contract."""

from __future__ import annotations

import os
import random
import secrets
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


P2_CHECKPOINT_SCHEMA_VERSION = 1


def load_p2_checkpoint(path: Path, device) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P2 resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("P2 checkpoint must contain a dictionary.")
    required = {
        "checkpoint_schema_version",
        "method",
        "stage",
        "iteration",
        "policy_state",
        "critic_state",
        "optimizer_state",
        "artifact_iterations",
        "runtime_config",
        "base_policy_sha256",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"P2 checkpoint is missing fields: {missing}.")
    if payload["checkpoint_schema_version"] != P2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported P2 checkpoint schema version.")
    if (
        payload["method"] != "psb_marl"
        or payload["stage"] != "p2_frozen_base_bifurcation"
    ):
        raise ValueError("Resume checkpoint is not a P2 checkpoint.")
    if type(payload["iteration"]) is not int or payload["iteration"] < 0:
        raise ValueError("P2 checkpoint iteration must be non-negative.")
    return payload


def restore_p2_rng_state(payload: Mapping[str, object]) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_p2_checkpoint(
    path: Path,
    *,
    iteration: int,
    policy,
    critic,
    optimizer,
    artifact_iterations,
    runtime_config: Mapping[str, object],
    base_policy_sha256: str,
    state_tracker=None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": P2_CHECKPOINT_SCHEMA_VERSION,
        "method": "psb_marl",
        "stage": "p2_frozen_base_bifurcation",
        "iteration": int(iteration),
        "policy_state": policy.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "artifact_iterations": list(artifact_iterations),
        "runtime_config": dict(runtime_config),
        "base_policy_sha256": str(base_policy_sha256),
        "terminal_state": (
            None if state_tracker is None else state_tracker.snapshot()
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(4)}.tmp"
    )
    torch.save(payload, temporary)
    os.replace(temporary, destination)

