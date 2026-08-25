"""M9 resumable training checkpoints."""

from __future__ import annotations

import os
import random
import secrets
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch


M9_CHECKPOINT_SCHEMA_VERSION = 1


def load_m9_checkpoint(path: Path, device) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"M9 resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("M9 checkpoint must contain a dictionary.")
    required = {
        "checkpoint_schema_version",
        "method",
        "stage",
        "iteration",
        "training_mode",
        "training_phase",
        "policy_state",
        "critic_state",
        "optimizer_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "artifact_iterations",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"M9 checkpoint is missing fields: {missing}")
    if payload["checkpoint_schema_version"] != M9_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported M9 checkpoint schema version.")
    if payload["method"] != "opinion_marl" or payload["stage"] != "m9_trainer":
        raise ValueError("Resume checkpoint is not an M9 trainer checkpoint.")
    if type(payload["iteration"]) is not int or payload["iteration"] < 0:
        raise ValueError("M9 checkpoint iteration must be a non-negative integer.")
    return payload


def restore_rng_state(payload: Mapping[str, object]) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_rng_state = payload.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state)


def save_m9_checkpoint(
    path: Path,
    *,
    iteration: int,
    training_mode: str,
    training_phase: str,
    policy,
    critic,
    optimizer,
    artifact_iterations,
    opinion_runtime_config: Mapping[str, object],
    state_tracker=None,
    base_source_state: Optional[Mapping[str, object]] = None,
    base_anchor_state: Optional[Mapping[str, object]] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": M9_CHECKPOINT_SCHEMA_VERSION,
        "method": "opinion_marl",
        "stage": "m9_trainer",
        "iteration": int(iteration),
        "training_mode": str(training_mode),
        "training_phase": str(training_phase),
        "policy_state": policy.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "artifact_iterations": list(artifact_iterations),
        "opinion_runtime_config": dict(opinion_runtime_config),
        "terminal_opinion_state": (
            None if state_tracker is None else state_tracker.snapshot()
        ),
        "base_source_state": base_source_state,
        "base_anchor_state": base_anchor_state,
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
