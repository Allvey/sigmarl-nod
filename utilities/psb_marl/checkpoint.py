"""Checkpoint integrity helpers for PSB-MARL."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_checkpoint_exact(source: Path, destination: Path) -> str:
    """Atomically copy a checkpoint and prove byte-for-byte equality."""

    source = Path(source).expanduser().resolve()
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.copying")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if destination_hash != source_hash:
        raise RuntimeError(
            "Checkpoint copy failed its SHA-256 equality check: "
            f"source={source}, destination={destination}."
        )
    return source_hash


def save_p0_training_checkpoint(
    path: Path,
    *,
    policy_checkpoint: Path,
    critic_checkpoint: Path,
    policy_sha256: str,
    critic_sha256: str,
    source_base_run: Path,
    config_fingerprint: str,
) -> None:
    """Save a self-contained P0 training container for future stage loading."""

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": "p0_base_passthrough",
        "iteration": 0,
        "policy_state": torch.load(policy_checkpoint, map_location="cpu"),
        "critic_state": torch.load(critic_checkpoint, map_location="cpu"),
        "policy_sha256": policy_sha256,
        "critic_sha256": critic_sha256,
        "source_base_run": str(Path(source_base_run).resolve()),
        "config_fingerprint": config_fingerprint,
    }
    path = Path(path)
    temporary = path.with_name(f".{path.name}.saving")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_p1_layer_checkpoint(path: Path, runtime_config: Dict[str, Any]) -> None:
    """Save the fixed, parameter-free P1 sidecar configuration."""

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": "p1_zero_control_equivalence",
        "trainable_parameters": 0,
        "runtime_config": dict(runtime_config),
    }
    path = Path(path)
    temporary = path.with_name(f".{path.name}.saving")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_p1_training_checkpoint(
    path: Path,
    *,
    policy_checkpoint: Path,
    critic_checkpoint: Path,
    policy_sha256: str,
    critic_sha256: str,
    parent_run: Path,
    runtime_config: Dict[str, Any],
    config_fingerprint: str,
) -> None:
    """Save the Base weights and fixed P1 runtime in one future-stage container."""

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": "p1_zero_control_equivalence",
        "iteration": 0,
        "policy_state": torch.load(policy_checkpoint, map_location="cpu"),
        "critic_state": torch.load(critic_checkpoint, map_location="cpu"),
        "policy_sha256": policy_sha256,
        "critic_sha256": critic_sha256,
        "parent_run": str(Path(parent_run).resolve()),
        "runtime_config": dict(runtime_config),
        "config_fingerprint": config_fingerprint,
    }
    path = Path(path)
    temporary = path.with_name(f".{path.name}.saving")
    torch.save(payload, temporary)
    os.replace(temporary, path)
