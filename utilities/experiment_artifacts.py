"""Artifact management for reproducible SigmaRL training runs.

This module is intentionally independent from the PPO implementation.  It
creates isolated run directories and writes metadata without changing the
collector, losses, or optimizer update order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ARTIFACT_SCHEMA_VERSION = 1
_INTERMEDIATE_POLICY_PATTERN = re.compile(
    r"^reward(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))_policy\.pth$"
)
_INTERMEDIATE_EVIDENCE_PATTERN = re.compile(
    r"^reward(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))_evidence_net\.pth$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically replace a JSON file in its destination directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def create_run_directory(output_root: str, method: str, seed: int) -> Path:
    """Create and return a unique, non-reused run directory."""

    root = Path(output_root).expanduser().resolve()
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{method}-seed{seed}-{timestamp}-{secrets.token_hex(4)}"
    run_directory = runs_root / run_id
    run_directory.mkdir(parents=False, exist_ok=False)
    return run_directory


def initialize_run(
    run_directory: Path,
    source_config: Dict[str, Any],
    resolved_config: Dict[str, Any],
    method: str = "base_mappo",
    stage: str = "base",
) -> None:
    """Write immutable input snapshots and initial run metadata."""

    run_directory = Path(run_directory)
    atomic_write_json(run_directory / "config_source.json", source_config)
    atomic_write_json(run_directory / "config_resolved.json", resolved_config)
    atomic_write_json(
        run_directory / "validation_protocol.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "method": method,
            "stage": stage,
            "automated_performance_validation": False,
            "performance_validation_owner": "user",
            "note": (
                "R1 preserves the original SigmaRL 1.2.0 training path."
                if method == "base_mappo"
                else "Opinion-MARL performance validation is performed manually."
            ),
        },
    )
    write_training_status(run_directory, status="running", iteration=0)


def write_training_status(
    run_directory: Path,
    status: str,
    iteration: Optional[int],
    error: Optional[str] = None,
) -> None:
    status_path = Path(run_directory) / "training_status.json"
    if iteration is None and status_path.is_file():
        with status_path.open("r", encoding="utf-8") as file:
            iteration = int(json.load(file).get("iteration", 0))
    if iteration is None:
        iteration = 0
    payload: Dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "iteration": int(iteration),
        "updated_at": _utc_now(),
    }
    if error is not None:
        payload["error"] = error
    atomic_write_json(status_path, payload)


def write_metrics(
    run_directory: Path,
    iterations: List[Dict[str, Any]],
    method: str = "base_mappo",
    stage: str = "base",
) -> None:
    atomic_write_json(
        Path(run_directory) / "metrics.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "method": method,
            "stage": stage,
            "iterations": iterations,
        },
    )


def write_timing(
    run_directory: Path,
    iterations: Iterable[Dict[str, Any]],
    total_seconds: float,
) -> None:
    timing_iterations = [
        {
            "iteration": item["iteration"],
            "rollout_seconds": item["rollout_seconds"],
            "optimization_seconds": item["optimization_seconds"],
            "iteration_seconds": item["iteration_seconds"],
        }
        for item in iterations
    ]
    atomic_write_json(
        Path(run_directory) / "timing.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "total_seconds": float(total_seconds),
            "total_hours": float(total_seconds) / 3600.0,
            "iterations": timing_iterations,
        },
    )


def save_training_curves(
    run_directory: Path, iterations: List[Dict[str, Any]]
) -> None:
    """Save the final Base training curves as one multi-panel PDF."""

    if not iterations:
        return

    import matplotlib.pyplot as plt

    x = [item["iteration"] for item in iterations]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    axes[0, 0].plot(x, [item["episode_reward_mean"] for item in iterations])
    axes[0, 0].set_title("Episode reward")

    axes[0, 1].plot(
        x,
        [item["collision_with_agents_rate"] for item in iterations],
        label="agents",
    )
    axes[0, 1].plot(
        x,
        [item["collision_with_lanelets_rate"] for item in iterations],
        label="lanelets",
    )
    axes[0, 1].plot(
        x,
        [item["total_collision_rate"] for item in iterations],
        label="total",
    )
    axes[0, 1].set_title("Collision rates")
    axes[0, 1].legend()

    axes[1, 0].plot(
        x, [item["loss_objective"] for item in iterations], label="actor"
    )
    axes[1, 0].plot(x, [item["loss_critic"] for item in iterations], label="critic")
    axes[1, 0].plot(x, [item["loss_entropy"] for item in iterations], label="entropy")
    axes[1, 0].set_title("PPO losses")
    axes[1, 0].legend()

    axes[1, 1].plot(
        x, [item["rollout_seconds"] for item in iterations], label="rollout"
    )
    axes[1, 1].plot(
        x,
        [item["optimization_seconds"] for item in iterations],
        label="optimization",
    )
    axes[1, 1].set_title("Wall time per iteration")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(Path(run_directory) / "training_curves.pdf", bbox_inches="tight")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_manifest(run_directory: Path) -> None:
    run_directory = Path(run_directory)
    artifacts = []
    for path in sorted(run_directory.iterdir()):
        if not path.is_file() or path.name == "artifacts_manifest.json":
            continue
        artifacts.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    atomic_write_json(
        run_directory / "artifacts_manifest.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "artifacts": artifacts,
        },
    )


def mark_latest_completed_run(output_root: str, run_directory: Path) -> None:
    atomic_write_json(
        Path(output_root).expanduser().resolve() / "latest_run.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "completed",
            "run_directory": str(Path(run_directory).resolve()),
            "updated_at": _utc_now(),
        },
    )


def resolve_latest_run(output_root: str) -> Path:
    """Resolve the completed run selected by the stable latest-run pointer."""

    root = Path(output_root).expanduser().resolve()
    pointer_path = root / "latest_run.json"
    if pointer_path.is_file():
        with pointer_path.open("r", encoding="utf-8") as file:
            pointer = json.load(file)
        run_directory = Path(pointer["run_directory"]).expanduser().resolve()
        if pointer.get("status") != "completed":
            raise RuntimeError(f"Latest run is not complete: {pointer_path}")
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_directory}")
        return run_directory

    # Compatibility with a legacy output directory that directly contains
    # SigmaRL artifacts rather than R1 runs/<run_id>/.
    if (root / "final_policy.pth").is_file():
        return root
    raise FileNotFoundError(
        f"No completed Base run found under {root}. Run python main_training.py first."
    )


def resolve_policy_checkpoint(
    run_directory: Path,
    checkpoint_path: Optional[Path] = None,
) -> Path:
    """Resolve the policy state dict used for visualization/evaluation.

    A final policy is preferred.  While training is still running, SigmaRL's
    reward-named best intermediate policy is a valid testing fallback.
    """

    run_directory = Path(run_directory).expanduser().resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_directory}")

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if checkpoint.parent != run_directory:
            raise ValueError(
                "The explicit checkpoint must be directly inside the selected "
                f"run directory: checkpoint={checkpoint}, run={run_directory}"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Policy checkpoint does not exist: {checkpoint}")
        if checkpoint.name != "final_policy.pth" and not (
            _INTERMEDIATE_POLICY_PATTERN.fullmatch(checkpoint.name)
        ):
            raise ValueError(
                "A testable policy checkpoint must be final_policy.pth or "
                f"reward<value>_policy.pth, got: {checkpoint.name}"
            )
        return checkpoint

    final_policy = run_directory / "final_policy.pth"
    if final_policy.is_file():
        return final_policy

    intermediate_policies = []
    for candidate in run_directory.glob("reward*_policy.pth"):
        match = _INTERMEDIATE_POLICY_PATTERN.fullmatch(candidate.name)
        if match and candidate.is_file():
            intermediate_policies.append((float(match.group(1)), candidate))
    if intermediate_policies:
        # Intermediate files represent best-so-far snapshots.  Select the
        # highest reward, with filename as a deterministic tie breaker.
        return max(intermediate_policies, key=lambda item: (item[0], item[1].name))[1]

    raise FileNotFoundError(
        "No testable policy checkpoint found in "
        f"{run_directory}. Expected final_policy.pth or reward<value>_policy.pth."
    )


def resolve_policy_critic_pair(
    run_directory: Path,
    checkpoint_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Resolve a policy and its reward/final-matched critic checkpoint."""

    policy_checkpoint = resolve_policy_checkpoint(run_directory, checkpoint_path)
    policy_suffix = "_policy.pth"
    checkpoint_prefix = policy_checkpoint.name[: -len(policy_suffix)]
    critic_checkpoint = policy_checkpoint.with_name(
        f"{checkpoint_prefix}_critic.pth"
    )
    if not critic_checkpoint.is_file():
        raise FileNotFoundError(
            "The selected Base policy has no matching critic checkpoint: "
            f"policy={policy_checkpoint}, expected_critic={critic_checkpoint}"
        )
    return policy_checkpoint, critic_checkpoint


def resolve_evidence_critic_pair(run_directory: Path) -> tuple[Path, Path]:
    """Resolve M5 EvidenceNet and its final/reward-matched critic."""

    run_directory = Path(run_directory).expanduser().resolve()
    final_evidence = run_directory / "final_evidence_net.pth"
    if final_evidence.is_file():
        critic = run_directory / "final_critic.pth"
        if not critic.is_file():
            raise FileNotFoundError(
                f"Final EvidenceNet has no matching critic: {critic}"
            )
        return final_evidence, critic

    candidates = []
    for candidate in run_directory.glob("reward*_evidence_net.pth"):
        match = _INTERMEDIATE_EVIDENCE_PATTERN.fullmatch(candidate.name)
        if match and candidate.is_file():
            candidates.append((float(match.group(1)), candidate))
    if not candidates:
        raise FileNotFoundError(
            f"No standalone M5 EvidenceNet found in {run_directory}. Expected "
            "final_evidence_net.pth or reward<value>_evidence_net.pth. Restart "
            "M5 with the current saving logic before starting M6."
        )
    _, evidence = max(candidates, key=lambda item: (item[0], item[1].name))
    reward_prefix = evidence.name[: -len("_evidence_net.pth")]
    critic = evidence.with_name(f"{reward_prefix}_critic.pth")
    if not critic.is_file():
        raise FileNotFoundError(
            "The selected M5 EvidenceNet has no matching critic: "
            f"evidence={evidence}, expected_critic={critic}"
        )
    return evidence, critic


def resolve_latest_evidence_run(output_root: str) -> Path:
    """Resolve a completed or newest M5 run with standalone Evidence/Critic."""

    root = Path(output_root).expanduser().resolve()
    pointer_path = root / "latest_run.json"
    checked = set()
    if pointer_path.is_file():
        completed_run = resolve_latest_run(output_root)
        checked.add(completed_run)
        try:
            resolve_evidence_critic_pair(completed_run)
            return completed_run
        except FileNotFoundError:
            pass

    runs_root = root / "runs"
    candidates = (
        sorted(
            (path.resolve() for path in runs_root.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        if runs_root.is_dir()
        else []
    )
    for candidate in candidates:
        if candidate in checked or not (
            candidate / "config_resolved.json"
        ).is_file():
            continue
        try:
            resolve_evidence_critic_pair(candidate)
            return candidate
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"No M5 run with a standalone EvidenceNet/Critic pair was found under "
        f"{root}. Restart M5 with the current saving logic first."
    )


def resolve_latest_testable_run(output_root: str) -> Path:
    """Resolve a completed run, or the newest run with an intermediate policy."""

    root = Path(output_root).expanduser().resolve()

    # Preserve the stable completed-run behavior whenever its pointer exists.
    pointer_path = root / "latest_run.json"
    if pointer_path.is_file():
        completed_run = resolve_latest_run(output_root)
        resolve_policy_checkpoint(completed_run)
        return completed_run

    # Compatibility with legacy flat output directories.
    if root.is_dir():
        try:
            resolve_policy_checkpoint(root)
            return root
        except FileNotFoundError:
            pass

    runs_root = root / "runs"
    candidates = (
        sorted(
            (path for path in runs_root.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        if runs_root.is_dir()
        else []
    )
    for candidate in candidates:
        if not (candidate / "config_resolved.json").is_file():
            continue
        try:
            resolve_policy_checkpoint(candidate)
            return candidate
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"No testable run found under {root}. A run becomes testable after its "
        "first reward<value>_policy.pth snapshot is saved; it does not need to "
        "have final_policy.pth yet."
    )
