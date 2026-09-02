"""P3.0 immutable P2 bridge and paired Candidate/Base rollout contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from utilities.experiment_artifacts import (
    atomic_write_json,
    create_run_directory,
    initialize_run,
    mark_latest_completed_run,
    resolve_latest_testable_run,
    resolve_policy_checkpoint,
    write_artifact_manifest,
    write_training_status,
)
from utilities.psb_marl.checkpoint import (
    copy_checkpoint_exact,
    save_p3_pairing_checkpoint,
    sha256_file,
)
from utilities.psb_marl.config import PSBConfigError, load_psb_experiment


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def package_p3_pairing(
    experiment,
    *,
    resume_checkpoint: Optional[Path],
    iterations_override: Optional[int],
) -> Path:
    """Package a certified P2.1-U candidate without changing model bytes."""

    if resume_checkpoint is not None:
        raise PSBConfigError("P3.0 is immutable and does not support resume.")
    if iterations_override is not None:
        raise PSBConfigError(
            "P3.0 performs no optimization; --iterations is invalid."
        )
    assert experiment.parent_run is not None
    output_root = str(Path(experiment.output_root).expanduser().resolve())
    run_directory = create_run_directory(
        output_root=output_root,
        method="psb-p3-paired",
        seed=experiment.effective_training_seed,
    )
    resolved_base = dict(experiment.base_run_config)
    resolved_base.update(
        {
            "seed": experiment.effective_training_seed,
            "where_to_save": str(run_directory) + os.sep,
            "output_root": output_root,
            "run_id": run_directory.name,
            "artifact_logging_enabled": True,
        }
    )
    initialize_run(
        run_directory=run_directory,
        source_config=experiment.source_config,
        resolved_config=resolved_base,
        method="psb_marl",
        stage=experiment.stage,
    )
    try:
        source_policy = experiment.parent_run / "candidate_policy.pth"
        source_critic = experiment.parent_run / "candidate_critic.pth"
        candidate_policy = run_directory / "candidate_policy.pth"
        candidate_critic = run_directory / "candidate_critic.pth"
        policy_hash = copy_checkpoint_exact(source_policy, candidate_policy)
        critic_hash = copy_checkpoint_exact(source_critic, candidate_critic)
        copy_checkpoint_exact(
            source_policy, run_directory / "source_p2_policy.pth"
        )
        copy_checkpoint_exact(
            source_critic, run_directory / "source_p2_critic.pth"
        )
        base_policy_hash = copy_checkpoint_exact(
            experiment.base.policy_checkpoint,
            run_directory / "base_fallback_policy.pth",
        )
        base_critic_hash = copy_checkpoint_exact(
            experiment.base.critic_checkpoint,
            run_directory / "base_fallback_critic.pth",
        )
        copy_checkpoint_exact(
            experiment.base.policy_checkpoint,
            run_directory / "final_policy.pth",
        )
        copy_checkpoint_exact(
            experiment.base.critic_checkpoint,
            run_directory / "final_critic.pth",
        )
        runtime = experiment.p3_runtime_config()
        config_fingerprint = _fingerprint(experiment.source_config)
        save_p3_pairing_checkpoint(
            run_directory / "final_checkpoint.pt",
            policy_checkpoint=candidate_policy,
            critic_checkpoint=candidate_critic,
            policy_sha256=policy_hash,
            critic_sha256=critic_hash,
            parent_run=experiment.parent_run,
            runtime_config=runtime,
            config_fingerprint=config_fingerprint,
        )
        equivalence = {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": experiment.stage,
            "equivalence_kind": "byte_exact_p2_candidate_passthrough",
            "learning_enabled": False,
            "parent_run": str(experiment.parent_run),
            "source_policy_sha256": sha256_file(source_policy),
            "source_critic_sha256": sha256_file(source_critic),
            "candidate_policy_sha256": policy_hash,
            "candidate_critic_sha256": critic_hash,
            "policy_bytes_identical_to_source_p2": True,
            "critic_bytes_identical_to_source_p2": True,
            "base_policy_sha256": base_policy_hash,
            "base_critic_sha256": base_critic_hash,
        }
        atomic_write_json(run_directory / "p3_0_equivalence.json", equivalence)
        atomic_write_json(
            run_directory / "psb_config_resolved.json",
            {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_parent_run": str(experiment.parent_run),
                "runtime_config": runtime,
                "config_fingerprint": config_fingerprint,
            },
        )
        atomic_write_json(
            run_directory / "deployment_manifest.json",
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "selected": "base_fallback_p3_pairing_only",
                "learning_enabled": False,
                "policy_checkpoint": "final_policy.pth",
                "critic_checkpoint": "final_critic.pth",
                "candidate_policy": "candidate_policy.pth",
                "candidate_critic": "candidate_critic.pth",
                "source_p2_policy": "source_p2_policy.pth",
                "source_p2_critic": "source_p2_critic.pth",
                "base_fallback_policy": "base_fallback_policy.pth",
                "base_fallback_critic": "base_fallback_critic.pth",
                "candidate_policy_sha256": policy_hash,
                "candidate_critic_sha256": critic_hash,
                "base_policy_sha256": base_policy_hash,
                "base_critic_sha256": base_critic_hash,
            },
        )
        write_training_status(
            run_directory, status="completed", iteration=0
        )
        write_artifact_manifest(run_directory)
        mark_latest_completed_run(output_root, run_directory)
    except BaseException as error:
        write_training_status(
            run_directory,
            status="failed",
            iteration=0,
            error=f"{type(error).__name__}: {error}",
        )
        write_artifact_manifest(run_directory)
        raise
    return run_directory


@dataclass(frozen=True)
class P3PairedBatch:
    """Paired physical-state contract consumed by the differential critic."""

    candidate_observation: torch.Tensor
    base_observation: torch.Tensor
    candidate_branch_state: torch.Tensor
    candidate_edge_mask: torch.Tensor
    candidate_control: torch.Tensor
    candidate_action: torch.Tensor
    base_action: torch.Tensor
    candidate_reward: torch.Tensor
    base_reward: torch.Tensor
    candidate_done: torch.Tensor
    base_done: torch.Tensor
    candidate_terminated: torch.Tensor
    base_terminated: torch.Tensor
    candidate_vehicle_collision: torch.Tensor
    base_vehicle_collision: torch.Tensor
    candidate_lane_collision: torch.Tensor
    base_lane_collision: torch.Tensor
    candidate_vehicle_risk: torch.Tensor
    base_vehicle_risk: torch.Tensor
    candidate_lane_clearance: torch.Tensor
    base_lane_clearance: torch.Tensor
    delta_reward: torch.Tensor
    delta_vehicle_collision: torch.Tensor
    delta_lane_collision: torch.Tensor
    delta_total_collision: torch.Tensor

    def summary(self, seed: int) -> Dict[str, object]:
        tensors = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        return {
            "seed": int(seed),
            "contract_version": 2,
            "common_random_numbers": True,
            "finite": all(bool(torch.isfinite(item).all()) for item in tensors),
            "candidate_observation_shape": list(
                self.candidate_observation.shape
            ),
            "base_observation_shape": list(self.base_observation.shape),
            "candidate_branch_state_shape": list(
                self.candidate_branch_state.shape
            ),
            "candidate_edge_mask_shape": list(self.candidate_edge_mask.shape),
            "candidate_action_shape": list(self.candidate_action.shape),
            "base_action_shape": list(self.base_action.shape),
            "reward_shape": list(self.delta_reward.shape),
            "done_shape": list(self.candidate_done.shape),
            "delta_reward_mean": float(self.delta_reward.float().mean().item()),
            "delta_vehicle_collision_mean": float(
                self.delta_vehicle_collision.float().mean().item()
            ),
            "delta_lane_collision_mean": float(
                self.delta_lane_collision.float().mean().item()
            ),
            "delta_total_collision_mean": float(
                self.delta_total_collision.float().mean().item()
            ),
            "candidate_vehicle_risk_mean": float(
                self.candidate_vehicle_risk.float().mean().item()
            ),
            "base_vehicle_risk_mean": float(
                self.base_vehicle_risk.float().mean().item()
            ),
            "delta_vehicle_risk_mean": float(
                (
                    self.candidate_vehicle_risk - self.base_vehicle_risk
                ).float().mean().item()
            ),
            "candidate_lane_clearance_mean": float(
                self.candidate_lane_clearance.float().mean().item()
            ),
            "base_lane_clearance_mean": float(
                self.base_lane_clearance.float().mean().item()
            ),
            "delta_lane_clearance_mean": float(
                (
                    self.candidate_lane_clearance - self.base_lane_clearance
                ).float().mean().item()
            ),
        }


@dataclass(frozen=True)
class P3SynchronizedBoundary:
    """One lock-step Candidate/Base episode boundary."""

    done: torch.Tensor
    candidate_synthetic_truncation: torch.Tensor
    base_synthetic_truncation: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.done.reshape(self.done.shape[0], -1).any(dim=-1).sum())


def synchronize_paired_transition_boundaries(
    candidate_transition,
    base_transition,
) -> P3SynchronizedBoundary:
    """Apply a union done boundary and synthetic truncation to a paired step.

    A physical termination remains a termination on the side where it occurred.
    The still-running counterpart is truncated, never falsely terminated.  Both
    transitions consequently expose the same ``done`` mask and can be reset from
    one shared random seed before the next physical step.
    """

    candidate_next = candidate_transition.get("next")
    base_next = base_transition.get("next")
    candidate_done = candidate_next.get("done").bool()
    base_done = base_next.get("done").bool()
    if candidate_done.shape != base_done.shape:
        raise ValueError("P3.3 paired done masks must have identical shapes.")
    candidate_terminated = candidate_next.get("terminated").bool()
    base_terminated = base_next.get("terminated").bool()
    if candidate_terminated.shape != candidate_done.shape:
        raise ValueError("P3.3 Candidate terminated mask does not match done.")
    if base_terminated.shape != base_done.shape:
        raise ValueError("P3.3 Base terminated mask does not match done.")
    candidate_truncated = candidate_next.get(
        "truncated", candidate_done & ~candidate_terminated
    ).bool()
    base_truncated = base_next.get(
        "truncated", base_done & ~base_terminated
    ).bool()
    if candidate_truncated.shape != candidate_done.shape:
        raise ValueError("P3.3 Candidate truncated mask does not match done.")
    if base_truncated.shape != base_done.shape:
        raise ValueError("P3.3 Base truncated mask does not match done.")

    synchronized_done = candidate_done | base_done
    candidate_synthetic = synchronized_done & ~candidate_done
    base_synthetic = synchronized_done & ~base_done
    candidate_next.set("done", synchronized_done.clone())
    base_next.set("done", synchronized_done.clone())
    candidate_next.set(
        "truncated", candidate_truncated | candidate_synthetic
    )
    base_next.set("truncated", base_truncated | base_synthetic)
    candidate_next.set("terminated", candidate_terminated)
    base_next.set("terminated", base_terminated)
    return P3SynchronizedBoundary(
        done=synchronized_done,
        candidate_synthetic_truncation=candidate_synthetic,
        base_synthetic_truncation=base_synthetic,
    )


def _collision_tensors(rollout, reward: torch.Tensor):
    shape = reward.shape[:-1]
    try:
        vehicle = rollout.get(
            ("next", "agents", "info", "is_collision_with_agents")
        ).bool()
        lane = rollout.get(
            ("next", "agents", "info", "is_collision_with_lanelets")
        ).bool()
    except KeyError:
        vehicle = torch.zeros(shape, dtype=torch.bool, device=reward.device)
        lane = torch.zeros_like(vehicle)
    if vehicle.shape == reward.shape:
        vehicle = vehicle.squeeze(-1)
    if lane.shape == reward.shape:
        lane = lane.squeeze(-1)
    if vehicle.shape != shape or lane.shape != shape:
        raise ValueError("Collision tensors do not align with paired rewards.")
    return vehicle, lane, vehicle | lane


def _agent_scalar(
    value: torch.Tensor,
    reward: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    """Normalize an info scalar to the paired reward's ``[E,T,N]`` shape."""

    expected = reward.shape[:-1]
    if value.shape == reward.shape:
        value = value.squeeze(-1)
    if value.shape != expected:
        raise ValueError(
            f"{label} must align with paired rewards; expected {expected}, "
            f"found {value.shape}."
        )
    if not value.is_floating_point():
        value = value.to(torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values.")
    return value


def _next_or_current(rollout, suffix: tuple[str, ...]) -> torch.Tensor:
    """Read transition-aligned info, retaining a compatibility fallback."""

    try:
        return rollout.get(("next", "agents", "info", *suffix))
    except KeyError:
        return rollout.get(("agents", "info", *suffix))


def _continuous_safety_tensors(rollout, reward: torch.Tensor):
    """Return dense vehicle conflict risk and geometric lane clearance.

    Vehicle risk is the strongest active CPA conflict incident to each agent.
    Multiplication by confidence prevents distant, weakly observed candidates
    from becoming artificial safety costs. Lane clearance is already
    normalized by ``3 * lane_width`` by the road environment.
    """

    urgency = _next_or_current(rollout, ("urgency",))
    confidence = _next_or_current(rollout, ("confidence",))
    pair_mask = _next_or_current(rollout, ("pair_mask",)).bool()
    if (
        urgency.shape != confidence.shape
        or urgency.shape != pair_mask.shape
        or urgency.ndim != reward.ndim
        or urgency.shape[:-1] != reward.shape[:-1]
    ):
        raise ValueError(
            "Continuous vehicle-risk tensors must have shape [E,T,N,K]."
        )
    vehicle_risk = (
        urgency.clamp(0.0, 1.0)
        * confidence.clamp(0.0, 1.0)
        * pair_mask.to(urgency.dtype)
    ).amax(dim=-1)
    left = _agent_scalar(
        _next_or_current(rollout, ("distance_left_b",)),
        reward,
        label="distance_left_b",
    )
    right = _agent_scalar(
        _next_or_current(rollout, ("distance_right_b",)),
        reward,
        label="distance_right_b",
    )
    lane_clearance = torch.minimum(left, right).clamp_min(0.0)
    return vehicle_risk, lane_clearance


def dense_edge_mask(
    neighbor_ids: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert local candidate slots into a symmetric dense edge mask."""

    if neighbor_ids.shape != pair_mask.shape or neighbor_ids.ndim < 3:
        raise ValueError("neighbor_ids and pair_mask must have shape [...,N,K].")
    if pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must be boolean.")
    n_agents = int(neighbor_ids.shape[-2])
    leading_ones = [1] * (neighbor_ids.ndim - 2)
    ego_ids = torch.arange(
        n_agents,
        device=neighbor_ids.device,
        dtype=neighbor_ids.dtype,
    ).view(*leading_ones, n_agents, 1)
    valid_ids = (neighbor_ids >= 0) & (neighbor_ids < n_agents)
    safe_ids = torch.where(valid_ids, neighbor_ids, ego_ids)
    valid = valid_ids & (safe_ids != ego_ids) & pair_mask
    directed = torch.zeros(
        *neighbor_ids.shape[:-2],
        n_agents,
        n_agents,
        dtype=torch.int64,
        device=neighbor_ids.device,
    ).scatter_add(-1, safe_ids, valid.to(torch.int64))
    result = (directed > 0) | (directed.transpose(-1, -2) > 0)
    diagonal = torch.eye(
        n_agents, dtype=torch.bool, device=neighbor_ids.device
    )
    return result & ~diagonal


def build_paired_batch(candidate_rollout, base_rollout) -> P3PairedBatch:
    """Build exact Candidate-minus-Base tensors from common-random-number runs."""

    candidate_observation = candidate_rollout.get(("agents", "observation"))
    base_observation = base_rollout.get(("agents", "observation"))
    candidate_branch_state = candidate_rollout.get(
        ("agents", "psb", "z_next_dense")
    )
    candidate_edge_mask = dense_edge_mask(
        candidate_rollout.get(("agents", "info", "neighbor_ids")).to(
            torch.long
        ),
        candidate_rollout.get(("agents", "info", "pair_mask")).bool(),
    )
    candidate_control = candidate_rollout.get(("agents", "psb", "b"))
    candidate_action = candidate_rollout.get(("agents", "action"))
    base_action = base_rollout.get(("agents", "action"))
    candidate_reward = candidate_rollout.get(("next", "agents", "reward"))
    base_reward = base_rollout.get(("next", "agents", "reward"))
    candidate_done = candidate_rollout.get(("next", "done")).bool()
    base_done = base_rollout.get(("next", "done")).bool()
    candidate_terminated = candidate_rollout.get(
        ("next", "terminated")
    ).bool()
    base_terminated = base_rollout.get(("next", "terminated")).bool()
    if candidate_action.shape != base_action.shape:
        raise ValueError("Paired Candidate/Base action shapes do not match.")
    if candidate_observation.shape != base_observation.shape:
        raise ValueError(
            "Paired Candidate/Base observation shapes do not match."
        )
    if candidate_reward.shape != base_reward.shape:
        raise ValueError("Paired Candidate/Base reward shapes do not match.")
    if candidate_done.shape != base_done.shape:
        raise ValueError("Paired Candidate/Base done shapes do not match.")
    if candidate_terminated.shape != base_terminated.shape:
        raise ValueError(
            "Paired Candidate/Base terminated shapes do not match."
        )
    candidate_vehicle, candidate_lane, candidate_total = _collision_tensors(
        candidate_rollout, candidate_reward
    )
    base_vehicle, base_lane, base_total = _collision_tensors(
        base_rollout, base_reward
    )
    candidate_vehicle_risk, candidate_lane_clearance = (
        _continuous_safety_tensors(candidate_rollout, candidate_reward)
    )
    base_vehicle_risk, base_lane_clearance = _continuous_safety_tensors(
        base_rollout, base_reward
    )
    return P3PairedBatch(
        candidate_observation=candidate_observation,
        base_observation=base_observation,
        candidate_branch_state=candidate_branch_state,
        candidate_edge_mask=candidate_edge_mask,
        candidate_control=candidate_control,
        candidate_action=candidate_action,
        base_action=base_action,
        candidate_reward=candidate_reward,
        base_reward=base_reward,
        candidate_done=candidate_done,
        base_done=base_done,
        candidate_terminated=candidate_terminated,
        base_terminated=base_terminated,
        candidate_vehicle_collision=candidate_vehicle,
        base_vehicle_collision=base_vehicle,
        candidate_lane_collision=candidate_lane,
        base_lane_collision=base_lane,
        candidate_vehicle_risk=candidate_vehicle_risk,
        base_vehicle_risk=base_vehicle_risk,
        candidate_lane_clearance=candidate_lane_clearance,
        base_lane_clearance=base_lane_clearance,
        delta_reward=candidate_reward - base_reward,
        delta_vehicle_collision=(
            candidate_vehicle.to(torch.float32) - base_vehicle.to(torch.float32)
        ),
        delta_lane_collision=(
            candidate_lane.to(torch.float32) - base_lane.to(torch.float32)
        ),
        delta_total_collision=(
            candidate_total.to(torch.float32) - base_total.to(torch.float32)
        ),
    )


def _verify_p3_run(experiment, run: Path, checkpoint_path: Optional[Path]):
    assert experiment.parent_run is not None
    manifest = _load_json(run / "deployment_manifest.json", "P3.0 manifest")
    proof = _load_json(run / "p3_0_equivalence.json", "P3.0 proof")
    resolved = _load_json(run / "psb_config_resolved.json", "P3.0 runtime")
    status = _load_json(run / "training_status.json", "P3.0 status")
    candidate = run / "candidate_policy.pth"
    checkpoint = resolve_policy_checkpoint(
        run, candidate if checkpoint_path is None else checkpoint_path
    )
    if checkpoint.name != "candidate_policy.pth":
        raise PSBConfigError("P3.0 testing accepts only candidate_policy.pth.")
    required = {
        "candidate_critic.pth",
        "source_p2_policy.pth",
        "source_p2_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
        "final_policy.pth",
        "final_critic.pth",
    }
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise PSBConfigError(f"P3.0 run is missing artifacts: {missing}.")
    candidate_hash = sha256_file(checkpoint)
    critic_hash = sha256_file(run / "candidate_critic.pth")
    source_policy_hash = sha256_file(experiment.parent_run / "candidate_policy.pth")
    source_critic_hash = sha256_file(experiment.parent_run / "candidate_critic.pth")
    base_policy_hash = sha256_file(experiment.base.policy_checkpoint)
    base_critic_hash = sha256_file(experiment.base.critic_checkpoint)
    checks = {
        "method_matches": manifest.get("method") == "psb_marl",
        "stage_matches": manifest.get("stage") == experiment.stage,
        "selection_is_safe_fallback": manifest.get("selected")
        == "base_fallback_p3_pairing_only",
        "learning_disabled": manifest.get("learning_enabled") is False,
        "packaging_completed": status.get("status") == "completed"
        and status.get("iteration") == 0,
        "proof_kind_matches": proof.get("equivalence_kind")
        == "byte_exact_p2_candidate_passthrough",
        "proof_learning_disabled": proof.get("learning_enabled") is False,
        "runtime_matches": resolved.get("runtime_config")
        == experiment.p3_runtime_config(),
        "candidate_policy_matches_source": candidate_hash == source_policy_hash,
        "candidate_critic_matches_source": critic_hash == source_critic_hash,
        "source_policy_copy_matches": sha256_file(run / "source_p2_policy.pth")
        == source_policy_hash,
        "source_critic_copy_matches": sha256_file(run / "source_p2_critic.pth")
        == source_critic_hash,
        "base_policy_matches": sha256_file(run / "base_fallback_policy.pth")
        == base_policy_hash,
        "base_critic_matches": sha256_file(run / "base_fallback_critic.pth")
        == base_critic_hash,
        "final_policy_is_base_fallback": sha256_file(run / "final_policy.pth")
        == base_policy_hash,
        "final_critic_is_base_fallback": sha256_file(run / "final_critic.pth")
        == base_critic_hash,
        "manifest_candidate_policy_hash": manifest.get(
            "candidate_policy_sha256"
        )
        == candidate_hash,
        "manifest_candidate_critic_hash": manifest.get(
            "candidate_critic_sha256"
        )
        == critic_hash,
        "proof_candidate_policy_hash": proof.get("candidate_policy_sha256")
        == candidate_hash,
        "proof_candidate_critic_hash": proof.get("candidate_critic_sha256")
        == critic_hash,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P3.0 artifact equivalence failed: {failed}")
    return checkpoint, checks


def test_p3_pairing(
    experiment,
    *,
    run_directory: Optional[Path],
    checkpoint_path: Optional[Path],
    scenario_type: Optional[str],
    max_steps: int,
    episodes: int,
    seeds: Optional[Sequence[int]],
    render: bool,
    save_simulation_video: bool,
    compare_base: bool,
    promote_if_noninferior: bool,
    psb_action_projection: Optional[str],
    report_label: Optional[str],
) -> Dict[str, object]:
    """Prove source equivalence and materialize paired delta tensors."""

    if not compare_base:
        raise ValueError("P3.0 requires --compare-base.")
    if promote_if_noninferior:
        raise ValueError("P3.0 is read-only and cannot be promoted.")
    if psb_action_projection is not None:
        raise ValueError("P3.0 action projection is fixed by its P2.1-U parent.")
    if report_label is not None and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", report_label
    ) is None:
        raise ValueError("P3.0 report label is unsafe.")
    selected_run = (
        resolve_latest_testable_run(experiment.output_root)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    checkpoint, artifact_checks = _verify_p3_run(
        experiment, selected_run, checkpoint_path
    )
    selected_seeds = tuple(seeds) if seeds is not None else (experiment.seed,)
    if not selected_seeds or any(
        type(seed) is not int or seed < 0 for seed in selected_seeds
    ):
        raise ValueError("P3.0 seeds must be non-negative integers.")
    if (
        type(episodes) is not int
        or episodes <= 0
        or type(max_steps) is not int
        or max_steps <= 1
    ):
        raise ValueError("P3.0 episodes and max_steps are invalid.")
    if render and (episodes != 1 or len(selected_seeds) != 1):
        raise ValueError("P3.0 rendering requires one episode and one seed.")
    if save_simulation_video:
        raise ValueError("P3.0 paired equivalence does not support video capture.")

    from main_testing import test_base
    from utilities.psb_marl.evaluator import (
        _paired_equivalence,
        _paired_performance,
        _rollout_summary,
    )

    assert experiment.parent_run is not None
    assert experiment.conflict_graph is not None
    p2_runtime = experiment.source_p2_runtime_config()
    candidate_rollouts = []
    source_rollouts = []
    base_rollouts = []
    source_equivalence = []
    paired_batches = []
    paired_comparisons = []
    for seed in selected_seeds:
        common = {
            "scenario_type": scenario_type,
            "max_steps": max_steps,
            "episodes": episodes,
            "seed": seed,
            "render": render,
            "opinion_pair_info_config": experiment.conflict_graph.to_dict(),
            "psb_runtime_config": p2_runtime,
        }
        candidate_td = test_base(
            experiment.output_root,
            selected_run,
            checkpoint,
            save_simulation_video=False,
            **common,
        )
        source_td = test_base(
            str(experiment.parent_run.parent.parent),
            experiment.parent_run,
            experiment.parent_run / "candidate_policy.pth",
            save_simulation_video=False,
            **{**common, "render": False},
        )
        base_td = test_base(
            str(experiment.base.run_directory),
            experiment.base.run_directory,
            experiment.base.policy_checkpoint,
            save_simulation_video=False,
            scenario_type=scenario_type,
            max_steps=max_steps,
            episodes=episodes,
            seed=seed,
            render=False,
            opinion_pair_info_config=experiment.conflict_graph.to_dict(),
        )
        candidate_summary = _rollout_summary(
            candidate_td, seed, p2_runtime_config=p2_runtime
        )
        source_summary = _rollout_summary(
            source_td, seed, p2_runtime_config=p2_runtime
        )
        base_summary = _rollout_summary(base_td, seed)
        equivalence = _paired_equivalence(candidate_td, source_td, seed)
        paired = build_paired_batch(candidate_td, base_td)
        candidate_rollouts.append(candidate_summary)
        source_rollouts.append(source_summary)
        base_rollouts.append(base_summary)
        source_equivalence.append(equivalence)
        paired_batches.append(paired.summary(seed))
        paired_comparisons.append(
            _paired_performance(candidate_summary, base_summary)
        )

    exact_source = all(
        item["shape_match"]
        and item["actions_exactly_equal"]
        and item["rewards_exactly_equal"]
        for item in source_equivalence
    )
    paired_contract = all(item["finite"] for item in paired_batches)
    passed = exact_source and paired_contract
    report_name = "p3_0_paired_equivalence.json"
    if report_label is not None:
        report_name = f"{Path(report_name).stem}_{report_label}.json"
    report = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": experiment.stage,
        "passed": bool(passed),
        "run_directory": str(selected_run),
        "checkpoint": str(checkpoint),
        "source_p2_run": str(experiment.parent_run),
        "report_label": report_label,
        "learning_enabled": False,
        "comparison_mode": "three_way_common_random_number_equivalence",
        "evaluation_protocol": {
            "scenario_type": experiment.base_run_config["scenario_type"]
            if scenario_type is None
            else scenario_type,
            "max_steps": int(max_steps),
            "episodes": int(episodes),
            "seeds": [int(seed) for seed in selected_seeds],
            "compare_base": True,
        },
        "artifact_checks": artifact_checks,
        "source_equivalence_passed": bool(exact_source),
        "paired_contract_passed": bool(paired_contract),
        "source_equivalence": source_equivalence,
        "paired_batches": paired_batches,
        "candidate_rollouts": candidate_rollouts,
        "source_p2_rollouts": source_rollouts,
        "base_rollouts": base_rollouts,
        "paired_comparisons": paired_comparisons,
    }
    atomic_write_json(selected_run / report_name, report)
    write_artifact_manifest(selected_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise RuntimeError("P3.0 failed source or paired-batch equivalence.")
    return report
