"""Actor-frozen P3.1 paired collection and differential critic fitting."""

from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

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
from utilities.psb_marl.checkpoint import copy_checkpoint_exact, sha256_file
from utilities.psb_marl.config import PSBConfigError
from utilities.psb_marl.p3_critic import (
    TARGET_CHANNELS,
    BaseRelativeDifferentialCritic,
)
from utilities.psb_marl.p3_pairing import P3PairedBatch, build_paired_batch


@dataclass(frozen=True)
class P3CriticSamples:
    candidate_observation: torch.Tensor
    base_observation: torch.Tensor
    candidate_z: torch.Tensor
    edge_mask: torch.Tensor
    target: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.target.shape[0])

    def select(self, indices: torch.Tensor) -> "P3CriticSamples":
        return P3CriticSamples(
            candidate_observation=self.candidate_observation[indices],
            base_observation=self.base_observation[indices],
            candidate_z=self.candidate_z[indices],
            edge_mask=self.edge_mask[indices],
            target=self.target[indices],
        )

    def to(self, device: torch.device) -> "P3CriticSamples":
        return P3CriticSamples(
            candidate_observation=self.candidate_observation.detach().to(device),
            base_observation=self.base_observation.detach().to(device),
            candidate_z=self.candidate_z.detach().to(device),
            edge_mask=self.edge_mask.detach().to(device),
            target=self.target.detach().to(device),
        )


def _discounted_return(
    signal: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Monte Carlo return with independent Candidate/Base episode boundaries."""

    if signal.ndim != 4:
        raise ValueError("P3.1 return signal must have shape [E,T,N,C].")
    if done.shape != terminated.shape or done.shape[:2] != signal.shape[:2]:
        raise ValueError("P3.1 episode boundary tensors do not align.")
    boundary = (done | terminated).reshape(*done.shape[:2], -1).any(dim=-1)
    result = torch.empty_like(signal)
    running = torch.zeros_like(signal[:, 0])
    for time_index in range(signal.shape[1] - 1, -1, -1):
        continuation = (~boundary[:, time_index]).to(signal.dtype).view(
            signal.shape[0], 1, 1
        )
        running = signal[:, time_index] + float(gamma) * continuation * running
        result[:, time_index] = running
    return result


def paired_critic_samples(
    batch: P3PairedBatch,
    *,
    gamma: float,
    energy_coefficient: float,
    lane_safety_margin: float,
) -> P3CriticSamples:
    """Turn a paired rollout into reward and dense safety-return targets."""

    if lane_safety_margin <= 0.0:
        raise ValueError("P3.1 lane_safety_margin must be positive.")

    n_agents = int(batch.candidate_observation.shape[-2])
    upper = torch.triu(
        torch.ones(
            n_agents,
            n_agents,
            dtype=torch.bool,
            device=batch.candidate_control.device,
        ),
        diagonal=1,
    )
    edge_count = float(n_agents * (n_agents - 1) // 2)
    energy = (
        batch.candidate_control.square()[..., upper].sum(dim=-1) / edge_count
    )
    candidate_augmented_reward = batch.candidate_reward - (
        float(energy_coefficient) * energy.unsqueeze(-1).unsqueeze(-1)
    )
    candidate_reward_return = _discounted_return(
        candidate_augmented_reward,
        batch.candidate_done,
        batch.candidate_terminated,
        gamma,
    )
    base_reward_return = _discounted_return(
        batch.base_reward,
        batch.base_done,
        batch.base_terminated,
        gamma,
    )
    candidate_vehicle_cost = torch.maximum(
        batch.candidate_vehicle_risk,
        batch.candidate_vehicle_collision.to(torch.float32),
    )
    base_vehicle_cost = torch.maximum(
        batch.base_vehicle_risk,
        batch.base_vehicle_collision.to(torch.float32),
    )
    candidate_vehicle_return = _discounted_return(
        candidate_vehicle_cost.unsqueeze(-1),
        batch.candidate_done,
        batch.candidate_terminated,
        gamma,
    )
    base_vehicle_return = _discounted_return(
        base_vehicle_cost.unsqueeze(-1),
        batch.base_done,
        batch.base_terminated,
        gamma,
    )
    candidate_lane_cost = torch.maximum(
        (
            (float(lane_safety_margin) - batch.candidate_lane_clearance)
            / float(lane_safety_margin)
        ).clamp(0.0, 1.0),
        batch.candidate_lane_collision.to(torch.float32),
    )
    base_lane_cost = torch.maximum(
        (
            (float(lane_safety_margin) - batch.base_lane_clearance)
            / float(lane_safety_margin)
        ).clamp(0.0, 1.0),
        batch.base_lane_collision.to(torch.float32),
    )
    candidate_lane_return = _discounted_return(
        candidate_lane_cost.unsqueeze(-1),
        batch.candidate_done,
        batch.candidate_terminated,
        gamma,
    )
    base_lane_return = _discounted_return(
        base_lane_cost.unsqueeze(-1),
        batch.base_done,
        batch.base_terminated,
        gamma,
    )
    target = torch.cat(
        (
            candidate_reward_return - base_reward_return,
            candidate_vehicle_return - base_vehicle_return,
            candidate_lane_return - base_lane_return,
        ),
        dim=-1,
    )
    leading_count = int(
        batch.candidate_observation.shape[0]
        * batch.candidate_observation.shape[1]
    )
    samples = P3CriticSamples(
        candidate_observation=batch.candidate_observation.reshape(
            leading_count, *batch.candidate_observation.shape[2:]
        ).detach(),
        base_observation=batch.base_observation.reshape(
            leading_count, *batch.base_observation.shape[2:]
        ).detach(),
        candidate_z=batch.candidate_branch_state.reshape(
            leading_count, *batch.candidate_branch_state.shape[2:]
        ).detach(),
        edge_mask=batch.candidate_edge_mask.reshape(
            leading_count, *batch.candidate_edge_mask.shape[2:]
        ).detach(),
        target=target.reshape(leading_count, *target.shape[2:]).detach(),
    )
    tensors = tuple(getattr(samples, name) for name in samples.__dataclass_fields__)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError("P3.1 paired critic samples contain non-finite values.")
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("P3.1 paired critic samples must be detached.")
    return samples


def concatenate_samples(parts: Sequence[P3CriticSamples]) -> P3CriticSamples:
    if not parts:
        raise ValueError("P3.1 requires at least one paired sample set.")
    return P3CriticSamples(
        **{
            name: torch.cat([getattr(part, name) for part in parts], dim=0)
            for name in P3CriticSamples.__dataclass_fields__
        }
    )


def _collect_pair(
    experiment,
    run: Path,
    seed: int,
    *,
    scenario: Optional[str] = None,
    max_steps: Optional[int] = None,
    episodes: Optional[int] = None,
):
    from main_testing import test_base

    assert experiment.conflict_graph is not None
    assert experiment.source_p2_runtime is not None
    assert experiment.differential_critic is not None
    config = experiment.differential_critic
    common = {
        "scenario_type": (
            config.collection_scenario if scenario is None else scenario
        ),
        "max_steps": config.max_steps if max_steps is None else max_steps,
        "episodes": config.episodes if episodes is None else episodes,
        "seed": int(seed),
        "render": False,
    }
    with torch.no_grad():
        candidate = test_base(
            experiment.output_root,
            run,
            run / "candidate_policy.pth",
            opinion_pair_info_config=experiment.conflict_graph.to_dict(),
            psb_runtime_config=experiment.source_p2_runtime,
            save_simulation_video=False,
            **common,
        )
        base = test_base(
            str(experiment.base.run_directory),
            experiment.base.run_directory,
            experiment.base.policy_checkpoint,
            opinion_pair_info_config=experiment.conflict_graph.to_dict(),
            save_simulation_video=False,
            **common,
        )
    return build_paired_batch(candidate, base), candidate, base


def _predict(
    model: BaseRelativeDifferentialCritic,
    samples: P3CriticSamples,
) -> torch.Tensor:
    return model(
        samples.candidate_observation,
        samples.base_observation,
        samples.candidate_z,
        samples.edge_mask,
    )


def critic_metrics(
    model: BaseRelativeDifferentialCritic,
    samples: P3CriticSamples,
    *,
    minibatch_size: int,
    huber_delta: float,
) -> Dict[str, object]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, samples.count, minibatch_size):
            indices = torch.arange(
                start,
                min(start + minibatch_size, samples.count),
                device=samples.target.device,
            )
            predictions.append(_predict(model, samples.select(indices)))
    prediction = torch.cat(predictions, dim=0)
    error = prediction - samples.target
    normalized_error = error / model.target_scale.view(1, 1, -1)
    normalized_huber = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=float(huber_delta),
    )
    channel_metrics = {}
    for index, name in enumerate(TARGET_CHANNELS):
        target = samples.target[..., index]
        channel_error = error[..., index]
        normalized_channel_error = (
            channel_error / model.target_scale[index]
        )
        channel_huber = F.smooth_l1_loss(
            normalized_channel_error,
            torch.zeros_like(normalized_channel_error),
            beta=float(huber_delta),
        )
        target_variance = target.var(unbiased=False)
        error_variance = channel_error.var(unbiased=False)
        explained_variance = (
            0.0
            if float(target_variance) <= 1e-12
            else 1.0 - float((error_variance / target_variance).item())
        )
        nonzero = target.abs() > 1e-8
        sign_accuracy = (
            1.0
            if not bool(nonzero.any())
            else float(
                (
                    torch.sign(prediction[..., index][nonzero])
                    == torch.sign(target[nonzero])
                )
                .float()
                .mean()
                .item()
            )
        )
        channel_metrics[name] = {
            "mae": float(channel_error.abs().mean().item()),
            "rmse": float(channel_error.square().mean().sqrt().item()),
            "normalized_huber": float(channel_huber.item()),
            "target_mean": float(target.mean().item()),
            "target_std": float(target.std(unbiased=False).item()),
            "explained_variance": explained_variance,
            "nonzero_sign_accuracy": sign_accuracy,
        }
    return {
        "sample_count": samples.count,
        "normalized_huber": float(normalized_huber.item()),
        "finite": bool(torch.isfinite(prediction).all()),
        "channels": channel_metrics,
    }


def critic_channel_quality(
    metrics: Dict[str, object],
    *,
    baseline_metrics: Dict[str, object],
    minimum_target_std: float,
    minimum_explained_variance: float,
) -> Dict[str, object]:
    """Require every head to improve its own fitted loss over a constant."""

    channels = metrics.get("channels")
    baseline_channels = baseline_metrics.get("channels")
    if not isinstance(channels, dict) or not isinstance(
        baseline_channels, dict
    ):
        raise ValueError("P3.1 critic metrics are missing channel results.")
    results: Dict[str, object] = {}
    for name in TARGET_CHANNELS:
        channel = channels.get(name)
        baseline_channel = baseline_channels.get(name)
        if not isinstance(channel, dict) or not isinstance(
            baseline_channel, dict
        ):
            raise ValueError(f"P3.1 critic metrics are missing {name}.")
        target_std = float(channel["target_std"])
        explained_variance = float(channel["explained_variance"])
        normalized_huber = float(channel["normalized_huber"])
        baseline_normalized_huber = float(
            baseline_channel["normalized_huber"]
        )
        relative_improvement = (
            0.0
            if baseline_normalized_huber <= 1e-12
            else (
                baseline_normalized_huber - normalized_huber
            )
            / baseline_normalized_huber
        )
        informative = target_std >= float(minimum_target_std)
        predictable = explained_variance >= float(minimum_explained_variance)
        loss_noninferior = normalized_huber <= baseline_normalized_huber
        results[name] = {
            "passed": informative and loss_noninferior,
            "target_informative": informative,
            "loss_noninferiority_passed": loss_noninferior,
            "normalized_huber": normalized_huber,
            "baseline_normalized_huber": baseline_normalized_huber,
            "relative_huber_improvement": relative_improvement,
            "explained_variance_passed": predictable,
            "target_std": target_std,
            "minimum_target_std": float(minimum_target_std),
            "explained_variance": explained_variance,
            "minimum_explained_variance": float(
                minimum_explained_variance
            ),
        }
    return {
        "passed": all(bool(item["passed"]) for item in results.values()),
        "channels": results,
    }


def load_differential_critic(
    path: Path,
    *,
    device: torch.device = torch.device("cpu"),
    load_weights: bool = True,
) -> tuple[BaseRelativeDifferentialCritic, Dict[str, object]]:
    payload = torch.load(Path(path), map_location=device)
    supported_stages = {
        "p3_differential_critic",
        "p3_paired_differential_primal_dual_ppo",
        "p5_joint_psb_marl",
    }
    if (
        not isinstance(payload, dict)
        or payload.get("method") != "psb_marl"
        or payload.get("stage") not in supported_stages
    ):
        raise ValueError(
            "Checkpoint is not a supported PSB differential critic."
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Differential checkpoint is missing model_config.")
    if model_config.get("target_channels") != list(TARGET_CHANNELS):
        raise ValueError("Differential checkpoint target channels do not match.")
    model = BaseRelativeDifferentialCritic(
        observation_dim=int(model_config["observation_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_sizes=tuple(model_config["hidden_sizes"]),
    ).to(device)
    if load_weights:
        model.load_state_dict(payload["critic_state"], strict=True)
    model.eval()
    return model, payload


def _save_critic(
    path: Path,
    *,
    model: BaseRelativeDifferentialCritic,
    runtime: Dict[str, object],
    certification: Dict[str, object],
) -> None:
    payload = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": "p3_differential_critic",
        "model_config": model.model_config(),
        "critic_state": model.state_dict(),
        "runtime_config": runtime,
        "certification": certification,
    }
    temporary = Path(path).with_name(f".{Path(path).name}.saving")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_p31(
    experiment,
    *,
    resume_checkpoint: Optional[Path],
    iterations_override: Optional[int],
) -> Path:
    """Collect fixed-policy pairs and fit only the differential critic."""

    if resume_checkpoint is not None:
        raise PSBConfigError("P3.1 does not support resume.")
    if iterations_override is not None:
        raise PSBConfigError(
            "P3.1 uses a locked critic epoch budget; --iterations is invalid."
        )
    assert experiment.parent_run is not None
    assert experiment.differential_critic is not None
    assert experiment.source_p2_runtime is not None
    config = experiment.differential_critic
    random.seed(experiment.effective_training_seed)
    np.random.seed(experiment.effective_training_seed)
    torch.manual_seed(experiment.effective_training_seed)

    output_root = str(Path(experiment.output_root).expanduser().resolve())
    run = create_run_directory(
        output_root=output_root,
        method="psb-p3-critic",
        seed=experiment.effective_training_seed,
    )
    resolved_base = dict(experiment.base_run_config)
    resolved_base.update(
        {
            "seed": experiment.effective_training_seed,
            "where_to_save": str(run) + os.sep,
            "output_root": output_root,
            "run_id": run.name,
            "artifact_logging_enabled": True,
        }
    )
    initialize_run(
        run_directory=run,
        source_config=experiment.source_config,
        resolved_config=resolved_base,
        method="psb_marl",
        stage=experiment.stage,
    )
    started = time.time()
    try:
        source_policy = experiment.parent_run / "candidate_policy.pth"
        source_p2_critic = experiment.parent_run / "candidate_critic.pth"
        actor_hash = copy_checkpoint_exact(
            source_policy, run / "candidate_policy.pth"
        )
        copy_checkpoint_exact(
            source_policy, run / "source_p2_policy.pth"
        )
        source_p2_critic_hash = copy_checkpoint_exact(
            source_p2_critic, run / "source_p2_critic.pth"
        )
        base_policy_hash = copy_checkpoint_exact(
            experiment.base.policy_checkpoint,
            run / "base_fallback_policy.pth",
        )
        base_critic_hash = copy_checkpoint_exact(
            experiment.base.critic_checkpoint,
            run / "base_fallback_critic.pth",
        )
        copy_checkpoint_exact(
            experiment.base.policy_checkpoint, run / "final_policy.pth"
        )
        copy_checkpoint_exact(
            experiment.base.critic_checkpoint, run / "final_critic.pth"
        )
        runtime = experiment.p31_runtime_config()
        atomic_write_json(
            run / "psb_config_resolved.json",
            {
                **experiment.source_config,
                "resolved_base_config": str(experiment.base_config_path),
                "resolved_base_run": str(experiment.base.run_directory),
                "resolved_parent_run": str(experiment.parent_run),
                "runtime_config": runtime,
            },
        )

        energy_coefficient = float(
            experiment.source_p2_runtime["training"]["energy_coefficient"]
        )
        training_parts = []
        validation_parts = []
        collection = []
        for split, seeds, destination in (
            ("training", config.training_seeds, training_parts),
            ("validation", config.validation_seeds, validation_parts),
        ):
            for seed in seeds:
                print(f"[INFO] P3.1 collecting {split} pair for seed {seed}.")
                paired, _, _ = _collect_pair(experiment, run, seed)
                samples = paired_critic_samples(
                    paired,
                    gamma=config.gamma,
                    energy_coefficient=energy_coefficient,
                    lane_safety_margin=config.lane_safety_margin,
                )
                destination.append(samples.to(torch.device("cpu")))
                paired_summary = paired.summary(seed)
                collection.append(
                    {
                        "split": split,
                        "seed": seed,
                        "sample_count": samples.count,
                        "paired_batch": paired_summary,
                    }
                )
        training = concatenate_samples(training_parts)
        validation = concatenate_samples(validation_parts)
        dataset_path = run / "paired_critic_dataset.pt"
        temporary_dataset = dataset_path.with_name(
            f".{dataset_path.name}.saving"
        )
        torch.save(
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "runtime_config": runtime,
                "training": {
                    name: getattr(training, name)
                    for name in training.__dataclass_fields__
                },
                "validation": {
                    name: getattr(validation, name)
                    for name in validation.__dataclass_fields__
                },
            },
            temporary_dataset,
        )
        os.replace(temporary_dataset, dataset_path)
        print(
            "[INFO] P3.1 paired dataset saved: "
            f"training={training.count}, validation={validation.count}."
        )
        observation_dim = int(training.candidate_observation.shape[-1])
        if validation.candidate_observation.shape[-1] != observation_dim:
            raise RuntimeError("P3.1 train/validation observation dimensions differ.")
        device = torch.device(str(experiment.base_run_config["device"]))
        training = training.to(device)
        validation = validation.to(device)
        model = BaseRelativeDifferentialCritic(
            observation_dim=observation_dim,
            embedding_dim=config.embedding_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(device)
        center = training.target.mean(dim=(0, 1))
        scale = training.target.std(dim=(0, 1), unbiased=False).clamp_min(
            config.target_scale_floor
        )
        model.set_target_normalization(center, scale)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        baseline_metrics = critic_metrics(
            model,
            validation,
            minibatch_size=config.minibatch_size,
            huber_delta=config.huber_delta,
        )
        best_loss = float(baseline_metrics["normalized_huber"])
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale_epochs = 0
        epoch_metrics = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(experiment.effective_training_seed)
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(training.count, generator=generator)
            train_loss_sum = 0.0
            train_sample_count = 0
            gradient_norm_sum = 0.0
            update_count = 0
            for start in range(0, training.count, config.minibatch_size):
                indices = order[start : start + config.minibatch_size].to(device)
                mini_batch = training.select(indices)
                prediction = _predict(model, mini_batch)
                normalized_error = (
                    prediction - mini_batch.target
                ) / model.target_scale.view(1, 1, -1)
                loss = F.smooth_l1_loss(
                    normalized_error,
                    torch.zeros_like(normalized_error),
                    beta=config.huber_delta,
                )
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                optimizer.step()
                count = int(indices.numel())
                train_loss_sum += float(loss.detach().item()) * count
                train_sample_count += count
                gradient_norm_sum += float(gradient_norm)
                update_count += 1
            validation_metrics = critic_metrics(
                model,
                validation,
                minibatch_size=config.minibatch_size,
                huber_delta=config.huber_delta,
            )
            validation_loss = float(validation_metrics["normalized_huber"])
            epoch_metrics.append(
                {
                    "epoch": epoch,
                    "training_normalized_huber": train_loss_sum
                    / train_sample_count,
                    "validation_normalized_huber": validation_loss,
                    "gradient_norm": gradient_norm_sum / update_count,
                }
            )
            print(
                "[INFO] P3.1 critic epoch "
                f"{epoch}/{config.epochs}: "
                f"train={train_loss_sum / train_sample_count:.6f}, "
                f"validation={validation_loss:.6f}."
            )
            if validation_loss < best_loss - 1e-10:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            write_training_status(run, status="running", iteration=epoch)
            if stale_epochs >= config.early_stopping_patience:
                break
        model.load_state_dict(best_state, strict=True)
        training_metrics = critic_metrics(
            model,
            training,
            minibatch_size=config.minibatch_size,
            huber_delta=config.huber_delta,
        )
        validation_metrics = critic_metrics(
            model,
            validation,
            minibatch_size=config.minibatch_size,
            huber_delta=config.huber_delta,
        )
        baseline_loss = float(baseline_metrics["normalized_huber"])
        achieved_improvement = (
            0.0
            if baseline_loss <= 1e-12
            else (baseline_loss - best_loss) / baseline_loss
        )
        actor_unchanged = sha256_file(run / "candidate_policy.pth") == actor_hash
        channel_quality = critic_channel_quality(
            validation_metrics,
            baseline_metrics=baseline_metrics,
            minimum_target_std=config.minimum_target_std,
            minimum_explained_variance=(
                config.minimum_channel_explained_variance
            ),
        )
        certification = {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": experiment.stage,
            "passed": bool(
                actor_unchanged
                and validation_metrics["finite"]
                and channel_quality["passed"]
                and achieved_improvement
                >= config.required_relative_improvement
            ),
            "actor_bytes_unchanged": actor_unchanged,
            "actor_learning_enabled": False,
            "dual_learning_enabled": False,
            "baseline_validation_normalized_huber": baseline_loss,
            "best_validation_normalized_huber": best_loss,
            "best_epoch": best_epoch,
            "achieved_relative_improvement": achieved_improvement,
            "required_relative_improvement": (
                config.required_relative_improvement
            ),
            "channel_quality": channel_quality,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
        }
        _save_critic(
            run / "candidate_critic.pth",
            model=model,
            runtime=runtime,
            certification=certification,
        )
        critic_hash = copy_checkpoint_exact(
            run / "candidate_critic.pth",
            run / "differential_critic.pth",
        )
        _save_critic(
            run / "final_checkpoint.pt",
            model=model,
            runtime=runtime,
            certification=certification,
        )
        atomic_write_json(run / "p3_1_certification.json", certification)
        atomic_write_json(
            run / "metrics.json",
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "target_channels": list(TARGET_CHANNELS),
                "target_center": center.detach().cpu().tolist(),
                "target_scale": scale.detach().cpu().tolist(),
                "collection": collection,
                "epochs": epoch_metrics,
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
            },
        )
        atomic_write_json(
            run / "deployment_manifest.json",
            {
                "schema_version": 1,
                "method": "psb_marl",
                "stage": experiment.stage,
                "selected": "base_fallback_p3_critic_only",
                "actor_learning_enabled": False,
                "dual_learning_enabled": False,
                "policy_checkpoint": "final_policy.pth",
                "critic_checkpoint": "final_critic.pth",
                "candidate_policy": "candidate_policy.pth",
                "candidate_critic": "candidate_critic.pth",
                "differential_critic": "differential_critic.pth",
                "source_p2_policy": "source_p2_policy.pth",
                "source_p2_critic": "source_p2_critic.pth",
                "base_fallback_policy": "base_fallback_policy.pth",
                "base_fallback_critic": "base_fallback_critic.pth",
                "candidate_policy_sha256": actor_hash,
                "candidate_critic_sha256": critic_hash,
                "source_p2_critic_sha256": source_p2_critic_hash,
                "base_policy_sha256": base_policy_hash,
                "base_critic_sha256": base_critic_hash,
                "critic_certification_passed": certification["passed"],
            },
        )
        atomic_write_json(
            run / "timing.json",
            {
                "schema_version": 1,
                "total_seconds": time.time() - started,
                "collection_rollout_count": 2
                * (len(config.training_seeds) + len(config.validation_seeds)),
                "epochs_completed": len(epoch_metrics),
            },
        )
        write_training_status(
            run, status="completed", iteration=len(epoch_metrics)
        )
        write_artifact_manifest(run)
        mark_latest_completed_run(output_root, run)
    except BaseException as error:
        write_training_status(
            run,
            status="failed",
            iteration=None,
            error=f"{type(error).__name__}: {error}",
        )
        write_artifact_manifest(run)
        raise
    return run


def _load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _verify_p31_run(experiment, run: Path, checkpoint_path: Optional[Path]):
    assert experiment.parent_run is not None
    assert experiment.differential_critic is not None
    manifest = _load_json(run / "deployment_manifest.json", "P3.1 manifest")
    certification = _load_json(
        run / "p3_1_certification.json", "P3.1 certification"
    )
    resolved = _load_json(run / "psb_config_resolved.json", "P3.1 runtime")
    status = _load_json(run / "training_status.json", "P3.1 status")
    candidate = run / "candidate_policy.pth"
    checkpoint = resolve_policy_checkpoint(
        run, candidate if checkpoint_path is None else checkpoint_path
    )
    if checkpoint.name != "candidate_policy.pth":
        raise PSBConfigError("P3.1 testing accepts only candidate_policy.pth.")
    required = {
        "candidate_critic.pth",
        "differential_critic.pth",
        "source_p2_policy.pth",
        "source_p2_critic.pth",
        "base_fallback_policy.pth",
        "base_fallback_critic.pth",
        "final_policy.pth",
        "final_critic.pth",
    }
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise PSBConfigError(f"P3.1 run is missing artifacts: {missing}.")
    actor_hash = sha256_file(checkpoint)
    source_actor_hash = sha256_file(
        experiment.parent_run / "candidate_policy.pth"
    )
    critic_hash = sha256_file(run / "candidate_critic.pth")
    base_policy_hash = sha256_file(experiment.base.policy_checkpoint)
    base_critic_hash = sha256_file(experiment.base.critic_checkpoint)
    model, payload = load_differential_critic(run / "candidate_critic.pth")
    checks = {
        "method_matches": manifest.get("method") == "psb_marl",
        "stage_matches": manifest.get("stage") == experiment.stage,
        "selection_is_safe_fallback": manifest.get("selected")
        == "base_fallback_p3_critic_only",
        "actor_learning_disabled": manifest.get("actor_learning_enabled")
        is False,
        "dual_learning_disabled": manifest.get("dual_learning_enabled")
        is False,
        "training_completed": status.get("status") == "completed"
        and type(status.get("iteration")) is int
        and 0 < status.get("iteration") <= experiment.differential_critic.epochs,
        "runtime_matches": resolved.get("runtime_config")
        == experiment.p31_runtime_config(),
        "actor_matches_p3_source": actor_hash == source_actor_hash,
        "source_actor_copy_matches": sha256_file(run / "source_p2_policy.pth")
        == source_actor_hash,
        "critic_copy_matches": sha256_file(run / "differential_critic.pth")
        == critic_hash,
        "manifest_actor_hash_matches": manifest.get(
            "candidate_policy_sha256"
        )
        == actor_hash,
        "manifest_critic_hash_matches": manifest.get(
            "candidate_critic_sha256"
        )
        == critic_hash,
        "critic_certified": certification.get("passed") is True,
        "payload_certified": isinstance(payload.get("certification"), dict)
        and payload["certification"].get("passed") is True,
        "final_policy_is_base": sha256_file(run / "final_policy.pth")
        == base_policy_hash,
        "final_critic_is_base": sha256_file(run / "final_critic.pth")
        == base_critic_hash,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P3.1 artifact verification failed: {failed}")
    return checkpoint, model, payload, checks


def test_p31(
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
    """Evaluate critic calibration on unseen paired seeds without Actor updates."""

    if not compare_base:
        raise ValueError("P3.1 requires --compare-base.")
    if render or save_simulation_video:
        raise ValueError("P3.1 paired critic validation requires --no-render.")
    if promote_if_noninferior:
        raise ValueError("P3.1 is critic-only and cannot promote the Actor.")
    if psb_action_projection is not None:
        raise ValueError("P3.1 action projection is fixed by its P2.1-U source.")
    if report_label is not None and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", report_label
    ) is None:
        raise ValueError("P3.1 report label is unsafe.")
    if type(max_steps) is not int or max_steps <= 1:
        raise ValueError("P3.1 max_steps must be an integer greater than 1.")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("P3.1 episodes must be a positive integer.")
    assert experiment.differential_critic is not None
    assert experiment.source_p2_runtime is not None
    config = experiment.differential_critic
    selected_seeds = tuple(seeds) if seeds is not None else ()
    if len(selected_seeds) < 2 or any(
        type(seed) is not int or seed < 0 for seed in selected_seeds
    ):
        raise ValueError(
            "P3.1 requires at least two explicit non-negative holdout seeds."
        )
    consumed = set(config.training_seeds) | set(config.validation_seeds)
    if consumed & set(selected_seeds):
        raise ValueError("P3.1 manual test seeds must be unseen by critic training.")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("P3.1 manual test seeds must be distinct.")
    selected_run = (
        resolve_latest_testable_run(experiment.output_root)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    checkpoint, model, payload, artifact_checks = _verify_p31_run(
        experiment, selected_run, checkpoint_path
    )
    device = next(model.parameters()).device
    baseline = BaseRelativeDifferentialCritic(
        observation_dim=model.observation_dim,
        embedding_dim=model.embedding_dim,
        hidden_sizes=model.hidden_sizes,
    ).to(device)
    baseline.set_target_normalization(model.target_center, model.target_scale)
    energy_coefficient = float(
        experiment.source_p2_runtime["training"]["energy_coefficient"]
    )
    from utilities.psb_marl.evaluator import (
        _paired_performance,
        _rollout_summary,
    )

    parts = []
    paired_batches = []
    candidate_rollouts = []
    base_rollouts = []
    comparisons = []
    per_seed_critic_metrics = []
    scenario = config.collection_scenario if scenario_type is None else scenario_type
    for seed in selected_seeds:
        paired, candidate_td, base_td = _collect_pair(
            experiment,
            selected_run,
            seed,
            scenario=scenario,
            max_steps=max_steps,
            episodes=episodes,
        )
        samples = paired_critic_samples(
            paired,
            gamma=config.gamma,
            energy_coefficient=energy_coefficient,
            lane_safety_margin=config.lane_safety_margin,
        ).to(device)
        candidate_summary = _rollout_summary(
            candidate_td,
            seed,
            p2_runtime_config=experiment.source_p2_runtime,
        )
        base_summary = _rollout_summary(base_td, seed)
        parts.append(samples.to(torch.device("cpu")))
        paired_batches.append(paired.summary(seed))
        candidate_rollouts.append(candidate_summary)
        base_rollouts.append(base_summary)
        comparisons.append(_paired_performance(candidate_summary, base_summary))
        per_seed_critic_metrics.append(
            {
                "seed": seed,
                **critic_metrics(
                    model,
                    samples,
                    minibatch_size=config.minibatch_size,
                    huber_delta=config.huber_delta,
                ),
            }
        )
    holdout = concatenate_samples(parts).to(device)
    holdout_metrics = critic_metrics(
        model,
        holdout,
        minibatch_size=config.minibatch_size,
        huber_delta=config.huber_delta,
    )
    baseline_metrics = critic_metrics(
        baseline,
        holdout,
        minibatch_size=config.minibatch_size,
        huber_delta=config.huber_delta,
    )
    baseline_loss = float(baseline_metrics["normalized_huber"])
    holdout_loss = float(holdout_metrics["normalized_huber"])
    relative_improvement = (
        0.0
        if baseline_loss <= 1e-12
        else (baseline_loss - holdout_loss) / baseline_loss
    )
    holdout_channel_quality = critic_channel_quality(
        holdout_metrics,
        baseline_metrics=baseline_metrics,
        minimum_target_std=config.minimum_target_std,
        minimum_explained_variance=(
            config.minimum_channel_explained_variance
        ),
    )
    proximal = experiment.source_p2_runtime["proximal"]
    structural_checks = {
        "paired_batches_finite": all(item["finite"] for item in paired_batches),
        "rollouts_finite": all(
            item["nonfinite_action_count"] == 0
            and item["nonfinite_reward_count"] == 0
            and item.get("nonfinite_z_count", 0) == 0
            for item in candidate_rollouts
        ),
        "antisymmetry_exact": all(
            item.get("max_antisymmetry_error") == 0.0
            for item in candidate_rollouts
        ),
        "proximal_residual_bounded": all(
            item.get("max_root_residual", float("inf"))
            <= float(proximal["residual_tolerance"])
            for item in candidate_rollouts
        ),
        "proximal_denominator_positive": all(
            item.get("min_root_denominator", 0.0) > 0.0
            for item in candidate_rollouts
        ),
        "sector_bound_respected": all(
            item.get("rollout_sector_bound_max_violation", float("inf"))
            <= 1e-7
            for item in candidate_rollouts
        ),
        "steering_unchanged": all(
            item.get("rollout_delta_steering_abs_max") == 0.0
            for item in candidate_rollouts
        ),
        "scale_unchanged": all(
            item.get("rollout_scale_matches_base_exactly") is True
            and item.get("rollout_delta_log_scale_abs_max") == 0.0
            for item in candidate_rollouts
        ),
    }
    critic_passed = bool(
        holdout_metrics["finite"]
        and holdout_channel_quality["passed"]
        and relative_improvement >= config.required_relative_improvement
    )
    passed = all(structural_checks.values()) and critic_passed
    report_name = "p3_1_manual_validation.json"
    if report_label is not None:
        report_name = f"{Path(report_name).stem}_{report_label}.json"
    report = {
        "schema_version": 1,
        "method": "psb_marl",
        "stage": experiment.stage,
        "passed": passed,
        "run_directory": str(selected_run),
        "checkpoint": str(checkpoint),
        "report_label": report_label,
        "actor_learning_enabled": False,
        "dual_learning_enabled": False,
        "deployment": "base_fallback",
        "actor_noninferiority": "inherited_from_byte_exact_p2_2_r_source",
        "evaluation_protocol": {
            "scenario_type": scenario,
            "max_steps": max_steps,
            "episodes": episodes,
            "seeds": list(selected_seeds),
            "compare_base": True,
        },
        "artifact_checks": artifact_checks,
        "structural_checks": structural_checks,
        "critic_passed": critic_passed,
        "critic_holdout_metrics": holdout_metrics,
        "constant_baseline_metrics": baseline_metrics,
        "critic_relative_improvement": relative_improvement,
        "required_relative_improvement": config.required_relative_improvement,
        "critic_channel_quality": holdout_channel_quality,
        "per_seed_critic_metrics": per_seed_critic_metrics,
        "paired_batches": paired_batches,
        "candidate_rollouts": candidate_rollouts,
        "base_rollouts": base_rollouts,
        "paired_comparisons": comparisons,
        "training_certification": payload["certification"],
    }
    atomic_write_json(selected_run / report_name, report)
    write_artifact_manifest(selected_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise RuntimeError(
            "P3.1 failed unseen critic calibration or structural checks; "
            f"see {report_name}."
        )
    return report
