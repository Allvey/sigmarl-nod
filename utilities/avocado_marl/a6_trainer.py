"""A6 one-step PPO trainer for bounded AVOCADO cooperation correction."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torchrl.envs.utils import step_mdp
from torchrl.modules import MultiAgentMLP

from utilities.avocado.road_benchmark import A3ScenarioRoadTraffic
from utilities.avocado.road_config import A3RoadExperimentConfig, RoadCaseConfig
from utilities.avocado_marl.a5_config import A5ExperimentConfig
from utilities.avocado_marl.a6_config import A6ExperimentConfig
from utilities.avocado_marl.a6_policy import (
    A6ExecutionBridge,
    A6OneStepPolicy,
    A6RolloutController,
)
from utilities.avocado_marl.benchmark import _testing_parameters
from utilities.avocado_marl.config import A4ExperimentConfig
from utilities.avocado_marl.y_correction import YCorrectionNet
from utilities.experiment_artifacts import (
    atomic_write_json,
    create_run_directory,
    initialize_run,
    mark_latest_completed_run,
    resolve_latest_run,
    write_artifact_manifest,
    write_training_status,
)
from utilities.helper_training import get_observation_key
from utilities.mappo_cavs import mappo_cavs
from utilities.opinion.residual import OpinionResidual
from utilities.constants import SCENARIOS


@dataclass(frozen=True)
class A6IterationMetrics:
    iteration: int
    mean_reward: float
    actor_loss: float
    critic_loss: float
    approximate_entropy: float
    approximate_kl: float
    clip_fraction: float
    correction_mean_absolute: float
    correction_maximum_absolute: float
    correction_saturation_rate: float
    correction_sign_switch_rate: float
    fusion_saturation_rate: float
    y_gradient_norm: float
    critic_gradient_norm: float
    agent_collision_rate: float
    lane_collision_rate: float
    shield_intervention_rate: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _reset_mask(scenario: A3ScenarioRoadTraffic) -> Tensor:
    return (
        scenario.a3_last_agent_collisions
        | scenario.a3_last_lane_collisions
        | scenario.a3_last_wrong_entries
        | scenario.a3_last_route_completions
    )


def _critic_network(
    observation_dim: int,
    n_agents: int,
    device: str,
    checkpoint: Path,
) -> MultiAgentMLP:
    network = MultiAgentMLP(
        n_agent_inputs=observation_dim,
        n_agent_outputs=1,
        n_agents=n_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    saved = torch.load(checkpoint, map_location=device)
    prefix = "module."
    state = {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in saved.items()
    }
    network.load_state_dict(state, strict=True)
    return network


def _stack(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _gradient_norm(parameters) -> float:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().norm(2).square()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


class A6OneStepTrainer:
    """Train only YCorrectionNet; temporal AVOCADO state is detached per step."""

    def __init__(
        self,
        config: A6ExperimentConfig,
        *,
        environment_steps_override: Optional[int] = None,
        parallel_environments_override: Optional[int] = None,
        scenario_type_override: Optional[str] = None,
        seed_override: Optional[int] = None,
        load_critic: bool = True,
        render_live: bool = False,
    ) -> None:
        self.config = config
        self.config_fingerprint = hashlib.sha256(
            repr(config).encode("utf-8")
        ).hexdigest()
        self.a5_config = A5ExperimentConfig.from_json(config.a5_config)
        self.a4_config = A4ExperimentConfig.from_json(self.a5_config.a4_config)
        self.a3_config = A3RoadExperimentConfig.from_json(self.a4_config.a3_config)
        training = config.training
        seed = training.seed if seed_override is None else seed_override
        if type(seed) is not int or seed < 0:
            raise ValueError("seed_override must be a non-negative integer.")
        self.a3_config = replace(
            self.a3_config,
            simulation=replace(self.a3_config.simulation, seed=seed),
        )
        scenario_type = (
            training.scenario_type
            if scenario_type_override is None
            else str(scenario_type_override)
        )
        if scenario_type not in SCENARIOS:
            raise ValueError(f"Unknown A6 evaluation scenario: {scenario_type}")
        n_agents = int(SCENARIOS[scenario_type]["n_agents"])
        parallel_environments = (
            training.parallel_environments
            if parallel_environments_override is None
            else int(parallel_environments_override)
        )
        if parallel_environments <= 0:
            raise ValueError("parallel_environments_override must be positive.")
        if load_critic and n_agents != training.n_agents:
            raise ValueError(
                "Training must keep the configured agent count because the "
                "Central Critic is agent-count specific."
            )
        case = RoadCaseConfig(
            name=f"{scenario_type}_{n_agents}",
            scenario_type=scenario_type,
            n_agents=n_agents,
        )
        environment_steps = (
            training.rollout_steps
            if environment_steps_override is None
            else int(environment_steps_override)
        )
        if environment_steps <= 0:
            raise ValueError("environment_steps_override must be positive.")
        parameters = _testing_parameters(
            config.base_policy.run_directory,
            self.a3_config,
            case,
            parallel_environments,
            environment_steps,
            render_live=render_live,
        )
        parameters.seed = seed
        parameters.frames_per_batch = parallel_environments * environment_steps
        parameters.num_vmas_envs = parallel_environments
        parameters.max_steps = environment_steps
        if render_live:
            parameters.render_title = f"A6 MARL--AVOCADO | {scenario_type}"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.scenario = A3ScenarioRoadTraffic()
        self.env, base_policy, priority, self.parameters = mappo_cavs(
            parameters=parameters,
            policy_checkpoint_path=config.base_policy.policy_checkpoint,
            scenario_override=self.scenario,
        )
        if priority is not None:
            raise NotImplementedError("A6 does not support prioritized MARL.")
        self.observation_key = get_observation_key(parameters)
        self.action_key = self.env.action_key
        base_policy_net = base_policy.module[0].module
        self.base_policy_net = base_policy_net
        self.base_actor_source_hash = _state_hash(base_policy_net)

        correction = config.y_correction
        self.y_correction_net = YCorrectionNet(
            feature_dim=correction.feature_dim,
            hidden_sizes=correction.hidden_sizes,
            maximum_correction=correction.maximum_correction,
            temperature=correction.temperature,
            strict_zero=correction.strict_zero,
            freeze=correction.freeze,
        ).to(parameters.device)
        residual_config = config.opinion_residual
        residual = OpinionResidual(
            opinion_scale=residual_config.opinion_scale,
            gain=residual_config.gain,
            max_abs=residual_config.maximum_absolute_residual,
            action_index=0,
        ).to(parameters.device)
        self.policy = A6OneStepPolicy(
            base_policy_net,
            self.y_correction_net,
            residual,
            self.a3_config,
        ).to(parameters.device)
        execution = A6ExecutionBridge(
            self.scenario,
            self.a3_config,
            config,
            velocity_continuity_weight=(
                self.a4_config.coupling.velocity_continuity_weight
            ),
            speed_tolerance=(
                self.a4_config.diagnostics.speed_intervention_tolerance_mps
            ),
            steering_tolerance_degrees=(
                self.a4_config.diagnostics.steering_intervention_tolerance_degrees
            ),
        )
        action_space = self.env.unbatched_action_spec[self.action_key].space
        self.rollout = A6RolloutController(
            self.policy,
            execution,
            self.observation_key,
            torch.as_tensor(action_space.low, device=parameters.device),
            torch.as_tensor(action_space.high, device=parameters.device),
            correction.candidate_count,
        )
        observation_dim = int(
            self.env.observation_spec[self.observation_key].shape[-1]
        )
        self.critic = (
            _critic_network(
                observation_dim,
                n_agents,
                parameters.device,
                config.base_policy.critic_checkpoint,
            )
            if load_critic
            else None
        )
        self.optimizer = (
            torch.optim.Adam(
                [
                    {
                        "params": list(self.y_correction_net.parameters()),
                        "lr": training.y_learning_rate,
                        "group_name": "y_correction",
                    },
                    {
                        "params": list(self.critic.parameters()),
                        "lr": training.critic_learning_rate,
                        "group_name": "critic",
                    },
                ]
            )
            if self.critic is not None
            else None
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def load_checkpoint(self, checkpoint: Path, *, load_optimizer: bool = False) -> int:
        payload = torch.load(Path(checkpoint), map_location=self.parameters.device)
        if payload.get("stage") != "a6":
            raise ValueError("Checkpoint is not an A6 checkpoint.")
        if payload.get("base_actor_source_hash") != self.base_actor_source_hash:
            raise ValueError("A6 checkpoint Base Actor source does not match config.")
        if payload.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError("A6 checkpoint configuration does not match config.")
        self.y_correction_net.load_state_dict(
            payload["y_correction_state"], strict=True
        )
        if self.critic is not None:
            self.critic.load_state_dict(payload["critic_state"], strict=True)
        if load_optimizer:
            if self.optimizer is None or self.critic is None:
                raise ValueError("Optimizer restore requires a training-mode trainer.")
            self.optimizer.load_state_dict(payload["optimizer_state"])
            torch.set_rng_state(payload["torch_rng_state"].cpu())
            np.random.set_state(payload["numpy_rng_state"])
            random.setstate(payload["python_rng_state"])
            self.rollout.controller.generator.set_state(
                payload["controller_generator_state"].cpu()
            )
        return int(payload["iteration"])

    def _value(self, observations: Tensor) -> Tensor:
        if self.critic is None:
            raise RuntimeError("The Central Critic is unavailable in evaluation mode.")
        return self.critic(observations)

    @torch.no_grad()
    def collect(self) -> dict[str, Tensor]:
        training = self.config.training
        self.rollout.reset_all()
        tensordict = self.env.reset()
        records = []
        previous_correction = None
        sign_switches = 0
        sign_comparisons = 0
        collisions = 0
        lane_collisions = 0
        shield_interventions = 0
        for _ in range(training.rollout_steps):
            observations = tensordict.get(self.observation_key)
            value = self._value(observations)
            rollout_step = self.rollout.step(tensordict, deterministic=False)
            step_tensordict = self.env.step(rollout_step.tensordict)
            reward = step_tensordict.get(("next", "agents", "reward"))
            environment_done = step_tensordict.get(("next", "done")).to(
                torch.bool
            )
            environment_done = environment_done.view(
                environment_done.shape[0], 1, 1
            ).expand(-1, training.n_agents, -1)
            reset_mask = _reset_mask(self.scenario)
            transition_done = environment_done | reset_mask.unsqueeze(-1)
            current_correction = rollout_step.policy_output.correction
            if previous_correction is not None:
                active = rollout_step.pair_mask & (previous_correction != 0)
                sign_switches += int(
                    ((current_correction * previous_correction < 0) & active).sum()
                )
                sign_comparisons += int(active.sum())
            previous_correction = current_correction
            bridge_last = self.rollout.execution_bridge.last
            if bridge_last is None:
                raise RuntimeError("A6 execution diagnostics are missing.")
            if bridge_last.shield_result is not None:
                shield_interventions += int(
                    bridge_last.shield_result.intervention_mask.sum()
                )
            collisions += int(self.scenario.a3_last_agent_collisions.sum())
            lane_collisions += int(self.scenario.a3_last_lane_collisions.sum())
            records.append(
                {
                    "observation": observations.detach().clone(),
                    "features": rollout_step.features,
                    "confidence": rollout_step.confidence,
                    "pair_mask": rollout_step.pair_mask,
                    "attention": rollout_step.prospective_attention,
                    "heuristic": rollout_step.heuristic_estimate,
                    "z_prev": rollout_step.z_prev,
                    "action": rollout_step.nominal_action,
                    "old_log_prob": rollout_step.old_log_prob,
                    "reward": reward.detach().clone(),
                    "done": transition_done.detach().clone(),
                    "value": value.detach().clone(),
                    "correction": current_correction.detach().clone(),
                    "fused": rollout_step.policy_output.fused_estimate,
                }
            )
            self.rollout.reset_agents(reset_mask)
            tensordict = step_mdp(
                step_tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.env.reward_keys,
                action_keys=self.env.action_keys,
                done_keys=self.env.done_keys,
            )
        final_value = self._value(tensordict.get(self.observation_key)).detach()
        result = _stack(records)
        result["final_value"] = final_value
        result["sign_switch_rate"] = torch.tensor(
            sign_switches / max(sign_comparisons, 1),
            device=final_value.device,
        )
        sample_count = (
            training.rollout_steps
            * training.parallel_environments
            * training.n_agents
        )
        result["agent_collision_rate"] = torch.tensor(
            collisions / max(sample_count, 1), device=final_value.device
        )
        result["lane_collision_rate"] = torch.tensor(
            lane_collisions / max(sample_count, 1), device=final_value.device
        )
        result["shield_intervention_rate"] = torch.tensor(
            shield_interventions / max(sample_count, 1), device=final_value.device
        )
        return result

    def _advantages(self, rollout: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        training = self.config.training
        rewards = rollout["reward"]
        dones = rollout["done"].to(rewards.dtype)
        values = rollout["value"]
        next_values = torch.cat((values[1:], rollout["final_value"].unsqueeze(0)))
        advantage = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for index in range(training.rollout_steps - 1, -1, -1):
            nonterminal = 1.0 - dones[index]
            delta = (
                rewards[index]
                + training.gamma * next_values[index] * nonterminal
                - values[index]
            )
            running = (
                delta
                + training.gamma * training.gae_lambda * nonterminal * running
            )
            advantage[index] = running
        returns = advantage + values
        normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-8)
        return normalized, returns

    def update(self, rollout: dict[str, Tensor]) -> dict[str, float]:
        training = self.config.training
        advantages, returns = self._advantages(rollout)
        frame_count = training.rollout_steps * training.parallel_environments
        flattened = {
            key: value.reshape(frame_count, *value.shape[2:])
            for key, value in rollout.items()
            if key
            in {
                "observation",
                "features",
                "confidence",
                "pair_mask",
                "attention",
                "heuristic",
                "z_prev",
                "action",
                "old_log_prob",
            }
        }
        flat_advantage = advantages.reshape(frame_count, *advantages.shape[2:]).squeeze(-1)
        flat_returns = returns.reshape(frame_count, *returns.shape[2:])
        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip": 0.0,
            "y_grad": 0.0,
            "critic_grad": 0.0,
        }
        updates = 0
        for _ in range(training.epochs):
            permutation = torch.randperm(frame_count, device=flat_advantage.device)
            for start in range(0, frame_count, training.minibatch_size):
                indices = permutation[start : start + training.minibatch_size]
                output = self.policy(
                    flattened["observation"][indices],
                    flattened["features"][indices],
                    flattened["confidence"][indices],
                    flattened["pair_mask"][indices],
                    flattened["attention"][indices],
                    flattened["heuristic"][indices],
                    flattened["z_prev"][indices].detach(),
                )
                distribution = self.rollout.distribution(output)
                new_log_prob = distribution.log_prob(flattened["action"][indices])
                old_log_prob = flattened["old_log_prob"][indices]
                ratio = (new_log_prob - old_log_prob).exp()
                advantage = flat_advantage[indices]
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - training.clip_epsilon,
                    1.0 + training.clip_epsilon,
                ) * advantage
                ppo_loss = -torch.minimum(unclipped, clipped).mean()
                approximate_entropy_loss = training.entropy_coefficient * new_log_prob.mean()
                active = flattened["pair_mask"][indices]
                active_float = active.to(output.correction.dtype)
                active_count = active_float.sum().clamp_min(1.0)
                correction_penalty = (
                    output.correction.square() * active_float
                ).sum() / active_count
                saturation_penalty = torch.relu(
                    output.fused_estimate.abs() - training.soft_fusion_limit
                ).square()
                saturation_penalty = (
                    saturation_penalty * active_float
                ).sum() / active_count
                actor_loss = (
                    ppo_loss
                    + approximate_entropy_loss
                    + training.correction_regularization * correction_penalty
                    + training.saturation_regularization * saturation_penalty
                )
                predicted_value = self._value(flattened["observation"][indices])
                critic_loss = torch.nn.functional.mse_loss(
                    predicted_value, flat_returns[indices]
                )
                loss = actor_loss + critic_loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("A6 PPO produced a non-finite loss.")
                self.optimizer.zero_grad()
                loss.backward()
                y_grad = _gradient_norm(self.y_correction_net.parameters())
                critic_grad = _gradient_norm(self.critic.parameters())
                torch.nn.utils.clip_grad_norm_(
                    self.y_correction_net.parameters(),
                    training.maximum_gradient_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), training.maximum_gradient_norm
                )
                self.optimizer.step()
                if any(parameter.grad is not None for parameter in self.base_policy_net.parameters()):
                    raise RuntimeError("Frozen Base Actor received an A6 gradient.")
                with torch.no_grad():
                    log_ratio = new_log_prob - old_log_prob
                    approximate_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > training.clip_epsilon
                    ).float().mean()
                totals["actor_loss"] += float(actor_loss.detach())
                totals["critic_loss"] += float(critic_loss.detach())
                totals["entropy"] += float((-new_log_prob).mean().detach())
                totals["kl"] += float(approximate_kl)
                totals["clip"] += float(clip_fraction)
                totals["y_grad"] += y_grad
                totals["critic_grad"] += critic_grad
                updates += 1
        return {key: value / max(updates, 1) for key, value in totals.items()}

    def train_iteration(self, iteration: int) -> A6IterationMetrics:
        rollout = self.collect()
        update = self.update(rollout)
        active = rollout["pair_mask"]
        correction = rollout["correction"]
        active_values = correction[active]
        maximum = self.config.y_correction.maximum_correction
        fused_active = rollout["fused"][active]
        return A6IterationMetrics(
            iteration=iteration,
            mean_reward=float(rollout["reward"].mean()),
            actor_loss=update["actor_loss"],
            critic_loss=update["critic_loss"],
            approximate_entropy=update["entropy"],
            approximate_kl=update["kl"],
            clip_fraction=update["clip"],
            correction_mean_absolute=(
                float(active_values.abs().mean()) if active_values.numel() else 0.0
            ),
            correction_maximum_absolute=(
                float(active_values.abs().max()) if active_values.numel() else 0.0
            ),
            correction_saturation_rate=(
                float((active_values.abs() >= 0.99 * maximum).float().mean())
                if active_values.numel()
                else 0.0
            ),
            correction_sign_switch_rate=float(rollout["sign_switch_rate"]),
            fusion_saturation_rate=(
                float((fused_active.abs() >= 0.99).float().mean())
                if fused_active.numel()
                else 0.0
            ),
            y_gradient_norm=update["y_grad"],
            critic_gradient_norm=update["critic_grad"],
            agent_collision_rate=float(rollout["agent_collision_rate"]),
            lane_collision_rate=float(rollout["lane_collision_rate"]),
            shield_intervention_rate=float(rollout["shield_intervention_rate"]),
        )

    def checkpoint(self, iteration: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stage": "a6",
            "iteration": int(iteration),
            "y_correction_state": self.y_correction_net.state_dict(),
            "critic_state": self.critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "base_actor_source_hash": self.base_actor_source_hash,
            "config_fingerprint": self.config_fingerprint,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "controller_generator_state": self.rollout.controller.generator.get_state(),
        }


def train_a6(
    config_path: Path,
    *,
    iterations_override: Optional[int] = None,
    resume_checkpoint: Optional[Path] = None,
) -> Path:
    config_path = Path(config_path)
    config = A6ExperimentConfig.from_json(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        source_config = json.load(stream)
    iterations = (
        config.training.iterations
        if iterations_override is None
        else int(iterations_override)
    )
    if iterations <= 0:
        raise ValueError("iterations_override must be positive.")
    if resume_checkpoint is None:
        run_directory = create_run_directory(
            output_root=config.output_root,
            method="a6-y-correction",
            seed=config.training.seed,
        )
    else:
        resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"A6 resume checkpoint not found: {resume_checkpoint}")
        run_directory = resume_checkpoint.parent
    resolved = copy.deepcopy(source_config)
    resolved["training"]["iterations"] = iterations
    if resume_checkpoint is None:
        initialize_run(
            run_directory=run_directory,
            source_config=source_config,
            resolved_config=resolved,
            method="avocado_marl",
            stage="a6",
        )
    else:
        write_training_status(run_directory, status="running", iteration=None)
    trainer = None
    metrics = []
    try:
        trainer = A6OneStepTrainer(config)
        if resume_checkpoint is None:
            torch.save(
                trainer.base_policy_net.state_dict(),
                run_directory / "source_base_actor.pth",
            )
            start_iteration = 1
        else:
            restored_iteration = trainer.load_checkpoint(
                resume_checkpoint, load_optimizer=True
            )
            start_iteration = restored_iteration + 1
            metrics_path = run_directory / "metrics.json"
            if metrics_path.is_file():
                with metrics_path.open("r", encoding="utf-8") as stream:
                    metrics = list(json.load(stream).get("iterations", []))
        if start_iteration > iterations:
            raise ValueError(
                "A6 target iterations must exceed the checkpoint iteration."
            )
        training = config.training
        frames_per_iteration = (
            training.parallel_environments * training.rollout_steps
        )
        minibatches_per_epoch = (
            frames_per_iteration + training.minibatch_size - 1
        ) // training.minibatch_size
        optimizer_steps_per_iteration = training.epochs * minibatches_per_epoch
        remaining_iterations = iterations - start_iteration + 1
        print(f"[A6] Run directory: {run_directory}", flush=True)
        print(
            "[A6] Training plan: "
            f"iterations={start_iteration}..{iterations}, "
            f"frames/iteration={frames_per_iteration}, "
            f"optimizer-steps/iteration={optimizer_steps_per_iteration}, "
            f"remaining-frames={remaining_iterations * frames_per_iteration}",
            flush=True,
        )
        if resume_checkpoint is not None:
            print(
                f"[A6] Resumed from iteration {start_iteration - 1}: "
                f"{resume_checkpoint}",
                flush=True,
            )
        session_started_at = time.perf_counter()
        for iteration in range(start_iteration, iterations + 1):
            iteration_started_at = time.perf_counter()
            item = trainer.train_iteration(iteration)
            metrics.append(item.to_dict())
            atomic_write_json(run_directory / "metrics.json", {"iterations": metrics})
            torch.save(
                trainer.checkpoint(iteration),
                run_directory / "latest_checkpoint.pt",
            )
            write_training_status(
                run_directory, status="running", iteration=iteration
            )
            elapsed = time.perf_counter() - iteration_started_at
            completed_this_session = iteration - start_iteration + 1
            mean_iteration_seconds = (
                time.perf_counter() - session_started_at
            ) / completed_this_session
            eta_seconds = mean_iteration_seconds * (iterations - iteration)
            print(
                f"[A6 {iteration}/{iterations}] "
                f"reward={item.mean_reward:.5f} "
                f"actor={item.actor_loss:.6f} "
                f"critic={item.critic_loss:.6f} "
                f"|dy|mean={item.correction_mean_absolute:.6f} "
                f"|dy|max={item.correction_maximum_absolute:.6f} "
                f"y_grad={item.y_gradient_norm:.3e} "
                f"collision={item.agent_collision_rate:.4f} "
                f"shield={item.shield_intervention_rate:.4f} "
                f"time={elapsed:.1f}s eta={eta_seconds:.1f}s",
                flush=True,
            )
        if _state_hash(trainer.base_policy_net) != trainer.base_actor_source_hash:
            raise RuntimeError("A6 modified the frozen Base Actor.")
        torch.save(
            trainer.y_correction_net.state_dict(),
            run_directory / "final_y_correction.pth",
        )
        torch.save(trainer.critic.state_dict(), run_directory / "final_critic.pth")
        torch.save(
            trainer.checkpoint(iterations), run_directory / "final_checkpoint.pt"
        )
        write_training_status(
            run_directory, status="completed", iteration=iterations
        )
        write_artifact_manifest(run_directory)
        mark_latest_completed_run(config.output_root, run_directory)
    except BaseException as error:
        write_training_status(
            run_directory,
            status="failed",
            iteration=None,
            error=f"{type(error).__name__}: {error}",
        )
        write_artifact_manifest(run_directory)
        raise
    finally:
        if trainer is not None:
            trainer.close()
    return run_directory


def resolve_latest_a6_checkpoint(output_root: str) -> Path:
    """Resolve an A6 checkpoint without applying Base-policy filename rules."""

    root = Path(output_root).expanduser().resolve()

    def checkpoint_in(run_directory: Path) -> Optional[Path]:
        for name in ("final_checkpoint.pt", "latest_checkpoint.pt"):
            candidate = run_directory / name
            if candidate.is_file():
                return candidate
        return None

    try:
        completed_run = resolve_latest_run(output_root)
    except FileNotFoundError:
        completed_run = None
    if completed_run is not None:
        checkpoint = checkpoint_in(completed_run)
        if checkpoint is not None:
            return checkpoint

    checkpoint = checkpoint_in(root)
    if checkpoint is not None:
        return checkpoint

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
    for run_directory in candidates:
        checkpoint = checkpoint_in(run_directory)
        if checkpoint is not None:
            return checkpoint
    raise FileNotFoundError(
        f"No A6 final_checkpoint.pt or latest_checkpoint.pt found under {root}."
    )


@torch.no_grad()
def evaluate_a6(
    config_path: Path,
    checkpoint: Path,
    *,
    max_steps: Optional[int] = None,
    render_live: bool = False,
    scenario_type: Optional[str] = None,
) -> dict[str, object]:
    """Run a deterministic A6 rollout with the exact frozen Base source."""

    config = A6ExperimentConfig.from_json(Path(config_path))
    steps = config.training.rollout_steps if max_steps is None else int(max_steps)
    if steps <= 0:
        raise ValueError("max_steps must be positive.")
    trainer = A6OneStepTrainer(
        config,
        environment_steps_override=steps,
        parallel_environments_override=(
            1 if render_live else config.training.parallel_environments
        ),
        scenario_type_override=scenario_type,
        load_critic=False,
        render_live=render_live,
    )
    try:
        restored_iteration = trainer.load_checkpoint(Path(checkpoint))
        trainer.rollout.reset_all()
        tensordict = trainer.env.reset()
        reward_sum = 0.0
        corrections = []
        fused = []
        opinions = []
        collisions = 0
        lane_collisions = 0
        route_completions = 0
        shield_interventions = 0
        action_samples = 0
        for step_index in range(steps):
            rollout_step = trainer.rollout.step(tensordict, deterministic=True)
            step_tensordict = trainer.env.step(rollout_step.tensordict)
            reward_sum += float(
                step_tensordict.get(("next", "agents", "reward")).sum()
            )
            reset_mask = _reset_mask(trainer.scenario)
            collisions += int(trainer.scenario.a3_last_agent_collisions.sum())
            lane_collisions += int(trainer.scenario.a3_last_lane_collisions.sum())
            route_completions += int(
                trainer.scenario.a3_last_route_completions.sum()
            )
            bridge_last = trainer.rollout.execution_bridge.last
            if bridge_last is not None and bridge_last.shield_result is not None:
                shield_interventions += int(
                    bridge_last.shield_result.intervention_mask.sum()
                )
            corrections.append(rollout_step.policy_output.correction.cpu())
            fused.append(rollout_step.policy_output.fused_estimate.cpu())
            opinions.append(trainer.rollout.controller.opinion.detach().cpu().clone())
            action_samples += (
                trainer.scenario.world.batch_dim * trainer.scenario.n_agents
            )
            if render_live:
                bridge_last = trainer.rollout.execution_bridge.last
                nominal = rollout_step.nominal_action[0, 0]
                executed = (
                    bridge_last.executed_action[0, 0]
                    if bridge_last is not None
                    else nominal
                )
                active = rollout_step.pair_mask[0]
                active_correction = rollout_step.policy_output.correction[0][active]
                active_fused = rollout_step.policy_output.fused_estimate[0][active]
                active_opinion = trainer.rollout.controller.opinion[0][active]
                lines = [
                    "stage: A6 learned y-correction",
                    f"step: {step_index + 1}/{steps}",
                    f"checkpoint iteration: {restored_iteration}",
                    f"nominal: {float(nominal[0]):.2f} m/s, "
                    f"{math.degrees(float(nominal[1])):.1f} deg",
                    f"executed: {float(executed[0]):.2f} m/s, "
                    f"{math.degrees(float(executed[1])):.1f} deg",
                    "max |Delta y|: "
                    f"{float(active_correction.abs().max()) if active_correction.numel() else 0.0:.4f}",
                    "mean |yF|: "
                    f"{float(active_fused.abs().mean()) if active_fused.numel() else 0.0:.3f}",
                    "mean |z|: "
                    f"{float(active_opinion.abs().mean()) if active_opinion.numel() else 0.0:.3f}",
                    "active VOs: "
                    f"{int(trainer.rollout.controller.last_active_vo_count[0].sum())}",
                    f"resets: {int(reset_mask[0].sum())}",
                ]
                if bridge_last is not None and bridge_last.shield_result is not None:
                    lines.append(
                        "TTC shield: "
                        f"{int(bridge_last.shield_result.intervention_mask[0].sum())}"
                    )
                trainer.scenario.set_opinion_visualization(lines)
                trainer.env.render(mode="rgb_array", visualize_when_rgb=True)
            trainer.rollout.reset_agents(reset_mask)
            tensordict = step_mdp(
                step_tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=trainer.env.reward_keys,
                action_keys=trainer.env.action_keys,
                done_keys=trainer.env.done_keys,
            )
        correction_tensor = torch.stack(corrections)
        fused_tensor = torch.stack(fused)
        opinion_tensor = torch.stack(opinions)
        return {
            "stage": "a6",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_iteration": restored_iteration,
            "steps": steps,
            "parallel_environments": trainer.scenario.world.batch_dim,
            "scenario_type": trainer.parameters.scenario_type,
            "rendered": bool(render_live),
            "mean_reward_per_agent_step": reward_sum / max(action_samples, 1),
            "agent_collision_rate": collisions / max(action_samples, 1),
            "lane_collision_rate": lane_collisions / max(action_samples, 1),
            "route_completion_rate": route_completions / max(action_samples, 1),
            "shield_intervention_rate": shield_interventions / max(action_samples, 1),
            "mean_absolute_correction": float(correction_tensor.abs().mean()),
            "maximum_absolute_correction": float(correction_tensor.abs().max()),
            "correction_saturation_rate": float(
                (
                    correction_tensor.abs()
                    >= 0.99 * config.y_correction.maximum_correction
                )
                .float()
                .mean()
            ),
            "mean_absolute_fused_estimate": float(fused_tensor.abs().mean()),
            "mean_absolute_opinion": float(opinion_tensor.abs().mean()),
            "base_actor_source_hash": trainer.base_actor_source_hash,
        }
    finally:
        trainer.close()
