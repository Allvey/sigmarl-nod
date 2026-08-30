"""One-step PPO trainer for A6-Action preferred-action learning."""

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

from utilities.avocado.road_benchmark import A3ScenarioRoadTraffic
from utilities.avocado.road_config import A3RoadExperimentConfig, RoadCaseConfig
from utilities.avocado_marl.a5_config import A5ExperimentConfig
from utilities.avocado_marl.a5_benchmark import run_a5_rollout
from utilities.avocado_marl.a6_action_config import A6ActionExperimentConfig
from utilities.avocado_marl.a6_action_policy import (
    A6ActionExecutionBridge,
    A6ActionPolicy,
    A6ActionRolloutController,
    InteractionActionNet,
)
from utilities.avocado_marl.a6_trainer import (
    _critic_network,
    _gradient_norm,
    _reset_mask,
    _stack,
    _state_hash,
)
from utilities.avocado_marl.benchmark import _testing_parameters
from utilities.avocado_marl.config import A4ExperimentConfig
from utilities.constants import SCENARIOS
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


@dataclass(frozen=True)
class A6ActionIterationMetrics:
    iteration: int
    mean_reward: float
    actor_loss: float
    critic_loss: float
    approximate_entropy: float
    approximate_kl: float
    clip_fraction: float
    loc_correction_mean_absolute: float
    loc_correction_maximum_absolute: float
    loc_correction_saturation_rate: float
    active_agent_rate: float
    action_gradient_norm: float
    critic_gradient_norm: float
    agent_collision_rate: float
    lane_collision_rate: float
    shield_intervention_rate: float
    action_intervention_rate: float
    mean_nominal_executed_speed_difference_mps: float
    mean_nominal_executed_steering_difference_degrees: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class A6ActionZeroEquivalence:
    scenario_type: str
    seed: int
    steps: int
    passed: bool
    maximum_differences: Dict[str, float]
    maximum_absolute_loc_correction: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class A6ActionOneStepTrainer:
    """Freeze Base Actor and safety chain; train only preferred-action head."""

    def __init__(
        self,
        config: A6ActionExperimentConfig,
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
            raise ValueError(
                f"Unknown A6-Action evaluation scenario: {scenario_type}"
            )
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
            parameters.render_title = f"A6-Action | {scenario_type}"
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
            raise NotImplementedError("A6-Action does not support prioritized MARL.")
        self.observation_key = get_observation_key(parameters)
        self.action_key = self.env.action_key
        self.base_policy_net = base_policy.module[0].module
        self.base_actor_source_hash = _state_hash(self.base_policy_net)

        action_config = config.action_policy
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.action_net = InteractionActionNet(
                feature_dim=action_config.feature_dim,
                hidden_sizes=action_config.hidden_sizes,
                maximum_loc_correction=(
                    action_config.maximum_loc_correction
                ),
                zero_initialization=action_config.zero_initialization,
                freeze=action_config.freeze,
            ).to(parameters.device)
        self.policy = A6ActionPolicy(
            self.base_policy_net,
            self.action_net,
        ).to(parameters.device)
        execution = A6ActionExecutionBridge(
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
        self.rollout = A6ActionRolloutController(
            self.policy,
            execution,
            self.observation_key,
            torch.as_tensor(action_space.low, device=parameters.device),
            torch.as_tensor(action_space.high, device=parameters.device),
            action_config.candidate_count,
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
                        "params": list(self.action_net.parameters()),
                        "lr": training.action_learning_rate,
                        "group_name": "interaction_action",
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

    def load_checkpoint(
        self, checkpoint: Path, *, load_optimizer: bool = False
    ) -> int:
        payload = torch.load(Path(checkpoint), map_location=self.parameters.device)
        if payload.get("stage") != "a6_action":
            raise ValueError("Checkpoint is not an A6-Action checkpoint.")
        if payload.get("base_actor_source_hash") != self.base_actor_source_hash:
            raise ValueError(
                "A6-Action checkpoint Base Actor source does not match config."
            )
        if payload.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError("A6-Action checkpoint configuration does not match.")
        self.action_net.load_state_dict(payload["action_net_state"], strict=True)
        if self.critic is not None:
            self.critic.load_state_dict(payload["critic_state"], strict=True)
        if load_optimizer:
            if self.optimizer is None or self.critic is None:
                raise ValueError("Optimizer restore requires training mode.")
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
            raise RuntimeError("The Central Critic is unavailable in evaluation.")
        return self.critic(observations)

    @torch.no_grad()
    def collect(self) -> dict[str, Tensor]:
        training = self.config.training
        self.rollout.reset_all()
        tensordict = self.env.reset()
        records = []
        collisions = 0
        lane_collisions = 0
        shield_interventions = 0
        action_interventions = 0
        speed_difference_sum = 0.0
        steering_difference_sum_degrees = 0.0
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
            bridge_last = self.rollout.execution_bridge.last
            if bridge_last is None:
                raise RuntimeError("A6-Action execution diagnostics are missing.")
            if bridge_last.shield_result is not None:
                shield_interventions += int(
                    bridge_last.shield_result.intervention_mask.sum()
                )
            action_interventions += int(bridge_last.intervention_mask.sum())
            speed_difference_sum += float(
                (
                    bridge_last.executed_action[..., 0]
                    - bridge_last.nominal_action[..., 0]
                )
                .abs()
                .sum()
            )
            steering_difference_sum_degrees += float(
                torch.rad2deg(
                    (
                        bridge_last.executed_action[..., 1]
                        - bridge_last.nominal_action[..., 1]
                    ).abs()
                ).sum()
            )
            collisions += int(self.scenario.a3_last_agent_collisions.sum())
            lane_collisions += int(self.scenario.a3_last_lane_collisions.sum())
            records.append(
                {
                    "observation": observations.detach().clone(),
                    "features": rollout_step.features,
                    "confidence": rollout_step.confidence,
                    "pair_mask": rollout_step.pair_mask,
                    "action": rollout_step.nominal_action,
                    "old_log_prob": rollout_step.old_log_prob,
                    "reward": reward.detach().clone(),
                    "done": transition_done.detach().clone(),
                    "value": value.detach().clone(),
                    "loc_correction": (
                        rollout_step.policy_output.loc_correction
                    ),
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
        sample_count = (
            training.rollout_steps
            * training.parallel_environments
            * training.n_agents
        )
        device = final_value.device
        result["agent_collision_rate"] = torch.tensor(
            collisions / max(sample_count, 1), device=device
        )
        result["lane_collision_rate"] = torch.tensor(
            lane_collisions / max(sample_count, 1), device=device
        )
        result["shield_intervention_rate"] = torch.tensor(
            shield_interventions / max(sample_count, 1), device=device
        )
        result["action_intervention_rate"] = torch.tensor(
            action_interventions / max(sample_count, 1), device=device
        )
        result["mean_speed_difference"] = torch.tensor(
            speed_difference_sum / max(sample_count, 1), device=device
        )
        result["mean_steering_difference_degrees"] = torch.tensor(
            steering_difference_sum_degrees / max(sample_count, 1),
            device=device,
        )
        return result

    def _advantages(self, rollout: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        training = self.config.training
        rewards = rollout["reward"]
        dones = rollout["done"].to(rewards.dtype)
        values = rollout["value"]
        next_values = torch.cat(
            (values[1:], rollout["final_value"].unsqueeze(0))
        )
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
        normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(
            1e-8
        )
        return normalized, returns

    def update(self, rollout: dict[str, Tensor]) -> dict[str, float]:
        if self.optimizer is None or self.critic is None:
            raise RuntimeError("A6-Action update requires training mode.")
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
                "action",
                "old_log_prob",
            }
        }
        flat_advantage = advantages.reshape(
            frame_count, *advantages.shape[2:]
        ).squeeze(-1)
        flat_returns = returns.reshape(frame_count, *returns.shape[2:])
        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip": 0.0,
            "action_grad": 0.0,
            "critic_grad": 0.0,
        }
        updates = 0
        for _ in range(training.epochs):
            permutation = torch.randperm(
                frame_count, device=flat_advantage.device
            )
            for start in range(0, frame_count, training.minibatch_size):
                indices = permutation[start : start + training.minibatch_size]
                output = self.policy(
                    flattened["observation"][indices],
                    flattened["features"][indices],
                    flattened["confidence"][indices],
                    flattened["pair_mask"][indices],
                )
                distribution = self.rollout.distribution(output)
                new_log_prob = distribution.log_prob(
                    flattened["action"][indices]
                )
                old_log_prob = flattened["old_log_prob"][indices]
                ratio = (new_log_prob - old_log_prob).exp()
                advantage = flat_advantage[indices]
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - training.clip_epsilon,
                    1.0 + training.clip_epsilon,
                ) * advantage
                ppo_loss = -torch.minimum(unclipped, clipped).mean()
                entropy_loss = (
                    training.entropy_coefficient * new_log_prob.mean()
                )
                active = flattened["pair_mask"][indices].any(dim=-1)
                active_float = active.to(output.loc_correction.dtype)
                active_count = active_float.sum().clamp_min(1.0)
                action_penalty = (
                    output.loc_correction.square().sum(dim=-1) * active_float
                ).sum() / active_count
                actor_loss = (
                    ppo_loss
                    + entropy_loss
                    + training.action_regularization * action_penalty
                )
                predicted_value = self._value(
                    flattened["observation"][indices]
                )
                critic_loss = torch.nn.functional.mse_loss(
                    predicted_value, flat_returns[indices]
                )
                loss = actor_loss + critic_loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("A6-Action PPO produced a non-finite loss.")
                self.optimizer.zero_grad()
                loss.backward()
                action_grad = _gradient_norm(self.action_net.parameters())
                critic_grad = _gradient_norm(self.critic.parameters())
                torch.nn.utils.clip_grad_norm_(
                    self.action_net.parameters(),
                    training.maximum_gradient_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), training.maximum_gradient_norm
                )
                self.optimizer.step()
                if any(
                    parameter.grad is not None
                    for parameter in self.base_policy_net.parameters()
                ):
                    raise RuntimeError(
                        "Frozen Base Actor received an A6-Action gradient."
                    )
                with torch.no_grad():
                    log_ratio = new_log_prob - old_log_prob
                    approximate_kl = (
                        (log_ratio.exp() - 1.0) - log_ratio
                    ).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > training.clip_epsilon
                    ).float().mean()
                totals["actor_loss"] += float(actor_loss.detach())
                totals["critic_loss"] += float(critic_loss.detach())
                totals["entropy"] += float((-new_log_prob).mean().detach())
                totals["kl"] += float(approximate_kl)
                totals["clip"] += float(clip_fraction)
                totals["action_grad"] += action_grad
                totals["critic_grad"] += critic_grad
                updates += 1
        return {key: value / max(updates, 1) for key, value in totals.items()}

    def train_iteration(self, iteration: int) -> A6ActionIterationMetrics:
        rollout = self.collect()
        update = self.update(rollout)
        active = rollout["pair_mask"].any(dim=-1)
        correction = rollout["loc_correction"]
        active_values = correction[active]
        maximum = self.action_net.maximum_loc_correction.to(correction)
        saturated = correction.abs() >= 0.99 * maximum
        return A6ActionIterationMetrics(
            iteration=iteration,
            mean_reward=float(rollout["reward"].mean()),
            actor_loss=update["actor_loss"],
            critic_loss=update["critic_loss"],
            approximate_entropy=update["entropy"],
            approximate_kl=update["kl"],
            clip_fraction=update["clip"],
            loc_correction_mean_absolute=(
                float(active_values.abs().mean())
                if active_values.numel()
                else 0.0
            ),
            loc_correction_maximum_absolute=(
                float(active_values.abs().max())
                if active_values.numel()
                else 0.0
            ),
            loc_correction_saturation_rate=(
                float(saturated[active].float().mean())
                if bool(active.any())
                else 0.0
            ),
            active_agent_rate=float(active.float().mean()),
            action_gradient_norm=update["action_grad"],
            critic_gradient_norm=update["critic_grad"],
            agent_collision_rate=float(rollout["agent_collision_rate"]),
            lane_collision_rate=float(rollout["lane_collision_rate"]),
            shield_intervention_rate=float(rollout["shield_intervention_rate"]),
            action_intervention_rate=float(rollout["action_intervention_rate"]),
            mean_nominal_executed_speed_difference_mps=float(
                rollout["mean_speed_difference"]
            ),
            mean_nominal_executed_steering_difference_degrees=float(
                rollout["mean_steering_difference_degrees"]
            ),
        )

    def checkpoint(self, iteration: int) -> dict[str, object]:
        if self.critic is None or self.optimizer is None:
            raise RuntimeError("A training checkpoint requires critic and optimizer.")
        return {
            "schema_version": 1,
            "stage": "a6_action",
            "iteration": int(iteration),
            "action_net_state": self.action_net.state_dict(),
            "critic_state": self.critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "base_actor_source_hash": self.base_actor_source_hash,
            "config_fingerprint": self.config_fingerprint,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "controller_generator_state": (
                self.rollout.controller.generator.get_state()
            ),
        }


def train_a6_action(
    config_path: Path,
    *,
    iterations_override: Optional[int] = None,
    resume_checkpoint: Optional[Path] = None,
) -> Path:
    config_path = Path(config_path)
    config = A6ActionExperimentConfig.from_json(config_path)
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
            method="a6-action",
            seed=config.training.seed,
        )
    else:
        resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"A6-Action resume checkpoint not found: {resume_checkpoint}"
            )
        run_directory = resume_checkpoint.parent
    resolved = copy.deepcopy(source_config)
    resolved["training"]["iterations"] = iterations
    if resume_checkpoint is None:
        initialize_run(
            run_directory=run_directory,
            source_config=source_config,
            resolved_config=resolved,
            method="avocado_marl",
            stage="a6_action",
        )
    else:
        write_training_status(run_directory, status="running", iteration=None)
    trainer = None
    metrics = []
    try:
        trainer = A6ActionOneStepTrainer(config)
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
                "A6-Action target iterations must exceed checkpoint iteration."
            )
        training = config.training
        frames_per_iteration = (
            training.parallel_environments * training.rollout_steps
        )
        minibatches_per_epoch = (
            frames_per_iteration + training.minibatch_size - 1
        ) // training.minibatch_size
        optimizer_steps_per_iteration = (
            training.epochs * minibatches_per_epoch
        )
        remaining_iterations = iterations - start_iteration + 1
        print(f"[A6-Action] Run directory: {run_directory}", flush=True)
        print(
            "[A6-Action] Training plan: "
            f"iterations={start_iteration}..{iterations}, "
            f"frames/iteration={frames_per_iteration}, "
            f"optimizer-steps/iteration={optimizer_steps_per_iteration}, "
            f"remaining-frames={remaining_iterations * frames_per_iteration}",
            flush=True,
        )
        session_started_at = time.perf_counter()
        for iteration in range(start_iteration, iterations + 1):
            iteration_started_at = time.perf_counter()
            item = trainer.train_iteration(iteration)
            metrics.append(item.to_dict())
            atomic_write_json(
                run_directory / "metrics.json", {"iterations": metrics}
            )
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
                f"[A6-Action {iteration}/{iterations}] "
                f"reward={item.mean_reward:.5f} "
                f"actor={item.actor_loss:.6f} "
                f"critic={item.critic_loss:.6f} "
                f"|dloc|mean={item.loc_correction_mean_absolute:.5f} "
                f"|dloc|max={item.loc_correction_maximum_absolute:.5f} "
                f"action_grad={item.action_gradient_norm:.3e} "
                f"collision={item.agent_collision_rate:.4f} "
                f"shield={item.shield_intervention_rate:.4f} "
                f"time={elapsed:.1f}s eta={eta_seconds:.1f}s",
                flush=True,
            )
        if _state_hash(trainer.base_policy_net) != trainer.base_actor_source_hash:
            raise RuntimeError("A6-Action modified the frozen Base Actor.")
        torch.save(
            trainer.action_net.state_dict(),
            run_directory / "final_action_net.pth",
        )
        torch.save(
            trainer.critic.state_dict(), run_directory / "final_critic.pth"
        )
        torch.save(
            trainer.checkpoint(iterations),
            run_directory / "final_checkpoint.pt",
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


def resolve_latest_a6_action_checkpoint(output_root: str) -> Path:
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
        f"No A6-Action checkpoint found under {root}."
    )


@torch.no_grad()
def verify_a6_action_zero_equivalence(
    config_path: Path,
    *,
    steps: int = 4,
    scenario_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> A6ActionZeroEquivalence:
    """Prove that the zero-initialized A6-Action closed loop equals A5."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    config = A6ActionExperimentConfig.from_json(Path(config_path))
    selected_scenario = scenario_type or config.training.scenario_type
    selected_seed = config.training.seed if seed is None else seed
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if selected_scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario type: {selected_scenario}")
    n_agents = int(SCENARIOS[selected_scenario]["n_agents"])
    case = RoadCaseConfig(
        name=f"{selected_scenario}_{n_agents}",
        scenario_type=selected_scenario,
        n_agents=n_agents,
    )
    a5_result = run_a5_rollout(
        A5ExperimentConfig.from_json(config.a5_config),
        case,
        run_directory=config.base_policy.run_directory,
        checkpoint=config.base_policy.policy_checkpoint,
        episodes_override=1,
        max_steps_override=steps,
        seed_override=selected_seed,
    )
    trainer = A6ActionOneStepTrainer(
        config,
        environment_steps_override=steps,
        parallel_environments_override=1,
        scenario_type_override=selected_scenario,
        seed_override=selected_seed,
        load_critic=False,
    )
    corrections = []
    try:
        trainer.rollout.reset_all()
        tensordict = trainer.env.reset()
        for step_index in range(steps):
            rollout_step = trainer.rollout.step(tensordict, deterministic=True)
            corrections.append(rollout_step.policy_output.loc_correction.cpu())
            step_tensordict = trainer.env.step(rollout_step.tensordict)
            reset_mask = _reset_mask(trainer.scenario)
            trainer.rollout.reset_agents(reset_mask)
            if step_index + 1 < steps:
                tensordict = step_mdp(
                    step_tensordict,
                    keep_other=True,
                    exclude_action=False,
                    exclude_reward=True,
                    reward_keys=trainer.env.reward_keys,
                    action_keys=trainer.env.action_keys,
                    done_keys=trainer.env.done_keys,
                )
        action_trace = trainer.rollout.execution_bridge.trace()
    finally:
        trainer.close()

    compared_fields = (
        "nominal_action",
        "pre_shield_action",
        "executed_action",
        "heuristic_estimate",
        "estimate_correction",
        "fused_estimate",
        "opinion",
        "attention",
        "pair_mask",
        "reset_mask",
    )
    differences = {}
    for field in compared_fields:
        a5_value = getattr(a5_result.trace, field)
        action_value = getattr(action_trace, field)
        if a5_value.shape != action_value.shape:
            differences[field] = float("inf")
        elif a5_value.dtype == torch.bool:
            differences[field] = float((a5_value != action_value).any())
        else:
            differences[field] = float((a5_value - action_value).abs().max())
    maximum_correction = float(torch.stack(corrections).abs().max())
    passed = maximum_correction == 0.0 and all(
        difference == 0.0 for difference in differences.values()
    )
    return A6ActionZeroEquivalence(
        scenario_type=selected_scenario,
        seed=selected_seed,
        steps=steps,
        passed=passed,
        maximum_differences=differences,
        maximum_absolute_loc_correction=maximum_correction,
    )


@torch.no_grad()
def evaluate_a6_action(
    config_path: Path,
    checkpoint: Path,
    *,
    max_steps: Optional[int] = None,
    render_live: bool = False,
    scenario_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, object]:
    config = A6ActionExperimentConfig.from_json(Path(config_path))
    steps = config.training.rollout_steps if max_steps is None else int(max_steps)
    if steps <= 0:
        raise ValueError("max_steps must be positive.")
    trainer = A6ActionOneStepTrainer(
        config,
        environment_steps_override=steps,
        parallel_environments_override=(
            1 if render_live else config.training.parallel_environments
        ),
        scenario_type_override=scenario_type,
        seed_override=seed,
        load_critic=False,
        render_live=render_live,
    )
    try:
        restored_iteration = trainer.load_checkpoint(Path(checkpoint))
        trainer.rollout.reset_all()
        tensordict = trainer.env.reset()
        reward_sum = 0.0
        corrections = []
        active_masks = []
        collisions = 0
        lane_collisions = 0
        route_completions = 0
        shield_interventions = 0
        action_interventions = 0
        action_samples = 0
        for step_index in range(steps):
            rollout_step = trainer.rollout.step(tensordict, deterministic=True)
            step_tensordict = trainer.env.step(rollout_step.tensordict)
            reward_sum += float(
                step_tensordict.get(("next", "agents", "reward")).sum()
            )
            reset_mask = _reset_mask(trainer.scenario)
            collisions += int(trainer.scenario.a3_last_agent_collisions.sum())
            lane_collisions += int(
                trainer.scenario.a3_last_lane_collisions.sum()
            )
            route_completions += int(
                trainer.scenario.a3_last_route_completions.sum()
            )
            bridge_last = trainer.rollout.execution_bridge.last
            if bridge_last is None:
                raise RuntimeError("A6-Action execution diagnostics are missing.")
            action_interventions += int(bridge_last.intervention_mask.sum())
            if bridge_last.shield_result is not None:
                shield_interventions += int(
                    bridge_last.shield_result.intervention_mask.sum()
                )
            corrections.append(
                rollout_step.policy_output.loc_correction.cpu()
            )
            active_masks.append(rollout_step.pair_mask.any(dim=-1).cpu())
            action_samples += (
                trainer.scenario.world.batch_dim * trainer.scenario.n_agents
            )
            if render_live:
                nominal = rollout_step.nominal_action[0, 0]
                executed = bridge_last.executed_action[0, 0]
                correction = rollout_step.policy_output.loc_correction[0, 0]
                lines = [
                    "stage: A6-Action preferred action",
                    f"step: {step_index + 1}/{steps}",
                    f"checkpoint iteration: {restored_iteration}",
                    f"nominal: {float(nominal[0]):.2f} m/s, "
                    f"{math.degrees(float(nominal[1])):.1f} deg",
                    f"executed: {float(executed[0]):.2f} m/s, "
                    f"{math.degrees(float(executed[1])):.1f} deg",
                    f"Delta loc: [{float(correction[0]):.3f}, "
                    f"{float(correction[1]):.3f}]",
                    "active VOs: "
                    f"{int(trainer.rollout.controller.last_active_vo_count[0].sum())}",
                    f"resets: {int(reset_mask[0].sum())}",
                ]
                if bridge_last.shield_result is not None:
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
        active_tensor = torch.stack(active_masks)
        active_corrections = correction_tensor[active_tensor]
        maximum = trainer.action_net.maximum_loc_correction.cpu()
        return {
            "stage": "a6_action",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_iteration": restored_iteration,
            "steps": steps,
            "parallel_environments": trainer.scenario.world.batch_dim,
            "scenario_type": trainer.parameters.scenario_type,
            "seed": config.training.seed if seed is None else seed,
            "rendered": bool(render_live),
            "mean_reward_per_agent_step": reward_sum / max(action_samples, 1),
            "agent_collision_rate": collisions / max(action_samples, 1),
            "lane_collision_rate": lane_collisions / max(action_samples, 1),
            "route_completion_rate": route_completions / max(action_samples, 1),
            "shield_intervention_rate": (
                shield_interventions / max(action_samples, 1)
            ),
            "action_intervention_rate": (
                action_interventions / max(action_samples, 1)
            ),
            "active_agent_rate": float(active_tensor.float().mean()),
            "mean_absolute_loc_correction": float(
                active_corrections.abs().mean()
                if active_corrections.numel()
                else 0.0
            ),
            "maximum_absolute_loc_correction": float(
                active_corrections.abs().max()
                if active_corrections.numel()
                else 0.0
            ),
            "loc_correction_saturation_rate": float(
                (
                    active_corrections.abs()
                    >= 0.99 * maximum.view(1, 2)
                )
                .float()
                .mean()
                if active_corrections.numel()
                else 0.0
            ),
            "base_actor_source_hash": trainer.base_actor_source_hash,
        }
    finally:
        trainer.close()
