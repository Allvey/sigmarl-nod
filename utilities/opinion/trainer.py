"""Three-stage optimizer contracts and compact Opinion-MARL trainer."""

from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torchrl.envs.utils import step_mdp

from utilities.opinion.checkpoint import load_opinion_checkpoint, save_opinion_checkpoint
from utilities.opinion.collector import OpinionStatefulCollector
from utilities.opinion.config import OpinionConfig
from utilities.opinion.policy import OpinionAugmentedPolicyCore, OpinionTanhNormalPolicy
from utilities.opinion.ppo_loss import OpinionCentralizedCritic, OpinionSequencePPOLoss
from utilities.opinion.sequence_buffer import OpinionSequenceBuffer
from utilities.opinion.diagnostics import OpinionDiagnostics


def build_stage_optimizers(
    *, core: OpinionAugmentedPolicyCore, critic: OpinionCentralizedCritic, config: OpinionConfig
) -> Dict[str, torch.optim.Optimizer]:
    actor_parameters = list(core.base_actor.parameters())
    evidence_parameters = list(core.evidence_net.parameters())
    critic_parameters = list(critic.parameters())
    for parameter in actor_parameters:
        parameter.requires_grad_(config.stage in ("base", "joint"))
    for parameter in evidence_parameters:
        parameter.requires_grad_(config.stage in ("evidence", "joint"))
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    optimizers = {}
    if config.stage in ("base", "joint"):
        optimizers["actor"] = torch.optim.Adam(actor_parameters, lr=config.lr_actor)
    if config.stage in ("evidence", "joint"):
        optimizers["evidence"] = torch.optim.Adam(
            evidence_parameters, lr=config.lr_evidence
        )
    optimizers["critic"] = torch.optim.Adam(critic_parameters, lr=config.lr_critic)
    parameter_sets = [
        {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        for optimizer in optimizers.values()
    ]
    for left in range(len(parameter_sets)):
        for right in range(left + 1, len(parameter_sets)):
            if not parameter_sets[left].isdisjoint(parameter_sets[right]):
                raise RuntimeError("optimizer parameter groups must be disjoint")
    return optimizers


def compute_gae(reward, value, done, next_value, *, gamma, lmbda):
    """Time-major generalized advantage estimation with truncated bootstrap."""
    if reward.shape != value.shape:
        raise ValueError("reward and value must have equal [T,E,N] shapes")
    if done.shape != reward.shape[:2]:
        raise ValueError("done must have shape [T,E]")
    if next_value.shape != reward.shape[1:]:
        raise ValueError("next_value must have shape [E,N]")
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(next_value)
    for index in range(reward.shape[0] - 1, -1, -1):
        not_done = (~done[index]).unsqueeze(-1)
        following_value = next_value if index == reward.shape[0] - 1 else value[index + 1]
        delta = reward[index] + gamma * following_value * not_done - value[index]
        running = delta + gamma * lmbda * not_done * running
        advantage[index] = running
    return advantage, advantage + value


class OpinionTrainer:
    """Minimal end-to-end trainer retaining contiguous opinion replay."""

    def __init__(
        self,
        *,
        env,
        policy: OpinionTanhNormalPolicy,
        critic: OpinionCentralizedCritic,
        config: OpinionConfig,
        parameters,
        output_dir,
    ) -> None:
        self.env = env
        self.policy = policy
        self.critic = critic
        self.config = config
        self.parameters = parameters
        self.output_dir = Path(output_dir)
        self.optimizers = build_stage_optimizers(
            core=policy.core, critic=critic, config=config
        )
        self.loss_module = OpinionSequencePPOLoss(
            policy=policy,
            critic=critic,
            clip_epsilon=parameters.clip_epsilon,
            entropy_eps=parameters.entropy_eps,
            neutral_loss_weight=config.neutral_loss_weight,
            magnitude_loss_weight=config.magnitude_loss_weight,
        )
        self.collector = OpinionStatefulCollector(
            policy=policy,
            n_envs=env.num_envs,
            n_agents=parameters.n_agents,
            device=torch.device(parameters.device),
        )
        self.metrics = []

    def load_stage_weights(self, checkpoint):
        """Start a new stage from network weights at an episode boundary."""
        allowed_previous = {
            "base": {"base"},
            "evidence": {"base", "evidence"},
            "joint": {"evidence", "joint"},
        }[self.config.stage]
        metadata = load_opinion_checkpoint(
            checkpoint,
            policy=self.policy,
            critic=self.critic,
            optimizers=None,
            map_location=self.parameters.device,
            expected_stages=allowed_previous,
        )
        self.collector.reset_all()
        return metadata

    def residual_scale(self, iteration):
        if self.config.stage == "base":
            return 0.0
        warmup_iters = max(
            1, int(math.ceil(self.parameters.n_iters * self.config.residual_warmup_fraction))
        )
        fraction = min(1.0, (iteration + 1) / warmup_iters)
        return self.config.residual_scale_start + fraction * (
            self.config.residual_scale_target - self.config.residual_scale_start
        )

    def collect_iteration(self, iteration):
        td = self.env.reset()
        self.collector.reset_all()
        buffer = OpinionSequenceBuffer(
            n_envs=self.env.num_envs, n_agents=self.parameters.n_agents
        )
        steps = self.parameters.frames_per_batch // self.env.num_envs
        scale = self.residual_scale(iteration)
        diagnostics = OpinionDiagnostics(
            b_max=self.config.b_max, z_clip=self.config.z_clip
        )
        for step_index in range(steps):
            observation = td["agents", "observation"]
            info = td["agents", "info"]
            dense_before = self.collector.z_dense.detach().clone()
            value = self.critic(observation, dense_before).detach()
            output = self.collector.step(
                step_id=step_index,
                observation=observation,
                pair_features=info["pair_features"],
                neighbor_ids=info["neighbor_ids"],
                pair_mask=info["pair_mask"].bool(),
                urgency=info["urgency"],
                confidence=info["confidence"],
                agent_reset_mask=info["agent_reset_mask"],
                environment_done=td["done"],
                residual_scale=scale,
            )
            td.set(("agents", "action"), output.action)
            transition = self.env.step(td)
            reward = transition["next", "agents", "reward"].squeeze(-1)
            done = transition["next", "done"].squeeze(-1).bool()
            next_info = transition["next", "agents", "info"]
            diagnostics.update(
                reward=reward,
                collision_agents=next_info["is_collision_with_agents"].squeeze(-1),
                collision_lanelets=next_info["is_collision_with_lanelets"].squeeze(-1),
                raw_b=output.raw_b,
                b=output.b,
                z_prev=output.z_prev,
                z_next=output.z_next,
                residual=output.residual,
                pair_mask=output.pair_mask,
                agent_reset_mask=info["agent_reset_mask"].squeeze(-1),
                residual_scale=scale,
            )
            buffer.append(
                observation=observation,
                action=output.action,
                old_log_prob=output.log_prob,
                reward=reward,
                done=done,
                pair_features=info["pair_features"],
                neighbor_ids=output.neighbor_ids,
                pair_mask=output.pair_mask,
                urgency=info["urgency"],
                confidence=info["confidence"],
                agent_reset_mask=info["agent_reset_mask"].squeeze(-1).bool(),
                z_dense_prev=output.z_dense_prev,
                value=value,
            )
            td = step_mdp(
                transition,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.env.reward_keys,
                done_keys=self.env.done_keys,
            )
        rollout = buffer.as_rollout()
        next_value = self.critic(
            td["agents", "observation"], self.collector.z_dense
        ).detach()
        advantage, returns = compute_gae(
            rollout["reward"],
            rollout["value"],
            rollout["done"],
            next_value,
            gamma=self.parameters.gamma,
            lmbda=self.parameters.lmbda,
        )
        advantage = (advantage - advantage.mean()) / advantage.std(
            unbiased=False
        ).clamp_min(1e-6)
        buffer.add_rollout_fields(returns=returns, advantage=advantage)
        return buffer, scale, diagnostics

    def optimize(self, buffer, residual_scale):
        losses = []
        gradient_norms = {"actor": 0.0, "evidence": 0.0, "critic": 0.0}
        chunks = list(buffer.iter_chunks(chunk_length=self.config.chunk_length))
        for _ in range(self.parameters.num_epochs):
            order = torch.randperm(len(chunks)).tolist()
            for offset in range(0, len(order), self.config.chunks_per_minibatch):
                minibatch = [
                    chunks[index]
                    for index in order[offset : offset + self.config.chunks_per_minibatch]
                ]
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                outputs = [
                    self.loss_module(chunk, residual_scale=residual_scale)
                    for chunk in minibatch
                ]
                loss = torch.stack([output.total_loss for output in outputs]).mean()
                loss.backward()
                for optimizer in self.optimizers.values():
                    parameters = [
                        p
                        for group in optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                    if parameters:
                        norm = clip_grad_norm_(parameters, self.parameters.max_grad_norm)
                        name = (
                            "evidence" if optimizer is self.optimizers.get("evidence")
                            else "actor" if optimizer is self.optimizers.get("actor")
                            else "critic"
                        )
                        gradient_norms[name] = max(gradient_norms[name], float(norm))
                    optimizer.step()
                losses.append(float(loss.detach()))
        return sum(losses) / max(1, len(losses)), gradient_norms

    def fit(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for iteration in range(self.parameters.n_iters):
            buffer, scale, diagnostics = self.collect_iteration(iteration)
            loss, gradient_norms = self.optimize(buffer, scale)
            summary = diagnostics.summary()
            summary.update(
                {
                    "iteration": iteration + 1,
                    "loss": loss,
                    "stage": self.config.stage,
                    "actor_gradient_norm": gradient_norms["actor"],
                    "evidence_gradient_norm": gradient_norms["evidence"],
                    "critic_gradient_norm": gradient_norms["critic"],
                }
            )
            self.metrics.append(summary)
        checkpoint = save_opinion_checkpoint(
            self.output_dir / "final_opinion.pt",
            policy=self.policy,
            critic=self.critic,
            optimizers=self.optimizers,
            resolved_config={
                **self.parameters.to_dict(),
                "opinion_config": self.config.to_dict(),
            },
            stage=self.config.stage,
            iteration=self.parameters.n_iters,
        )
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2), encoding="utf-8"
        )
        return checkpoint


def build_opinion_trainer(loaded, *, smoke=False, output_dir=None):
    """Construct the independent road-traffic runtime from validated config."""
    from torchrl.envs.libs.vmas import VmasEnv

    from scenarios.road_traffic import ScenarioRoadTraffic

    parameters = copy.deepcopy(loaded.parameters)
    config = loaded.opinion
    if smoke:
        parameters.n_iters = 2
        parameters.max_steps = 4
        parameters.frames_per_batch = 8
        parameters.num_vmas_envs = 2
        parameters.num_epochs = 1
    else:
        parameters.num_vmas_envs = parameters.frames_per_batch // parameters.max_steps
    if output_dir is None:
        output_dir = Path(parameters.where_to_save) / (
            "smoke" if smoke else config.stage
        )
    parameters.where_to_save = str(output_dir)
    torch.manual_seed(parameters.seed)
    scenario = ScenarioRoadTraffic()
    scenario.parameters = parameters
    env = VmasEnv(
        scenario=scenario,
        num_envs=parameters.num_vmas_envs,
        continuous_actions=True,
        max_steps=parameters.max_steps,
        device=parameters.device,
        n_agents=parameters.n_agents,
    )
    td = env.reset()
    observation_dim = td["agents", "observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    core = OpinionAugmentedPolicyCore.from_config(
        observation_dim=observation_dim,
        action_dim=action_dim,
        config=config,
        dt=parameters.dt,
    ).to(parameters.device)
    bounds = env.unbatched_action_spec[env.action_key].space
    policy = OpinionTanhNormalPolicy(
        core=core,
        action_low=bounds.low.to(parameters.device),
        action_high=bounds.high.to(parameters.device),
    )
    critic = OpinionCentralizedCritic(
        observation_dim=observation_dim,
        n_agents=parameters.n_agents,
        include_z=config.include_z_in_critic,
    ).to(parameters.device)
    return OpinionTrainer(
        env=env,
        policy=policy,
        critic=critic,
        config=config,
        parameters=parameters,
        output_dir=output_dir,
    )
