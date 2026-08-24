# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import time
import random
from pathlib import Path
from typing import Mapping, Optional

from termcolor import colored, cprint

# Torch
import torch
import numpy as np

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from utilities.helper_training import SyncDataCollectorCustom, PriorityModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum
from torchrl.envs.utils import (
    check_env_specs,
)

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from tqdm import tqdm

import os

import matplotlib.pyplot as plt

# Scientific plotting
import scienceplots  # Do not remove (https://github.com/garrettj403/SciencePlots)

plt.rcParams.update(
    {"figure.dpi": "100"}
)  # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(
    ["science", "ieee"]
)  # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

from torchrl.envs.libs.vmas import VmasEnv

# Import custom classes
from utilities.helper_training import (
    Parameters,
    SaveData,
    TransformedEnvCustom,
    get_path_to_save_model,
    find_the_highest_reward_among_all_models,
    save,
    compute_td_error,
    get_observation_key,
)

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.experiment_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    save_training_curves,
    write_metrics,
    write_timing,
    write_training_status,
)


def mappo_cavs(
    parameters: Parameters,
    opinion_pair_info_config: Optional[Mapping[str, object]] = None,
    opinion_policy_config: Optional[Mapping[str, object]] = None,
    artifact_method: str = "base_mappo",
    artifact_stage: str = "base",
    policy_checkpoint_path: Optional[Path] = None,
):
    # Preserve the upstream default (seed 0) while making it explicit in the
    # resolved configuration for reproducible Base runs.
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)

    scenario = ScenarioRoadTraffic()

    scenario.parameters = parameters
    if opinion_pair_info_config is not None:
        scenario.configure_opinion_pair_info(dict(opinion_pair_info_config))

    # Using multi-threads to handle file writing
    # pool = ThreadPoolExecutor(128)

    env = VmasEnv(
        scenario=scenario,
        num_envs=parameters.num_vmas_envs,
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=parameters.max_steps,
        device=parameters.device,
        # Scenario kwargs
        n_agents=parameters.n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    )

    save_data = SaveData(
        parameters=parameters,
        episode_reward_mean_list=[],
    )
    artifact_logging_enabled = parameters.artifact_logging_enabled
    artifact_run_directory = Path(parameters.where_to_save)
    artifact_iterations = []

    reward_sum = RewardSum(
        in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]
    )
    emits_opinion_pair_info = bool(
        opinion_pair_info_config
        and opinion_pair_info_config.get("emit_pair_info", False)
    )
    if emits_opinion_pair_info:
        # TorchRL 0.2.1's VMAS wrapper casts every info leaf to float32.
        # Restore the M4 public contract after the wrapper, while leaving the
        # standard Base transform stack byte-for-byte equivalent in behavior.
        from torchrl.envs.transforms import Compose, DTypeCastTransform

        env_transform = Compose(
            reward_sum,
            DTypeCastTransform(
                torch.float32,
                torch.long,
                in_keys=[("agents", "info", "neighbor_ids")],
                in_keys_inv=[],
            ),
            DTypeCastTransform(
                torch.float32,
                torch.bool,
                in_keys=[
                    ("agents", "info", "pair_mask"),
                    ("agents", "info", "agent_reset_mask"),
                ],
                in_keys_inv=[],
            ),
        )
    else:
        env_transform = reward_sum

    env = TransformedEnvCustom(env, env_transform)

    check_env_specs(env)

    observation_key = get_observation_key(parameters)

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[observation_key].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=(2 * env.action_spec.shape[-1]),  # 2 * n_actions_per_agents
            n_agents=env.n_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=True,  # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
            device=parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
    )

    # print("policy_net:", policy_net, "\n")

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[observation_key],
        out_keys=[
            ("agents", "loc"),
            ("agents", "scale"),
        ],  # represents the parameters of the policy distribution for each agent
    )

    # Use a probabilistic actor allows for exploration
    base_policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[env.action_key].space.low,
            "max": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=(
            "agents",
            "sample_log_prob",
        ),  # log probability favors numerical stability and gradient calculation
    )  # we'll need the log-prob for the PPO loss

    is_direct_opinion = bool(
        opinion_policy_config
        and opinion_policy_config.get("mode") == "direct_evidence"
    )
    opinion_bridge = None
    base_actor_source_state = None
    if is_direct_opinion:
        if not emits_opinion_pair_info:
            raise ValueError("The M5 policy bridge requires M4 pair info.")
        if not parameters.is_load_model:
            base_actor_checkpoint = Path(
                str(opinion_policy_config["base_actor_checkpoint"])
            )
            if not base_actor_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Base Actor checkpoint not found: {base_actor_checkpoint}"
                )
            base_actor_source_state = torch.load(
                base_actor_checkpoint,
                map_location=parameters.device,
            )
            base_policy.load_state_dict(base_actor_source_state, strict=True)
            print(
                colored(
                    f"[INFO] Loaded frozen Base Actor: {base_actor_checkpoint}",
                    "red",
                )
            )

        from utilities.opinion.config import EvidenceConfig, ResidualConfig
        from utilities.opinion.evidence_net import OpinionEvidenceNet
        from utilities.opinion.policy import DirectEvidencePolicyBridge
        from utilities.opinion.residual import OpinionResidual

        evidence_config = EvidenceConfig.from_dict(
            opinion_policy_config["evidence"]
        )
        residual_config = ResidualConfig.from_dict(
            opinion_policy_config["residual"]
        )
        evidence_net = OpinionEvidenceNet.from_config(
            pair_feature_dim=int(opinion_pair_info_config["pair_feature_dim"]),
            config=evidence_config,
        ).to(parameters.device)
        residual = OpinionResidual.from_config(residual_config).to(parameters.device)
        opinion_bridge = DirectEvidencePolicyBridge(
            base_policy_net=policy_net,
            evidence_net=evidence_net,
            residual=residual,
            freeze_base_actor=bool(
                opinion_policy_config.get("freeze_base_actor", True)
            ),
        ).to(parameters.device)
        opinion_policy_module = TensorDictModule(
            opinion_bridge,
            in_keys=[
                observation_key,
                ("agents", "info", "pair_features"),
                ("agents", "info", "urgency"),
                ("agents", "info", "confidence"),
                ("agents", "info", "pair_mask"),
            ],
            out_keys=[
                ("agents", "loc"),
                ("agents", "scale"),
                ("agents", "opinion", "base_loc"),
                ("agents", "opinion", "raw_b"),
                ("agents", "opinion", "b"),
                ("agents", "opinion", "direct_z"),
                ("agents", "opinion", "residual"),
            ],
        )
        policy = ProbabilisticActor(
            module=opinion_policy_module,
            spec=env.unbatched_action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": env.unbatched_action_spec[env.action_key].space.low,
                "max": env.unbatched_action_spec[env.action_key].space.high,
            },
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )
    else:
        policy = base_policy

    mappo = True  # IPPO (Independent PPO) if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[observation_key].shape[
            -1
        ],  # Number of observations
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
        share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
        device=parameters.device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[
            observation_key
        ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
        out_keys=[("agents", "state_value")],
    )

    if is_direct_opinion and not parameters.is_load_model:
        base_critic_checkpoint = Path(
            str(opinion_policy_config["base_critic_checkpoint"])
        )
        if not base_critic_checkpoint.is_file():
            raise FileNotFoundError(
                f"Base Critic checkpoint not found: {base_critic_checkpoint}"
            )
        critic.load_state_dict(
            torch.load(base_critic_checkpoint, map_location=parameters.device),
            strict=True,
        )
        print(
            colored(
                f"[INFO] Initialized M5 Critic from: {base_critic_checkpoint}",
                "red",
            )
        )

    # Instantiate the priority module
    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        priority_module = PriorityModule(env=env, mappo=mappo)
    else:
        priority_module = None

    # Check if the directory defined to store the model exists and create it if not
    if not os.path.exists(parameters.where_to_save):
        os.makedirs(parameters.where_to_save)
        print(
            colored(
                "[INFO] Created a new directory to save the trained model:", "black"
            ),
            colored(f"{parameters.where_to_save}", "blue"),
        )

    # Load an existing model or train a new model?
    if parameters.is_load_model:
        if policy_checkpoint_path is not None:
            checkpoint = Path(policy_checkpoint_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint}")
            PATH_POLICY = str(checkpoint)
            policy.load_state_dict(
                torch.load(PATH_POLICY, map_location=parameters.device)
            )
            print(colored(f"[INFO] Loaded policy checkpoint: {PATH_POLICY}", "blue"))

            checkpoint_suffix = "_policy.pth"
            checkpoint_prefix = checkpoint.name[: -len(checkpoint_suffix)]
            PATH_CRITIC = str(
                checkpoint.with_name(f"{checkpoint_prefix}_critic.pth")
            )
            if priority_module:
                PATH_PRIORITY_POLICY = str(
                    checkpoint.with_name(
                        f"{checkpoint_prefix}_priority_policy.pth"
                    )
                )
                PATH_PRIORITY_CRITIC = str(
                    checkpoint.with_name(
                        f"{checkpoint_prefix}_priority_critic.pth"
                    )
                )
                if not os.path.isfile(PATH_PRIORITY_POLICY):
                    raise FileNotFoundError(
                        f"Priority policy checkpoint not found: {PATH_PRIORITY_POLICY}"
                    )
                priority_module.policy.load_state_dict(
                    torch.load(PATH_PRIORITY_POLICY, map_location=parameters.device)
                )
        elif parameters.is_load_final_model:
            PATH_POLICY = parameters.where_to_save + "final_policy.pth"
            PATH_CRITIC = parameters.where_to_save + "final_critic.pth"
            if not os.path.isfile(PATH_POLICY):
                raise FileNotFoundError(f"Final policy not found: {PATH_POLICY}")
            policy.load_state_dict(
                torch.load(PATH_POLICY, map_location=parameters.device)
            )
            print(colored(f"[INFO] Loaded final policy: {PATH_POLICY}", "red"))

            if priority_module:
                PATH_PRIORITY_POLICY = (
                    parameters.where_to_save + "final_priority_policy.pth"
                )
                PATH_PRIORITY_CRITIC = (
                    parameters.where_to_save + "final_priority_critic.pth"
                )
                priority_module.policy.load_state_dict(
                    torch.load(PATH_PRIORITY_POLICY, map_location=parameters.device)
                )
        else:
            highest_reward = find_the_highest_reward_among_all_models(
                parameters.where_to_save
            )
            if highest_reward == float("-inf"):
                raise ValueError(
                    f"No intermediate model found in {parameters.where_to_save}."
                )
            parameters.episode_reward_mean_current = highest_reward
            paths = get_path_to_save_model(parameters=parameters)

            if priority_module:
                (
                    PATH_POLICY,
                    PATH_CRITIC,
                    PATH_PRIORITY_POLICY,
                    PATH_PRIORITY_CRITIC,
                    PATH_FIG,
                    PATH_JSON,
                ) = paths
            else:
                PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = paths

            policy.load_state_dict(
                torch.load(PATH_POLICY, map_location=parameters.device)
            )
            print(colored(f"[INFO] Loaded intermediate policy: {PATH_POLICY}", "blue"))

            if priority_module:
                priority_module.policy.load_state_dict(
                    torch.load(PATH_PRIORITY_POLICY, map_location=parameters.device)
                )

        if not parameters.is_continue_train:
            print(colored("[INFO] Training will not continue.", "blue"))

            return env, policy, priority_module, parameters
        else:
            print(
                colored("[INFO] Training will continue with the loaded model.", "red")
            )
            if not os.path.isfile(PATH_CRITIC):
                raise FileNotFoundError(f"Critic not found: {PATH_CRITIC}")
            critic.load_state_dict(
                torch.load(PATH_CRITIC, map_location=parameters.device)
            )

            if priority_module:
                priority_module.critic.load_state_dict(
                    torch.load(PATH_PRIORITY_CRITIC, map_location=parameters.device)
                )

    collector = SyncDataCollectorCustom(
        env,
        policy,
        priority_module=priority_module,
        device=parameters.device,
        storing_device=parameters.device,
        frames_per_batch=parameters.frames_per_batch,
        total_frames=parameters.total_frames,
    )

    if parameters.is_prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.6,
            storage=LazyTensorStorage(
                parameters.frames_per_batch, device=parameters.device
            ),
            batch_size=parameters.minibatch_size,
            priority_key="td_error",
        )
    else:
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                parameters.frames_per_batch, device=parameters.device
            ),  # We store the frames_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=parameters.minibatch_size,  # We will sample minibatches of this size
        )

    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=parameters.clip_epsilon,
        entropy_coef=parameters.entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )

    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=parameters.gamma, lmbda=parameters.lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator  # Generalized Advantage Estimation

    if is_direct_opinion:
        actor_trainable_parameters = [
            parameter
            for parameter in loss_module.actor_params.values(True, True)
            if parameter.requires_grad
        ]
        critic_trainable_parameters = [
            parameter
            for parameter in loss_module.critic_params.values(True, True)
            if parameter.requires_grad
        ]
        if not actor_trainable_parameters:
            raise RuntimeError("M5 EvidenceNet has no trainable parameters.")
        evidence_lr_scale = float(
            opinion_policy_config["evidence_learning_rate_scale"]
        )
        optim = torch.optim.Adam(
            [
                {
                    "params": actor_trainable_parameters,
                    "lr": parameters.lr * evidence_lr_scale,
                    "lr_scale": evidence_lr_scale,
                    "group_name": "evidence",
                },
                {
                    "params": critic_trainable_parameters,
                    "lr": parameters.lr,
                    "lr_scale": 1.0,
                    "group_name": "critic",
                },
            ]
        )
    else:
        optim = torch.optim.Adam(loss_module.parameters(), parameters.lr)
    optimization_parameters = [
        parameter for parameter in loss_module.parameters() if parameter.requires_grad
    ]

    pbar = tqdm(total=parameters.n_iters, desc="epi_rew_mean = 0")

    episode_reward_mean_list = []

    t_start = time.time()
    iteration_cycle_start = t_start
    for tensordict_data in collector:
        rollout_finished_at = time.time()
        rollout_seconds = rollout_finished_at - iteration_cycle_start
        optimization_started_at = rollout_finished_at
        loss_objective_sum = 0.0
        loss_critic_sum = 0.0
        loss_entropy_sum = 0.0
        loss_update_count = 0

        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_params,
                target_params=loss_module.target_critic_params,
            )  # Compute GAE and add it to the data

            if priority_module:
                priority_module.GAE(
                    tensordict_data,
                    params=priority_module.loss_module.critic_params,
                    target_params=priority_module.loss_module.target_critic_params,
                )

        # Update sample priorities
        if parameters.is_prb:
            td_error = compute_td_error(tensordict_data, gamma=0.9)
            tensordict_data.set(
                ("td_error"), td_error
            )  # Adding TD error to the tensordict_data

            assert (
                tensordict_data["td_error"].min() >= 0
            ), "TD error must be greater than 0"

        data_view = tensordict_data.reshape(
            -1
        )  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)
        # replay_buffer.update_tensordict_priority() # Not necessary, as priorities were updated automatically when calling `replay_buffer.extend()`

        for _ in range(parameters.num_epochs):
            # print("[DEBUG] for _ in range(parameters.num_epochs):")
            for _ in range(parameters.frames_per_batch // parameters.minibatch_size):
                # sample a batch of data
                mini_batch_data, info = replay_buffer.sample(return_info=True)

                loss_vals = loss_module(mini_batch_data)

                loss_objective_sum += (
                    loss_vals["loss_objective"].detach().mean().item()
                )
                loss_critic_sum += loss_vals["loss_critic"].detach().mean().item()
                loss_entropy_sum += loss_vals["loss_entropy"].detach().mean().item()
                loss_update_count += 1

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                assert not loss_value.isnan().any()
                assert not loss_value.isinf().any()

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    optimization_parameters, parameters.max_grad_norm
                )  # Optional

                optim.step()
                optim.zero_grad()

                if priority_module:
                    priority_module.compute_losses_and_optimize(mini_batch_data)

                if parameters.is_prb:
                    # Recalculate loss
                    with torch.no_grad():
                        GAE(
                            mini_batch_data,
                            params=loss_module.critic_params,
                            target_params=loss_module.target_critic_params,
                        )
                        if parameters.is_using_prioritized_marl:
                            priority_module.GAE(
                                tensordict_data,
                                params=priority_module.loss_module.critic_params,
                                target_params=priority_module.loss_module.target_critic_params,
                            )
                    # Recalculate the TD errors of the sampled minibatch with updated model weights and update priorities in the buffer
                    new_td_errors = compute_td_error(mini_batch_data, gamma=0.9)
                    mini_batch_data.set("td_error", new_td_errors)
                    replay_buffer.update_tensordict_priority(mini_batch_data)
        optimization_seconds = time.time() - optimization_started_at
        collector.update_policy_weights_()  # Updates the policy weights if the policy of the data collector and the trained policy live on different devices

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done]
            .mean()
            .item()
        )
        episode_reward_mean = round(episode_reward_mean, 2)
        episode_reward_mean_list.append(episode_reward_mean)

        collision_with_agents = tensordict_data.get(
            ("next", "agents", "info", "is_collision_with_agents")
        ).bool()
        collision_with_lanelets = tensordict_data.get(
            ("next", "agents", "info", "is_collision_with_lanelets")
        ).bool()
        collision_with_agents_rate = (
            collision_with_agents.float().mean().item()
        )
        collision_with_lanelets_rate = (
            collision_with_lanelets.float().mean().item()
        )
        total_collision_rate = (
            (collision_with_agents | collision_with_lanelets).float().mean().item()
        )
        opinion_iteration_metrics = {}
        if is_direct_opinion:
            raw_b_collected = tensordict_data.get(
                ("agents", "opinion", "raw_b")
            )
            gated_b_collected = tensordict_data.get(
                ("agents", "opinion", "b")
            )
            residual_collected = tensordict_data.get(
                ("agents", "opinion", "residual")
            )
            pair_mask_collected = tensordict_data.get(
                ("agents", "info", "pair_mask")
            )
            opinion_iteration_metrics = {
                "raw_b_abs_mean": float(raw_b_collected.abs().mean().item()),
                "gated_b_abs_mean": float(gated_b_collected.abs().mean().item()),
                "speed_residual_abs_mean": float(
                    residual_collected.abs().mean().item()
                ),
                "active_pair_fraction": float(
                    pair_mask_collected.float().mean().item()
                ),
            }
        pbar.set_description(
            f"Episode mean reward = {episode_reward_mean:.2f}", refresh=False
        )

        # env.scenario.iter = pbar.n # A way to pass the information from the training algorithm to the environment

        if parameters.is_save_intermediate_model:
            # Update the current mean episode reward
            parameters.episode_reward_mean_current = episode_reward_mean
            save_data.episode_reward_mean_list = episode_reward_mean_list

            if episode_reward_mean > parameters.episode_reward_intermediate:
                # Save the model if it improves the mean episode reward sufficiently enough
                parameters.episode_reward_intermediate = episode_reward_mean

                if (
                    parameters.is_using_prioritized_marl
                    and parameters.prioritization_method.lower() == "marl"
                ):
                    save(
                        parameters=parameters,
                        save_data=save_data,
                        policy=policy,
                        critic=critic,
                        priority_policy=priority_module.policy,
                        priority_critic=priority_module.critic,
                    )
                else:
                    save(
                        parameters=parameters,
                        save_data=save_data,
                        policy=policy,
                        critic=critic,
                    )
                if is_direct_opinion:
                    if opinion_bridge is None:
                        raise RuntimeError(
                            "M5 EvidenceNet is unavailable while saving a checkpoint."
                        )
                    torch.save(
                        opinion_bridge.evidence_net.state_dict(),
                        parameters.where_to_save
                        + parameters.model_name
                        + "_evidence_net.pth",
                    )
            else:
                # Save only the mean episode reward list and parameters
                parameters.episode_reward_mean_current = (
                    parameters.episode_reward_intermediate
                )
                save(
                    parameters=parameters,
                    save_data=save_data,
                    policy=None,
                    critic=None,
                    priority_policy=None,
                    priority_critic=None,
                )

        # Learning rate schedule
        for param_group in optim.param_groups:
            # Linear decay to lr_min
            lr_decay = (parameters.lr - parameters.lr_min) * (
                1 - (pbar.n / parameters.n_iters)
            )
            scheduled_lr = parameters.lr_min + lr_decay
            param_group["lr"] = scheduled_lr * param_group.get("lr_scale", 1.0)
            if pbar.n % 10 == 0:
                print(f"Learning rate updated to {param_group['lr']}.")

        pbar.update()

        if artifact_logging_enabled:
            if loss_update_count == 0:
                raise RuntimeError("No PPO updates were executed in this iteration.")
            iteration_seconds = time.time() - iteration_cycle_start
            artifact_iterations.append(
                {
                    "iteration": int(pbar.n),
                    "episode_reward_mean": float(episode_reward_mean),
                    "collision_with_agents_rate": float(
                        collision_with_agents_rate
                    ),
                    "collision_with_lanelets_rate": float(
                        collision_with_lanelets_rate
                    ),
                    "total_collision_rate": float(total_collision_rate),
                    "loss_objective": loss_objective_sum / loss_update_count,
                    "loss_critic": loss_critic_sum / loss_update_count,
                    "loss_entropy": loss_entropy_sum / loss_update_count,
                    "learning_rate": float(
                        optim.param_groups[-1]["lr"]
                        if is_direct_opinion
                        else optim.param_groups[0]["lr"]
                    ),
                    **(
                        {
                            "evidence_learning_rate": float(
                                optim.param_groups[0]["lr"]
                            ),
                            "critic_learning_rate": float(
                                optim.param_groups[1]["lr"]
                            ),
                        }
                        if is_direct_opinion
                        else {}
                    ),
                    **opinion_iteration_metrics,
                    "rollout_seconds": float(rollout_seconds),
                    "optimization_seconds": float(optimization_seconds),
                    "iteration_seconds": float(iteration_seconds),
                }
            )
            write_metrics(
                artifact_run_directory,
                artifact_iterations,
                method=artifact_method,
                stage=artifact_stage,
            )
            write_training_status(
                artifact_run_directory,
                status="running",
                iteration=pbar.n,
            )

        iteration_cycle_start = time.time()

    # Save the final model
    torch.save(policy.state_dict(), parameters.where_to_save + "final_policy.pth")
    torch.save(critic.state_dict(), parameters.where_to_save + "final_critic.pth")
    if is_direct_opinion:
        if opinion_bridge is None:
            raise RuntimeError(
                "M5 EvidenceNet is unavailable while saving the final model."
            )
        torch.save(
            opinion_bridge.evidence_net.state_dict(),
            parameters.where_to_save + "final_evidence_net.pth",
        )

    if artifact_logging_enabled:
        if is_direct_opinion:
            if base_actor_source_state is None or opinion_bridge is None:
                raise RuntimeError("M5 checkpoint source state is unavailable.")
            torch.save(
                base_actor_source_state,
                artifact_run_directory / "final_base_actor.pth",
            )
            torch.save(
                policy.state_dict(),
                artifact_run_directory / "final_opinion_policy.pth",
            )
            torch.save(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "method": "opinion_marl",
                    "stage": "evidence_direct",
                    "iteration": int(pbar.n),
                    "base_actor_state": base_actor_source_state,
                    "evidence_state": opinion_bridge.evidence_net.state_dict(),
                    "opinion_policy_state": policy.state_dict(),
                    "critic_state": critic.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "resolved_base_config": dict(parameters.to_dict()),
                    "opinion_runtime_config": dict(opinion_policy_config),
                    "torch_rng_state": torch.get_rng_state(),
                },
                artifact_run_directory / "final_checkpoint.pt",
            )
        else:
            # Keep the upstream filename for main_testing.py and expose a
            # stable bridge for later Opinion training stages.
            torch.save(
                policy.state_dict(),
                artifact_run_directory / "final_base_actor.pth",
            )
            torch.save(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "method": "base_mappo",
                    "iteration": int(pbar.n),
                    "base_actor_state": policy.state_dict(),
                    "critic_state": critic.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "resolved_config": dict(parameters.to_dict()),
                    "torch_rng_state": torch.get_rng_state(),
                },
                artifact_run_directory / "final_checkpoint.pt",
            )

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        torch.save(
            priority_module.policy.state_dict(),
            parameters.where_to_save + "final_priority_policy.pth",
        )
        torch.save(
            priority_module.critic.state_dict(),
            parameters.where_to_save + "final_priority_critic.pth",
        )

    print(
        colored("[INFO] All files have been saved under:", "black"),
        colored(f"{parameters.where_to_save}", "red"),
    )
    # plt.show()

    training_duration_seconds = time.time() - t_start
    training_duration = training_duration_seconds / 3600  # seconds to hours
    print(colored(f"[INFO] Training duration: {training_duration:.2f} hours.", "blue"))

    if artifact_logging_enabled:
        write_timing(
            artifact_run_directory,
            artifact_iterations,
            total_seconds=training_duration_seconds,
        )
        save_training_curves(artifact_run_directory, artifact_iterations)
        atomic_write_json(
            artifact_run_directory / "comparison_to_base.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "reference": "self",
                "status": "base_reference_created",
                "automated_performance_validation": False,
                "note": "Performance comparison is performed manually by the user.",
            },
        )

    return env, policy, priority_module, parameters


if __name__ == "__main__":
    config_file = "config.json"
    parameters = Parameters.from_json(config_file)
    env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)
