# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import time
import random
import copy
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
    psb_runtime_config: Optional[Mapping[str, object]] = None,
    psb_action_projection: Optional[str] = None,
    artifact_method: str = "base_mappo",
    artifact_stage: str = "base",
    policy_checkpoint_path: Optional[Path] = None,
    training_resume_checkpoint: Optional[Path] = None,
    scenario_override: Optional[ScenarioRoadTraffic] = None,
):
    # Preserve the upstream default (seed 0) while making it explicit in the
    # resolved configuration for reproducible Base runs.
    random.seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)

    psb_stage = (
        str(psb_runtime_config.get("stage")) if psb_runtime_config else ""
    )
    is_psb_p1 = psb_stage == "p1_zero_control_equivalence"
    is_psb_p2 = psb_stage == "p2_frozen_base_bifurcation"
    if psb_runtime_config is not None and not (is_psb_p1 or is_psb_p2):
        raise ValueError(f"Unsupported PSB runtime stage: {psb_stage!r}.")
    if psb_action_projection not in {None, "full", "longitudinal_only"}:
        raise ValueError("Unsupported PSB action projection.")
    if psb_action_projection not in {None, "full"} and not is_psb_p2:
        raise ValueError("PSB action projection requires a P2 policy.")
    if psb_action_projection not in {None, "full"} and not bool(
        parameters.is_testing_mode
    ):
        raise ValueError("PSB action projection is inference-only.")

    resume_payload = None
    resume_start_iteration = 0
    if training_resume_checkpoint is not None:
        if is_psb_p2:
            from utilities.psb_marl.p2_checkpoint import load_p2_checkpoint

            resume_payload = load_p2_checkpoint(
                training_resume_checkpoint, parameters.device
            )
        else:
            from utilities.opinion.checkpoint import load_m9_checkpoint

            resume_payload = load_m9_checkpoint(
                training_resume_checkpoint, parameters.device
            )
        resume_start_iteration = int(resume_payload["iteration"])
        if resume_start_iteration >= parameters.n_iters:
            raise ValueError(
                "Resume checkpoint iteration must be smaller than configured n_iters."
            )

    # A4 reuses the exact Base-MAPPO construction while attaching additional
    # pre-reset diagnostics to the road scenario.  The default remains
    # byte-for-byte compatible with all training and legacy testing callers.
    scenario = (
        ScenarioRoadTraffic() if scenario_override is None else scenario_override
    )

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
        # VMAS maps seed=None to seed 0 and resets the global Python, NumPy,
        # and Torch RNGs during construction.  Passing the experiment seed is
        # therefore required both for environment diversity and for the P2
        # modules initialized immediately after the environment.
        seed=parameters.seed,
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
        from torchrl.envs.transforms import Compose
        from utilities.opinion.transforms import DiscreteDTypeCastTransform

        env_transform = Compose(
            reward_sum,
            DiscreteDTypeCastTransform(
                torch.float32,
                torch.long,
                n=parameters.n_agents,
                in_keys=[("agents", "info", "neighbor_ids")],
                in_keys_inv=[],
            ),
            DiscreteDTypeCastTransform(
                torch.float32,
                torch.bool,
                n=2,
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

    # TorchRL's default is seed=0 and the check mutates the environment plus
    # the global RNG state.  Preserve the configured experiment seed instead
    # of silently collapsing all nonzero training seeds back to zero.
    check_env_specs(env, seed=parameters.seed)

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

    opinion_mode = (
        str(opinion_policy_config.get("mode"))
        if opinion_policy_config
        else ""
    )
    is_direct_opinion = opinion_mode == "direct_evidence"
    is_stateful_opinion = opinion_mode == "stateful_opinion"
    is_opinion_policy = is_direct_opinion or is_stateful_opinion
    use_base_ppo_update = bool(
        opinion_policy_config
        and opinion_policy_config.get("use_base_ppo_update", False)
    )
    is_sequence_buffer = bool(
        is_stateful_opinion
        and opinion_policy_config
        and opinion_policy_config.get("sequence_buffer_enabled", False)
        and not use_base_ppo_update
    )
    is_sequence_evidence_training = bool(
        is_sequence_buffer
        and opinion_policy_config.get("sequence_evidence_training", False)
    )
    trainer_runtime_config = (
        dict(opinion_policy_config.get("trainer", {}))
        if opinion_policy_config
        else {}
    )
    is_m9_trainer = bool(trainer_runtime_config.get("enabled", False))
    initialize_from_scratch = bool(
        is_m9_trainer
        and opinion_policy_config.get("initialize_from_scratch", False)
    )
    if (
        training_resume_checkpoint is not None
        and not is_m9_trainer
        and not is_psb_p2
    ):
        raise ValueError("--resume is supported only by M9 and PSB P2 trainers.")
    if resume_payload is not None:
        if is_psb_p2:
            if resume_payload.get("runtime_config") != dict(psb_runtime_config):
                raise ValueError(
                    "P2 resume checkpoint runtime does not match the current config."
                )
        else:
            saved_runtime = resume_payload.get("opinion_runtime_config", {})
            if saved_runtime.get("trainer") != trainer_runtime_config:
                raise ValueError(
                    "Resume checkpoint Trainer configuration does not match the "
                    "current configuration."
                )
    opinion_bridge = None
    state_tracker = None
    base_actor_source_state = None
    psb_bridge = None
    psb_tracker = None
    if is_opinion_policy:
        if not emits_opinion_pair_info:
            raise ValueError("The Opinion policy bridge requires M4 pair info.")
        if not parameters.is_load_model:
            if initialize_from_scratch:
                base_actor_source_state = copy.deepcopy(base_policy.state_dict())
                if artifact_logging_enabled:
                    torch.save(
                        base_actor_source_state,
                        artifact_run_directory / "source_base_actor.pth",
                    )
                print(
                    colored(
                        "[INFO] Randomly initialized Base Actor for M9 "
                        "joint-from-scratch training.",
                        "red",
                    )
                )
            else:
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
                base_actor_source_state = copy.deepcopy(base_actor_source_state)
                if artifact_logging_enabled:
                    # Preserve the exact frozen source immediately so an interrupted
                    # M5/M6 run never depends on a later-deleted reward checkpoint.
                    torch.save(
                        base_actor_source_state,
                        artifact_run_directory / "source_base_actor.pth",
                    )
                print(
                    colored(
                        f"[INFO] Loaded "
                        f"{'initial' if is_m9_trainer else 'frozen'} Base Actor: "
                        f"{base_actor_checkpoint}",
                        "red",
                    )
                )

        from utilities.opinion.config import (
            DynamicsConfig,
            EvidenceConfig,
            ResidualConfig,
        )
        from utilities.opinion.dynamics import OpinionDynamics
        from utilities.opinion.evidence_net import OpinionEvidenceNet
        from utilities.opinion.policy import (
            DirectEvidencePolicyBridge,
            StatefulOpinionPolicyBridge,
            StatefulOpinionPolicyController,
        )
        from utilities.opinion.residual import OpinionResidual
        from utilities.opinion.state import OpinionStateTracker

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
        if (
            is_stateful_opinion
            and not parameters.is_load_model
            and not opinion_policy_config.get("initialize_evidence_random", False)
            and opinion_policy_config.get("evidence_checkpoint") is not None
        ):
            evidence_checkpoint = Path(
                str(opinion_policy_config["evidence_checkpoint"])
            )
            if not evidence_checkpoint.is_file():
                raise FileNotFoundError(
                    f"M5 EvidenceNet checkpoint not found: {evidence_checkpoint}"
                )
            evidence_net.load_state_dict(
                torch.load(evidence_checkpoint, map_location=parameters.device),
                strict=True,
            )
            print(
                colored(
                    f"[INFO] Loaded frozen M5 EvidenceNet: {evidence_checkpoint}",
                    "red",
                )
            )
        residual = OpinionResidual.from_config(residual_config).to(parameters.device)
        if is_stateful_opinion:
            dynamics_config = DynamicsConfig.from_dict(
                opinion_policy_config["dynamics"]
            )
            dynamics = OpinionDynamics.from_config(dynamics_config).to(
                parameters.device
            )
            opinion_bridge = StatefulOpinionPolicyBridge(
                base_policy_net=policy_net,
                evidence_net=evidence_net,
                dynamics=dynamics,
                residual=residual,
                dt=parameters.dt,
                freeze_base_actor=bool(
                    opinion_policy_config.get("freeze_base_actor", True)
                ),
                freeze_evidence=bool(
                    opinion_policy_config.get("freeze_evidence", True)
                ),
            ).to(parameters.device)
            opinion_in_keys = [
                observation_key,
                ("agents", "info", "pair_features"),
                ("agents", "info", "urgency"),
                ("agents", "info", "confidence"),
                ("agents", "info", "pair_mask"),
                ("agents", "opinion", "z_prev"),
            ]
            opinion_out_keys = [
                ("agents", "loc"),
                ("agents", "scale"),
                ("agents", "opinion", "base_loc"),
                ("agents", "opinion", "raw_b"),
                ("agents", "opinion", "b"),
                ("agents", "opinion", "z_next"),
                ("agents", "opinion", "q"),
                ("agents", "opinion", "normalized_weights"),
                ("agents", "opinion", "aggregate"),
                ("agents", "opinion", "residual"),
            ]
        else:
            opinion_bridge = DirectEvidencePolicyBridge(
                base_policy_net=policy_net,
                evidence_net=evidence_net,
                residual=residual,
                freeze_base_actor=bool(
                    opinion_policy_config.get("freeze_base_actor", True)
                ),
            ).to(parameters.device)
            opinion_in_keys = [
                observation_key,
                ("agents", "info", "pair_features"),
                ("agents", "info", "urgency"),
                ("agents", "info", "confidence"),
                ("agents", "info", "pair_mask"),
            ]
            opinion_out_keys = [
                ("agents", "loc"),
                ("agents", "scale"),
                ("agents", "opinion", "base_loc"),
                ("agents", "opinion", "raw_b"),
                ("agents", "opinion", "b"),
                ("agents", "opinion", "direct_z"),
                ("agents", "opinion", "residual"),
            ]
        opinion_policy_module = TensorDictModule(
            opinion_bridge,
            in_keys=opinion_in_keys,
            out_keys=opinion_out_keys,
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
        if (
            is_sequence_buffer
            and not parameters.is_load_model
            and opinion_policy_config.get("initial_policy_checkpoint") is not None
        ):
            initial_policy_checkpoint = Path(
                str(opinion_policy_config["initial_policy_checkpoint"])
            )
            if not initial_policy_checkpoint.is_file():
                raise FileNotFoundError(
                    "M6 source policy checkpoint not found: "
                    f"{initial_policy_checkpoint}"
                )
            policy.load_state_dict(
                torch.load(
                    initial_policy_checkpoint,
                    map_location=parameters.device,
                ),
                strict=True,
            )
            # The M6 full policy is authoritative for M7. Refresh the
            # separately saved Base state after the strict full-policy load.
            base_actor_source_state = copy.deepcopy(base_policy.state_dict())
            if artifact_logging_enabled:
                torch.save(
                    base_actor_source_state,
                    artifact_run_directory / "source_base_actor.pth",
                )
            print(
                colored(
                    "[INFO] Initialized sequence policy from its source: "
                    f"{initial_policy_checkpoint}",
                    "red",
                )
            )
        policy_for_collection = policy
        if is_stateful_opinion:
            decay_factor = 1.0 - (
                float(parameters.dt)
                * float(dynamics_config.response_rate)
                * float(dynamics_config.decay_rate)
            )
            state_tracker = OpinionStateTracker(
                n_agents=parameters.n_agents,
                decay_factor=decay_factor,
                zero_threshold=float(opinion_policy_config["zero_threshold"]),
            )
            policy_for_collection = StatefulOpinionPolicyController(
                policy=policy,
                state_tracker=state_tracker,
            ).to(parameters.device)
    else:
        policy = base_policy
        policy_for_collection = policy

    if psb_runtime_config is not None:
        if is_opinion_policy:
            raise ValueError(
                "PSB and legacy Opinion policy paths are mutually exclusive."
            )
        if not emits_opinion_pair_info:
            raise ValueError("The PSB runtime requires local conflict pair info.")
        proximal_config = dict(psb_runtime_config["proximal"])
        from utilities.psb_marl.proximal import ProximalSaturatingBifurcation

        proximal_layer = ProximalSaturatingBifurcation.from_runtime_config(
            proximal_config
        ).to(parameters.device)
        if is_psb_p1:
            from utilities.psb_marl.policy import (
                P1ZeroControlPolicyController,
                validate_p1_runtime_contract,
            )
            from utilities.psb_marl.state import P1ZeroControlStateTracker

            validate_p1_runtime_contract(psb_runtime_config, parameters.n_agents)
            psb_tracker = P1ZeroControlStateTracker(
                n_agents=parameters.n_agents,
                proximal=proximal_layer,
                zero_threshold=float(proximal_config["zero_threshold"]),
            )
            policy_for_collection = P1ZeroControlPolicyController(
                policy=policy,
                tracker=psb_tracker,
            ).to(parameters.device)
        else:
            from utilities.psb_marl.p2_network import (
                AntisymmetricBifurcationControl,
                BranchContextEncoder,
                BranchDistributionAdapter,
            )
            from utilities.psb_marl.p2_policy import (
                FrozenBaseBifurcationPolicyBridge,
                P2PolicyController,
                validate_p2_runtime_contract,
            )
            from utilities.psb_marl.p2_state import P2EdgeStateTracker

            validate_p2_runtime_contract(psb_runtime_config, parameters.n_agents)
            base_actor_checkpoint = Path(
                str(psb_runtime_config["base_policy_checkpoint"])
            ).expanduser().resolve()
            if not base_actor_checkpoint.is_file():
                raise FileNotFoundError(
                    f"P2 Base Actor checkpoint not found: {base_actor_checkpoint}"
                )
            base_actor_source_state = torch.load(
                base_actor_checkpoint, map_location=parameters.device
            )
            base_policy.load_state_dict(base_actor_source_state, strict=True)
            base_actor_source_state = copy.deepcopy(base_actor_source_state)
            if artifact_logging_enabled:
                torch.save(
                    base_actor_source_state,
                    artifact_run_directory / "source_base_actor.pth",
                )

            control_config = dict(psb_runtime_config["control"])
            adapter_config = dict(psb_runtime_config["branch_adapter"])
            configured_action_projection = str(
                adapter_config.get("action_projection", "full")
            )
            if configured_action_projection not in {
                "full",
                "longitudinal_only",
            }:
                raise ValueError("Unsupported configured P2 action projection.")
            effective_action_projection = (
                configured_action_projection
                if psb_action_projection is None
                else psb_action_projection
            )
            control_net = AntisymmetricBifurcationControl(
                pair_feature_dim=int(
                    opinion_pair_info_config["pair_feature_dim"]
                ),
                hidden_sizes=tuple(control_config["hidden_sizes"]),
                b_max=float(proximal_config["b_max"]),
                temperature=float(control_config["temperature"]),
                support_power=float(control_config["support_power"]),
                critical_gate_enabled=bool(
                    control_config["critical_gate_enabled"]
                ),
                critical_width=float(control_config["critical_width"]),
                critical_floor=float(control_config["critical_floor"]),
                final_layer_gain=float(control_config["final_layer_gain"]),
                rho_c=float(proximal_config["rho_c"]),
                rho_max=float(proximal_config["rho_max"]),
            ).to(parameters.device)
            branch_encoder = BranchContextEncoder(
                pair_feature_dim=int(
                    opinion_pair_info_config["pair_feature_dim"]
                ),
                hidden_sizes=tuple(adapter_config["pair_hidden_sizes"]),
                context_dim=int(adapter_config["context_dim"]),
                z_scale=float(adapter_config["z_scale"]),
                rho_max=float(proximal_config["rho_max"]),
                conditioning_mode=str(
                    adapter_config.get("conditioning_mode", "general")
                ),
            ).to(parameters.device)
            observation_dim = int(
                env.observation_spec[observation_key].shape[-1]
            )
            action_dim = int(env.action_spec.shape[-1])
            mean_action_mask = None
            if effective_action_projection == "longitudinal_only":
                if action_dim != 2:
                    raise ValueError(
                        "Longitudinal projection requires native "
                        "[speed, steering] actions."
                    )
                mean_action_mask = (1.0, 0.0)
            adapter = BranchDistributionAdapter(
                observation_dim=observation_dim,
                context_dim=int(adapter_config["context_dim"]),
                action_dim=action_dim,
                hidden_sizes=tuple(adapter_config["adapter_hidden_sizes"]),
                max_delta_loc=float(adapter_config["max_delta_loc"]),
                max_delta_log_scale=float(
                    adapter_config["max_delta_log_scale"]
                ),
                conditioning_mode=str(
                    adapter_config.get("conditioning_mode", "general")
                ),
                mean_action_mask=mean_action_mask,
            ).to(parameters.device)
            psb_bridge = FrozenBaseBifurcationPolicyBridge(
                base_policy_net=policy_net,
                control_net=control_net,
                proximal=proximal_layer,
                branch_encoder=branch_encoder,
                adapter=adapter,
                n_agents=parameters.n_agents,
            ).to(parameters.device)
            psb_module = TensorDictModule(
                psb_bridge,
                in_keys=[
                    observation_key,
                    ("agents", "info", "pair_features"),
                    ("agents", "info", "neighbor_ids"),
                    ("agents", "info", "urgency"),
                    ("agents", "info", "confidence"),
                    ("agents", "info", "pair_mask"),
                    ("agents", "psb", "z_prev_dense"),
                ],
                out_keys=[
                    ("agents", "loc"),
                    ("agents", "scale"),
                    ("agents", "psb", "base_loc"),
                    ("agents", "psb", "base_scale"),
                    ("agents", "psb", "raw_b"),
                    ("agents", "psb", "b_candidates"),
                    ("agents", "psb", "b"),
                    ("agents", "psb", "rho"),
                    ("agents", "psb", "z_next_dense"),
                    ("agents", "psb", "z_next"),
                    ("agents", "psb", "q"),
                    ("agents", "psb", "attention"),
                    ("agents", "psb", "branch_context"),
                    ("agents", "psb", "branch_activity"),
                    ("agents", "psb", "delta_loc"),
                    ("agents", "psb", "delta_log_scale"),
                    ("agents", "psb", "root_residual"),
                    ("agents", "psb", "root_denominator"),
                ],
            )
            policy = ProbabilisticActor(
                module=psb_module,
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
            psb_tracker = P2EdgeStateTracker(
                n_agents=parameters.n_agents,
                zero_threshold=float(proximal_config["zero_threshold"]),
            )
            policy_for_collection = P2PolicyController(
                policy=policy,
                tracker=psb_tracker,
            ).to(parameters.device)

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

    # The critic is unused during pure policy evaluation.  Skipping this
    # training-only warm start also permits a decentralized P2 actor trained
    # with N agents to be evaluated in a different-N scenario: the Base
    # centralized critic has an N-dependent input shape, while the actor and
    # pairwise PSB modules do not.
    if is_psb_p2 and not parameters.is_load_model:
        base_critic_checkpoint = Path(
            str(psb_runtime_config["base_critic_checkpoint"])
        ).expanduser().resolve()
        if not base_critic_checkpoint.is_file():
            raise FileNotFoundError(
                f"P2 Base Critic checkpoint not found: {base_critic_checkpoint}"
            )
        critic.load_state_dict(
            torch.load(base_critic_checkpoint, map_location=parameters.device),
            strict=True,
        )
        from utilities.psb_marl.p2_critic import AugmentedCentralCritic

        augmented_critic_net = AugmentedCentralCritic(
            base_critic_net=critic_net,
            n_agents=parameters.n_agents,
            observation_dim=int(
                env.observation_spec[observation_key].shape[-1]
            ),
            candidate_count=int(
                opinion_pair_info_config["candidate_count"]
            ),
        ).to(parameters.device)
        critic = TensorDictModule(
            module=augmented_critic_net,
            in_keys=[
                observation_key,
                ("agents", "psb", "z_prev_dense"),
                ("agents", "info", "pair_mask"),
            ],
            out_keys=[("agents", "state_value")],
        )

    if (
        is_opinion_policy
        and not parameters.is_load_model
        and not initialize_from_scratch
    ):
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
                f"[INFO] Initialized Opinion Critic from: {base_critic_checkpoint}",
                "red",
            )
        )
    elif is_opinion_policy and initialize_from_scratch:
        print(
            colored(
                "[INFO] Randomly initialized Central Critic for M9 "
                "joint-from-scratch training.",
                "red",
            )
        )

    if resume_payload is not None:
        if not is_psb_p2 and (
            resume_payload["training_mode"] != trainer_runtime_config.get("mode")
        ):
            raise ValueError(
                "Resume checkpoint training mode does not match current config."
            )
        policy.load_state_dict(resume_payload["policy_state"], strict=True)
        critic.load_state_dict(resume_payload["critic_state"], strict=True)
        if (
            not is_psb_p2
            and resume_payload.get("base_source_state") is not None
        ):
            base_actor_source_state = copy.deepcopy(
                resume_payload["base_source_state"]
            )
        print(
            colored(
                f"[INFO] Restored {'P2' if is_psb_p2 else 'M9'} model state at iteration "
                f"{resume_start_iteration}: {training_resume_checkpoint}",
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

            return env, policy_for_collection, priority_module, parameters
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
        policy_for_collection,
        priority_module=priority_module,
        device=parameters.device,
        storing_device=parameters.device,
        frames_per_batch=parameters.frames_per_batch,
        total_frames=(
            parameters.frames_per_batch
            * (parameters.n_iters - resume_start_iteration)
        ),
    )

    uses_sequence_buffer = is_sequence_buffer or is_psb_p2
    if uses_sequence_buffer:
        replay_buffer = None
    elif parameters.is_prb:
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

    base_anchor_net = None
    if is_m9_trainer and float(
        trainer_runtime_config.get("base_anchor_coefficient", 0.0)
    ) > 0.0:
        base_anchor_net = copy.deepcopy(policy_net).to(parameters.device)
        if resume_payload is not None and resume_payload.get("base_anchor_state"):
            base_anchor_net.load_state_dict(
                resume_payload["base_anchor_state"], strict=True
            )
        base_anchor_net.eval()
        for parameter in base_anchor_net.parameters():
            parameter.requires_grad_(False)

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

    sequence_ppo_loss = None
    p2_sequence_loss = None
    if is_sequence_evidence_training:
        from utilities.opinion.sequence_ppo import OpinionSequencePPOLoss

        sequence_ppo_loss = OpinionSequencePPOLoss(
            actor=policy,
            bridge=opinion_bridge,
            observation_key=observation_key,
            action_key=env.action_key,
            advantage_key=loss_module.tensor_keys.advantage,
            n_agents=parameters.n_agents,
            clip_epsilon=parameters.clip_epsilon,
            entropy_coefficient=parameters.entropy_eps,
            neutral_loss_coefficient=float(
                opinion_policy_config["neutral_loss_coefficient"]
            ),
            magnitude_loss_coefficient=float(
                opinion_policy_config["magnitude_loss_coefficient"]
            ),
            decay_factor=1.0
            - (
                float(parameters.dt)
                * float(dynamics_config.response_rate)
                * float(dynamics_config.decay_rate)
            ),
            zero_threshold=float(opinion_policy_config["zero_threshold"]),
            base_anchor_net=base_anchor_net,
            base_anchor_coefficient=float(
                trainer_runtime_config.get("base_anchor_coefficient", 0.0)
            ),
        )

    if is_psb_p2:
        from utilities.psb_marl.p2_loss import P2SequencePPOLoss

        p2_training_config = dict(psb_runtime_config["training"])
        p2_sequence_loss = P2SequencePPOLoss(
            actor=policy,
            bridge=psb_bridge,
            observation_key=observation_key,
            action_key=env.action_key,
            advantage_key=loss_module.tensor_keys.advantage,
            n_agents=parameters.n_agents,
            clip_epsilon=parameters.clip_epsilon,
            entropy_coefficient=parameters.entropy_eps,
            energy_coefficient=float(
                p2_training_config["energy_coefficient"]
            ),
            control_trust_region_coefficient=float(
                p2_training_config["control_trust_region_coefficient"]
            ),
            saturation_coefficient=float(
                p2_training_config["saturation_coefficient"]
            ),
            saturation_fraction=float(
                p2_training_config["saturation_fraction"]
            ),
        )

    training_schedule = None
    current_training_phase = None
    if is_psb_p2:
        trainable_groups = psb_bridge.trainable_groups()
        control_parameters = trainable_groups["control"]
        adapter_parameters = trainable_groups["adapter"]
        critic_trainable_parameters = [
            parameter
            for parameter in loss_module.critic_params.values(True, True)
            if parameter.requires_grad
        ]
        if not control_parameters or not adapter_parameters:
            raise RuntimeError("P2 Actor parameter groups must be non-empty.")
        if not critic_trainable_parameters:
            raise RuntimeError("P2 Critic parameter group must be non-empty.")
        if any(
            parameter.requires_grad
            for parameter in psb_bridge.base_policy_net.parameters()
        ):
            raise RuntimeError("P2 Base Actor must remain frozen.")
        optim = torch.optim.Adam(
            [
                {
                    "params": control_parameters,
                    "lr": parameters.lr
                    * float(
                        p2_training_config["control_learning_rate_scale"]
                    ),
                    "lr_scale": float(
                        p2_training_config["control_learning_rate_scale"]
                    ),
                    "group_name": "control",
                },
                {
                    "params": adapter_parameters,
                    "lr": parameters.lr
                    * float(
                        p2_training_config["adapter_learning_rate_scale"]
                    ),
                    "lr_scale": float(
                        p2_training_config["adapter_learning_rate_scale"]
                    ),
                    "group_name": "adapter",
                },
                {
                    "params": critic_trainable_parameters,
                    "lr": parameters.lr
                    * float(
                        p2_training_config["critic_learning_rate_scale"]
                    ),
                    "lr_scale": float(
                        p2_training_config["critic_learning_rate_scale"]
                    ),
                    "group_name": "critic",
                },
            ]
        )
        if resume_payload is not None:
            from utilities.psb_marl.p2_checkpoint import restore_p2_rng_state

            optim.load_state_dict(resume_payload["optimizer_state"])
            restore_p2_rng_state(resume_payload)
    elif is_m9_trainer:
        from utilities.opinion.trainer import (
            OpinionTrainingSchedule,
            clip_m9_gradients,
        )

        schedule_config = dict(trainer_runtime_config)
        schedule_config["evidence_learning_rate_scale"] = float(
            opinion_policy_config["evidence_learning_rate_scale"]
        )
        training_schedule = OpinionTrainingSchedule(schedule_config)
        base_actor_parameters = list(opinion_bridge.base_policy_net.parameters())
        evidence_parameters = list(opinion_bridge.evidence_net.parameters())
        critic_trainable_parameters = [
            parameter
            for parameter in loss_module.critic_params.values(True, True)
            if parameter.requires_grad
        ]
        if not base_actor_parameters or not evidence_parameters:
            raise RuntimeError("M9 Actor parameter groups must be non-empty.")
        if not critic_trainable_parameters:
            raise RuntimeError("M9 Critic parameter group must be non-empty.")
        optim = torch.optim.Adam(
            [
                {
                    "params": base_actor_parameters,
                    "lr": parameters.lr
                    * training_schedule.base_actor_lr_scale,
                    "lr_scale": training_schedule.base_actor_lr_scale,
                    "group_name": "base_actor",
                },
                {
                    "params": evidence_parameters,
                    "lr": parameters.lr * training_schedule.evidence_lr_scale,
                    "lr_scale": training_schedule.evidence_lr_scale,
                    "group_name": "evidence",
                },
                {
                    "params": critic_trainable_parameters,
                    "lr": parameters.lr * training_schedule.critic_lr_scale,
                    "lr_scale": training_schedule.critic_lr_scale,
                    "group_name": "critic",
                },
            ]
        )
        if resume_payload is not None:
            optim.load_state_dict(resume_payload["optimizer_state"])
        current_training_phase = training_schedule.phase_for_iteration(
            resume_start_iteration + 1
        )
        training_schedule.apply(
            current_training_phase, opinion_bridge, optim
        )
        if resume_payload is not None:
            from utilities.opinion.checkpoint import restore_rng_state

            restore_rng_state(resume_payload)
    elif is_direct_opinion:
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
    elif is_stateful_opinion and is_sequence_evidence_training:
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
        evidence_parameters = list(opinion_bridge.evidence_net.parameters())
        if not actor_trainable_parameters or not evidence_parameters:
            raise RuntimeError("M8 EvidenceNet has no trainable parameters.")
        if {id(parameter) for parameter in actor_trainable_parameters} != {
            id(parameter) for parameter in evidence_parameters
        }:
            raise RuntimeError(
                "M8 actor parameters must contain only the shared EvidenceNet."
            )
        if any(
            parameter.requires_grad
            for parameter in opinion_bridge.base_policy_net.parameters()
        ):
            raise RuntimeError("M8 Base Actor must remain frozen.")
        evidence_lr_scale = float(
            opinion_policy_config["evidence_learning_rate_scale"]
        )
        optim = torch.optim.Adam(
            [
                {
                    "params": evidence_parameters,
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
    elif is_stateful_opinion:
        critic_trainable_parameters = [
            parameter
            for parameter in loss_module.critic_params.values(True, True)
            if parameter.requires_grad
        ]
        if not critic_trainable_parameters:
            raise RuntimeError("M6/M7 Critic has no trainable parameters.")
        if any(
            parameter.requires_grad
            for parameter in loss_module.actor_params.values(True, True)
        ):
            raise RuntimeError(
                "M6/M7 require Base Actor and EvidenceNet to remain frozen."
            )
        optim = torch.optim.Adam(
            [
                {
                    "params": critic_trainable_parameters,
                    "lr": parameters.lr,
                    "lr_scale": 1.0,
                    "group_name": "critic",
                }
            ]
        )
    else:
        optim = torch.optim.Adam(loss_module.parameters(), parameters.lr)
    optimization_parameters = [
        parameter
        for group in optim.param_groups
        for parameter in group["params"]
    ]

    pbar = tqdm(
        total=parameters.n_iters,
        initial=resume_start_iteration,
        desc="epi_rew_mean = 0",
    )

    episode_reward_mean_list = []

    t_start = time.time()
    iteration_cycle_start = t_start
    if resume_payload is not None:
        artifact_iterations = list(resume_payload["artifact_iterations"])
        episode_reward_mean_list = [
            float(item["episode_reward_mean"]) for item in artifact_iterations
        ]

    for tensordict_data in collector:
        rollout_finished_at = time.time()
        rollout_seconds = rollout_finished_at - iteration_cycle_start
        optimization_started_at = rollout_finished_at
        p2_rollout_diagnostics = {}
        loss_objective_sum = 0.0
        loss_critic_sum = 0.0
        loss_entropy_sum = 0.0
        loss_regularization_sum = 0.0
        loss_base_anchor_sum = 0.0
        evidence_gradient_norm_sum = 0.0
        base_actor_gradient_norm_sum = 0.0
        sequence_approx_kl_sum = 0.0
        sequence_clip_fraction_sum = 0.0
        sequence_log_prob_error_sum = 0.0
        sequence_state_replay_error_sum = 0.0
        sequence_neutral_penalty_sum = 0.0
        sequence_magnitude_penalty_sum = 0.0
        p2_control_energy_sum = 0.0
        p2_control_trust_sum = 0.0
        p2_saturation_penalty_sum = 0.0
        p2_max_root_residual = 0.0
        p2_min_root_denominator = float("inf")
        p2_mean_abs_b_sum = 0.0
        p2_mean_abs_z_sum = 0.0
        p2_mean_abs_delta_loc_sum = 0.0
        p2_control_gradient_norm_sum = 0.0
        p2_adapter_gradient_norm_sum = 0.0
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

        if is_psb_p2:
            from utilities.psb_marl.p2_diagnostics import (
                p2_state_diagnostics,
                p2_zero_branch_counterfactual_diagnostics,
            )
            from utilities.psb_marl.p2_state import P2EdgeStateTracker

            if psb_bridge is None:
                raise RuntimeError("P2 diagnostics require the policy bridge.")
            p2_rollout_diagnostics = p2_state_diagnostics(
                tensordict_data,
                rho_c=float(psb_runtime_config["proximal"]["rho_c"]),
                z_scale=float(
                    psb_runtime_config["branch_adapter"]["z_scale"]
                ),
            )
            p2_rollout_diagnostics.update(
                p2_zero_branch_counterfactual_diagnostics(
                    tensordict_data,
                    bridge=psb_bridge,
                )
            )

            task_reward = tensordict_data.get(("next", env.reward_key))
            tensordict_data.set(
                ("next", "agents", "psb", "task_reward"),
                task_reward.detach().clone(),
            )
            collected_b = tensordict_data.get(("agents", "psb", "b"))
            upper_mask = torch.triu(
                torch.ones(
                    parameters.n_agents,
                    parameters.n_agents,
                    dtype=torch.bool,
                    device=collected_b.device,
                ),
                diagonal=1,
            )
            edge_count = float(
                parameters.n_agents * (parameters.n_agents - 1) // 2
            )
            rollout_control_energy = (
                collected_b.square()[..., upper_mask].sum(dim=-1)
                / edge_count
            )
            reward_penalty = (
                float(p2_training_config["energy_coefficient"])
                * rollout_control_energy.unsqueeze(-1).unsqueeze(-1)
            )
            tensordict_data.set(
                ("next", env.reward_key),
                task_reward - reward_penalty.expand_as(task_reward),
            )

            z_for_next_value = P2EdgeStateTracker.apply_resets(
                tensordict_data.get(("agents", "psb", "z_next_dense")),
                tensordict_data.get(
                    ("next", "agents", "info", "agent_reset_mask")
                ),
                tensordict_data.get(("next", "done")),
            )
            tensordict_data.set(
                ("next", "agents", "psb", "z_prev_dense"),
                z_for_next_value,
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

        sequence_iteration_metrics = {}
        sequence_buffer = None
        if uses_sequence_buffer:
            if is_psb_p2:
                from utilities.psb_marl.p2_buffer import P2SequenceBuffer

                sequence_buffer = P2SequenceBuffer(
                    tensordict_data,
                    chunk_length=int(p2_training_config["chunk_length"]),
                )
            else:
                from utilities.opinion.sequence_buffer import OpinionSequenceBuffer

                sequence_buffer = OpinionSequenceBuffer(
                    tensordict_data,
                    chunk_length=int(opinion_policy_config["chunk_length"]),
                )
            sequence_iteration_metrics = sequence_buffer.diagnostics()
        else:
            data_view = tensordict_data.reshape(
                -1
            )  # Flatten the batch size to shuffle data
            replay_buffer.extend(data_view)
            # replay_buffer.update_tensordict_priority() # Not necessary, as priorities were updated automatically when calling `replay_buffer.extend()`

        for _ in range(parameters.num_epochs):
            if uses_sequence_buffer:
                if is_sequence_evidence_training:
                    mini_batches = sequence_buffer.iter_sequence_minibatches(
                        minibatch_size=parameters.minibatch_size,
                    )
                else:
                    mini_batches = sequence_buffer.iter_minibatches(
                        minibatch_size=parameters.minibatch_size,
                    )
            else:
                def replay_minibatches():
                    for _ in range(
                        parameters.frames_per_batch
                        // parameters.minibatch_size
                    ):
                        mini_batch, _ = replay_buffer.sample(return_info=True)
                        yield mini_batch

                mini_batches = replay_minibatches()
            for mini_batch_data in mini_batches:

                if is_psb_p2:
                    loss_vals = p2_sequence_loss(mini_batch_data)
                    critic_batch = mini_batch_data.tensordict.reshape(-1)
                    loss_vals["loss_critic"] = loss_module.loss_critic(
                        critic_batch
                    ).mean()
                elif is_sequence_evidence_training:
                    loss_vals = sequence_ppo_loss(mini_batch_data)
                    critic_batch = mini_batch_data.tensordict.reshape(-1)
                    loss_vals["loss_critic"] = loss_module.loss_critic(
                        critic_batch
                    ).mean()
                else:
                    loss_vals = loss_module(mini_batch_data)

                loss_objective_sum += (
                    loss_vals["loss_objective"].detach().mean().item()
                )
                loss_critic_sum += loss_vals["loss_critic"].detach().mean().item()
                loss_entropy_sum += loss_vals["loss_entropy"].detach().mean().item()
                if is_sequence_evidence_training:
                    loss_regularization_sum += (
                        loss_vals["loss_regularization"].detach().item()
                    )
                    loss_base_anchor_sum += (
                        loss_vals["loss_base_anchor"].detach().item()
                    )
                    sequence_approx_kl_sum += loss_vals["approx_kl"].item()
                    sequence_clip_fraction_sum += loss_vals["clip_fraction"].item()
                    sequence_log_prob_error_sum += loss_vals[
                        "log_prob_abs_error"
                    ].item()
                    sequence_state_replay_error_sum += loss_vals[
                        "state_replay_abs_error"
                    ].item()
                    sequence_neutral_penalty_sum += loss_vals[
                        "neutral_penalty"
                    ].item()
                    sequence_magnitude_penalty_sum += loss_vals[
                        "magnitude_penalty"
                    ].item()
                if is_psb_p2:
                    loss_regularization_sum += float(
                        loss_vals["loss_regularization"].detach().item()
                    )
                    sequence_approx_kl_sum += float(
                        loss_vals["approx_kl"].item()
                    )
                    sequence_clip_fraction_sum += float(
                        loss_vals["clip_fraction"].item()
                    )
                    sequence_log_prob_error_sum += float(
                        loss_vals["log_prob_abs_error"].item()
                    )
                    sequence_state_replay_error_sum += float(
                        loss_vals["state_replay_abs_error"].item()
                    )
                    p2_control_energy_sum += float(
                        loss_vals["control_energy"].item()
                    )
                    p2_control_trust_sum += float(
                        loss_vals["control_trust"].item()
                    )
                    p2_saturation_penalty_sum += float(
                        loss_vals["saturation_penalty"].item()
                    )
                    p2_max_root_residual = max(
                        p2_max_root_residual,
                        float(loss_vals["max_root_residual"].item()),
                    )
                    p2_min_root_denominator = min(
                        p2_min_root_denominator,
                        float(loss_vals["min_root_denominator"].item()),
                    )
                    p2_mean_abs_b_sum += float(
                        loss_vals["mean_abs_b"].item()
                    )
                    p2_mean_abs_z_sum += float(
                        loss_vals["mean_abs_z"].item()
                    )
                    p2_mean_abs_delta_loc_sum += float(
                        loss_vals["mean_abs_delta_loc"].item()
                    )
                loss_update_count += 1

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                    + loss_vals.get("loss_regularization", 0.0)
                    + loss_vals.get("loss_base_anchor", 0.0)
                )

                assert not loss_value.isnan().any()
                assert not loss_value.isinf().any()

                loss_value.backward()

                if is_sequence_evidence_training:
                    squared_gradient_norm = torch.zeros(
                        (), device=parameters.device
                    )
                    for parameter in opinion_bridge.evidence_net.parameters():
                        if parameter.grad is not None:
                            squared_gradient_norm = squared_gradient_norm + (
                                parameter.grad.detach().norm(2).square()
                            )
                    evidence_gradient_norm_sum += float(
                        squared_gradient_norm.sqrt().item()
                    )
                if is_m9_trainer:
                    squared_base_gradient_norm = torch.zeros(
                        (), device=parameters.device
                    )
                    for parameter in opinion_bridge.base_policy_net.parameters():
                        if parameter.grad is not None:
                            squared_base_gradient_norm = (
                                squared_base_gradient_norm
                                + parameter.grad.detach().norm(2).square()
                            )
                    base_actor_gradient_norm_sum += float(
                        squared_base_gradient_norm.sqrt().item()
                    )
                if is_psb_p2:
                    squared_control_gradient_norm = torch.zeros(
                        (), device=parameters.device
                    )
                    for parameter in control_parameters:
                        if parameter.grad is not None:
                            squared_control_gradient_norm += (
                                parameter.grad.detach().norm(2).square()
                            )
                    p2_control_gradient_norm_sum += float(
                        squared_control_gradient_norm.sqrt().item()
                    )
                    squared_adapter_gradient_norm = torch.zeros(
                        (), device=parameters.device
                    )
                    for parameter in adapter_parameters:
                        if parameter.grad is not None:
                            squared_adapter_gradient_norm += (
                                parameter.grad.detach().norm(2).square()
                            )
                    p2_adapter_gradient_norm_sum += float(
                        squared_adapter_gradient_norm.sqrt().item()
                    )

                if is_m9_trainer:
                    clip_m9_gradients(optim, parameters.max_grad_norm)
                else:
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
        if is_opinion_policy:
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
            if is_stateful_opinion:
                z_prev_collected = tensordict_data.get(
                    ("agents", "opinion", "z_prev")
                )
                z_next_collected = tensordict_data.get(
                    ("agents", "opinion", "z_next")
                )
                opinion_iteration_metrics.update(
                    {
                        "stateful_z_abs_mean": float(
                            z_next_collected.abs().mean().item()
                        ),
                        "stateful_z_max_abs": float(
                            z_next_collected.abs().max().item()
                        ),
                        "stateful_delta_z_abs_mean": float(
                            (z_next_collected - z_prev_collected)
                            .abs()
                            .mean()
                            .item()
                        ),
                        "agent_reset_fraction": float(
                            tensordict_data.get(
                                ("agents", "info", "agent_reset_mask")
                            )
                            .float()
                            .mean()
                            .item()
                        ),
                    }
                )
        p2_iteration_metrics = {}
        if is_psb_p2:
            p2_collected_b = tensordict_data.get(("agents", "psb", "b"))
            p2_collected_z = tensordict_data.get(
                ("agents", "psb", "z_next_dense")
            )
            p2_collected_delta = tensordict_data.get(
                ("agents", "psb", "delta_loc")
            )
            p2_collected_activity = tensordict_data.get(
                ("agents", "psb", "branch_activity")
            )
            p2_collected_delta_log_scale = tensordict_data.get(
                ("agents", "psb", "delta_log_scale")
            )
            p2_collected_scale = tensordict_data.get(("agents", "scale"))
            p2_collected_base_scale = tensordict_data.get(
                ("agents", "psb", "base_scale")
            )
            p2_collected_root = tensordict_data.get(
                ("agents", "psb", "root_residual")
            )
            p2_collected_denominator = tensordict_data.get(
                ("agents", "psb", "root_denominator")
            )
            p2_iteration_metrics = {
                "rollout_control_energy": float(
                    rollout_control_energy.mean().item()
                ),
                "rollout_b_abs_mean": float(p2_collected_b.abs().mean().item()),
                "rollout_z_abs_mean": float(p2_collected_z.abs().mean().item()),
                "rollout_delta_loc_abs_mean": float(
                    p2_collected_delta.abs().mean().item()
                ),
                "rollout_branch_activity_mean": float(
                    p2_collected_activity.mean().item()
                ),
                "rollout_branch_activity_max": float(
                    p2_collected_activity.max().item()
                ),
                "rollout_delta_log_scale_abs_mean": float(
                    p2_collected_delta_log_scale.abs().mean().item()
                ),
                "rollout_delta_log_scale_abs_max": float(
                    p2_collected_delta_log_scale.abs().max().item()
                ),
                "rollout_scale_matches_base_exactly": bool(
                    torch.equal(p2_collected_scale, p2_collected_base_scale)
                ),
                "rollout_max_root_residual": float(
                    p2_collected_root.abs().max().item()
                ),
                "rollout_min_root_denominator": float(
                    p2_collected_denominator.min().item()
                ),
                "rollout_z_antisymmetry_error": float(
                    (
                        p2_collected_z
                        + p2_collected_z.transpose(-1, -2)
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "base_actor_frozen": True,
                **p2_rollout_diagnostics,
            }
            if str(
                adapter_config.get("conditioning_mode", "general")
            ) in {"sector_q_gate", "supported_sector_q_gate"}:
                p2_sector_bound = (
                    float(adapter_config["max_delta_loc"])
                    * p2_collected_activity
                )
                p2_iteration_metrics[
                    "rollout_sector_bound_max_violation"
                ] = float(
                    (
                        p2_collected_delta.abs() - p2_sector_bound
                    )
                    .clamp_min(0.0)
                    .max()
                    .item()
                )
            if p2_collected_delta.shape[-1] == 2:
                p2_speed_delta = p2_collected_delta[..., 0].abs()
                p2_steering_delta = p2_collected_delta[..., 1].abs()
                p2_iteration_metrics.update(
                    {
                        "rollout_delta_speed_abs_mean": float(
                            p2_speed_delta.mean().item()
                        ),
                        "rollout_delta_speed_abs_p95": float(
                            torch.quantile(p2_speed_delta, 0.95).item()
                        ),
                        "rollout_delta_speed_abs_max": float(
                            p2_speed_delta.max().item()
                        ),
                        "rollout_delta_steering_abs_mean": float(
                            p2_steering_delta.mean().item()
                        ),
                        "rollout_delta_steering_abs_p95": float(
                            torch.quantile(p2_steering_delta, 0.95).item()
                        ),
                        "rollout_delta_steering_abs_max": float(
                            p2_steering_delta.max().item()
                        ),
                    }
                )
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
                if is_opinion_policy:
                    if opinion_bridge is None:
                        raise RuntimeError(
                            "Opinion EvidenceNet is unavailable while saving a checkpoint."
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
                    **(
                        {
                            "loss_evidence_regularization": (
                                loss_regularization_sum / loss_update_count
                            ),
                            "loss_base_anchor": (
                                loss_base_anchor_sum / loss_update_count
                            ),
                            "evidence_gradient_norm": (
                                evidence_gradient_norm_sum / loss_update_count
                            ),
                            "sequence_approx_kl": (
                                sequence_approx_kl_sum / loss_update_count
                            ),
                            "sequence_clip_fraction": (
                                sequence_clip_fraction_sum / loss_update_count
                            ),
                            "sequence_log_prob_abs_error": (
                                sequence_log_prob_error_sum / loss_update_count
                            ),
                            "sequence_state_replay_abs_error": (
                                sequence_state_replay_error_sum
                                / loss_update_count
                            ),
                            "evidence_neutral_penalty": (
                                sequence_neutral_penalty_sum
                                / loss_update_count
                            ),
                            "evidence_magnitude_penalty": (
                                sequence_magnitude_penalty_sum
                                / loss_update_count
                            ),
                        }
                        if is_sequence_evidence_training
                        else {}
                    ),
                    **(
                        {
                            "loss_psb_regularization": (
                                loss_regularization_sum / loss_update_count
                            ),
                            "sequence_approx_kl": (
                                sequence_approx_kl_sum / loss_update_count
                            ),
                            "sequence_clip_fraction": (
                                sequence_clip_fraction_sum / loss_update_count
                            ),
                            "sequence_log_prob_abs_error": (
                                sequence_log_prob_error_sum / loss_update_count
                            ),
                            "sequence_state_replay_abs_error": (
                                sequence_state_replay_error_sum
                                / loss_update_count
                            ),
                            "control_energy": (
                                p2_control_energy_sum / loss_update_count
                            ),
                            "control_trust": (
                                p2_control_trust_sum / loss_update_count
                            ),
                            "saturation_penalty": (
                                p2_saturation_penalty_sum / loss_update_count
                            ),
                            "max_root_residual": p2_max_root_residual,
                            "min_root_denominator": p2_min_root_denominator,
                            "mean_abs_b": p2_mean_abs_b_sum / loss_update_count,
                            "mean_abs_z": p2_mean_abs_z_sum / loss_update_count,
                            "mean_abs_delta_loc": (
                                p2_mean_abs_delta_loc_sum / loss_update_count
                            ),
                            "control_gradient_norm": (
                                p2_control_gradient_norm_sum / loss_update_count
                            ),
                            "adapter_gradient_norm": (
                                p2_adapter_gradient_norm_sum / loss_update_count
                            ),
                            "control_learning_rate": float(
                                next(
                                    group["lr"]
                                    for group in optim.param_groups
                                    if group["group_name"] == "control"
                                )
                            ),
                            "adapter_learning_rate": float(
                                next(
                                    group["lr"]
                                    for group in optim.param_groups
                                    if group["group_name"] == "adapter"
                                )
                            ),
                            "critic_learning_rate": float(
                                next(
                                    group["lr"]
                                    for group in optim.param_groups
                                    if group["group_name"] == "critic"
                                )
                            ),
                        }
                        if is_psb_p2
                        else {}
                    ),
                    "learning_rate": float(
                        optim.param_groups[-1]["lr"]
                        if is_opinion_policy
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
                        if (
                            (is_direct_opinion or is_sequence_evidence_training)
                            and not is_m9_trainer
                        )
                        else {}
                    ),
                    **(
                        {
                            "evidence_learning_rate": (
                                float(optim.param_groups[0]["lr"])
                                if is_sequence_evidence_training
                                else 0.0
                            ),
                            "critic_learning_rate": float(
                                optim.param_groups[
                                    1 if is_sequence_evidence_training else 0
                                ]["lr"]
                            ),
                            "stateful_evidence_frozen": (
                                not is_sequence_evidence_training
                            ),
                        }
                        if is_stateful_opinion and not is_m9_trainer
                        else {}
                    ),
                    **(
                        {
                            "training_mode": training_schedule.mode,
                            "training_phase": current_training_phase.name,
                            "base_actor_trainable": (
                                current_training_phase.train_base_actor
                            ),
                            "base_actor_learning_rate": float(
                                next(
                                    group["lr"]
                                    for group in optim.param_groups
                                    if group["group_name"] == "base_actor"
                                )
                            ),
                            "evidence_learning_rate": float(
                                next(
                                    group["lr"]
                                    for group in optim.param_groups
                                    if group["group_name"] == "evidence"
                                )
                            ),
                            "critic_learning_rate": float(
                                next(
                                    group["lr"]
                                    for group in optim.param_groups
                                    if group["group_name"] == "critic"
                                )
                            ),
                            "base_actor_gradient_norm": (
                                base_actor_gradient_norm_sum / loss_update_count
                            ),
                        }
                        if is_m9_trainer
                        else {}
                    ),
                    **opinion_iteration_metrics,
                    **p2_iteration_metrics,
                    **sequence_iteration_metrics,
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
            if is_psb_p2:
                from utilities.psb_marl.checkpoint import sha256_file
                from utilities.psb_marl.p2_checkpoint import save_p2_checkpoint

                checkpoint_arguments = {
                    "iteration": int(pbar.n),
                    "policy": policy,
                    "critic": critic,
                    "optimizer": optim,
                    "artifact_iterations": artifact_iterations,
                    "runtime_config": psb_runtime_config,
                    "base_policy_sha256": sha256_file(
                        Path(str(psb_runtime_config["base_policy_checkpoint"]))
                    ),
                    "state_tracker": psb_tracker,
                }
                save_p2_checkpoint(
                    artifact_run_directory / "latest_checkpoint.pt",
                    **checkpoint_arguments,
                )
                checkpoint_interval = int(
                    p2_training_config["checkpoint_interval"]
                )
                if pbar.n % checkpoint_interval == 0:
                    save_p2_checkpoint(
                        artifact_run_directory
                        / f"checkpoint_iteration_{pbar.n:06d}.pt",
                        **checkpoint_arguments,
                    )
            elif is_m9_trainer:
                from utilities.opinion.checkpoint import save_m9_checkpoint

                checkpoint_arguments = {
                    "iteration": int(pbar.n),
                    "training_mode": training_schedule.mode,
                    "training_phase": current_training_phase.name,
                    "policy": policy,
                    "critic": critic,
                    "optimizer": optim,
                    "artifact_iterations": artifact_iterations,
                    "opinion_runtime_config": opinion_policy_config,
                    "state_tracker": state_tracker,
                    "base_source_state": base_actor_source_state,
                    "base_anchor_state": (
                        None
                        if base_anchor_net is None
                        else base_anchor_net.state_dict()
                    ),
                }
                save_m9_checkpoint(
                    artifact_run_directory / "latest_checkpoint.pt",
                    **checkpoint_arguments,
                )
                checkpoint_interval = int(
                    trainer_runtime_config["checkpoint_interval"]
                )
                if pbar.n % checkpoint_interval == 0:
                    save_m9_checkpoint(
                        artifact_run_directory
                        / f"checkpoint_iteration_{pbar.n:06d}.pt",
                        **checkpoint_arguments,
                    )

        if is_m9_trainer and pbar.n < parameters.n_iters:
            current_training_phase = training_schedule.phase_for_iteration(
                int(pbar.n) + 1
            )
            training_schedule.apply(
                current_training_phase, opinion_bridge, optim
            )

        iteration_cycle_start = time.time()

    # P2 keeps the learned policy quarantined as a candidate.  The stable
    # final_* deployment pair remains byte-identical to Base until the manual
    # paired non-inferiority gate explicitly promotes the candidate.
    if is_psb_p2:
        from utilities.psb_marl.checkpoint import copy_checkpoint_exact

        torch.save(
            policy.state_dict(),
            parameters.where_to_save + "candidate_policy.pth",
        )
        torch.save(
            critic.state_dict(),
            parameters.where_to_save + "candidate_critic.pth",
        )
        source_policy = Path(
            str(psb_runtime_config["base_policy_checkpoint"])
        )
        source_critic = Path(
            str(psb_runtime_config["base_critic_checkpoint"])
        )
        copy_checkpoint_exact(
            source_policy,
            Path(parameters.where_to_save) / "base_fallback_policy.pth",
        )
        copy_checkpoint_exact(
            source_critic,
            Path(parameters.where_to_save) / "base_fallback_critic.pth",
        )
        copy_checkpoint_exact(
            source_policy,
            Path(parameters.where_to_save) / "final_policy.pth",
        )
        copy_checkpoint_exact(
            source_critic,
            Path(parameters.where_to_save) / "final_critic.pth",
        )
        torch.save(
            psb_bridge.control_net.state_dict(),
            parameters.where_to_save + "final_control_net.pth",
        )
        torch.save(
            {
                "branch_encoder": psb_bridge.branch_encoder.state_dict(),
                "distribution_adapter": psb_bridge.adapter.state_dict(),
            },
            parameters.where_to_save + "final_branch_adapter.pth",
        )
        torch.save(
            psb_tracker.snapshot(),
            parameters.where_to_save + "final_psb_state.pt",
        )
    else:
        torch.save(
            policy.state_dict(), parameters.where_to_save + "final_policy.pth"
        )
        torch.save(
            critic.state_dict(), parameters.where_to_save + "final_critic.pth"
        )
    if is_opinion_policy:
        if opinion_bridge is None:
            raise RuntimeError(
                "Opinion EvidenceNet is unavailable while saving the final model."
            )
        torch.save(
            opinion_bridge.evidence_net.state_dict(),
            parameters.where_to_save + "final_evidence_net.pth",
        )
    if is_stateful_opinion:
        if state_tracker is None:
            raise RuntimeError(
                "M6 state tracker is unavailable while saving the final model."
            )
        torch.save(
            state_tracker.snapshot(),
            parameters.where_to_save + "final_opinion_state.pt",
        )

    if artifact_logging_enabled:
        if is_psb_p2:
            from utilities.psb_marl.checkpoint import sha256_file
            from utilities.psb_marl.p2_checkpoint import save_p2_checkpoint

            save_p2_checkpoint(
                artifact_run_directory / "final_checkpoint.pt",
                iteration=int(pbar.n),
                policy=policy,
                critic=critic,
                optimizer=optim,
                artifact_iterations=artifact_iterations,
                runtime_config=psb_runtime_config,
                base_policy_sha256=sha256_file(
                    Path(str(psb_runtime_config["base_policy_checkpoint"]))
                ),
                state_tracker=psb_tracker,
            )
        elif is_opinion_policy:
            if base_actor_source_state is None or opinion_bridge is None:
                raise RuntimeError("Opinion checkpoint source state is unavailable.")
            torch.save(
                base_policy.state_dict(),
                artifact_run_directory / "final_base_actor.pth",
            )
            torch.save(
                policy.state_dict(),
                artifact_run_directory / "final_opinion_policy.pth",
            )
            if is_m9_trainer:
                from utilities.opinion.checkpoint import save_m9_checkpoint

                save_m9_checkpoint(
                    artifact_run_directory / "final_checkpoint.pt",
                    iteration=int(pbar.n),
                    training_mode=training_schedule.mode,
                    training_phase=current_training_phase.name,
                    policy=policy,
                    critic=critic,
                    optimizer=optim,
                    artifact_iterations=artifact_iterations,
                    opinion_runtime_config=opinion_policy_config,
                    state_tracker=state_tracker,
                    base_source_state=base_actor_source_state,
                    base_anchor_state=(
                        None
                        if base_anchor_net is None
                        else base_anchor_net.state_dict()
                    ),
                )
            else:
                torch.save(
                    {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "method": "opinion_marl",
                        "stage": artifact_stage,
                        "iteration": int(pbar.n),
                        "base_actor_state": base_actor_source_state,
                        "evidence_state": opinion_bridge.evidence_net.state_dict(),
                        "opinion_policy_state": policy.state_dict(),
                        "critic_state": critic.state_dict(),
                        "optimizer_state": optim.state_dict(),
                        "resolved_base_config": dict(parameters.to_dict()),
                        "opinion_runtime_config": dict(opinion_policy_config),
                        "terminal_opinion_state": (
                            state_tracker.snapshot()
                            if state_tracker is not None
                            else None
                        ),
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
        comparison_payload = (
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "reference": "recorded_base_source",
                "status": "pending_manual_paired_validation",
                "automated_performance_validation": False,
                "deployment": "base_fallback",
                "candidate_checkpoint": "candidate_policy.pth",
                "note": (
                    "Run main_testing.py with --compare-base; deployment remains "
                    "Base until an explicit promotion passes the configured gate."
                ),
            }
            if is_psb_p2
            else {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "reference": "self",
                "status": "base_reference_created",
                "automated_performance_validation": False,
                "note": "Performance comparison is performed manually by the user.",
            }
        )
        atomic_write_json(
            artifact_run_directory / "comparison_to_base.json",
            comparison_payload,
        )

    return env, policy, priority_module, parameters


if __name__ == "__main__":
    config_file = "config.json"
    parameters = Parameters.from_json(config_file)
    env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)
