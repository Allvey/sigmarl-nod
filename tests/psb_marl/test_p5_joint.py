"""P5 single-stage joint-training contracts."""

from pathlib import Path
import unittest

import torch
from torch import nn

from utilities.psb_marl.config import (
    PSBConfigError,
    PSBP5JointTrainingConfig,
    load_psb_experiment,
)
from utilities.psb_marl.p2_policy import (
    FrozenBaseBifurcationPolicyBridge,
    validate_p2_runtime_contract,
)
from utilities.psb_marl.p5_schedule import (
    blend_actor_advantages,
    scratch_phase,
)


CONFIG = Path("configs/psb_marl/p5_joint_psb_marl.json")
SCRATCH_CONFIG = Path("configs/psb_marl/p5_scratch_joint_psb_marl.json")
SCRATCH_V2_CONFIG = Path(
    "configs/psb_marl/p5_scratch_v2_joint_psb_marl.json"
)
BASE_ONLY_250_CONFIG = Path(
    "configs/psb_marl/p5_scratch_v2_base_only_250.json"
)


class P5JointTrainingTests(unittest.TestCase):
    def test_p5_runtime_unfreezes_only_candidate_backbone(self):
        experiment = load_psb_experiment(CONFIG)
        runtime = experiment.p5_runtime_config()
        self.assertEqual(experiment.stage, "p5_joint_psb_marl")
        self.assertEqual(runtime["stage"], "p2_frozen_base_bifurcation")
        self.assertEqual(runtime["p5_stage"], "p5_joint_psb_marl")
        self.assertFalse(runtime["freeze_base_actor"])
        self.assertEqual(runtime["joint_training"]["ppo_mode"], "transition")
        self.assertEqual(runtime["joint_training"]["ppo_epochs"], 15)
        self.assertEqual(runtime["joint_training"]["minibatch_size"], 1024)
        self.assertEqual(runtime["joint_training"]["target_kl"], 0.03)
        self.assertEqual(
            runtime["joint_training"]["initialization_mode"], "warm_start"
        )
        self.assertEqual(
            runtime["branch_adapter"]["action_projection"],
            "longitudinal_only",
        )
        self.assertEqual(runtime["branch_adapter"]["max_delta_log_scale"], 0.0)
        self.assertEqual(runtime["primal_dual"]["active_constraints"], ["vehicle"])
        self.assertTrue(
            runtime["initial_policy_checkpoint"].endswith("candidate_policy.pth")
        )
        self.assertTrue(
            runtime["initial_scalar_critic_checkpoint"].endswith(
                "candidate_critic.pth"
            )
        )
        self.assertTrue(
            runtime["p3_differential_critic_checkpoint"].endswith(
                "candidate_differential_critic.pth"
            )
        )
        validate_p2_runtime_contract(runtime, environment_n_agents=4)

    def test_joint_config_rejects_full_rate_backbone_updates(self):
        payload = {
            "ppo_mode": "transition",
            "ppo_epochs": 15,
            "minibatch_size": 1024,
            "target_kl": 0.03,
            "initialization_mode": "warm_start",
            "base_pretrain_iterations": 0,
            "base_pretrain_ppo_epochs": 0,
            "base_pretrain_minibatch_size": 0,
            "base_pretrain_target_kl": 0.0,
            "absolute_advantage_warmup_iterations": 0,
            "advantage_blend_iterations": 0,
            "dual_warmup_iterations": 0,
            "branch_bootstrap_iterations": 0,
            "branch_activity_bootstrap_offset": 0.0,
            "base_actor_learning_rate_scale": 1.01,
            "absolute_critic_learning_rate_scale": 1.0,
            "absolute_critic_loss_coefficient": 1.0,
            "base_anchor_coefficient": 0.01,
        }
        with self.assertRaises(PSBConfigError):
            PSBP5JointTrainingConfig.from_dict(payload)

    def test_joint_config_rejects_sequence_ppo(self):
        payload = {
            "ppo_mode": "sequence",
            "ppo_epochs": 15,
            "minibatch_size": 1024,
            "target_kl": 0.03,
            "initialization_mode": "warm_start",
            "base_pretrain_iterations": 0,
            "base_pretrain_ppo_epochs": 0,
            "base_pretrain_minibatch_size": 0,
            "base_pretrain_target_kl": 0.0,
            "absolute_advantage_warmup_iterations": 0,
            "advantage_blend_iterations": 0,
            "dual_warmup_iterations": 0,
            "branch_bootstrap_iterations": 0,
            "branch_activity_bootstrap_offset": 0.0,
            "base_actor_learning_rate_scale": 0.1,
            "absolute_critic_learning_rate_scale": 1.0,
            "absolute_critic_loss_coefficient": 1.0,
            "base_anchor_coefficient": 0.01,
        }
        with self.assertRaises(PSBConfigError):
            PSBP5JointTrainingConfig.from_dict(payload)

    def test_scratch_runtime_uses_locked_curriculum(self):
        experiment = load_psb_experiment(SCRATCH_CONFIG)
        runtime = experiment.p5_runtime_config()
        joint = runtime["joint_training"]

        self.assertEqual(experiment.stage, "p5_joint_psb_marl")
        self.assertEqual(joint["initialization_mode"], "scratch")
        self.assertEqual(joint["absolute_advantage_warmup_iterations"], 100)
        self.assertEqual(joint["advantage_blend_iterations"], 100)
        self.assertEqual(joint["dual_warmup_iterations"], 100)
        self.assertEqual(joint["base_actor_learning_rate_scale"], 1.0)
        self.assertEqual(joint["base_anchor_coefficient"], 0.0)
        self.assertEqual(
            runtime["primal_dual"]["initial_vehicle_multiplier"], 0.0
        )
        validate_p2_runtime_contract(runtime, environment_n_agents=4)

    def test_scratch_v2_starts_with_unpaired_base_pretraining(self):
        experiment = load_psb_experiment(SCRATCH_V2_CONFIG)
        joint = experiment.p5_runtime_config()["joint_training"]

        self.assertEqual(joint["base_pretrain_iterations"], 60)
        self.assertEqual(joint["base_pretrain_ppo_epochs"], 30)
        self.assertEqual(joint["base_pretrain_minibatch_size"], 512)
        self.assertEqual(joint["base_pretrain_target_kl"], 0.0)
        self.assertEqual(joint["absolute_advantage_warmup_iterations"], 40)
        self.assertEqual(joint["branch_bootstrap_iterations"], 40)
        self.assertEqual(joint["branch_activity_bootstrap_offset"], 0.05)

    def test_scratch_base_only_ablation_can_use_the_complete_run(self):
        experiment = load_psb_experiment(BASE_ONLY_250_CONFIG)
        joint = experiment.p5_runtime_config()["joint_training"]

        self.assertEqual(experiment.primal_dual.iterations, 250)
        self.assertEqual(joint["base_pretrain_iterations"], 250)
        self.assertEqual(joint["base_pretrain_ppo_epochs"], 60)
        self.assertEqual(joint["base_pretrain_minibatch_size"], 512)
        self.assertEqual(joint["base_pretrain_target_kl"], 0.0)
        phase = scratch_phase(
            250,
            base_pretrain_iterations=joint["base_pretrain_iterations"],
            absolute_warmup_iterations=joint[
                "absolute_advantage_warmup_iterations"
            ],
            advantage_blend_iterations=joint["advantage_blend_iterations"],
            dual_warmup_iterations=joint["dual_warmup_iterations"],
        )
        self.assertEqual(phase.name, "base_actor_pretrain")
        self.assertFalse(phase.paired_learning_enabled)
        self.assertFalse(phase.psb_learning_enabled)

    def test_scratch_schedule_warms_blends_then_uses_differential_signal(self):
        warm = scratch_phase(
            100,
            absolute_warmup_iterations=100,
            advantage_blend_iterations=100,
            dual_warmup_iterations=100,
        )
        blend = scratch_phase(
            150,
            absolute_warmup_iterations=100,
            advantage_blend_iterations=100,
            dual_warmup_iterations=100,
        )
        differential = scratch_phase(
            201,
            absolute_warmup_iterations=100,
            advantage_blend_iterations=100,
            dual_warmup_iterations=100,
        )

        self.assertEqual(
            (warm.absolute_weight, warm.differential_weight), (1.0, 0.0)
        )
        self.assertFalse(warm.dual_update_enabled)
        self.assertEqual(
            (blend.absolute_weight, blend.differential_weight), (0.5, 0.5)
        )
        self.assertTrue(blend.dual_update_enabled)
        self.assertEqual(
            (differential.absolute_weight, differential.differential_weight),
            (0.0, 1.0),
        )

    def test_scratch_v2_schedule_defers_pairing_and_bootstraps_psb(self):
        pretrain = scratch_phase(
            1,
            base_pretrain_iterations=60,
            absolute_warmup_iterations=40,
            advantage_blend_iterations=100,
            dual_warmup_iterations=100,
            branch_bootstrap_iterations=40,
            branch_activity_bootstrap_offset=0.05,
        )
        bootstrap = scratch_phase(
            61,
            base_pretrain_iterations=60,
            absolute_warmup_iterations=40,
            advantage_blend_iterations=100,
            dual_warmup_iterations=100,
            branch_bootstrap_iterations=40,
            branch_activity_bootstrap_offset=0.05,
        )
        blend = scratch_phase(
            101,
            base_pretrain_iterations=60,
            absolute_warmup_iterations=40,
            advantage_blend_iterations=100,
            dual_warmup_iterations=100,
            branch_bootstrap_iterations=40,
            branch_activity_bootstrap_offset=0.05,
        )

        self.assertEqual(pretrain.name, "base_actor_pretrain")
        self.assertFalse(pretrain.paired_learning_enabled)
        self.assertFalse(pretrain.psb_learning_enabled)
        self.assertEqual(pretrain.branch_activity_offset, 0.0)
        self.assertEqual(bootstrap.name, "absolute_warmup")
        self.assertTrue(bootstrap.paired_learning_enabled)
        self.assertTrue(bootstrap.psb_learning_enabled)
        self.assertAlmostEqual(bootstrap.branch_activity_offset, 0.05)
        self.assertEqual(blend.name, "absolute_differential_blend")
        self.assertTrue(blend.dual_update_enabled)
        self.assertEqual(blend.branch_activity_offset, 0.0)

    def test_advantage_blend_normalizes_each_agent(self):
        absolute = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4, 1)
        differential = torch.flip(absolute, dims=(0, 1))
        differential = (
            differential - differential.mean(dim=(0, 1), keepdim=True)
        ) / differential.std(dim=(0, 1), unbiased=False, keepdim=True)

        mixed, metrics = blend_actor_advantages(
            absolute,
            differential,
            differential_weight=0.25,
            scale_floor=1e-4,
        )

        self.assertTrue(
            torch.allclose(
                mixed.mean(dim=(0, 1)), torch.zeros(4, 1), atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                mixed.std(dim=(0, 1), unbiased=False),
                torch.ones(4, 1),
                atol=1e-6,
            )
        )
        self.assertIn("mixed_advantage_scale", metrics)

    def test_policy_bridge_exposes_trainable_candidate_backbone_group(self):
        base = nn.Linear(3, 2)
        bridge = FrozenBaseBifurcationPolicyBridge(
            base_policy_net=base,
            control_net=nn.Linear(1, 1),
            proximal=nn.Identity(),
            branch_encoder=nn.Linear(1, 1),
            adapter=nn.Linear(1, 1),
            n_agents=2,
            freeze_base_actor=False,
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in base.parameters())
        )
        self.assertEqual(
            bridge.trainable_groups()["base_actor"], list(base.parameters())
        )
        sum(parameter.sum() for parameter in base.parameters()).backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in base.parameters())
        )
