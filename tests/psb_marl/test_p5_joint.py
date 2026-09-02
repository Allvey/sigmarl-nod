"""P5 single-stage joint-training contracts."""

from pathlib import Path
import unittest

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


CONFIG = Path("configs/psb_marl/p5_joint_psb_marl.json")


class P5JointTrainingTests(unittest.TestCase):
    def test_p5_runtime_unfreezes_only_candidate_backbone(self):
        experiment = load_psb_experiment(CONFIG)
        runtime = experiment.p5_runtime_config()
        self.assertEqual(experiment.stage, "p5_joint_psb_marl")
        self.assertEqual(runtime["stage"], "p2_frozen_base_bifurcation")
        self.assertEqual(runtime["p5_stage"], "p5_joint_psb_marl")
        self.assertFalse(runtime["freeze_base_actor"])
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
            "base_actor_learning_rate_scale": 1.01,
            "absolute_critic_learning_rate_scale": 1.0,
            "absolute_critic_loss_coefficient": 1.0,
            "base_anchor_coefficient": 0.01,
        }
        with self.assertRaises(PSBConfigError):
            PSBP5JointTrainingConfig.from_dict(payload)

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
