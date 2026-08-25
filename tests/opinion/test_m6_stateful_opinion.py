"""Reference checks for M6 global-ID state and pure stateful policy mapping.

These tests are provided for manual execution and are not run by this session.
"""

import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn

from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.policy import StatefulOpinionPolicyBridge
from utilities.opinion.policy import StatefulOpinionPolicyController
from utilities.opinion.residual import OpinionResidual
from utilities.opinion.state import OpinionStateTracker
from utilities.opinion.transforms import DiscreteDTypeCastTransform
from main_training_opinion import _resolve_m6_base_actor_source
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.collectors.collectors import _policy_is_tensordict_compatible


class M6CheckpointCompatibilityTests(unittest.TestCase):
    def test_completed_legacy_m5_uses_final_base_actor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            final_base_actor = run_directory / "final_base_actor.pth"
            final_base_actor.touch()
            unrelated_current = run_directory / "newer_base_policy.pth"
            unrelated_current.touch()

            source, checkpoint_kind = _resolve_m6_base_actor_source(
                evidence_run_directory=run_directory,
                recorded_base_actor=run_directory / "deleted_reward_policy.pth",
                current_base_actor=unrelated_current,
            )

            self.assertEqual(source, final_base_actor.resolve())
            self.assertEqual(checkpoint_kind, "m5_final_base_actor")


class M6TorchRLCompatibilityTests(unittest.TestCase):
    def test_integer_and_boolean_info_specs_can_be_sampled(self):
        float_spec = UnboundedContinuousTensorSpec(
            shape=(4, 2), dtype=torch.float32
        )
        id_transform = DiscreteDTypeCastTransform(
            torch.float32,
            torch.long,
            n=4,
            in_keys=[("agents", "info", "neighbor_ids")],
            in_keys_inv=[],
        )
        bool_transform = DiscreteDTypeCastTransform(
            torch.float32,
            torch.bool,
            n=2,
            in_keys=[("agents", "info", "pair_mask")],
            in_keys_inv=[],
        )

        id_spec = id_transform._transform_spec(float_spec)
        bool_spec = bool_transform._transform_spec(float_spec)

        self.assertEqual(id_spec.rand().dtype, torch.long)
        self.assertEqual(bool_spec.rand().dtype, torch.bool)

    def test_stateful_controller_is_recognized_as_tensordict_policy(self):
        policy = nn.Linear(1, 1)
        policy.in_keys = [("agents", "observation")]
        policy.out_keys = [("agents", "action")]
        controller = StatefulOpinionPolicyController(
            policy=policy,
            state_tracker=OpinionStateTracker(3, 0.95),
        )

        self.assertTrue(_policy_is_tensordict_compatible(controller))


class OpinionStateTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = OpinionStateTracker(
            n_agents=3,
            decay_factor=0.95,
            zero_threshold=1e-6,
        )
        self.reset = torch.zeros(1, 3, dtype=torch.bool)
        self.reference = torch.zeros(1, 3, 2)

    def test_state_follows_global_id_when_candidate_slots_swap(self):
        ids = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        z_prev = self.tracker.prepare_step(
            ids, mask, self.reset, self.reference
        )
        self.assertEqual(torch.count_nonzero(z_prev).item(), 0)
        first_state = torch.tensor(
            [[[0.2, 0.7], [-0.2, 0.4], [-0.7, -0.4]]]
        )
        self.tracker.commit_step(ids, first_state)

        swapped_ids = ids.flip(-1)
        swapped_prev = self.tracker.prepare_step(
            swapped_ids, mask, self.reset, self.reference
        )
        torch.testing.assert_close(swapped_prev, first_state.flip(-1))

    def test_agent_reset_clears_incoming_and_outgoing_state(self):
        ids = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        self.tracker.prepare_step(ids, mask, self.reset, self.reference)
        self.tracker.commit_step(ids, torch.ones_like(self.reference))

        reset_agent_one = self.reset.clone()
        reset_agent_one[0, 1] = True
        self.tracker.prepare_step(
            ids,
            torch.zeros_like(mask),
            reset_agent_one,
            self.reference,
        )
        self.assertEqual(torch.count_nonzero(self.tracker.z_dense[:, 1, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(self.tracker.z_dense[:, :, 1]).item(), 0)

    def test_torchrl_scalar_reset_trailing_dimension_is_normalized(self):
        ids = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        reset_with_scalar_dimension = self.reset.unsqueeze(-1)

        z_prev = self.tracker.prepare_step(
            ids,
            mask,
            reset_with_scalar_dimension,
            self.reference,
        )

        self.assertEqual(z_prev.shape, self.reference.shape)

    def test_reappearing_conflict_edge_starts_from_zero(self):
        ids = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        active = torch.ones_like(ids, dtype=torch.bool)
        self.tracker.prepare_step(ids, active, self.reset, self.reference)
        self.tracker.commit_step(ids, torch.ones_like(self.reference))

        inactive = active.clone()
        inactive[0, 0, 0] = False
        decaying = self.tracker.prepare_step(
            ids, inactive, self.reset, self.reference
        )
        decayed_next = decaying * 0.95
        self.tracker.commit_step(ids, decayed_next)

        reappeared = self.tracker.prepare_step(
            ids, active, self.reset, self.reference
        )
        self.assertEqual(reappeared[0, 0, 0].item(), 0.0)

    def test_environment_done_does_not_clear_another_environment(self):
        tracker = OpinionStateTracker(3, 0.95, 1e-6)
        ids = torch.tensor(
            [
                [[1, 2], [0, 2], [0, 1]],
                [[1, 2], [0, 2], [0, 1]],
            ]
        )
        mask = torch.ones_like(ids, dtype=torch.bool)
        reset = torch.zeros(2, 3, dtype=torch.bool)
        reference = torch.zeros(2, 3, 2)
        tracker.prepare_step(ids, mask, reset, reference)
        tracker.commit_step(ids, torch.ones_like(reference))

        tracker.prepare_step(
            ids,
            torch.zeros_like(mask),
            reset,
            reference,
            environment_done=torch.tensor([[True], [False]]),
        )
        self.assertEqual(torch.count_nonzero(tracker.z_dense[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(tracker.z_dense[1]).item(), 0)


class FakeBasePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, observation):
        parameters = self.linear(observation)
        return parameters[..., :2], parameters[..., 2:].exp()


class StatefulOpinionPolicyBridgeTests(unittest.TestCase):
    def test_bridge_is_pure_and_freezes_actor_and_evidence(self):
        torch.manual_seed(11)
        base = FakeBasePolicyNet()
        evidence = OpinionEvidenceNet(10, (16, 16), 1.0, 1.0)
        bridge = StatefulOpinionPolicyBridge(
            base_policy_net=base,
            evidence_net=evidence,
            dynamics=OpinionDynamics(0.5, 1.0, 0.5, 1.0),
            residual=OpinionResidual(1.0, 0.1, 0.25),
            dt=0.05,
            freeze_base_actor=True,
            freeze_evidence=True,
        )
        observation = torch.randn(2, 3, 3)
        features = torch.randn(2, 3, 2, 10)
        urgency = torch.ones(2, 3, 2)
        confidence = torch.ones(2, 3, 2)
        mask = torch.ones(2, 3, 2, dtype=torch.bool)
        z_prev = torch.full((2, 3, 2), 0.2)

        first = bridge(
            observation, features, urgency, confidence, mask, z_prev
        )
        second = bridge(
            observation, features, urgency, confidence, mask, z_prev
        )
        for first_tensor, second_tensor in zip(first, second):
            torch.testing.assert_close(first_tensor, second_tensor)
        final_loc, scale, base_loc, _, _, z_next, *_ = first
        torch.testing.assert_close(final_loc[..., 1], base_loc[..., 1])
        self.assertEqual(scale.shape, base_loc.shape)
        self.assertFalse(torch.equal(z_next, z_prev))
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        self.assertTrue(all(not p.requires_grad for p in evidence.parameters()))


if __name__ == "__main__":
    unittest.main()
