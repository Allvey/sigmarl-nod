"""Reference checks for the M7 contiguous sequence-buffer contract."""

import unittest

import torch
from tensordict import TensorDict
from pathlib import Path

from utilities.opinion.config import (
    load_opinion_experiment,
    require_m7_supported_mode,
)
from utilities.opinion.sequence_buffer import OpinionSequenceBuffer


def make_rollout(environments=2, steps=7, agents=3, candidates=2):
    done = torch.zeros(environments, steps, 1, dtype=torch.bool)
    trajectory_ids = torch.zeros(environments, steps, dtype=torch.long)
    z_dense = torch.arange(
        environments * steps * agents * agents,
        dtype=torch.float32,
    ).reshape(environments, steps, agents, agents)
    edge_active = torch.zeros_like(z_dense, dtype=torch.bool)
    neighbor_ids = torch.zeros(
        environments, steps, agents, candidates, dtype=torch.long
    )
    pair_mask = torch.ones_like(neighbor_ids, dtype=torch.bool)
    reset_mask = torch.zeros(
        environments, steps, agents, dtype=torch.bool
    )
    old_log_prob = torch.zeros(environments, steps, agents)
    observation = torch.randn(environments, steps, agents, 4)
    return TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": observation,
                    "sample_log_prob": old_log_prob,
                    "info": TensorDict(
                        {
                            "neighbor_ids": neighbor_ids,
                            "pair_mask": pair_mask,
                            "agent_reset_mask": reset_mask,
                        },
                        batch_size=[environments, steps, agents],
                    ),
                    "opinion": TensorDict(
                        {
                            "z_dense_prev": z_dense,
                            "edge_active_prev": edge_active,
                        },
                        batch_size=[environments, steps, agents],
                    ),
                },
                batch_size=[environments, steps, agents],
            ),
            "next": TensorDict(
                {"done": done}, batch_size=[environments, steps]
            ),
            "collector": TensorDict(
                {"traj_ids": trajectory_ids},
                batch_size=[environments, steps],
            ),
        },
        batch_size=[environments, steps],
    )


class OpinionSequenceBufferTests(unittest.TestCase):
    def test_committed_m7_config_enables_sequence_only_after_stateful(self):
        project_root = Path(__file__).resolve().parents[2]
        experiment = load_opinion_experiment(
            project_root / "configs/opinion/m7_sequence_buffer.json"
        )
        require_m7_supported_mode(experiment)

        self.assertEqual(experiment.config.stage, "sequence")
        self.assertTrue(experiment.config.opinion.stateful.enabled)
        self.assertTrue(experiment.config.opinion.sequence_ppo.enabled)
        self.assertEqual(experiment.config.opinion.sequence_ppo.chunk_length, 16)

    def test_chunks_do_not_cross_done_or_environment(self):
        rollout = make_rollout()
        rollout.set_at_(
            ("next", "done"),
            torch.tensor([True]),
            (0, 2),
        )
        rollout.set_at_(
            ("collector", "traj_ids"),
            torch.tensor(1),
            (0, slice(3, None)),
        )

        buffer = OpinionSequenceBuffer(rollout, chunk_length=4)

        self.assertEqual(buffer.boundary_violation_count, 0)
        self.assertEqual(buffer.valid_steps, 14)
        self.assertTrue(
            all(chunk.length <= 4 for chunk in buffer.chunks)
        )
        self.assertFalse(
            any(
                chunk.environment_index == 0
                and chunk.start < 3 < chunk.start + chunk.length
                for chunk in buffer.chunks
            )
        )

    def test_chunk_initial_state_matches_first_physical_step(self):
        rollout = make_rollout(environments=1, steps=5)
        buffer = OpinionSequenceBuffer(rollout, chunk_length=3)

        expected = torch.stack(
            [
                rollout.get(("agents", "opinion", "z_dense_prev"))[
                    chunk.environment_index, chunk.start
                ]
                for chunk in buffer.chunks
            ]
        )
        torch.testing.assert_close(buffer.z_init, expected)

    def test_short_tail_has_padding_mask_but_loss_batch_has_valid_steps_only(self):
        rollout = make_rollout(environments=1, steps=5)
        buffer = OpinionSequenceBuffer(rollout, chunk_length=4)

        self.assertEqual(buffer.valid_step_mask.tolist(), [
            [True, True, True, True],
            [True, False, False, False],
        ])
        sampled_steps = sum(
            int(batch.batch_size[0])
            for batch in buffer.iter_minibatches(minibatch_size=4)
        )
        self.assertEqual(sampled_steps, 5)

    def test_nan_old_log_prob_is_rejected(self):
        rollout = make_rollout(environments=1, steps=4)
        rollout.get(("agents", "sample_log_prob"))[0, 1, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "log-prob"):
            OpinionSequenceBuffer(rollout, chunk_length=2)


if __name__ == "__main__":
    unittest.main()
