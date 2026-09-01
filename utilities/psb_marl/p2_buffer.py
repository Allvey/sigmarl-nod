"""Contiguous, boundary-safe P2 sequence buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass(frozen=True)
class P2Chunk:
    environment_index: int
    start: int
    length: int


@dataclass(frozen=True)
class P2SequenceMiniBatch:
    tensordict: object
    z_init: torch.Tensor
    chunk_indices: torch.Tensor


class P2SequenceBuffer:
    Z_KEY = ("agents", "psb", "z_prev_dense")
    OLD_LOG_PROB_KEY = ("agents", "sample_log_prob")

    def __init__(self, rollout, chunk_length: int) -> None:
        if type(chunk_length) is not int or chunk_length < 2:
            raise ValueError("P2 chunk_length must be an integer >= 2.")
        if len(rollout.batch_size) != 2:
            raise ValueError("P2 rollout must have [environment,time] batch shape.")
        self.rollout = rollout.detach()
        self.chunk_length = chunk_length
        self.n_environments = int(rollout.batch_size[0])
        self.n_steps = int(rollout.batch_size[1])
        for key in (
            self.Z_KEY,
            self.OLD_LOG_PROB_KEY,
            ("next", "done"),
            ("agents", "info", "pair_features"),
            ("agents", "info", "neighbor_ids"),
            ("agents", "info", "pair_mask"),
            ("agents", "info", "urgency"),
            ("agents", "info", "confidence"),
            ("agents", "info", "agent_reset_mask"),
        ):
            tensor = self.rollout.get(key)
            if tensor.requires_grad:
                raise ValueError(f"P2 rollout tensor {key} must be detached.")
        self.chunks = self._build_chunks()
        if not self.chunks:
            raise ValueError("P2 sequence buffer received an empty rollout.")
        self.z_init = torch.stack(
            [
                self.rollout.get(self.Z_KEY)[chunk.environment_index, chunk.start]
                for chunk in self.chunks
            ],
            dim=0,
        ).detach()

    def _done(self) -> torch.Tensor:
        done = self.rollout.get(("next", "done")).to(torch.bool)
        return done.reshape(self.n_environments, self.n_steps, -1).any(dim=-1)

    def _trajectory_ids(self) -> Optional[torch.Tensor]:
        try:
            ids = self.rollout.get(("collector", "traj_ids"))
        except KeyError:
            return None
        return ids.reshape(self.n_environments, self.n_steps, -1)[..., 0]

    def _build_chunks(self) -> list[P2Chunk]:
        done = self._done()
        trajectory_ids = self._trajectory_ids()
        chunks = []
        for environment in range(self.n_environments):
            segment_start = 0
            for time in range(self.n_steps):
                changes = (
                    trajectory_ids is not None
                    and time + 1 < self.n_steps
                    and trajectory_ids[environment, time + 1]
                    != trajectory_ids[environment, time]
                )
                if bool(done[environment, time]) or changes:
                    self._append(chunks, environment, segment_start, time + 1)
                    segment_start = time + 1
            self._append(chunks, environment, segment_start, self.n_steps)
        return chunks

    def _append(
        self,
        chunks: list[P2Chunk],
        environment: int,
        start: int,
        end: int,
    ) -> None:
        cursor = start
        while cursor < end:
            length = min(self.chunk_length, end - cursor)
            chunks.append(P2Chunk(environment, cursor, length))
            cursor += length

    def iter_minibatches(
        self,
        minibatch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[P2SequenceMiniBatch]:
        if minibatch_size < self.chunk_length:
            raise ValueError("P2 minibatch_size must be >= chunk_length.")
        buckets: dict[int, list[int]] = {}
        for index, chunk in enumerate(self.chunks):
            buckets.setdefault(chunk.length, []).append(index)
        lengths = list(buckets)
        for length_offset in torch.randperm(
            len(lengths), generator=generator, device="cpu"
        ).tolist():
            length = lengths[length_offset]
            indices = buckets[length]
            chunks_per_batch = max(1, minibatch_size // length)
            order = torch.randperm(
                len(indices), generator=generator, device="cpu"
            ).tolist()
            for offset in range(0, len(order), chunks_per_batch):
                selected = [
                    indices[position]
                    for position in order[offset : offset + chunks_per_batch]
                ]
                views = [
                    self.rollout[
                        self.chunks[index].environment_index,
                        self.chunks[index].start : (
                            self.chunks[index].start + self.chunks[index].length
                        ),
                    ]
                    for index in selected
                ]
                yield P2SequenceMiniBatch(
                    tensordict=torch.stack(views, dim=0).detach(),
                    z_init=self.z_init[selected].detach(),
                    chunk_indices=torch.tensor(selected, dtype=torch.long),
                )

    def diagnostics(self) -> dict[str, object]:
        steps = sum(chunk.length for chunk in self.chunks)
        return {
            "sequence_chunk_count": len(self.chunks),
            "sequence_valid_steps": steps,
            "sequence_boundary_violation_count": 0,
            "sequence_state_memory_mb": (
                self.z_init.numel() * self.z_init.element_size() / (1024.0**2)
            ),
        }

