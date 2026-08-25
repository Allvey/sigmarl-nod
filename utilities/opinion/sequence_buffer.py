"""M7 contiguous chunk buffer for stateful Opinion rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch


@dataclass(frozen=True)
class SequenceChunk:
    environment_index: int
    start: int
    length: int
    trajectory_id: Optional[int]


@dataclass(frozen=True)
class SequenceMiniBatch:
    """One batch of equal-length chunks for truncated sequence PPO."""

    tensordict: object
    z_init: torch.Tensor
    edge_active_init: torch.Tensor
    chunk_indices: torch.Tensor

    @property
    def chunk_length(self) -> int:
        return int(self.tensordict.batch_size[1])

    @property
    def valid_steps(self) -> int:
        return int(self.tensordict.batch_size[0] * self.chunk_length)


class OpinionSequenceBuffer:
    """Index one ``[environment,time]`` rollout without destroying time order.

    Chunks are independently shuffleable, but steps inside each chunk remain
    contiguous. Episode boundaries terminate a chunk. Short tails are retained
    conceptually through ``valid_step_mask`` and are materialized without
    padding when passed to the current stateless Critic loss.
    """

    Z_DENSE_KEY = ("agents", "opinion", "z_dense_prev")
    EDGE_ACTIVE_KEY = ("agents", "opinion", "edge_active_prev")
    OLD_LOG_PROB_KEY = ("agents", "sample_log_prob")

    def __init__(self, rollout, chunk_length: int) -> None:
        if type(chunk_length) is not int or chunk_length < 2:
            raise ValueError("chunk_length must be an integer >= 2.")
        if len(rollout.batch_size) != 2:
            raise ValueError(
                "M7 rollout must have batch shape [environment,time], got "
                f"{tuple(rollout.batch_size)}."
            )
        self.rollout = rollout.detach()
        self.chunk_length = chunk_length
        self.n_environments = int(rollout.batch_size[0])
        self.n_steps = int(rollout.batch_size[1])
        self._validate_required_tensors()
        self.chunks = self._build_chunks()
        if not self.chunks:
            raise ValueError("M7 sequence buffer received an empty rollout.")
        self.valid_step_mask = self._build_valid_step_mask()
        self.z_init = torch.stack(
            [
                self.rollout.get(self.Z_DENSE_KEY)[
                    chunk.environment_index, chunk.start
                ]
                for chunk in self.chunks
            ],
            dim=0,
        ).detach()
        self.edge_active_init = torch.stack(
            [
                self.rollout.get(self.EDGE_ACTIVE_KEY)[
                    chunk.environment_index, chunk.start
                ]
                for chunk in self.chunks
            ],
            dim=0,
        ).detach()
        self._assert_boundary_contract()

    def _validate_required_tensors(self) -> None:
        for key in (
            self.Z_DENSE_KEY,
            self.EDGE_ACTIVE_KEY,
            self.OLD_LOG_PROB_KEY,
            ("next", "done"),
            ("agents", "info", "neighbor_ids"),
            ("agents", "info", "pair_mask"),
            ("agents", "info", "agent_reset_mask"),
        ):
            try:
                tensor = self.rollout.get(key)
            except KeyError as error:
                raise KeyError(f"M7 rollout is missing required key {key}.") from error
            if tensor.requires_grad:
                raise ValueError(f"M7 rollout tensor {key} must be detached.")
        if not torch.isfinite(self.rollout.get(self.OLD_LOG_PROB_KEY)).all():
            raise ValueError("M7 old rollout log-prob contains NaN or Inf.")

    def _done_by_environment_and_time(self) -> torch.Tensor:
        done = self.rollout.get(("next", "done")).to(dtype=torch.bool)
        if done.shape[:2] != (self.n_environments, self.n_steps):
            raise ValueError("M7 done tensor must start with [environment,time].")
        return done.reshape(self.n_environments, self.n_steps, -1).any(dim=-1)

    def _trajectory_ids(self) -> Optional[torch.Tensor]:
        try:
            trajectory_ids = self.rollout.get(("collector", "traj_ids"))
        except KeyError:
            return None
        if trajectory_ids.shape[:2] != (self.n_environments, self.n_steps):
            raise ValueError(
                "M7 trajectory IDs must start with [environment,time]."
            )
        return trajectory_ids.reshape(
            self.n_environments, self.n_steps, -1
        )[..., 0]

    def _append_segment_chunks(
        self,
        chunks: List[SequenceChunk],
        environment_index: int,
        start: int,
        end: int,
        trajectory_ids: Optional[torch.Tensor],
    ) -> None:
        cursor = start
        while cursor < end:
            length = min(self.chunk_length, end - cursor)
            trajectory_id = (
                None
                if trajectory_ids is None
                else int(trajectory_ids[environment_index, cursor].item())
            )
            chunks.append(
                SequenceChunk(
                    environment_index=environment_index,
                    start=cursor,
                    length=length,
                    trajectory_id=trajectory_id,
                )
            )
            cursor += length

    def _build_chunks(self) -> List[SequenceChunk]:
        done = self._done_by_environment_and_time()
        trajectory_ids = self._trajectory_ids()
        chunks: List[SequenceChunk] = []
        for environment_index in range(self.n_environments):
            segment_start = 0
            for time_index in range(self.n_steps):
                trajectory_changes = (
                    trajectory_ids is not None
                    and time_index + 1 < self.n_steps
                    and trajectory_ids[environment_index, time_index + 1]
                    != trajectory_ids[environment_index, time_index]
                )
                if bool(done[environment_index, time_index]) or trajectory_changes:
                    self._append_segment_chunks(
                        chunks,
                        environment_index,
                        segment_start,
                        time_index + 1,
                        trajectory_ids,
                    )
                    segment_start = time_index + 1
            self._append_segment_chunks(
                chunks,
                environment_index,
                segment_start,
                self.n_steps,
                trajectory_ids,
            )
        return chunks

    def _build_valid_step_mask(self) -> torch.Tensor:
        mask = torch.zeros(
            (len(self.chunks), self.chunk_length),
            device=self.rollout.device,
            dtype=torch.bool,
        )
        for index, chunk in enumerate(self.chunks):
            mask[index, : chunk.length] = True
        return mask

    def _assert_boundary_contract(self) -> None:
        done = self._done_by_environment_and_time()
        trajectory_ids = self._trajectory_ids()
        for chunk in self.chunks:
            if chunk.length <= 0 or chunk.length > self.chunk_length:
                raise RuntimeError("M7 constructed an invalid chunk length.")
            stop = chunk.start + chunk.length
            if bool(done[chunk.environment_index, chunk.start : stop - 1].any()):
                raise RuntimeError("M7 chunk crosses an episode done boundary.")
            if trajectory_ids is not None:
                ids = trajectory_ids[
                    chunk.environment_index, chunk.start : stop
                ]
                if not bool((ids == ids[0]).all()):
                    raise RuntimeError("M7 chunk crosses a trajectory boundary.")

    @property
    def valid_steps(self) -> int:
        return int(self.valid_step_mask.sum().item())

    @property
    def padded_steps(self) -> int:
        return int(self.valid_step_mask.numel() - self.valid_steps)

    @property
    def valid_step_fraction(self) -> float:
        return float(self.valid_steps / self.valid_step_mask.numel())

    @property
    def boundary_violation_count(self) -> int:
        return 0

    @property
    def state_memory_bytes(self) -> int:
        return (
            self.z_init.numel() * self.z_init.element_size()
            + self.edge_active_init.numel()
            * self.edge_active_init.element_size()
            + self.valid_step_mask.numel()
            * self.valid_step_mask.element_size()
        )

    def iter_minibatches(
        self,
        minibatch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator:
        """Yield detached valid steps from shuffled groups of whole chunks."""

        if type(minibatch_size) is not int or minibatch_size < self.chunk_length:
            raise ValueError("minibatch_size must be >= chunk_length for M7.")
        chunks_per_minibatch = max(1, minibatch_size // self.chunk_length)
        order = torch.randperm(
            len(self.chunks),
            generator=generator,
            device="cpu",
        ).tolist()
        for offset in range(0, len(order), chunks_per_minibatch):
            selected = [self.chunks[index] for index in order[offset : offset + chunks_per_minibatch]]
            views = [
                self.rollout[
                    chunk.environment_index,
                    chunk.start : chunk.start + chunk.length,
                ]
                for chunk in selected
            ]
            yield torch.cat(views, dim=0).reshape(-1).detach()

    def iter_sequence_minibatches(
        self,
        minibatch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[SequenceMiniBatch]:
        """Yield time-preserving ``[chunk,time]`` mini-batches.

        Chunks are bucketed by their real length.  This keeps every tensor
        rectangular without allowing padded steps to enter the PPO objective,
        and retains a vectorized chunk dimension while the dynamics unrolls
        only over the short time dimension.
        """

        if type(minibatch_size) is not int or minibatch_size < self.chunk_length:
            raise ValueError("minibatch_size must be >= chunk_length for M8.")

        buckets = {}
        for index, chunk in enumerate(self.chunks):
            buckets.setdefault(chunk.length, []).append(index)

        bucket_lengths = torch.randperm(
            len(buckets), generator=generator, device="cpu"
        ).tolist()
        lengths = list(buckets)
        for length_offset in bucket_lengths:
            length = lengths[length_offset]
            indices = buckets[length]
            chunks_per_minibatch = max(1, minibatch_size // length)
            order = torch.randperm(
                len(indices), generator=generator, device="cpu"
            ).tolist()
            for offset in range(0, len(order), chunks_per_minibatch):
                selected_indices = [
                    indices[position]
                    for position in order[offset : offset + chunks_per_minibatch]
                ]
                views = []
                for index in selected_indices:
                    chunk = self.chunks[index]
                    views.append(
                        self.rollout[
                            chunk.environment_index,
                            chunk.start : chunk.start + chunk.length,
                        ]
                    )
                yield SequenceMiniBatch(
                    tensordict=torch.stack(views, dim=0).detach(),
                    z_init=self.z_init[selected_indices].detach(),
                    edge_active_init=self.edge_active_init[
                        selected_indices
                    ].detach(),
                    chunk_indices=torch.tensor(
                        selected_indices, dtype=torch.long, device="cpu"
                    ),
                )

    def diagnostics(self) -> dict:
        return {
            "sequence_chunk_count": int(len(self.chunks)),
            "sequence_valid_steps": int(self.valid_steps),
            "sequence_padded_steps": int(self.padded_steps),
            "sequence_valid_step_fraction": float(self.valid_step_fraction),
            "sequence_boundary_violation_count": int(
                self.boundary_violation_count
            ),
            "sequence_state_memory_mb": float(
                self.state_memory_bytes / (1024.0 * 1024.0)
            ),
        }
