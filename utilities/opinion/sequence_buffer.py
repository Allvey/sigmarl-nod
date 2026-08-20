"""Time-major rollout storage and episode-safe contiguous chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch
from torch import Tensor


@dataclass(frozen=True)
class SequenceChunk:
    data: Dict[str, Tensor]
    z_init: Tensor
    env_index: int
    start: int
    end: int


class OpinionSequenceBuffer:
    def __init__(self, *, n_envs: int, n_agents: int) -> None:
        if type(n_envs) is not int or n_envs <= 0:
            raise ValueError("n_envs must be a positive int")
        if type(n_agents) is not int or n_agents <= 1:
            raise ValueError("n_agents must be an int greater than one")
        self.n_envs = n_envs
        self.n_agents = n_agents
        self._steps: List[Dict[str, Tensor]] = []
        self._fields = None

    def __len__(self) -> int:
        return len(self._steps)

    def clear(self) -> None:
        self._steps.clear()
        self._fields = None

    def append(self, **transition: Tensor) -> None:
        if not transition:
            raise ValueError("transition must contain tensor fields")
        fields = frozenset(transition)
        if self._fields is None:
            self._fields = fields
        elif fields != self._fields:
            raise ValueError("all transitions must contain the same fields")
        stored = {}
        for name, value in transition.items():
            if not torch.is_tensor(value):
                raise TypeError(f"transition field {name!r} must be a tensor")
            if value.ndim < 1 or value.shape[0] != self.n_envs:
                raise ValueError(f"transition field {name!r} must start with [E]")
            stored[name] = value.detach().clone()
        done = stored.get("done")
        if done is None or done.shape != (self.n_envs,) or done.dtype is not torch.bool:
            raise ValueError("transition done must be bool [E]")
        z_dense = stored.get("z_dense_prev")
        if z_dense is None or z_dense.shape != (
            self.n_envs,
            self.n_agents,
            self.n_agents,
        ):
            raise ValueError("z_dense_prev must have shape [E, N, N]")
        self._steps.append(stored)

    def as_rollout(self) -> Dict[str, Tensor]:
        if not self._steps:
            raise RuntimeError("cannot materialize an empty sequence buffer")
        return {
            name: torch.stack([step[name] for step in self._steps], dim=0)
            for name in self._fields
        }

    def add_rollout_fields(self, **fields: Tensor) -> None:
        """Attach derived time-major fields such as advantage and returns."""
        if not self._steps:
            raise RuntimeError("cannot add fields to an empty sequence buffer")
        expected_prefix = (len(self._steps), self.n_envs)
        for name, value in fields.items():
            if name in self._fields:
                raise ValueError(f"rollout field {name!r} already exists")
            if not torch.is_tensor(value) or value.shape[:2] != expected_prefix:
                raise ValueError(
                    f"rollout field {name!r} must start with [T, E]={expected_prefix}"
                )
        for time_index, step in enumerate(self._steps):
            for name, value in fields.items():
                step[name] = value[time_index].detach().clone()
        self._fields = frozenset(set(self._fields).union(fields))

    def iter_chunks(self, *, chunk_length: int) -> Iterator[SequenceChunk]:
        if type(chunk_length) is not int or chunk_length <= 0:
            raise ValueError("chunk_length must be a positive int")
        rollout = self.as_rollout()
        n_steps = len(self._steps)
        done = rollout["done"]
        for env_index in range(self.n_envs):
            segment_start = 0
            for time_index in range(n_steps):
                is_boundary = bool(done[time_index, env_index]) or time_index == n_steps - 1
                if not is_boundary:
                    continue
                segment_end = time_index + 1
                for start in range(segment_start, segment_end, chunk_length):
                    end = min(start + chunk_length, segment_end)
                    data = {
                        name: value[start:end, env_index]
                        for name, value in rollout.items()
                    }
                    yield SequenceChunk(
                        data=data,
                        z_init=rollout["z_dense_prev"][start, env_index]
                        .detach()
                        .clone(),
                        env_index=env_index,
                        start=start,
                        end=end,
                    )
                segment_start = segment_end
