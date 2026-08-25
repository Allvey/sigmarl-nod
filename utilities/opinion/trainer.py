"""M9 parameter-group scheduling for independent and joint training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    train_base_actor: bool
    train_evidence: bool = True
    train_critic: bool = True


class OpinionTrainingSchedule:
    """Resolve and apply the M9 phase for each one-based iteration."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.mode = str(config["mode"])
        self.warmup_iterations = int(config["evidence_warmup_iterations"])
        self.base_actor_lr_scale = float(
            config["base_actor_learning_rate_scale"]
        )
        self.evidence_lr_scale = float(config["evidence_learning_rate_scale"])
        self.critic_lr_scale = float(config["critic_learning_rate_scale"])
        if self.mode not in {"evidence_only", "joint", "warmup_then_joint"}:
            raise ValueError(f"Unsupported M9 trainer mode: {self.mode}")

    def phase_for_iteration(self, iteration: int) -> TrainingPhase:
        if type(iteration) is not int or iteration < 1:
            raise ValueError("iteration must be a positive integer.")
        if self.mode == "evidence_only":
            return TrainingPhase("evidence", train_base_actor=False)
        if self.mode == "joint":
            return TrainingPhase("joint", train_base_actor=True)
        if iteration <= self.warmup_iterations:
            return TrainingPhase("evidence_warmup", train_base_actor=False)
        return TrainingPhase("joint", train_base_actor=True)

    def apply(self, phase: TrainingPhase, bridge, optimizer) -> None:
        for parameter in bridge.base_policy_net.parameters():
            parameter.requires_grad_(phase.train_base_actor)
        for parameter in bridge.evidence_net.parameters():
            parameter.requires_grad_(phase.train_evidence)

        groups = {group.get("group_name"): group for group in optimizer.param_groups}
        required = {"base_actor", "evidence", "critic"}
        if set(groups) != required:
            raise RuntimeError(
                "M9 optimizer must contain exactly base_actor/evidence/critic groups."
            )
        groups["base_actor"]["lr_scale"] = (
            self.base_actor_lr_scale if phase.train_base_actor else 0.0
        )
        groups["evidence"]["lr_scale"] = self.evidence_lr_scale
        groups["critic"]["lr_scale"] = self.critic_lr_scale
        reference_lr = groups["critic"]["lr"] / self.critic_lr_scale
        groups["base_actor"]["lr"] = (
            reference_lr * self.base_actor_lr_scale
            if phase.train_base_actor
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "evidence_warmup_iterations": self.warmup_iterations,
            "base_actor_learning_rate_scale": self.base_actor_lr_scale,
            "evidence_learning_rate_scale": self.evidence_lr_scale,
            "critic_learning_rate_scale": self.critic_lr_scale,
        }
