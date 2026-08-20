"""Checkpoint-backed Opinion-MARL evaluation rollout."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchrl.envs.utils import step_mdp

from utilities.opinion.checkpoint import load_opinion_checkpoint
from utilities.opinion.diagnostics import OpinionDiagnostics
from utilities.opinion.trainer import build_opinion_trainer


def evaluate_opinion_checkpoint(
    loaded,
    *,
    checkpoint,
    steps=None,
    smoke=False,
    output_path=None,
):
    trainer = build_opinion_trainer(loaded, smoke=smoke, output_dir=Path(checkpoint).parent)
    load_opinion_checkpoint(
        checkpoint,
        policy=trainer.policy,
        critic=trainer.critic,
        optimizers=None,
        map_location=trainer.parameters.device,
        expected_stages={loaded.opinion.stage},
    )
    if steps is None:
        steps = 4 if smoke else trainer.parameters.max_steps
    if type(steps) is not int or steps <= 0:
        trainer.env.close()
        raise ValueError("evaluation steps must be a positive int")
    diagnostics = OpinionDiagnostics(
        b_max=loaded.opinion.b_max, z_clip=loaded.opinion.z_clip
    )
    td = trainer.env.reset()
    trainer.collector.reset_all()
    scale = 0.0 if loaded.opinion.stage == "base" else loaded.opinion.residual_scale_target
    try:
        for step_index in range(steps):
            observation = td["agents", "observation"]
            info = td["agents", "info"]
            output = trainer.collector.step(
                step_id=step_index,
                observation=observation,
                pair_features=info["pair_features"],
                neighbor_ids=info["neighbor_ids"],
                pair_mask=info["pair_mask"].bool(),
                urgency=info["urgency"],
                confidence=info["confidence"],
                agent_reset_mask=info["agent_reset_mask"],
                environment_done=td["done"],
                residual_scale=scale,
            )
            td.set(("agents", "action"), output.action)
            transition = trainer.env.step(td)
            next_info = transition["next", "agents", "info"]
            diagnostics.update(
                reward=transition["next", "agents", "reward"].squeeze(-1),
                collision_agents=next_info["is_collision_with_agents"].squeeze(-1),
                collision_lanelets=next_info["is_collision_with_lanelets"].squeeze(-1),
                raw_b=output.raw_b,
                b=output.b,
                z_prev=output.z_prev,
                z_next=output.z_next,
                residual=output.residual,
                pair_mask=output.pair_mask,
                agent_reset_mask=info["agent_reset_mask"].squeeze(-1),
                residual_scale=scale,
            )
            td = step_mdp(
                transition,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=trainer.env.reward_keys,
                done_keys=trainer.env.done_keys,
            )
    finally:
        trainer.env.close()
    summary = diagnostics.summary()
    summary.update(
        {
            "stage": loaded.opinion.stage,
            "steps": steps,
            "checkpoint": str(Path(checkpoint).resolve()),
        }
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
