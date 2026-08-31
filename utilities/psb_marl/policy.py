"""P1 rollout adapter: update bifurcation state without altering Base actions."""

from __future__ import annotations

import torch
from torch import nn

from utilities.psb_marl.state import P1ZeroControlStateTracker


def validate_p1_runtime_contract(runtime_config, environment_n_agents: int) -> None:
    """Validate P1 invariants without requiring the deployment fleet size.

    ``runtime_config['n_agents']`` records the Base source run.  The shared
    decentralized Actor and the parameter-free P1 edge tracker are both
    cardinality agnostic, so a testing scenario may contain a different
    number of agents.
    """

    if runtime_config.get("stage") != "p1_zero_control_equivalence":
        raise ValueError("Unsupported PSB runtime stage.")
    if runtime_config.get("control_mode") != "zero":
        raise ValueError("P1 requires zero bifurcation control.")
    if float(runtime_config.get("actor_context_gain", -1.0)) != 0.0:
        raise ValueError("P1 requires actor_context_gain=0.")
    source_n_agents = runtime_config.get("n_agents")
    if type(source_n_agents) is not int or source_n_agents < 2:
        raise ValueError("P1 source n_agents metadata must be an integer >= 2.")
    if type(environment_n_agents) is not int or environment_n_agents < 2:
        raise ValueError("P1 environment n_agents must be an integer >= 2.")


class P1ZeroControlPolicyController(nn.Module):
    """Run the P1 state side path, then call the untouched Base policy once."""

    _DIAGNOSTIC_KEYS = (
        ("agents", "psb", "z_prev_dense"),
        ("agents", "psb", "z_next_dense"),
        ("agents", "psb", "z_next"),
        ("agents", "psb", "rho"),
        ("agents", "psb", "b"),
        ("agents", "psb", "root_residual"),
        ("agents", "psb", "root_denominator"),
    )

    def __init__(self, policy: nn.Module, tracker: P1ZeroControlStateTracker) -> None:
        super().__init__()
        self.policy = policy
        self.tracker = tracker
        self.in_keys = list(getattr(policy, "in_keys", ()))
        for key in (
            ("agents", "info", "neighbor_ids"),
            ("agents", "info", "pair_mask"),
            ("agents", "info", "urgency"),
            ("agents", "info", "agent_reset_mask"),
        ):
            if key not in self.in_keys:
                self.in_keys.append(key)
        self.out_keys = list(getattr(policy, "out_keys", ()))
        for key in self._DIAGNOSTIC_KEYS:
            if key not in self.out_keys:
                self.out_keys.append(key)

    @property
    def spec(self):
        return getattr(self.policy, "spec", None)

    @staticmethod
    def _optional(tensordict, key):
        try:
            return tensordict.get(key)
        except KeyError:
            return None

    def forward(self, tensordict):
        step = self.tracker.step(
            neighbor_ids=tensordict.get(("agents", "info", "neighbor_ids")),
            pair_mask=tensordict.get(("agents", "info", "pair_mask")),
            urgency=tensordict.get(("agents", "info", "urgency")),
            agent_reset_mask=tensordict.get(
                ("agents", "info", "agent_reset_mask")
            ),
            environment_done=self._optional(tensordict, "done"),
        )
        tensordict.set(self._DIAGNOSTIC_KEYS[0], step.z_prev_dense)
        tensordict.set(self._DIAGNOSTIC_KEYS[1], step.z_next_dense)
        tensordict.set(self._DIAGNOSTIC_KEYS[2], step.z_next_candidates)
        tensordict.set(self._DIAGNOSTIC_KEYS[3], step.rho_dense)
        tensordict.set(self._DIAGNOSTIC_KEYS[4], step.b_dense)
        tensordict.set(self._DIAGNOSTIC_KEYS[5], step.residual_dense)
        tensordict.set(self._DIAGNOSTIC_KEYS[6], step.denominator_dense)
        # P1's sole action-affecting operation is this unchanged Base call.
        return self.policy(tensordict)

    def reset_state(self) -> None:
        self.tracker.reset_all()
