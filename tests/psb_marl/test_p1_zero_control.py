"""P1 proximal mathematics, packaging, and zero-action-path contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from utilities.psb_marl.certification import certify_p1_layer
from utilities.psb_marl.checkpoint import sha256_file
from utilities.psb_marl.config import PSBConfigError, load_psb_experiment
from utilities.psb_marl.evaluator import test_psb as run_psb_test
from utilities.psb_marl.policy import (
    P1ZeroControlPolicyController,
    validate_p1_runtime_contract,
)
from utilities.psb_marl.proximal import ProximalSaturatingBifurcation
from utilities.psb_marl.state import P1ZeroControlStateTracker
from utilities.psb_marl.trainer import train_psb


class FakeTensorDict:
    def __init__(self, values):
        self.values = dict(values)

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value
        return self


class DeterministicPolicy(nn.Module):
    in_keys = [("agents", "observation")]
    out_keys = [("agents", "action")]
    spec = None

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, tensordict):
        observation = tensordict.get(("agents", "observation"))
        return tensordict.set(("agents", "action"), observation[..., :2] + self.anchor)


def make_layer() -> ProximalSaturatingBifurcation:
    return ProximalSaturatingBifurcation(
        kappa=1.0,
        nu=1.0,
        alpha=1.0,
        rho_max=2.0,
        h_z=0.25,
        b_max=0.0,
        residual_tolerance=1e-8,
        max_iterations=64,
    )


class P1ProximalTests(unittest.TestCase):
    def test_runtime_allows_cross_scenario_agent_count(self):
        runtime = {
            "stage": "p1_zero_control_equivalence",
            "n_agents": 4,
            "actor_context_gain": 0.0,
            "control_mode": "zero",
        }
        self.assertIsNone(validate_p1_runtime_contract(runtime, 6))

    def test_runtime_still_rejects_action_affecting_configuration(self):
        runtime = {
            "stage": "p1_zero_control_equivalence",
            "n_agents": 4,
            "actor_context_gain": 0.1,
            "control_mode": "zero",
        }
        with self.assertRaisesRegex(ValueError, "actor_context_gain"):
            validate_p1_runtime_contract(runtime, 6)

    def test_certification_covers_implicit_gradient_and_energy(self):
        report = certify_p1_layer(make_layer())
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))

    def test_implicit_gradient_matches_closed_form(self):
        layer = make_layer()
        z_prev = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
        rho = torch.tensor([1.4], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        z_next = layer(z_prev, rho, b)
        z_next.backward()
        result = layer.solve_with_diagnostics(
            z_prev.detach(), rho.detach(), b.detach()
        )
        denominator = result.denominator.item()
        self.assertAlmostEqual(z_prev.grad.item(), 4.0 / denominator, places=12)
        self.assertAlmostEqual(b.grad.item(), 1.0 / denominator, places=12)

    def test_state_is_antisymmetric_and_zero_control(self):
        layer = make_layer()
        tracker = P1ZeroControlStateTracker(
            n_agents=3, proximal=layer, zero_threshold=1e-12
        )
        initial = torch.tensor(
            [[[0.0, 0.2, 0.0], [-0.2, 0.0, -0.1], [0.0, 0.1, 0.0]]],
            dtype=torch.float64,
        )
        tracker.set_state_for_testing(initial)
        step = tracker.step(
            neighbor_ids=torch.tensor([[[1, 2], [0, 2], [0, 1]]]),
            pair_mask=torch.ones(1, 3, 2, dtype=torch.bool),
            urgency=torch.tensor(
                [[[0.8, 0.2], [0.8, 0.6], [0.2, 0.6]]],
                dtype=torch.float64,
            ),
            agent_reset_mask=torch.zeros(1, 3, dtype=torch.bool),
        )
        self.assertTrue(
            torch.allclose(step.z_next_dense, -step.z_next_dense.transpose(-1, -2))
        )
        self.assertEqual(float(step.b_dense.abs().max().item()), 0.0)
        self.assertLessEqual(
            float(step.residual_dense.abs().max().item()),
            layer.residual_tolerance,
        )

    def test_policy_side_path_cannot_change_base_action(self):
        layer = make_layer()
        policy = DeterministicPolicy()
        controller = P1ZeroControlPolicyController(
            policy,
            P1ZeroControlStateTracker(
                n_agents=2, proximal=layer, zero_threshold=1e-12
            ),
        )
        observation = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        values = {
            ("agents", "observation"): observation,
            ("agents", "info", "neighbor_ids"): torch.tensor([[[1], [0]]]),
            ("agents", "info", "pair_mask"): torch.ones(1, 2, 1, dtype=torch.bool),
            ("agents", "info", "urgency"): torch.ones(1, 2, 1),
            ("agents", "info", "agent_reset_mask"): torch.zeros(
                1, 2, dtype=torch.bool
            ),
        }
        direct = policy(FakeTensorDict(values)).get(("agents", "action")).clone()
        wrapped = controller(FakeTensorDict(values)).get(("agents", "action"))
        self.assertTrue(torch.equal(direct, wrapped))


class P1ArtifactTests(unittest.TestCase):
    def _experiment(self, root: Path) -> tuple[Path, Path, Path]:
        repository = Path(__file__).resolve().parents[2]
        base_config = json.loads((repository / "config.json").read_text())
        base_config_path = root / "base.json"
        base_config_path.write_text(json.dumps(base_config), encoding="utf-8")
        base_run = root / "base-run"
        base_run.mkdir()
        (base_run / "config_resolved.json").write_text(
            json.dumps(base_config), encoding="utf-8"
        )
        policy = base_run / "final_policy.pth"
        critic = base_run / "final_critic.pth"
        torch.save({"weight": torch.arange(3)}, policy)
        torch.save({"weight": torch.arange(5)}, critic)
        p0_config = {
            "schema_version": 1,
            "method": "psb_marl",
            "stage": "p0_base_passthrough",
            "base_config": str(base_config_path),
            "output_root": str(root / "p0-output"),
            "base": {
                "run_directory": str(base_run),
                "policy_checkpoint": str(policy),
                "critic_checkpoint": str(critic),
            },
        }
        p0_path = root / "p0.json"
        p0_path.write_text(json.dumps(p0_config), encoding="utf-8")
        p0_run = train_psb(p0_path)
        (p0_run / "p0_manual_validation.json").write_text(
            json.dumps(
                {
                    "noninferiority_result": "proven_by_identical_policy_checkpoint",
                    "rollouts": [
                        {
                            "nonfinite_action_count": 0,
                            "nonfinite_reward_count": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        p1_config = {
            **p0_config,
            "stage": "p1_zero_control_equivalence",
            "output_root": str(root / "p1-output"),
            "parent_run": str(p0_run),
            "conflict_graph": {
                "emit_pair_info": True,
                "candidate_count": 2,
                "pair_feature_dim": 10,
                "prediction_horizon_seconds": 3.0,
                "conflict_distance_meters": 2.0,
                "sensing_distance_meters": 20.0,
                "cpa_epsilon": 1e-6,
                "urgency_time_scale_seconds": 3.0,
                "urgency_distance_scale_meters": 2.0,
            },
            "proximal": {
                "kappa": 1.0,
                "nu": 1.0,
                "alpha": 1.0,
                "rho_max": 2.0,
                "tau_z": 0.2,
                "b_max": 0.0,
                "residual_tolerance": 1e-6,
                "max_iterations": 64,
                "zero_threshold": 1e-8,
            },
        }
        p1_path = root / "p1.json"
        p1_path.write_text(json.dumps(p1_config), encoding="utf-8")
        return p1_path, policy, critic

    def test_p1_requires_validated_p0_and_strong_convexity(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._experiment(Path(directory))
            experiment = load_psb_experiment(config_path)
            self.assertGreater(
                experiment.proximal.convexity_margin(experiment.dt), 0.0
            )
            raw = json.loads(config_path.read_text())
            raw["proximal"]["tau_z"] = 0.01
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(PSBConfigError, "uniqueness"):
                load_psb_experiment(config_path)

    def test_p1_packaging_keeps_base_bytes_and_certifies_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, source_policy, source_critic = self._experiment(
                Path(directory)
            )
            run = train_psb(config_path)
            self.assertEqual(
                sha256_file(run / "final_policy.pth"), sha256_file(source_policy)
            )
            self.assertEqual(
                sha256_file(run / "final_critic.pth"), sha256_file(source_critic)
            )
            certification = json.loads((run / "p1_certification.json").read_text())
            equivalence = json.loads((run / "p1_equivalence.json").read_text())
            self.assertTrue(certification["passed"])
            self.assertEqual(equivalence["actor_context_gain"], 0.0)
            self.assertEqual(equivalence["trainable_psb_parameters"], 0)

    def test_p1_evaluator_requires_exact_paired_actions(self):
        class FakeRollout:
            def __init__(self, include_psb: bool):
                self.values = {
                    ("agents", "action"): torch.zeros(2, 3, 4, 2),
                    ("next", "agents", "reward"): torch.ones(2, 3, 4, 1),
                }
                if include_psb:
                    zeros = torch.zeros(2, 3, 4, 4)
                    self.values.update(
                        {
                            ("agents", "psb", "z_next_dense"): zeros,
                            ("agents", "psb", "root_residual"): zeros,
                            ("agents", "psb", "root_denominator"): zeros + 3.0,
                            ("agents", "psb", "b"): zeros,
                        }
                    )

            def get(self, key):
                return self.values[key]

        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._experiment(Path(directory))
            run = train_psb(config_path)

            def fake_test(*args, **kwargs):
                return FakeRollout(kwargs.get("psb_runtime_config") is not None)

            with patch("main_testing.test_base", side_effect=fake_test):
                report = run_psb_test(
                    config_path,
                    run_directory=run,
                    max_steps=4,
                    episodes=2,
                    seeds=(7,),
                    render=False,
                    compare_base=True,
                    promote_if_noninferior=True,
                )
            self.assertEqual(
                report["noninferiority_result"],
                "proven_by_exact_paired_actions",
            )
            self.assertTrue(report["paired_comparisons"][0]["actions_exactly_equal"])


if __name__ == "__main__":
    unittest.main()
