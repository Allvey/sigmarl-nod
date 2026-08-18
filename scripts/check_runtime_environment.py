#!/usr/bin/env python3
"""Run the reproducible M0 dependency and road-traffic smoke checks."""

from __future__ import annotations

import argparse
import importlib
import platform
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODULES = ("torch", "torchrl", "tensordict", "vmas")


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_required_pins(requirements_path: Path) -> Dict[str, str]:
    required_names = {
        _normalize_package_name(module_name): module_name
        for module_name in REQUIRED_MODULES
    }
    pins = {}
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            f"unable to read requirements pins from '{requirements_path}': {error}"
        ) from error

    exact_pin_pattern = re.compile(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)$"
    )
    package_pattern = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
    for line_number, raw_line in enumerate(lines, start=1):
        requirement = raw_line.split("#", 1)[0].strip()
        if not requirement:
            continue

        exact_match = exact_pin_pattern.fullmatch(requirement)
        if exact_match:
            package_name, version = exact_match.groups()
            normalized_name = _normalize_package_name(package_name)
            if normalized_name not in required_names:
                continue
            if normalized_name in pins:
                raise RuntimeError(
                    f"duplicate exact pin for package "
                    f"'{required_names[normalized_name]}' at line {line_number}"
                )
            pins[normalized_name] = version
            continue

        package_match = package_pattern.match(requirement)
        if package_match:
            normalized_name = _normalize_package_name(package_match.group(1))
            if normalized_name in required_names:
                raise RuntimeError(
                    f"package '{required_names[normalized_name]}' must use an exact "
                    f"== pin in '{requirements_path}' (line {line_number})"
                )

    for normalized_name, module_name in required_names.items():
        if normalized_name not in pins:
            raise RuntimeError(
                f"package '{module_name}' has no exact == pin in "
                f"'{requirements_path}'"
            )
    return pins


def runtime_versions(
    requirements_path: Path = PROJECT_ROOT / "requirements.txt",
) -> Dict[str, str]:
    """Import required modules and verify versions against requirements pins."""
    expected_pins = _read_required_pins(requirements_path)
    versions = {"python": platform.python_version()}
    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        actual_version = str(getattr(module, "__version__", "unknown"))
        expected_version = expected_pins[_normalize_package_name(module_name)]
        if actual_version != expected_version:
            raise RuntimeError(
                f"version mismatch for package '{module_name}': "
                f"expected={expected_version}, actual={actual_version}"
            )
        versions[module_name] = actual_version
    return versions


def assert_finite_tensors(tensordict, *, context: str) -> None:
    """Raise with a useful nested key when a floating tensor is non-finite."""
    import torch

    for key, value in tensordict.items(include_nested=True, leaves_only=True):
        if not torch.is_tensor(value):
            continue
        if not (value.is_floating_point() or value.is_complex()):
            continue
        if not torch.isfinite(value).all():
            key_path = "/".join(key) if isinstance(key, tuple) else str(key)
            raise RuntimeError(
                f"{context} contains a non-finite tensor at '{key_path}'"
            )


def check_tanh_normal() -> Dict[str, torch.Tensor]:
    """Exercise the TanhNormal sample and log-probability path used by PPO."""
    import torch
    from torchrl.modules import TanhNormal

    distribution = TanhNormal(
        loc=torch.zeros(2, 2),
        scale=torch.ones(2, 2),
        min=-1.0,
        max=1.0,
    )
    sample = distribution.rsample()
    log_prob = distribution.log_prob(sample)

    if not torch.isfinite(sample).all():
        raise RuntimeError("TanhNormal produced a non-finite sample")
    if not torch.isfinite(log_prob).all():
        raise RuntimeError("TanhNormal produced a non-finite log_prob")
    return {"sample": sample, "log_prob": log_prob}


def check_road_traffic_rollout(*, steps: int) -> int:
    """Construct the project road-traffic env and run random actions."""
    if not 2 <= steps <= 10:
        raise ValueError("rollout steps must be between 2 and 10")

    import torch
    from torchrl.envs.libs.vmas import VmasEnv
    from torchrl.envs.utils import step_mdp

    from scenarios.road_traffic import ScenarioRoadTraffic
    from utilities.helper_training import Parameters

    torch.manual_seed(0)
    parameters = Parameters(
        n_agents=4,
        frames_per_batch=8,
        max_steps=10,
        scenario_type="CPM_mixed",
        cpm_scenario_probabilities=[1.0, 0.0, 0.0],
        is_challenging_initial_state_buffer=False,
        is_add_noise=False,
        is_visualize_short_term_path=False,
        is_visualize_extra_info=False,
        is_visualize_agent_id=False,
    )
    scenario = ScenarioRoadTraffic()
    scenario.parameters = parameters
    env = VmasEnv(
        scenario=scenario,
        num_envs=1,
        continuous_actions=True,
        max_steps=10,
        device="cpu",
        n_agents=parameters.n_agents,
    )

    try:
        tensordict = env.reset()
        assert_finite_tensors(tensordict, context="road_traffic reset")
        for step_index in range(steps):
            tensordict = env.rand_step(tensordict)
            assert_finite_tensors(
                tensordict,
                context=f"road_traffic rollout step {step_index + 1}",
            )
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=env.reward_keys,
                done_keys=env.done_keys,
            )
    finally:
        env.close()

    return steps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="number of random road-traffic rollout steps (2-10; default: 3)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        versions = runtime_versions()
        tanh_result = check_tanh_normal()
        completed_steps = check_road_traffic_rollout(steps=args.steps)
    except Exception as error:
        print(f"[FAIL] M0 runtime check: {error}", file=sys.stderr)
        return 1

    print("[OK] Core runtime versions:")
    for name, version in versions.items():
        print(f"  {name}: {version}")
    print(
        "[OK] TanhNormal sample/log_prob are finite "
        f"(sample shape={tuple(tanh_result['sample'].shape)}, "
        f"log_prob shape={tuple(tanh_result['log_prob'].shape)})"
    )
    print(f"[OK] road_traffic random rollout completed {completed_steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
