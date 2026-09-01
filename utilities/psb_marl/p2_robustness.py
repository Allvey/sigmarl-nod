"""Locked P2.2-R holdout validation and multi-training-seed aggregation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from utilities.experiment_artifacts import atomic_write_json
from utilities.psb_marl.checkpoint import sha256_file
from utilities.psb_marl.config import load_psb_experiment


class P2RobustnessError(ValueError):
    """Raised when a run or report violates the locked robustness protocol."""


def _load_object(path: Path, label: str) -> Dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise P2RobustnessError(f"{label} must contain a JSON object: {source}")
    return value


def _protocol_path(protocol_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise P2RobustnessError("Protocol config paths must be non-empty strings.")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repository = Path(protocol_path).expanduser().resolve().parent.parents[1]
    return (repository / candidate).resolve()


def load_robustness_protocol(path: Path) -> Dict[str, object]:
    protocol_path = Path(path).expanduser().resolve()
    protocol = _load_object(protocol_path, "P2.2-R protocol")
    expected_keys = {
        "schema_version",
        "method",
        "protocol",
        "training_iterations",
        "training_runs",
        "evaluations",
        "required_gates",
    }
    if set(protocol) != expected_keys:
        raise P2RobustnessError(
            "P2.2-R protocol keys do not match the locked schema."
        )
    if (
        protocol["schema_version"] != 1
        or protocol["method"] != "psb_marl_robustness"
        or protocol["protocol"] != "p2_2_r_locked_holdout"
    ):
        raise P2RobustnessError("Unsupported P2.2-R protocol identity.")
    iterations = protocol["training_iterations"]
    if type(iterations) is not int or iterations <= 0:
        raise P2RobustnessError("training_iterations must be positive.")

    training_runs = protocol["training_runs"]
    if not isinstance(training_runs, list) or len(training_runs) < 3:
        raise P2RobustnessError("P2.2-R requires at least three training seeds.")
    training_seeds = []
    for item in training_runs:
        if not isinstance(item, dict) or set(item) != {"training_seed", "config"}:
            raise P2RobustnessError("Each training run needs seed and config.")
        seed = item["training_seed"]
        if type(seed) is not int or seed < 0:
            raise P2RobustnessError("Training seeds must be non-negative integers.")
        config_path = _protocol_path(protocol_path, item["config"])
        experiment = load_psb_experiment(config_path)
        if experiment.training_seed != seed:
            raise P2RobustnessError(
                f"Training config seed does not match protocol: {config_path}"
            )
        if experiment.training is None or experiment.training.iterations != iterations:
            raise P2RobustnessError(
                f"Training budget does not match protocol: {config_path}"
            )
        training_seeds.append(seed)
    if len(set(training_seeds)) != len(training_seeds):
        raise P2RobustnessError("P2.2-R training seeds must be unique.")

    evaluations = protocol["evaluations"]
    if not isinstance(evaluations, list) or len(evaluations) < 2:
        raise P2RobustnessError("P2.2-R requires training-domain and transfer tests.")
    evaluation_names = []
    all_evaluation_seeds = []
    for item in evaluations:
        required = {"name", "scenario", "max_steps", "episodes", "seeds"}
        if not isinstance(item, dict) or set(item) != required:
            raise P2RobustnessError("Evaluation entries violate the locked schema.")
        name = item["name"]
        seeds = item["seeds"]
        if not isinstance(name, str) or not name.startswith("holdout_"):
            raise P2RobustnessError("Evaluation names must start with holdout_.")
        if type(item["max_steps"]) is not int or item["max_steps"] <= 1:
            raise P2RobustnessError("Evaluation max_steps must exceed one.")
        if type(item["episodes"]) is not int or item["episodes"] <= 0:
            raise P2RobustnessError("Evaluation episodes must be positive.")
        if (
            not isinstance(seeds, list)
            or len(seeds) < 10
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise P2RobustnessError(
                "Each holdout evaluation requires at least ten unique seeds."
            )
        evaluation_names.append(name)
        all_evaluation_seeds.extend(seeds)
    if len(set(evaluation_names)) != len(evaluation_names):
        raise P2RobustnessError("Holdout evaluation names must be unique.")
    if len(set(all_evaluation_seeds)) != len(all_evaluation_seeds):
        raise P2RobustnessError("Holdout seed sets must be disjoint.")

    required_gates = protocol["required_gates"]
    expected_gates = {
        "reward_passed",
        "collision_passed",
        "lane_collision_passed",
        "structural_passed",
    }
    if not isinstance(required_gates, list) or set(required_gates) != expected_gates:
        raise P2RobustnessError("P2.2-R required gates cannot be weakened.")
    return protocol


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _validate_report(
    report: Mapping[str, object],
    *,
    training_seed: int,
    evaluation: Mapping[str, object],
    required_gates: Sequence[str],
) -> Dict[str, object]:
    expected_protocol = {
        "scenario_type": evaluation["scenario"],
        "max_steps": evaluation["max_steps"],
        "episodes": evaluation["episodes"],
        "seeds": evaluation["seeds"],
        "compare_base": True,
    }
    if report.get("report_label") != evaluation["name"]:
        raise P2RobustnessError("Holdout report label does not match protocol.")
    if report.get("training_seed") != training_seed:
        raise P2RobustnessError("Holdout report training seed is incorrect.")
    if report.get("evaluation_protocol") != expected_protocol:
        raise P2RobustnessError("Holdout report evaluation protocol is incorrect.")
    comparisons = report.get("paired_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(
        evaluation["seeds"]
    ):
        raise P2RobustnessError("Holdout report has the wrong paired seed count.")
    if [item.get("seed") for item in comparisons] != evaluation["seeds"]:
        raise P2RobustnessError("Holdout report seed order does not match protocol.")
    gate = report.get("noninferiority_gate")
    if not isinstance(gate, dict):
        raise P2RobustnessError("Holdout report is missing its gate result.")
    gate_results = {name: gate.get(name) is True for name in required_gates}
    metrics = {
        "reward_difference": statistics.fmean(
            float(item["reward_difference_candidate_minus_base"])
            for item in comparisons
        ),
        "vehicle_collision_difference": statistics.fmean(
            float(item["vehicle_collision_difference_candidate_minus_base"])
            for item in comparisons
        ),
        "lane_collision_difference": statistics.fmean(
            float(item["lane_collision_difference_candidate_minus_base"])
            for item in comparisons
        ),
        "total_collision_difference": statistics.fmean(
            float(item["collision_difference_candidate_minus_base"])
            for item in comparisons
        ),
    }
    return {
        "passed": bool(gate.get("passed") is True and all(gate_results.values())),
        "gate_results": gate_results,
        "metrics": {name: float(value) for name, value in metrics.items()},
        "confidence_bounds": {
            "reward_lower_bound": gate.get("reward_lower_bound"),
            "collision_upper_bound": gate.get("collision_upper_bound"),
            "lane_collision_upper_bound": gate.get(
                "lane_collision_upper_bound"
            ),
        },
    }


def aggregate_robustness(
    protocol_path: Path,
    run_directories: Sequence[Path],
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Validate locked reports and summarize variability across training seeds."""

    protocol_path = Path(protocol_path).expanduser().resolve()
    protocol = load_robustness_protocol(protocol_path)
    expected_runs = protocol["training_runs"]
    if len(run_directories) != len(expected_runs):
        raise P2RobustnessError(
            f"Expected {len(expected_runs)} run directories, got {len(run_directories)}."
        )

    runs_by_seed: Dict[int, Path] = {}
    policy_hashes_by_seed: Dict[int, str] = {}
    for raw_run in run_directories:
        run = Path(raw_run).expanduser().resolve()
        status = _load_object(run / "training_status.json", "training status")
        resolved = _load_object(run / "psb_config_resolved.json", "PSB config")
        runtime = resolved.get("runtime_config")
        if not isinstance(runtime, dict):
            raise P2RobustnessError("Run is missing its P2 runtime contract.")
        seed = runtime.get("training_seed")
        if type(seed) is not int or seed in runs_by_seed:
            raise P2RobustnessError("Run training seeds must be explicit and unique.")
        if status.get("status") != "completed" or status.get("iteration") != protocol[
            "training_iterations"
        ]:
            raise P2RobustnessError(f"Training run is incomplete: {run}")
        runs_by_seed[seed] = run
        policy_hashes_by_seed[seed] = sha256_file(run / "candidate_policy.pth")

    if len(set(policy_hashes_by_seed.values())) != len(policy_hashes_by_seed):
        raise P2RobustnessError(
            "Candidate policies are byte-identical across training seeds; "
            "the runs are not independent stochastic replications."
        )

    run_results = []
    for specification in expected_runs:
        seed = specification["training_seed"]
        if seed not in runs_by_seed:
            raise P2RobustnessError(f"Missing training seed {seed}.")
        run = runs_by_seed[seed]
        experiment = load_psb_experiment(
            _protocol_path(protocol_path, specification["config"])
        )
        resolved = _load_object(run / "psb_config_resolved.json", "PSB config")
        if resolved.get("runtime_config") != experiment.p2_runtime_config():
            raise P2RobustnessError(f"Run does not match seed-{seed} config.")
        candidate = run / "candidate_policy.pth"
        evaluations = {}
        for evaluation in protocol["evaluations"]:
            report_name = f"p2_manual_validation_{evaluation['name']}.json"
            report = _load_object(run / report_name, "holdout report")
            evaluations[evaluation["name"]] = _validate_report(
                report,
                training_seed=seed,
                evaluation=evaluation,
                required_gates=protocol["required_gates"],
            )
        run_results.append(
            {
                "training_seed": seed,
                "run_directory": str(run),
                "candidate_policy_sha256": policy_hashes_by_seed[seed],
                "passed": all(item["passed"] for item in evaluations.values()),
                "evaluations": evaluations,
            }
        )

    scenario_summaries = {}
    metric_names = (
        "reward_difference",
        "vehicle_collision_difference",
        "lane_collision_difference",
        "total_collision_difference",
    )
    for evaluation in protocol["evaluations"]:
        name = evaluation["name"]
        scenario_summaries[name] = {
            "scenario": evaluation["scenario"],
            "all_training_seeds_passed": all(
                run["evaluations"][name]["passed"] for run in run_results
            ),
            "across_training_seeds": {
                metric: _mean_std(
                    [
                        run["evaluations"][name]["metrics"][metric]
                        for run in run_results
                    ]
                )
                for metric in metric_names
            },
        }

    summary = {
        "schema_version": 1,
        "method": "psb_marl_robustness",
        "protocol": protocol["protocol"],
        "protocol_path": str(protocol_path),
        "passed": all(run["passed"] for run in run_results),
        "training_seed_count": len(run_results),
        "run_results": run_results,
        "scenario_summaries": scenario_summaries,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path).expanduser().resolve(), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = aggregate_robustness(
        arguments.protocol,
        arguments.run_dirs,
        output_path=arguments.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
