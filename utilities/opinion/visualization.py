"""Small live diagnostics panel for M4-M5 testing rollouts."""

from __future__ import annotations

from typing import Mapping

import torch


def _optional_tensor(tensordict, key):
    try:
        return tensordict.get(key)
    except KeyError:
        return None


def update_opinion_visualization(
    env,
    tensordict,
    config: Mapping[str, object],
) -> None:
    """Copy one environment/ego diagnostic snapshot into the scenario renderer."""

    neighbor_ids = _optional_tensor(
        tensordict, ("agents", "info", "neighbor_ids")
    )
    pair_features = _optional_tensor(
        tensordict, ("agents", "info", "pair_features")
    )
    pair_mask = _optional_tensor(tensordict, ("agents", "info", "pair_mask"))
    urgency = _optional_tensor(tensordict, ("agents", "info", "urgency"))
    confidence = _optional_tensor(
        tensordict, ("agents", "info", "confidence")
    )
    if any(
        tensor is None
        for tensor in (neighbor_ids, pair_features, pair_mask, urgency, confidence)
    ):
        return

    ego_id = int(config["agent_id"])
    horizon = float(config["prediction_horizon_seconds"])
    sensing_distance = float(config["sensing_distance_meters"])
    environment_id = 0

    raw_b = _optional_tensor(tensordict, ("agents", "opinion", "raw_b"))
    gated_b = _optional_tensor(tensordict, ("agents", "opinion", "b"))
    direct_z = _optional_tensor(tensordict, ("agents", "opinion", "direct_z"))
    z_prev = _optional_tensor(tensordict, ("agents", "opinion", "z_prev"))
    z_next = _optional_tensor(tensordict, ("agents", "opinion", "z_next"))
    q = _optional_tensor(tensordict, ("agents", "opinion", "q"))
    normalized_weights = _optional_tensor(
        tensordict, ("agents", "opinion", "normalized_weights")
    )
    base_loc = _optional_tensor(
        tensordict, ("agents", "opinion", "base_loc")
    )
    final_loc = _optional_tensor(tensordict, ("agents", "loc"))
    executed_action = _optional_tensor(
        tensordict, ("agents", "action")
    )
    residual = _optional_tensor(tensordict, ("agents", "opinion", "residual"))

    def scalar(tensor: torch.Tensor, *indices) -> float:
        return float(tensor[indices].detach().cpu().item())

    residual_text = (
        f"{scalar(residual, environment_id, ego_id, 0):+.4f}"
        if residual is not None
        else "N/A (M5)"
    )
    base_speed_text = (
        f"{scalar(base_loc, environment_id, ego_id, 0):+.4f}"
        if base_loc is not None
        else "N/A"
    )
    final_speed_text = (
        f"{scalar(final_loc, environment_id, ego_id, 0):+.4f}"
        if final_loc is not None
        else "N/A"
    )
    executed_speed_text = (
        f"{scalar(executed_action, environment_id, ego_id, 0):+.4f}"
        if executed_action is not None
        else "N/A"
    )
    lines = [
        f"Opinion | ego={ego_id}",
        f"speed loc | base={base_speed_text} residual={residual_text} "
        f"final={final_speed_text}",
        f"executed speed={executed_speed_text} m/s",
    ]
    candidate_count = neighbor_ids.shape[-1]
    for candidate_index in range(candidate_count):
        neighbor_id = int(
            neighbor_ids[environment_id, ego_id, candidate_index]
            .detach()
            .cpu()
            .item()
        )
        t_cpa = (
            scalar(pair_features, environment_id, ego_id, candidate_index, 8)
            * horizon
        )
        d_cpa = (
            scalar(pair_features, environment_id, ego_id, candidate_index, 9)
            * sensing_distance
        )
        is_active = bool(
            pair_mask[environment_id, ego_id, candidate_index]
            .detach()
            .cpu()
            .item()
        )
        lines.append(
            f"j={neighbor_id} | mask={int(is_active)} | "
            f"tCPA={t_cpa:.2f}s | dCPA={d_cpa:.2f}m"
        )
        if raw_b is None or gated_b is None:
            lines.append(
                f"  rho={scalar(urgency, environment_id, ego_id, candidate_index):.3f} "
                f"conf={scalar(confidence, environment_id, ego_id, candidate_index):.3f} "
                "raw_b=N/A b=N/A"
            )
            lines.append("  z_direct=N/A (M5) | z_stateful=N/A (M6)")
        else:
            lines.append(
                f"  rho={scalar(urgency, environment_id, ego_id, candidate_index):.3f} "
                f"conf={scalar(confidence, environment_id, ego_id, candidate_index):.3f} "
                f"raw_b={scalar(raw_b, environment_id, ego_id, candidate_index):+.3f} "
                f"b={scalar(gated_b, environment_id, ego_id, candidate_index):+.3f}"
            )
            if z_prev is not None and z_next is not None:
                previous = scalar(
                    z_prev, environment_id, ego_id, candidate_index
                )
                updated = scalar(
                    z_next, environment_id, ego_id, candidate_index
                )
                lines.append(
                    f"  z_prev={previous:+.3f} z_next={updated:+.3f} "
                    f"dz={updated - previous:+.3f}"
                )
                if q is not None and normalized_weights is not None:
                    lines.append(
                        f"  q={scalar(q, environment_id, ego_id, candidate_index):+.3f} "
                        f"weight={scalar(normalized_weights, environment_id, ego_id, candidate_index):.3f}"
                    )
            elif direct_z is not None:
                lines.append(
                    f"  z_direct={scalar(direct_z, environment_id, ego_id, candidate_index):+.3f} "
                    "| z_stateful=N/A (M6)"
                )
            else:
                lines.append("  z=N/A")

    base_env = env.base_env
    scenario = getattr(base_env, "scenario_name", None)
    if scenario is not None and hasattr(scenario, "set_opinion_visualization"):
        scenario.set_opinion_visualization(lines)
