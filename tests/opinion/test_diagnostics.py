import torch

from utilities.opinion.diagnostics import OpinionDiagnostics


def test_diagnostics_reports_required_finite_metrics():
    diagnostics = OpinionDiagnostics()
    diagnostics.update(
        reward=torch.tensor([[1.0, 2.0]]),
        collision_agents=torch.tensor([[False, True]]),
        collision_lanelets=torch.tensor([[True, True]]),
        raw_b=torch.tensor([[[0.5, -0.5], [0.0, 0.0]]]),
        b=torch.tensor([[[0.2, -0.2], [0.0, 0.0]]]),
        z_prev=torch.tensor([[[0.1, -0.1], [0.0, 0.0]]]),
        z_next=torch.tensor([[[-0.1, -0.2], [0.0, 0.0]]]),
        residual=torch.tensor([[0.05, 0.0]]),
        pair_mask=torch.tensor([[[True, True], [False, False]]]),
        agent_reset_mask=torch.tensor([[False, True]]),
        residual_scale=0.1,
    )

    summary = diagnostics.summary()

    required = {
        "reward_mean",
        "collision_agents_rate",
        "collision_lanelets_rate",
        "collision_total_rate",
        "raw_b_mean",
        "raw_b_variance",
        "b_mean",
        "b_variance",
        "z_mean",
        "z_variance",
        "z_abs_mean",
        "z_flip_rate",
        "residual_abs_mean",
        "edge_count_mean",
        "mask_ratio",
        "reset_count",
        "residual_scale",
    }
    assert required <= set(summary)
    assert summary["reset_count"] == 1
    assert summary["collision_agents_rate"] == 0.5
    assert summary["collision_lanelets_rate"] == 1.0
    assert summary["collision_total_rate"] == 1.0
    assert summary["z_flip_rate"] > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in summary.values())
