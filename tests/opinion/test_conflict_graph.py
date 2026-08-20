import pytest
import torch

from utilities.opinion.conflict_graph import (
    PAIR_FEATURE_NAMES,
    ConflictGraph,
)


def _graph(n_candidates=1):
    return ConflictGraph(
        n_candidates=n_candidates,
        ttc_horizon=5.0,
        safe_distance=1.0,
        urgency_time_scale=2.0,
        urgency_distance_temperature=0.2,
    )


def _two_agent_state(pos_j, vel_i, vel_j, *, visible=True):
    positions = torch.tensor([[[0.0, 0.0], pos_j]], dtype=torch.float32)
    velocities = torch.tensor([[vel_i, vel_j]], dtype=torch.float32)
    headings = torch.atan2(velocities[..., 1], velocities[..., 0])
    visibility = torch.tensor(
        [[[False, visible], [visible, False]]], dtype=torch.bool
    )
    return positions, velocities, headings, visibility


def test_pair_feature_schema_is_explicit_and_current_physics_only():
    assert PAIR_FEATURE_NAMES == (
        "relative_position_longitudinal",
        "relative_position_lateral",
        "relative_velocity_longitudinal",
        "relative_velocity_lateral",
        "distance",
        "closing_speed",
        "time_to_closest_approach",
        "distance_at_closest_approach",
        "heading_difference_sin",
        "heading_difference_cos",
        "ego_speed",
        "neighbor_speed",
    )
    assert not any("opinion" in name or "future" in name for name in PAIR_FEATURE_NAMES)


def test_head_on_approach_has_positive_urgency_and_finite_pair_features():
    state = _two_agent_state([4.0, 0.0], [1.0, 0.0], [-1.0, 0.0])

    result = _graph()(*state)

    assert result.neighbor_ids.tolist() == [[[1], [0]]]
    assert result.pair_mask.all()
    assert (result.urgency > 0).all()
    assert torch.isfinite(result.pair_features).all()
    t_cpa_index = PAIR_FEATURE_NAMES.index("time_to_closest_approach")
    d_cpa_index = PAIR_FEATURE_NAMES.index("distance_at_closest_approach")
    assert result.pair_features[0, 0, 0, t_cpa_index] == pytest.approx(2.0)
    assert result.pair_features[0, 0, 0, d_cpa_index] == pytest.approx(0.0)


def test_parallel_same_speed_is_not_an_active_conflict():
    state = _two_agent_state([0.0, 2.0], [1.0, 0.0], [1.0, 0.0])

    result = _graph()(*state)

    assert result.pair_mask.all()
    assert torch.equal(result.urgency, torch.zeros_like(result.urgency))


def test_diverging_pair_is_not_an_active_conflict():
    state = _two_agent_state([2.0, 0.0], [-1.0, 0.0], [1.0, 0.0])

    result = _graph()(*state)

    assert result.pair_mask.all()
    assert torch.equal(result.urgency, torch.zeros_like(result.urgency))


def test_crossing_approach_has_positive_urgency_and_zero_cpa_distance():
    positions = torch.tensor([[[-2.0, 0.0], [0.0, -2.0]]])
    velocities = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    headings = torch.tensor([[0.0, torch.pi / 2]])
    visibility = torch.tensor([[[False, True], [True, False]]])

    result = _graph()(positions, velocities, headings, visibility)

    d_cpa_index = PAIR_FEATURE_NAMES.index("distance_at_closest_approach")
    assert (result.urgency > 0).all()
    assert torch.allclose(
        result.pair_features[..., d_cpa_index], torch.zeros(1, 2, 1), atol=1e-6
    )


def test_approaching_stationary_neighbor_is_a_conflict():
    state = _two_agent_state([2.0, 0.0], [1.0, 0.0], [0.0, 0.0])

    result = _graph()(*state)

    assert result.urgency[0, 0, 0] > 0
    assert result.confidence[0, 0, 0] == 1


def test_invisible_neighbor_is_padded_and_has_no_evidence_weight():
    state = _two_agent_state([2.0, 0.0], [1.0, 0.0], [0.0, 0.0], visible=False)

    result = _graph()(*state)

    assert result.neighbor_ids.tolist() == [[[-1], [-1]]]
    assert not result.pair_mask.any()
    assert not result.urgency.any()
    assert not result.confidence.any()
    assert not result.pair_features.any()


def test_candidates_are_ranked_by_urgency_then_distance_with_stable_global_ids():
    positions = torch.tensor(
        [[[0.0, 0.0], [4.0, 0.0], [0.0, 2.0], [-4.0, 0.0]]]
    )
    velocities = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    headings = torch.atan2(velocities[..., 1], velocities[..., 0])
    visibility = ~torch.eye(4, dtype=torch.bool).unsqueeze(0)

    result = _graph(n_candidates=3)(positions, velocities, headings, visibility)

    # For ego 0, agent 1 is the only active conflict. The remaining zero-urgency
    # candidates are ordered by current distance, then global ID.
    assert result.neighbor_ids[0, 0].tolist() == [1, 2, 3]
    assert result.urgency[0, 0, 0] > result.urgency[0, 0, 1]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("positions", torch.zeros(2, 3), "positions"),
        ("velocities", torch.zeros(1, 2, 3), "velocities"),
        ("headings", torch.zeros(1, 2, 1), "headings"),
        ("visibility", torch.ones(1, 2, 2), "visibility_mask"),
    ),
)
def test_input_contract_rejects_wrong_shapes_or_dtypes(field, value, match):
    values = {
        "positions": torch.zeros(1, 2, 2),
        "velocities": torch.zeros(1, 2, 2),
        "headings": torch.zeros(1, 2),
        "visibility": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=match):
        _graph()(
            values["positions"],
            values["velocities"],
            values["headings"],
            values["visibility"],
        )
