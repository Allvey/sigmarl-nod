import torch
from torchrl.envs.libs.vmas import VmasEnv

from utilities.opinion.config import load_opinion_experiment_config
from scenarios.road_traffic import ScenarioRoadTraffic


OPINION_KEYS = {
    "pair_features",
    "neighbor_ids",
    "pair_mask",
    "urgency",
    "confidence",
    "agent_reset_mask",
}


def _make_env(*, num_envs=2, opinion=True, testing=False):
    loaded = load_opinion_experiment_config("config_opinion.json")
    parameters = loaded.parameters
    parameters.use_opinion_marl = opinion
    parameters.is_testing_mode = testing
    if not opinion:
        parameters.opinion_config = None
    scenario = ScenarioRoadTraffic()
    scenario.parameters = parameters
    env = VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=parameters.max_steps,
        device="cpu",
        n_agents=parameters.n_agents,
    )
    return env, scenario


def test_opinion_reset_info_has_fixed_stacked_shapes_and_global_ids():
    env, scenario = _make_env()
    try:
        td = env.reset()
        info = td["agents", "info"]

        assert OPINION_KEYS <= set(info.keys())
        assert info["pair_features"].shape == (2, 4, 3, 12)
        assert info["neighbor_ids"].shape == (2, 4, 3)
        assert info["pair_mask"].shape == (2, 4, 3)
        assert info["urgency"].shape == (2, 4, 3)
        assert info["confidence"].shape == (2, 4, 3)
        # TorchRL adds a final metric dimension to each scalar per-agent info.
        assert info["agent_reset_mask"].shape == (2, 4, 1)
        assert info["agent_reset_mask"].bool().all()

        ids = info["neighbor_ids"].long()
        mask = info["pair_mask"].bool()
        ego_ids = torch.arange(4).view(1, 4, 1).expand_as(ids)
        assert ((ids >= 0) & (ids < 4))[mask].all()
        assert (ids[mask] != ego_ids[mask]).all()
        assert scenario.conflict_graph.feature_dim == 12
    finally:
        env.close()


def test_conflict_info_is_finite_and_stable_across_repeated_current_state_reads():
    env, scenario = _make_env(num_envs=1)
    try:
        env.reset()
        first = scenario._build_opinion_conflict_graph()
        second = scenario._build_opinion_conflict_graph()

        for name in ("pair_features", "urgency", "confidence"):
            first_value = getattr(first, name)
            second_value = getattr(second, name)
            assert torch.isfinite(first_value).all()
            assert torch.equal(first_value, second_value)
        assert torch.equal(first.neighbor_ids, second.neighbor_ids)
        assert torch.equal(first.pair_mask, second.pair_mask)
    finally:
        env.close()


def test_conflict_info_does_not_read_short_term_future_reference_points():
    env, scenario = _make_env(num_envs=1)
    try:
        env.reset()
        before = scenario._build_opinion_conflict_graph()
        scenario.ref_paths_agent_related.short_term.fill_(123456.0)
        after = scenario._build_opinion_conflict_graph()

        assert torch.equal(before.pair_features, after.pair_features)
        assert torch.equal(before.neighbor_ids, after.neighbor_ids)
        assert torch.equal(before.urgency, after.urgency)
    finally:
        env.close()


def test_conflict_graph_is_independent_of_tsc_neighbor_selection_flags():
    env, scenario = _make_env(num_envs=1)
    try:
        env.reset()
        before = scenario._build_opinion_conflict_graph()
        scenario.parameters.use_topology_neighbor_selection = True
        scenario.parameters.n_topology_nearing_agents_observed = 1
        after = scenario._build_opinion_conflict_graph()

        assert torch.equal(before.pair_features, after.pair_features)
        assert torch.equal(before.neighbor_ids, after.neighbor_ids)
        assert torch.equal(before.pair_mask, after.pair_mask)
        assert torch.equal(before.urgency, after.urgency)
    finally:
        env.close()


def test_direct_single_agent_reset_marks_only_that_global_agent_once():
    env, scenario = _make_env(num_envs=2)
    try:
        env.reset()
        # reset() consumed the initial all-agent reset event through info().
        assert not scenario._opinion_agent_reset_mask.any()

        scenario.reset_world_at(env_index=1, agent_index=torch.tensor(2))
        raw_info = scenario.info(scenario.world.agents[2])

        assert raw_info["agent_reset_mask"].tolist() == [False, True]
        assert not scenario._opinion_agent_reset_mask[:, 2].any()
        for other_agent in (0, 1, 3):
            assert not scenario._opinion_agent_reset_mask[:, other_agent].any()
    finally:
        env.close()


def test_automatic_partial_reset_is_announced_before_done_without_duplicate_event():
    env, scenario = _make_env(num_envs=1)
    try:
        env.reset()
        scenario.timer.step[0] = 1
        scenario.collisions.with_exit_segments[0, 2] = True

        raw_info = scenario.info(scenario.world.agents[2])
        assert raw_info["agent_reset_mask"].item() is True

        done = scenario.done()
        assert not done.item()
        assert not scenario._opinion_agent_reset_mask.any()
        assert not scenario._opinion_preannounced_reset_mask.any()
    finally:
        env.close()


def test_testing_mode_collision_reset_is_reported_as_partial_not_environment_done():
    env, scenario = _make_env(num_envs=1, testing=True)
    try:
        env.reset()
        scenario.timer.step[0] = 1
        scenario.collisions.with_agents[0, 2, 0] = True

        raw_info = scenario.info(scenario.world.agents[2])

        assert raw_info["agent_reset_mask"].item() is True
    finally:
        env.close()


def test_base_scenario_does_not_construct_or_emit_opinion_components():
    env, scenario = _make_env(num_envs=1, opinion=False)
    try:
        td = env.reset()

        assert not hasattr(scenario, "conflict_graph")
        assert OPINION_KEYS.isdisjoint(td["agents", "info"].keys())
    finally:
        env.close()
