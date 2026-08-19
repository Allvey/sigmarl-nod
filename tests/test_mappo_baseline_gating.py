from types import SimpleNamespace

from utilities.mappo_cavs import _build_topology_manager, uses_tsc_components


def _parameters(**overrides):
    values = {
        "is_using_opponent_modeling": False,
        "use_topology_neighbor_selection": False,
        "topology_loss_weight": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_base_switches_do_not_use_tsc_components():
    assert uses_tsc_components(_parameters()) is False


def test_each_tsc_switch_activates_tsc_components():
    assert uses_tsc_components(_parameters(is_using_opponent_modeling=True)) is True
    assert uses_tsc_components(_parameters(use_topology_neighbor_selection=True)) is True
    assert uses_tsc_components(_parameters(topology_loss_weight=0.5)) is True


def test_base_does_not_construct_topology_manager():
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return object()

    result = _build_topology_manager(_parameters(), scenario=object(), factory=factory)

    assert result is None
    assert constructed == []


def test_tsc_constructs_topology_manager_once():
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return "manager"

    parameters = _parameters(topology_loss_weight=0.5)
    scenario = object()
    result = _build_topology_manager(parameters, scenario=scenario, factory=factory)

    assert result == "manager"
    assert constructed == [{"parameters": parameters, "scenario": scenario}]
