from src.environment import SyntheticMultiAgentEnv
from src.ra_mcts import RAMCTS


def test_budget_and_statistics_are_exposed():
    search = RAMCTS(SyntheticMultiAgentEnv(), budget=12, seed=4)
    trace, _ = search.search()
    assert len(trace) == 3
    assert search.simulations == 12
    assert search.pruned_nodes >= 1
