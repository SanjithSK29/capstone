from src.environment import SyntheticMultiAgentEnv
from src.ra_mcts import RAMCTS

def test_search_is_reproducible_and_bounded():
    a = RAMCTS(SyntheticMultiAgentEnv(), budget=10, seed=3).search()
    b = RAMCTS(SyntheticMultiAgentEnv(), budget=10, seed=3).search()
    assert a == b
    assert len(a[0]) == 3
