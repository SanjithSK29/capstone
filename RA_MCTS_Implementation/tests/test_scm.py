from src.environment import SyntheticMultiAgentEnv
from src.scm import SCM


def test_factual_and_counterfactual_share_context():
    env = SyntheticMultiAgentEnv()
    scm = SCM(env)
    trace = [(1, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert scm.factual(trace) == 1
    assert scm.intervene(trace, {(0, 0): 0}) == 0
