from src.environment import SyntheticMultiAgentEnv
from src.interventions import evaluate_intervention, intervention
from src.scm import SCM


def test_multiple_intervention_is_applied_at_requested_times():
    env = SyntheticMultiAgentEnv()
    scm = SCM(env)
    trace = [(1, 1, 0), (0, 0, 0), (0, 0, 0)]
    changes = intervention(trace, [(0, 0), (0, 1)])
    assert changes == {(0, 0): 0, (0, 1): 0}
    assert evaluate_intervention(scm, trace, [(0, 0), (0, 1)]) == 0
