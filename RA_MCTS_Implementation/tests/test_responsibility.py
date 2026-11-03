from src.environment import SyntheticMultiAgentEnv
from src.responsibility import minimal_causes
from src.scm import SCM

def test_joint_cause_is_minimal():
    env = SyntheticMultiAgentEnv()
    trace = [(1, 1, 0), (0, 0, 0), (0, 0, 0)]
    causes = minimal_causes(SCM(env), trace)
    assert any(len(c.variables) == 2 for c in causes)

def test_sequential_single_cause_is_witness():
    env = SyntheticMultiAgentEnv()
    trace = [(1, 0, 0), (0, 0, 0), (0, 0, 0)]
    causes = minimal_causes(SCM(env), trace)
    assert any(len(c.variables) == 1 and c.responsibility == 1.0 for c in causes)
