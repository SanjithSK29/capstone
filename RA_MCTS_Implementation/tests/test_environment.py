from src.environment import SyntheticMultiAgentEnv

def test_safe_trace_succeeds():
    env = SyntheticMultiAgentEnv()
    assert env.rollout(env.default_trace())[1] == 0

def test_joint_hazards_fail():
    env = SyntheticMultiAgentEnv()
    trace = [(1, 1, 0), (0, 0, 0), (0, 0, 0)]
    assert env.rollout(trace)[1] == 1
