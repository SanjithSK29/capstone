"""Minimal finite-horizon Dec-POMDP representation used by the experiments."""
from dataclasses import dataclass
from typing import Callable, List, Tuple
from .environment import SyntheticMultiAgentEnv, State, JointAction

@dataclass
class DecPOMDP:
    env: SyntheticMultiAgentEnv
    reward: Callable[[int], float] = lambda failure: -float(failure)

    def step(self, state: State, action: JointAction) -> Tuple[State, float, Tuple[int, ...]]:
        nxt = self.env.transition(state, action)
        return nxt, self.reward(self.env.outcome(nxt)), self.env.observe(action)

    def evaluate(self, trace: List[JointAction]) -> float:
        _, failure = self.env.rollout(trace)
        return self.reward(failure)
