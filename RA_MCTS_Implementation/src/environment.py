"""Deterministic synthetic multi-agent environment with stochastic observations."""
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

Action = int
JointAction = Tuple[Action, ...]

@dataclass(frozen=True)
class State:
    step: int
    progress: int
    failed: bool = False

class SyntheticMultiAgentEnv:
    """Three-agent, two-step task with single, redundant, joint and sequential causes."""
    def __init__(self, n_agents: int = 3, horizon: int = 3, seed: int = 0):
        if n_agents < 3:
            raise ValueError("at least three agents are required")
        self.n_agents, self.horizon, self.rng = n_agents, horizon, random.Random(seed)
        self.actions = (0, 1)  # 0=safe, 1=hazard
        self.initial_state = State(0, 0, False)

    def transition(self, state: State, action: JointAction) -> State:
        if len(action) != self.n_agents or any(a not in self.actions for a in action):
            raise ValueError("invalid joint action")
        if state.failed:
            return State(state.step + 1, state.progress, True)
        hazards = sum(action)
        progress = state.progress + (1 if hazards == 0 else 0)
        # A joint cause is two hazards; a sequential cause is hazard at step 0
        # followed by an otherwise safe action at step 1.
        joint = hazards >= 2
        sequential = state.step == 1 and state.progress == 0 and hazards == 0
        failed = joint or sequential
        return State(state.step + 1, progress, failed)

    def outcome(self, state: State) -> int:
        return int(state.failed or state.progress < self.horizon)

    def rollout(self, actions: List[JointAction]) -> Tuple[List[State], int]:
        state = self.initial_state
        states = [state]
        for action in actions[: self.horizon]:
            state = self.transition(state, action)
            states.append(state)
        return states, self.outcome(state)

    def observe(self, action: JointAction, noise: float = 0.1) -> Tuple[int, ...]:
        """Private noisy binary observations, useful as a Dec-POMDP observation."""
        return tuple(a if self.rng.random() >= noise else 1 - a for a in action)

    def default_trace(self) -> List[JointAction]:
        return [(0,) * self.n_agents for _ in range(self.horizon)]

    def action_space(self) -> List[JointAction]:
        return [tuple((mask >> i) & 1 for i in range(self.n_agents))
                for mask in range(2 ** self.n_agents)]
