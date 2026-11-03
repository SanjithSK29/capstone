"""Structural causal model facade around the deterministic environment."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .environment import SyntheticMultiAgentEnv, JointAction

@dataclass
class SCM:
    env: SyntheticMultiAgentEnv

    def factual(self, trace: List[JointAction]) -> int:
        return self.env.rollout(trace)[1]

    def intervene(self, trace: List[JointAction], changes: Dict[Tuple[int, int], int]) -> int:
        """do(action[t,agent]=value), preserving all non-intervened context."""
        edited = [list(a) for a in trace]
        for (step, agent), value in changes.items():
            if step < 0 or step >= len(edited) or agent < 0 or agent >= self.env.n_agents:
                raise IndexError("intervention outside trace")
            edited[step][agent] = value
        return self.env.rollout([tuple(a) for a in edited])[1]
