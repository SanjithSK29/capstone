"""Responsibility-aware MCTS with causal pruning and minimality-aware backup."""
import random
from typing import List, Tuple
from .environment import SyntheticMultiAgentEnv, JointAction
from .responsibility import minimal_causes
from .scm import SCM
from .search_tree import Node

class RAMCTS:
    def __init__(self, env: SyntheticMultiAgentEnv, budget: int = 100, seed: int = 0,
                 exploration: float = 1.414, scalar_weight: float = 0.5):
        if budget < 1:
            raise ValueError("budget must be positive")
        self.env, self.budget, self.rng = env, budget, random.Random(seed)
        self.exploration, self.scalar_weight = exploration, scalar_weight
        self.scm, self.root = SCM(env), Node()
        self.actions = env.action_space()
        self.simulations = 0
        self.pruned_nodes = 0

    def _complete(self, prefix: Tuple[JointAction, ...]) -> List[JointAction]:
        return list(prefix) + [self.rng.choice(self.actions) for _ in range(self.env.horizon - len(prefix))]

    def search(self) -> Tuple[List[JointAction], List]:
        for _ in range(self.budget):
            node = self.root
            while len(node.prefix) < self.env.horizon and node.children:
                node = node.best_child(self.exploration, self.scalar_weight)
            if len(node.prefix) < self.env.horizon:
                available = [a for a in self.actions if a not in node.children]
                action = self.rng.choice(available or self.actions)
                child = Node(node.prefix + (action,), node)
                node.children[action] = child
                node = child
            trace = self._complete(node.prefix)
            failure = self.scm.factual(trace)
            causes = minimal_causes(self.scm, trace)
            reward = -float(failure) + max((c.responsibility for c in causes), default=0.0)
            self.simulations += 1
            cur = node
            while cur:
                cur.update(reward, max((c.responsibility for c in causes), default=0.0))
                cur = cur.parent
            # A fully evaluated leaf cannot produce a new responsibility value.
            if node.depth == self.env.horizon and node.children == {}:
                node.pruned = True
                self.pruned_nodes += 1
        best = self.root.best_child(self.exploration, self.scalar_weight) if self.root.children else self.root
        trace = self._complete(best.prefix)
        return trace, minimal_causes(self.scm, trace)
