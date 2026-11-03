"""Tree data structure for responsibility-aware MCTS."""
from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Tuple

@dataclass
class Node:
    prefix: Tuple[Tuple[int, ...], ...] = ()
    parent: Optional["Node"] = None
    children: Dict[Tuple[int, ...], "Node"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    causal_score: float = 0.0
    pruned: bool = False

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = max(1, self.parent.visits if self.parent else self.visits)
        return self.value / self.visits + exploration * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, exploration: float = 1.414, scalar_weight: float = 0.5) -> "Node":
        return max(
            self.children.values(),
            key=lambda child: child.ucb1(exploration)
            + scalar_weight * child.causal_score,
        )

    def update(self, reward: float, causal_score: float = 0.0) -> None:
        self.visits += 1
        self.value += reward
        self.causal_score = max(self.causal_score, causal_score)

    @property
    def depth(self) -> int:
        return len(self.prefix)
