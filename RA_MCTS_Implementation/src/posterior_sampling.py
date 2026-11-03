"""Simple Beta posterior used for uncertainty-aware simulation."""
from dataclasses import dataclass
import random

@dataclass
class BernoulliPosterior:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, failure: bool) -> None:
        if failure:
            self.alpha += 1
        else:
            self.beta += 1
