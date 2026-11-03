"""Same-context interventions and brute-force cause candidates."""
from itertools import combinations
from typing import Dict, Iterable, List, Tuple
from .environment import JointAction
from .scm import SCM

Variable = Tuple[int, int]  # (time, agent)

def intervention(trace: List[JointAction], variables: Iterable[Variable], value: int = 0) -> Dict[Variable, int]:
    return {v: value for v in variables}

def candidate_sets(trace: List[JointAction], max_size: int = 3) -> List[Tuple[Variable, ...]]:
    variables = [(t, a) for t, action in enumerate(trace) for a, value in enumerate(action) if value == 1]
    return [combo for size in range(1, max_size + 1) for combo in combinations(variables, size)]

def evaluate_intervention(scm: SCM, trace: List[JointAction], variables: Iterable[Variable]) -> int:
    return scm.intervene(trace, intervention(trace, variables))
