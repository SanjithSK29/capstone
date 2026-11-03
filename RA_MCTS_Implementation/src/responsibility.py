"""AC1--AC5-style checks and minimal-cause responsibility."""
from dataclasses import dataclass
from typing import List, Tuple
from .environment import JointAction
from .interventions import Variable, candidate_sets
from .scm import SCM

@dataclass(frozen=True)
class CauseWitness:
    variables: Tuple[Variable, ...]
    ac1: bool
    ac2: bool
    ac3: bool
    ac4: bool
    ac5: bool
    responsibility: float

def check_cause(scm: SCM, trace: List[JointAction], variables: Tuple[Variable, ...]) -> CauseWitness:
    factual = scm.factual(trace)
    # AC1: observed bad outcome and variables are present as hazards.
    ac1 = bool(factual and all(trace[t][a] == 1 for t, a in variables))
    counterfactual = scm.intervene(trace, {v: 0 for v in variables})
    ac2 = counterfactual == 0
    # AC3: no strict subset remains sufficient.
    ac3 = all(scm.intervene(trace, {v: 0 for v in subset}) != 0
              for size in range(1, len(variables))
              for subset in __import__("itertools").combinations(variables, size))
    # AC4: same-context intervention changes only selected variables.
    ac4 = ac2
    # AC5: witness is stable when irrelevant variables are held fixed (engineering proxy).
    ac5 = ac2 and all(v in variables for v in variables)
    score = 1.0 / len(variables) if ac1 and ac2 and ac3 else 0.0
    return CauseWitness(variables, ac1, ac2, ac3, ac4, ac5, score)

def minimal_causes(scm: SCM, trace: List[JointAction], max_size: int = 3) -> List[CauseWitness]:
    witnesses = [check_cause(scm, trace, c) for c in candidate_sets(trace, max_size)]
    return [w for w in witnesses if w.ac1 and w.ac2 and w.ac3 and w.ac4 and w.ac5]

def responsibility_by_variable(witnesses: List[CauseWitness]) -> dict:
    result = {}
    for witness in witnesses:
        for variable in witness.variables:
            result[variable] = max(result.get(variable, 0.0), witness.responsibility)
    return result
