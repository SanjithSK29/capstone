"""Causal estimands and intervention planning for the temporal SALT SCM.

These are model-based quantities. They are not additional measurements from
the DeepSeek pilot. The purpose is to make the causal claim precise:
total effect, mediated effect through the persistent state, and the smallest
intervention that prevents or reverses that state.
"""
from __future__ import annotations

from itertools import combinations
from random import Random
from typing import Iterable

from scm import Exogenous, TemporalSCM


def estimate_acceptance(
    scm: TemporalSCM,
    inputs: dict[str, int],
    interventions: dict[str, int] | None = None,
    samples: int = 20_000,
    seed: int = 0,
) -> float:
    """Estimate P(A_t=1) under a fixed context and intervention."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = Random(seed)
    accepts = 0
    for _ in range(samples):
        result = scm.evaluate(
            **inputs,
            exogenous=Exogenous(rng.random(), rng.random()),
            intervention=interventions,
        )
        accepts += result["A_t"]
    return accepts / samples


def total_effect(
    scm: TemporalSCM,
    inputs: dict[str, int],
    cause: str,
    samples: int = 20_000,
    seed: int = 0,
) -> float:
    """P(accept | cause=1) - P(accept | cause=0), all pathways included."""
    one = estimate_acceptance(scm, inputs, {cause: 1}, samples, seed)
    zero = estimate_acceptance(scm, inputs, {cause: 0}, samples, seed)
    return one - zero


def mediated_state_effect(
    scm: TemporalSCM,
    inputs: dict[str, int],
    cause: str,
    samples: int = 20_000,
    seed: int = 0,
) -> float:
    """Effect transmitted through C_t, approximated by a state intervention.

    The cause is switched while the persistent state is held at the value
    produced under the cause=1 condition. This isolates the pathway through
    the state from the direct context-to-verdict pathway.
    """
    rng = Random(seed)
    effect = 0.0
    for _ in range(samples):
        exogenous = Exogenous(rng.random(), rng.random())
        factual = scm.evaluate(**inputs, exogenous=exogenous, intervention={cause: 1})
        cause_zero = scm.evaluate(**inputs, exogenous=exogenous, intervention={cause: 0})
        with_state = scm.evaluate(
            **inputs,
            exogenous=exogenous,
            intervention={cause: 0, "C_t": factual["C_t"]},
        )
        without_state = scm.evaluate(
            **inputs,
            exogenous=exogenous,
            intervention={cause: 0, "C_t": cause_zero["C_t"]},
        )
        effect += with_state["A_t"] - without_state["A_t"]
    return effect / samples


def direct_effect(
    scm: TemporalSCM,
    inputs: dict[str, int],
    cause: str,
    samples: int = 20_000,
    seed: int = 0,
) -> float:
    """Effect of a cause with the persistent state blocked at C_t=0."""
    one = estimate_acceptance(scm, inputs, {cause: 1, "C_t": 0}, samples, seed)
    zero = estimate_acceptance(scm, inputs, {cause: 0, "C_t": 0}, samples, seed)
    return one - zero


def minimal_interventions(
    scm: TemporalSCM,
    inputs: dict[str, int],
    candidates: Iterable[str],
    target_state: int,
    samples: int = 20_000,
    seed: int = 0,
) -> list[tuple[str, ...]]:
    """Find smallest context interventions that force the target state.

    This is a planning question: what is the least context change that
    prevents onset (target 0) or preserves/recreates collusion (target 1)?
    """
    candidates = tuple(candidates)
    answers = []
    for size in range(len(candidates) + 1):
        for names in combinations(candidates, size):
            intervention = {name: 0 for name in names} if target_state == 0 else {name: 1 for name in names}
            result = scm.evaluate(
                **inputs,
                exogenous=Exogenous(),
                intervention=intervention,
            )
            if result["C_t"] == target_state:
                answers.append(names)
        if answers:
            return answers
    return answers
