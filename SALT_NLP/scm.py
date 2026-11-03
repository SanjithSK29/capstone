"""A small explicit temporal SCM for SALT-NLP attribution.

This is a declared causal model, not a graph learned from the pilot data.
The graph represents the hypothesis that contextual channels affect a verdict
and that an earlier verdict can create a persistent collusion state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


NODES = (
    "M_t",
    "F_t",
    "R_t",
    "C_prev",
    "C_t",
    "A_t",
)

EDGES = (
    ("M_t", "A_t"),
    ("F_t", "A_t"),
    ("R_t", "A_t"),
    ("C_prev", "C_t"),
    ("F_t", "C_t"),
    ("C_t", "A_t"),
)


@dataclass(frozen=True)
class Exogenous:
    """Exogenous noise for one SCM evaluation."""

    learning: float = 0.0
    verdict: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.learning <= 1.0 or not 0.0 <= self.verdict <= 1.0:
            raise ValueError("exogenous values must be between 0 and 1")


@dataclass(frozen=True)
class StructuralParameters:
    """Probabilities defining the declared temporal mechanism."""

    learning_probability: float = 1.0
    baseline_accept_probability: float = 0.31
    persistent_accept_probability: float = 1.0
    message_accept_effect: float = 0.0
    reflection_accept_effect: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.learning_probability, self.baseline_accept_probability,
                      self.persistent_accept_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("structural probabilities must be between 0 and 1")


class TemporalSCM:
    """Evaluate the declared onset/persistence SCM under interventions.

    ``M_t``, ``F_t``, and ``R_t`` are binary context channels. ``C_t`` is the
    latent collusion state and ``A_t`` is Alice's binary accept outcome.
    """

    nodes = NODES
    edges = EDGES

    def __init__(self, parameters: StructuralParameters | None = None):
        self.parameters = parameters or StructuralParameters()

    def evaluate(
        self,
        *,
        M_t: int,
        F_t: int,
        R_t: int,
        C_prev: int,
        exogenous: Exogenous,
        intervention: Mapping[str, int] | None = None,
    ) -> dict[str, int]:
        values = {
            "M_t": M_t,
            "F_t": F_t,
            "R_t": R_t,
            "C_prev": C_prev,
        }
        intervention = intervention or {}
        for name, value in intervention.items():
            if name not in (*values, "C_t"):
                raise KeyError(f"unknown intervention node {name!r}")
            if value not in (0, 1):
                raise ValueError("binary interventions must be 0 or 1")
            if name in values:
                values[name] = value
        if any(value not in (0, 1) for value in values.values()):
            raise ValueError("SCM inputs must be binary")

        learned = (
            values["F_t"] == 1
            and exogenous.learning < self.parameters.learning_probability
        )
        values["C_t"] = int(values["C_prev"] == 1 or learned)
        if "C_t" in intervention:
            values["C_t"] = intervention["C_t"]
        accept_probability = (
            self.parameters.persistent_accept_probability
            if values["C_t"]
            else min(
                1.0,
                max(
                    0.0,
                    self.parameters.baseline_accept_probability
                    + values["M_t"] * self.parameters.message_accept_effect
                    + values["R_t"] * self.parameters.reflection_accept_effect,
                ),
            )
        )
        values["A_t"] = int(exogenous.verdict < accept_probability)
        return values


def graph_dot() -> str:
    """Return the SCM DAG in Graphviz DOT format."""
    lines = ["digraph TemporalSALT {", "  rankdir=LR;"]
    for node in NODES:
        shape = "ellipse" if node in {"C_t", "A_t"} else "box"
        lines.append(f'  "{node}" [shape={shape}];')
    for source, target in EDGES:
        lines.append(f'  "{source}" -> "{target}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def graph_mermaid() -> str:
    """Return a dependency graph that renders in Markdown."""
    lines = ["flowchart LR"]
    for source, target in EDGES:
        lines.append(f"    {source.replace('_', '')}[{source}] --> {target.replace('_', '')}[{target}]")
    return "\n".join(lines) + "\n"
