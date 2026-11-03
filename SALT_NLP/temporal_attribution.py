"""Stage-aware attribution for sequential agent failures.

The ordinary attribution functions answer whether a cause changes one outcome
table. This module answers a different question: does a cause affect the first
failure transition, later persistence, or both?

Input tables contain observed failure/acceptance probabilities by stage and
the probability after removing one cause. They can come from real replay
results or from a controlled simulator. No model calls are made here.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping


@dataclass(frozen=True)
class StageEffect:
    stage: str
    factual: float
    intervened: float
    effect: float


def _validate_rate(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")


def stage_effects(
    stages: Mapping[str, Mapping[str, float]],
    cause: str,
) -> list[StageEffect]:
    """Return factual-minus-intervened effects for one cause by ordered stage."""
    effects = []
    for stage, values in stages.items():
        factual = values["factual"]
        interventions = values["interventions"]
        if cause not in interventions:
            raise KeyError(f"missing intervention for {cause!r} in stage {stage!r}")
        intervened = interventions[cause]
        _validate_rate(factual, f"{stage}.factual")
        _validate_rate(intervened, f"{stage}.interventions[{cause}]")
        effects.append(StageEffect(stage, factual, intervened, factual - intervened))
    if not effects:
        raise ValueError("at least one stage is required")
    return effects


def temporal_profile(
    stages: Mapping[str, Mapping[str, float]],
    causes: tuple[str, ...],
    onset_stages: tuple[str, ...] = ("onset",),
    persistence_stages: tuple[str, ...] = ("persistence",),
) -> dict[str, dict[str, float | list[dict[str, float | str]]]]:
    """Calculate onset, persistence, and overall effects for each cause.

    Effects are reductions in the failure/acceptance probability used by the
    input. A positive effect means removing the cause reduced that outcome.
    """
    available = set(stages)
    if not set(onset_stages) <= available or not set(persistence_stages) <= available:
        raise KeyError("onset or persistence stage is missing")
    output = {}
    for cause in causes:
        effects = stage_effects(stages, cause)
        by_stage = {effect.stage: effect.effect for effect in effects}
        onset = mean(by_stage[stage] for stage in onset_stages)
        persistence = mean(by_stage[stage] for stage in persistence_stages)
        output[cause] = {
            "onset_effect": onset,
            "persistence_effect": persistence,
            "temporal_change": onset - persistence,
            "overall_effect": mean(effect.effect for effect in effects),
            "stage_effects": [
                {"stage": effect.stage, "factual": effect.factual,
                 "intervened": effect.intervened, "effect": effect.effect}
                for effect in effects
            ],
        }
    return output


def classify_profile(
    profile: Mapping[str, Mapping[str, float]],
    effect_threshold: float = 0.10,
) -> dict[str, str]:
    """Classify a cause as initiator, maintainer, both, or inactive."""
    if effect_threshold < 0:
        raise ValueError("effect_threshold must be non-negative")
    labels = {}
    for cause, values in profile.items():
        onset = values["onset_effect"] >= effect_threshold
        persistence = values["persistence_effect"] >= effect_threshold
        labels[cause] = (
            "initiator_and_maintainer" if onset and persistence else
            "initiator_only" if onset else
            "maintainer_only" if persistence else
            "inactive"
        )
    return labels


def profile_rows(
    profile: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, str],
) -> list[dict[str, float | str]]:
    """Flatten a profile for CSV/report generation."""
    return [
        {
            "cause": cause,
            "onset_effect": values["onset_effect"],
            "persistence_effect": values["persistence_effect"],
            "temporal_change": values["temporal_change"],
            "overall_effect": values["overall_effect"],
            "classification": labels[cause],
        }
        for cause, values in profile.items()
    ]
