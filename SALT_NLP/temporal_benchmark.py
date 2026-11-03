"""Validate stage-aware attribution on known temporal mechanisms."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from temporal_attribution import classify_profile, temporal_profile

CAUSES = ("M", "F")
SAMPLE_SIZES = (16, 32, 64)
REPETITIONS = 100
RESULTS = Path(__file__).parent / "results"

SCENARIOS = {
    "initiator_only": {"M": (0.90, 0.90), "F": (0.05, 0.90)},
    "maintainer_only": {"M": (0.90, 0.05), "F": (0.90, 0.90)},
    "both_stages": {"M": (0.05, 0.05), "F": (0.05, 0.05)},
    "inactive": {"M": (0.90, 0.90), "F": (0.90, 0.90)},
}
EXPECTED = {
    "initiator_only": {"M": "inactive", "F": "initiator_only"},
    "maintainer_only": {"M": "maintainer_only", "F": "inactive"},
    "both_stages": {"M": "initiator_and_maintainer", "F": "initiator_and_maintainer"},
    "inactive": {"M": "inactive", "F": "inactive"},
}


def _rate_table(scenario: str, n: int, rng: random.Random) -> dict:
    values = {}
    for stage_index, stage in enumerate(("onset", "persistence")):
        interventions = {}
        for cause in CAUSES:
            effect = SCENARIOS[scenario][cause][stage_index]
            interventions[cause] = sum(rng.random() < effect for _ in range(n)) / n
        values[stage] = {"factual": 0.90, "interventions": interventions}
    return values


def run() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for n in SAMPLE_SIZES:
            correct = 0
            for repetition in range(REPETITIONS):
                stages = _rate_table(scenario, n, random.Random(8011 + repetition))
                profile = temporal_profile(stages, CAUSES)
                labels = classify_profile(profile, effect_threshold=0.10)
                correct += labels == EXPECTED[scenario]
            rows.append({
                "scenario": scenario,
                "sample_size": n,
                "repetitions": REPETITIONS,
                "profile_exact_rate": correct / REPETITIONS,
            })
    return rows


def main() -> None:
    rows = run()
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "temporal_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} temporal benchmark rows to {RESULTS}")


if __name__ == "__main__":
    main()
