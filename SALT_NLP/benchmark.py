"""Offline benchmark for the SALT-NLP attribution methods.

The benchmark uses known Bernoulli mechanisms, so the expected cause is known
before the attribution methods run. It does not call a model or use an API.
Results are written below ``results/`` next to this file.

The separate ``REAL_PILOT`` table records the values reported in the local
``results_causal.md`` file. It is intentionally not combined with synthetic
scores.
"""
from __future__ import annotations

import csv
import json
import random
from itertools import combinations
from pathlib import Path

from attribution import _subsets, all_methods

CAUSES = ("M", "F", "R")
SAMPLE_SIZES = (8, 16, 32, 64)
REPETITIONS = 100
HI, LO = 0.5, 0.25

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def _mechanism(name: str, present: set[str]) -> float:
    if name == "single_feedback":
        return 0.90 if "F" in present else 0.05
    if name == "or_peer_feedback":
        return 0.90 if present & {"M", "F"} else 0.05
    if name == "and_peer_feedback":
        return 0.90 if {"M", "F"} <= present else 0.05
    if name == "partial_feedback":
        return 0.90 if "F" in present else 0.31
    if name == "three_way_or":
        return 0.90 if present else 0.05
    raise ValueError(f"unknown scenario: {name}")


SCENARIOS = {
    "single_feedback": {"truth": {"F"}, "description": "one cause"},
    "or_peer_feedback": {"truth": {"M", "F"}, "description": "redundant OR causes"},
    "and_peer_feedback": {"truth": {"M", "F"}, "description": "joint AND causes"},
    "partial_feedback": {"truth": {"F"}, "description": "partial effect, like the pilot"},
    "three_way_or": {"truth": {"M", "F", "R"}, "description": "three redundant causes"},
}

# Acceptance probabilities from Sanjith's local results_causal.md.
REAL_PILOT = [
    {"episode": "ep1", "factual": 1.00, "minus_M": 1.00, "minus_F": 0.31, "minus_R": 1.00, "minus_all": 0.25},
    {"episode": "ep2", "factual": 1.00, "minus_M": 1.00, "minus_F": 1.00, "minus_R": 1.00, "minus_all": 0.75},
    {"episode": "ep3", "factual": 1.00, "minus_M": 1.00, "minus_F": 1.00, "minus_R": 1.00, "minus_all": 1.00},
    {"episode": "ep4", "factual": 1.00, "minus_M": 1.00, "minus_F": 1.00, "minus_R": 1.00, "minus_all": 1.00},
]


def _sample_table(scenario: str, n: int, rng: random.Random) -> dict:
    table = {}
    for removed in _subsets(CAUSES):
        probability = _mechanism(scenario, set(CAUSES) - set(removed))
        successes = sum(rng.random() < probability for _ in range(n))
        table[removed] = (successes, n)
    return table


def _top_causes(scores: dict[str, float]) -> set[str]:
    maximum = max(scores.values())
    return {cause for cause, score in scores.items() if score == maximum and score > 0}


def _score_result(result: dict, truth: set[str]) -> tuple[bool, float]:
    scores = result["responsibility"]
    predicted = _top_causes(scores)
    exact = predicted == truth
    error = sum(abs(scores[cause] - (1.0 if cause in truth else 0.0)) for cause in CAUSES) / len(CAUSES)
    return exact, error


def run_benchmark() -> tuple[list[dict], list[dict]]:
    rows = []
    for scenario, metadata in SCENARIOS.items():
        truth = metadata["truth"]
        for n in SAMPLE_SIZES:
            exact_count = 0
            errors = []
            method_hits = {method: 0 for method in ("but_for", "ablation", "shapley", "hp", "responsibility")}
            for repetition in range(REPETITIONS):
                table = _sample_table(scenario, n, random.Random(20261103 + repetition))
                result = all_methods(table, CAUSES, HI, LO)
                exact, error = _score_result(result, truth)
                exact_count += exact
                errors.append(error)
                for method in method_hits:
                    scores = result[method]
                    if method == "but_for":
                        predicted = {cause for cause, score in scores.items() if score == 1.0}
                    elif method == "hp":
                        predicted = {cause for cause, details in scores.items() if details[0]}
                    else:
                        predicted = _top_causes(scores)
                    method_hits[method] += predicted == truth
            row = {
                "scenario": scenario,
                "description": metadata["description"],
                "sample_size": n,
                "repetitions": REPETITIONS,
                "responsibility_exact_rate": exact_count / REPETITIONS,
                "responsibility_mean_absolute_error": sum(errors) / len(errors),
            }
            row.update({f"{method}_exact_rate": count / REPETITIONS for method, count in method_hits.items()})
            rows.append(row)
    return rows, REAL_PILOT


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict]) -> None:
    width, height = 900, 520
    left, top, chart_w, chart_h = 80, 45, 770, 380
    methods = ("but_for", "ablation", "shapley", "hp", "responsibility")
    colors = ("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1")
    n = 64
    selected = [row for row in rows if row["sample_size"] == n]
    groups = len(selected)
    group_w = chart_w / groups
    bar_w = group_w / (len(methods) + 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="450" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Method accuracy at 64 samples</text>')
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.0%}</text>')
    for index, row in enumerate(selected):
        x0 = left + index * group_w
        label = row["scenario"].replace("_", " ")
        parts.append(f'<text x="{x0 + group_w / 2:.1f}" y="{top + chart_h + 25}" text-anchor="middle" font-family="sans-serif" font-size="10">{label}</text>')
        for method_index, (method, color) in enumerate(zip(methods, colors)):
            value = row[f"{method}_exact_rate"]
            bar_h = chart_h * value
            x = x0 + (method_index + 0.5) * bar_w
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.75:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>')
    for index, (method, color) in enumerate(zip(methods, colors)):
        x = left + index * 145
        parts.append(f'<rect x="{x}" y="475" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="486" font-family="sans-serif" font-size="11">{method}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_pilot_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pilot_plot(path: Path, rows: list[dict]) -> None:
    width, height = 760, 430
    left, top, chart_w, chart_h = 70, 40, 630, 300
    series = (("factual", "#4e79a7"), ("minus_M", "#f28e2b"), ("minus_F", "#e15759"),
              ("minus_R", "#59a14f"), ("minus_all", "#b07aa1"))
    group_w = chart_w / len(rows)
    bar_w = group_w / (len(series) + 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">Sanjith pilot: acceptance rates (copied report values)</text>')
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.0%}</text>')
    for index, row in enumerate(rows):
        x0 = left + index * group_w
        parts.append(f'<text x="{x0 + group_w / 2:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{row["episode"]}</text>')
        for series_index, (name, color) in enumerate(series):
            value = row[name]
            bar_h = chart_h * value
            x = x0 + (series_index + 0.5) * bar_w
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.75:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>')
    for index, (name, color) in enumerate(series):
        x = left + index * 145
        parts.append(f'<rect x="{x}" y="385" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 17}" y="396" font-family="sans-serif" font-size="11">-{name[5:] if name.startswith("minus_") else name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows, pilot = run_benchmark()
    _write_csv(RESULTS / "synthetic_summary.csv", rows)
    (RESULTS / "synthetic_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_pilot_csv(RESULTS / "sanjith_pilot_values.csv", pilot)
    (RESULTS / "sanjith_pilot_values.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    _write_plot(RESULTS / "method_accuracy_64_samples.svg", rows)
    _write_pilot_plot(RESULTS / "sanjith_pilot_acceptance.svg", pilot)
    print(f"Wrote {len(rows)} synthetic benchmark rows and {len(pilot)} pilot rows to {RESULTS}")
    print("Pilot values are copied unchanged; they are not synthetic benchmark scores.")


if __name__ == "__main__":
    main()
