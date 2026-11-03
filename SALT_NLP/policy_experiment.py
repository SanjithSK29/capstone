"""Causal intervention policy experiment for sequential collusion.

This experiment compares prevention and recovery policies under the declared
temporal SCM. It asks a question static blame scores do not answer:

    Which smallest intervention reliably prevents onset or restores a safe
    state after persistence has begun?

The results are model-based stress tests, not additional measurements from the
DeepSeek pilot.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from scm import Exogenous, StructuralParameters, TemporalSCM

RESULTS = Path(__file__).parent / "results"


@dataclass(frozen=True)
class Policy:
    name: str
    intervention: dict[str, int]
    cost: int


POLICIES = (
    Policy("remove_feedback", {"F_t": 0}, 1),
    Policy("remove_peer_messages", {"M_t": 0}, 1),
    Policy("remove_reflections", {"R_t": 0}, 1),
    Policy("remove_all_context", {"M_t": 0, "F_t": 0, "R_t": 0}, 3),
    Policy("reset_persistent_state", {"C_prev": 0}, 1),
    Policy("reset_state_and_feedback", {"C_prev": 0, "F_t": 0}, 2),
)


def _evaluate_policy(
    scm: TemporalSCM,
    inputs: dict[str, int],
    policy: Policy,
) -> dict[str, int]:
    result = scm.evaluate(
        **inputs,
        # Use a fixed high verdict-noise draw so the policy table isolates
        # state prevention/recovery rather than one random verdict sample.
        exogenous=Exogenous(verdict=0.99),
        intervention=policy.intervention,
    )
    return {
        "state": result["C_t"],
        "accept": result["A_t"],
    }


def run_policy_experiment() -> list[dict[str, int | float | str]]:
    scenarios = {
        "onset": {"M_t": 1, "F_t": 1, "R_t": 1, "C_prev": 0},
        "persistence": {"M_t": 1, "F_t": 0, "R_t": 0, "C_prev": 1},
    }
    parameter_sets = {
        "pilot_facing": StructuralParameters(),
        "moderate_learning": StructuralParameters(
            learning_probability=0.75,
            baseline_accept_probability=0.31,
        ),
        "message_sensitive": StructuralParameters(
            message_accept_effect=0.40,
        ),
    }
    rows = []
    for parameter_name, parameters in parameter_sets.items():
        scm = TemporalSCM(parameters)
        for stage, inputs in scenarios.items():
            for policy in POLICIES:
                result = _evaluate_policy(scm, inputs, policy)
                target_state = 0
                rows.append({
                    "parameter_set": parameter_name,
                    "stage": stage,
                    "policy": policy.name,
                    "cost": policy.cost,
                    "state_after": result["state"],
                    "accept_after": result["accept"],
                    "success": int(result["state"] == target_state),
                })
    return rows


def _best(rows: list[dict]) -> dict[str, dict[str, int | str]]:
    answers = {}
    for stage in ("onset", "persistence"):
        stage_rows = [row for row in rows if row["stage"] == stage and row["success"]]
        for parameter_set in sorted({row["parameter_set"] for row in stage_rows}):
            candidates = [row for row in stage_rows if row["parameter_set"] == parameter_set]
            answers[f"{parameter_set}:{stage}"] = min(
                candidates,
                key=lambda row: (row["cost"], row["policy"]),
            )
    return answers


def _write_svg(path: Path, rows: list[dict]) -> None:
    selected = [row for row in rows if row["parameter_set"] == "pilot_facing"]
    width, height = 850, 430
    left, top, chart_w, chart_h = 70, 45, 710, 280
    colors = {"onset": "#4e79a7", "persistence": "#e15759"}
    group_w = chart_w / len(POLICIES)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="425" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">Intervention policy success under pilot-facing SCM</text>')
    for tick in range(4):
        y = top + chart_h * (1 - tick / 3)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick / 3:.0%}</text>')
    for index, policy in enumerate(POLICIES):
        x0 = left + index * group_w
        parts.append(f'<text x="{x0 + group_w / 2:.1f}" y="{top + chart_h + 23}" text-anchor="middle" font-family="sans-serif" font-size="9" transform="rotate(-25 {x0 + group_w / 2:.1f} {top + chart_h + 23})">{policy.name}</text>')
        for offset, stage in enumerate(("onset", "persistence")):
            row = next(item for item in selected if item["policy"] == policy.name and item["stage"] == stage)
            bar_h = chart_h * row["success"]
            x = x0 + (offset + 0.8) * (group_w / 3)
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{group_w / 4:.1f}" height="{bar_h:.1f}" fill="{colors[stage]}"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>')
    for index, stage in enumerate(("onset", "persistence")):
        x = 300 + index * 150
        parts.append(f'<rect x="{x}" y="385" width="12" height="12" fill="{colors[stage]}"/>')
        parts.append(f'<text x="{x + 17}" y="396" font-family="sans-serif" font-size="11">{stage}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = run_policy_experiment()
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "policy_experiment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"best_policies": _best(rows), "rows": rows}
    (RESULTS / "policy_experiment.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_svg(RESULTS / "policy_success.svg", rows)
    for key, row in _best(rows).items():
        print(f"{key}: {row['policy']} (cost={row['cost']})")


if __name__ == "__main__":
    main()
