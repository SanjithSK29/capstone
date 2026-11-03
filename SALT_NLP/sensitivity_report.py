"""Sensitivity analysis for the declared temporal SCM."""
from __future__ import annotations

import csv
from pathlib import Path

from causal_estimands import direct_effect, mediated_state_effect, total_effect
from scm import StructuralParameters, TemporalSCM

RESULTS = Path(__file__).parent / "results"
INPUTS = {"M_t": 1, "F_t": 1, "R_t": 1, "C_prev": 0}


def main() -> None:
    rows = []
    for learning in (0.5, 0.75, 1.0):
        for baseline in (0.10, 0.31, 0.50):
            scm = TemporalSCM(StructuralParameters(
                learning_probability=learning,
                baseline_accept_probability=baseline,
            ))
            total = total_effect(scm, INPUTS, "F_t", samples=5000, seed=int(learning * 100))
            mediated = mediated_state_effect(scm, INPUTS, "F_t", samples=5000, seed=17)
            direct = direct_effect(scm, INPUTS, "F_t", samples=5000, seed=17)
            rows.append({
                "learning_probability": learning,
                "baseline_accept_probability": baseline,
                "feedback_total_effect": total,
                "feedback_mediated_effect": mediated,
                "feedback_direct_effect": direct,
            })
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "scm_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} SCM sensitivity rows to {RESULTS}")


if __name__ == "__main__":
    main()
