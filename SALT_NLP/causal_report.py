"""Generate model-based causal pathway and intervention tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from causal_estimands import direct_effect, mediated_state_effect, minimal_interventions, total_effect
from scm import StructuralParameters, TemporalSCM

RESULTS = Path(__file__).parent / "results"
INPUTS = {"M_t": 1, "F_t": 1, "R_t": 1, "C_prev": 0}
CAUSES = {"M_t": "M", "F_t": "F", "R_t": "R"}


def main() -> None:
    scm = TemporalSCM(StructuralParameters(message_accept_effect=0.0, reflection_accept_effect=0.0))
    rows = []
    for node, label in CAUSES.items():
        total = total_effect(scm, INPUTS, node)
        direct = direct_effect(scm, INPUTS, node)
        mediated = mediated_state_effect(scm, INPUTS, node)
        rows.append({
            "cause": label,
            "total_effect": total,
            "direct_effect": direct,
            "mediated_state_effect": mediated,
            "decomposition_error": total - (direct + mediated),
        })
    report = {
        "model": "declared temporal SALT SCM",
        "inputs": INPUTS,
        "pathway_rows": rows,
        "minimal_onset_prevention": minimal_interventions(scm, INPUTS, ("M_t", "F_t", "R_t"), 0),
        "minimal_state_creation": minimal_interventions(
            scm, {**INPUTS, "F_t": 0, "C_prev": 0}, ("M_t", "F_t", "R_t"), 1
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "causal_pathway_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (RESULTS / "causal_pathway_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote causal pathway report to", RESULTS)


if __name__ == "__main__":
    main()
