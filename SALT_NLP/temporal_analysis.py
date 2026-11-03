"""Analyze Sanjith's reported SALT-NLP pilot without changing its values."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from temporal_attribution import classify_profile, profile_rows, temporal_profile

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

PILOT_STAGES = {
    "onset": {
        "factual": 1.00,
        "interventions": {"M": 1.00, "F": 0.31, "R": 1.00},
    },
    "persistence": {
        "factual": 1.00,
        "interventions": {"M": 1.00, "F": 1.00, "R": 1.00},
    },
}


def main() -> None:
    profile = temporal_profile(PILOT_STAGES, ("M", "F", "R"))
    labels = classify_profile(profile, effect_threshold=0.10)
    rows = profile_rows(profile, labels)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sanjith_temporal_profile.json").write_text(
        json.dumps({"source": "results_causal.md", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (RESULTS / "sanjith_temporal_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_plot(RESULTS / "sanjith_temporal_profile.svg", rows)
    print("Wrote stage-aware pilot profile to", RESULTS)
    for row in rows:
        print(f'{row["cause"]}: {row["classification"]}, onset={row["onset_effect"]:.2f}, persistence={row["persistence_effect"]:.2f}')


def _write_plot(path: Path, rows: list[dict]) -> None:
    width, height = 700, 420
    left, top, chart_w, chart_h = 70, 45, 560, 285
    series = (("onset_effect", "#4e79a7"), ("persistence_effect", "#e15759"))
    group_w = chart_w / len(rows)
    bar_w = group_w / 3
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="350" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">Pilot temporal attribution profile</text>')
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.0%}</text>')
    for index, row in enumerate(rows):
        x0 = left + index * group_w
        parts.append(f'<text x="{x0 + group_w / 2:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["cause"]}</text>')
        for series_index, (name, color) in enumerate(series):
            value = row[name]
            bar_h = chart_h * value
            x = x0 + (series_index + 0.6) * bar_w
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.75:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>')
    for index, (name, color) in enumerate(series):
        x = left + index * 190
        parts.append(f'<rect x="{x}" y="370" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="381" font-family="sans-serif" font-size="11">{name.replace("_", " ")}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
