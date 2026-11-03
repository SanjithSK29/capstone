"""Reproducibility and small serialization helpers."""
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

def write_rows(path: Path, rows: Iterable[Mapping]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

def write_svg_bar_plot(path: Path, labels: Iterable[str], values: Iterable[float],
                       title: str) -> None:
    """Write a dependency-free SVG bar chart from measured experiment values."""
    labels, values = list(labels), list(values)
    if not values:
        return
    width, height, margin = 720, 420, 60
    maximum = max(max(values), 1.0)
    bar_width = (width - 2 * margin) / len(values)
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = (height - 2 * margin) * value / maximum
        x = margin + index * bar_width + bar_width * 0.1
        y = height - margin - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * .8:.1f}" '
            f'height="{bar_height:.1f}" fill="#3568a8"/>'
            f'<text x="{x + bar_width * .4:.1f}" y="{height - margin + 18}" '
            f'text-anchor="middle" font-size="12">{label}</text>'
        )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="16">{title}</text>'
           f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
           f'y2="{height - margin}" stroke="black"/>{"".join(bars)}</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
