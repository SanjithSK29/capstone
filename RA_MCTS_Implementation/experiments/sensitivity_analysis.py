"""Evaluate RA-MCTS over a grid of budgets and exploration constants."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.environment import SyntheticMultiAgentEnv
from src.ra_mcts import RAMCTS
from src.utils import write_svg_bar_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="10,30,100")
    parser.add_argument("--exploration", default="0.5,1.414,2.0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/tables/sensitivity.csv"))
    args = parser.parse_args()
    rows = []
    for budget in (int(value) for value in args.budgets.split(",")):
        for exploration in (float(value) for value in args.exploration.split(",")):
            env = SyntheticMultiAgentEnv(seed=args.seed)
            search = RAMCTS(env, budget=budget, seed=args.seed, exploration=exploration)
            _, causes = search.search()
            rows.append({"budget": budget, "exploration": exploration,
                         "responsibility": max((c.responsibility for c in causes), default=0.0),
                         "simulations": search.simulations, "pruned": search.pruned_nodes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    plot = args.output.parents[1] / "plots" / "sensitivity_responsibility.svg"
    write_svg_bar_plot(
        plot,
        [f"{row['budget']}/{row['exploration']}" for row in rows],
        [row["responsibility"] for row in rows],
        "RA-MCTS sensitivity",
    )
    print(f"wrote {len(rows)} sensitivity rows to {args.output}")


if __name__ == "__main__":
    main()
