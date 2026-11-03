"""Run the random-search baseline on the synthetic benchmark."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.environment import SyntheticMultiAgentEnv
from src.responsibility import minimal_causes
from src.scm import SCM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/tables/random_search.csv"))
    args = parser.parse_args()
    env = SyntheticMultiAgentEnv(seed=args.seed)
    rng = random.Random(args.seed)
    scm = SCM(env)
    rows = []
    for iteration in range(args.budget):
        trace = [rng.choice(env.action_space()) for _ in range(env.horizon)]
        causes = minimal_causes(scm, trace)
        rows.append({"iteration": iteration, "failure": scm.factual(trace),
                     "responsibility": max((c.responsibility for c in causes), default=0.0)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} random-search simulations to {args.output}")


if __name__ == "__main__":
    main()
