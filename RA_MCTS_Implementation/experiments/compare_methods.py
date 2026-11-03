"""Compare brute force, random search, and RA-MCTS on one fixed instance."""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.environment import SyntheticMultiAgentEnv
from src.ra_mcts import RAMCTS
from src.responsibility import minimal_causes
from src.scm import SCM
from src.utils import write_svg_bar_plot


def score(scm, trace):
    return max((c.responsibility for c in minimal_causes(scm, trace)), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/tables/comparison.csv"))
    args = parser.parse_args()
    env, scm = SyntheticMultiAgentEnv(seed=args.seed), None
    scm = SCM(env)
    methods = []
    started = time.perf_counter()
    brute_trace = max((list(t) for t in itertools.product(env.action_space(), repeat=env.horizon)),
                      key=lambda t: score(scm, t))
    methods.append(("bruteforce", score(scm, brute_trace), time.perf_counter() - started))
    rng = random.Random(args.seed)
    started = time.perf_counter()
    random_trace = max(([rng.choice(env.action_space()) for _ in range(env.horizon)]
                        for _ in range(args.budget)), key=lambda t: score(scm, t))
    methods.append(("random", score(scm, random_trace), time.perf_counter() - started))
    started = time.perf_counter()
    _, causes = RAMCTS(env, args.budget, args.seed).search()
    methods.append(("ra_mcts", max((c.responsibility for c in causes), default=0.0),
                    time.perf_counter() - started))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "responsibility", "runtime_seconds"])
        writer.writerows(methods)
    plot = args.output.parents[1] / "plots" / "method_responsibility.svg"
    write_svg_bar_plot(plot, [row[0] for row in methods],
                       [row[1] for row in methods], "Responsibility by method")
    print(f"wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
