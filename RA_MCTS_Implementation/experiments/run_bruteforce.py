"""Enumerate the finite benchmark and report the best responsibility."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.environment import SyntheticMultiAgentEnv
from src.responsibility import minimal_causes
from src.scm import SCM


def enumerate_traces(env: SyntheticMultiAgentEnv):
    return itertools.product(env.action_space(), repeat=env.horizon)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/tables/bruteforce.json"))
    args = parser.parse_args()
    env = SyntheticMultiAgentEnv(args.agents, args.horizon, args.seed)
    scm = SCM(env)
    best = None
    evaluated = 0
    for trace in enumerate_traces(env):
        causes = minimal_causes(scm, list(trace))
        score = max((c.responsibility for c in causes), default=0.0)
        candidate = (score, tuple(trace), causes)
        if best is None or candidate[0] > best[0]:
            best = candidate
        evaluated += 1
    assert best is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "evaluated_traces": evaluated, "best_score": best[0],
        "trace": best[1], "causes": [c.variables for c in best[2]],
    }, indent=2, default=list), encoding="utf-8")
    print(f"evaluated {evaluated} traces; best responsibility={best[0]:.3f}")


if __name__ == "__main__":
    main()
