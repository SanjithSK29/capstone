"""CLI comparing RA-MCTS with brute force and random search."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.environment import SyntheticMultiAgentEnv
from src.ra_mcts import RAMCTS
from src.responsibility import minimal_causes
from src.scm import SCM
from src.utils import write_json, write_rows

def brute(env):
    scm = SCM(env)
    traces = [[]]
    for _ in range(env.horizon):
        traces = [p + [a] for p in traces for a in env.action_space()]
    scored = [(len(minimal_causes(scm, t)), t) for t in traces]
    return max(scored, key=lambda x: x[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows, details = [], []
    for episode in range(args.episodes):
        env = SyntheticMultiAgentEnv(seed=args.seed + episode)
        trace, causes = RAMCTS(env, args.budget, args.seed + episode).search()
        _, brute_trace = brute(env)
        random_trace = [env.rng.choice(env.action_space()) for _ in range(env.horizon)]
        rows.append({"episode": episode, "mcts_causes": len(causes),
                     "brute_causes": len(minimal_causes(SCM(env), brute_trace)),
                     "random_failure": SCM(env).factual(random_trace)})
        details.append({"episode": episode, "trace": trace, "causes": [c.variables for c in causes]})
    root = Path(__file__).parents[1]
    write_rows(root / "results/tables/summary.csv", rows)
    write_json(root / "results/plots/summary.json", details)
    print(f"wrote {len(rows)} episodes to results/")

if __name__ == "__main__":
    main()
