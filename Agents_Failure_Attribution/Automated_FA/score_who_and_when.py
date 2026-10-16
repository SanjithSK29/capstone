"""
Score Who&When attribution predictions against the human labels.

Usage:
    python score_who_and_when.py all_at_once_qwen-7b_alg_generated.txt "Who&When/Algorithm-Generated"

Needs only Python 3 (no extra packages). Runs in about a second.

Reads each "Prediction for N.json" block (agent + step) from the predictions file,
opens N.json in the dataset folder, and compares with the labelled culprit:
  - agent accuracy: predicted agent == labelled agent
  - step accuracy:  predicted step  == labelled step (also reported allowing for
                    0- vs 1-based counting, so an indexing mismatch is visible)
Writes scored_predictions.csv with one row per log.
"""
import csv
import json
import os
import re
import sys

AGENT_KEYS = ["mistake_agent", "mistake_agent_name"]   # label field names (checked in order)
STEP_KEYS = ["mistake_step", "mistake_step_index"]


def parse_predictions(path):
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"^Prediction for ", text, flags=re.M)[1:]
    preds = {}
    for b in blocks:
        fname = b.split(":", 1)[0].strip()
        agent = re.search(r"Agent Name:\s*(.+)", b)
        step = re.search(r"Step Number:\s*(\d+)", b)
        preds[fname] = {
            "agent": agent.group(1).strip().strip(",").strip() if agent else None,
            "step": int(step.group(1)) if step else None,
        }
    return preds


def load_label(folder, fname):
    with open(os.path.join(folder, fname), encoding="utf-8") as f:
        d = json.load(f)
    agent = next((d[k] for k in AGENT_KEYS if k in d), None)
    step = next((d[k] for k in STEP_KEYS if k in d), None)
    if agent is None or step is None:
        raise KeyError(f"{fname}: label fields not found; keys are {list(d.keys())}")
    return str(agent).strip(), int(step)


def norm(a):
    return re.sub(r"[^a-z0-9]", "", (a or "").lower())


def main(pred_path, data_dir):
    preds = parse_predictions(pred_path)
    rows, missing = [], []
    for fname, p in sorted(preds.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
        if not os.path.exists(os.path.join(data_dir, fname)):
            missing.append(fname)
            continue
        true_agent, true_step = load_label(data_dir, fname)
        agent_ok = norm(p["agent"]) == norm(true_agent)
        step_ok = p["step"] is not None and p["step"] == true_step
        step_ok_shift = p["step"] is not None and p["step"] - 1 == true_step
        rows.append({"file": fname, "pred_agent": p["agent"], "true_agent": true_agent,
                     "pred_step": p["step"], "true_step": true_step,
                     "agent_correct": agent_ok, "step_correct": agent_ok and step_ok,
                     "step_correct_if_1_based": agent_ok and step_ok_shift})

    n = len(rows)
    if n == 0:
        sys.exit("No predictions matched any file in the dataset folder. Check the folder path.")
    agent_acc = sum(r["agent_correct"] for r in rows) / n
    step_acc = sum(r["step_correct"] for r in rows) / n
    step_acc_shift = sum(r["step_correct_if_1_based"] for r in rows) / n
    no_step = sum(r["pred_step"] is None for r in rows)

    print(f"Logs scored:        {n}  (predictions found: {len(preds)}, missing label files: {len(missing)})")
    print(f"Agent accuracy:     {agent_acc:.1%}")
    print(f"Step accuracy:      {step_acc:.1%}   (agent AND step correct)")
    print(f"  if model counted steps from 1: {step_acc_shift:.1%}   <- if this is much higher, fix the indexing")
    print(f"Unparseable steps:  {no_step} (counted as wrong)")

    with open("scored_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Per-log results saved to scored_predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
