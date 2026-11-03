# Capstone Phase 2: attribution on the SALT-NLP collusion data

Question: when Alice (an LLM agent) violates the verification protocol by accepting Bob's work without the raw log, which of the things in her context caused it, and which blame method gives a sensible answer when several causes could each be enough?

## Data
Pilot slice of SALT-NLP/agent-collusion (Hugging Face): condition `controlled_peer__gemini-3-1-flash-lite__summary--accept--full-feedback`. Bob is scripted (summaries, always accept); Alice is gemini-3.1-flash-lite. 500 episodes (50 runs x 10); Alice accepts in 41.2%, matching the paper. `get_pilot.py` downloads the episodes table and 5 traces.

## Candidate causes (parts of Alice's context at her verdict)
- M: the peer's messages in the current episode
- F: reward + verdict-review feedback from earlier episodes
- R: Alice's own private reflections from earlier episodes

Note: the "agents" in the blame sense are information sources acting on Alice (peer, reward/feedback channel, her own history). The dataset has only two agents, so this is attribution among influence channels in a two-agent system, not among many agents.

## Files
- `loader.py`: trace -> Episode objects (verdict, peer messages, history, message indices of each cause).
- `interventions.py`: remove any subset of {M, F, R} from Alice's context; `validate` checks the edited conversation is still legal.
- `replay.py`: re-sample Alice's verdict K times -> P(violation) with a Wilson interval. `LiteLLMSampler` is NOT yet tested against a live API.
- `attribution.py`: but-for, ablation, Shapley, Halpern-Pearl (original definition) and Chockler-Halpern responsibility 1/(1+|W|) from a table {removed set -> (k, n)}. Thresholds hi=0.5, lo=0.25 are parameters to report.
- `test_attribution.py`: synthetic tables with known structure (single, OR, AND, 3-way OR). A test of the code, not a result.

## Status
Tested offline on 2 of the 5 traces (20 episodes): loader matches the episodes table; all 160 edited contexts validate. Nothing has been run against a live model yet (needs an API key). With clean effects (0.90 vs 0.05) about 40 samples per cell are needed for responsibility to be right ~91-97% of the time; 20 gives ~50-70%.

## Run
```
python -m pip install -U datasets pyarrow huggingface_hub pandas litellm
python get_pilot.py                      # downloads the pilot episodes table + 5 traces (not stored in this repo)
python test_attribution.py               # synthetic checks (Windows PowerShell: $env:PYTHONPATH="."; python test_attribution.py)
```
`get_pilot.py` writes `pilot_episodes.parquet` and `traces/` next to itself. Keep them out of git.

Next: fidelity check (re-sample with nothing removed, compare to recorded rate; needs a Gemini API key, read from an environment variable, never committed), then pilot replay.

## Assumptions to state in the report
- Only the verdict is re-sampled; task and communication phases stay as recorded.
- R was written after seeing F, so removing F alone does not remove its information (direct vs total effect).
- This is interventional replay with the factual history held fixed, not a full counterfactual.

## Offline method benchmark

The API replay is expensive, so the attribution methods can also be evaluated
without a model:

```
python benchmark.py
```

This writes `results/synthetic_summary.csv`, `results/synthetic_summary.json`,
`results/sanjith_pilot_values.csv`, `results/sanjith_pilot_values.json`,
`results/method_accuracy_64_samples.svg`, and
`results/sanjith_pilot_acceptance.svg`. The synthetic benchmark has known
ground truth and covers a single cause, redundant causes, joint causes, a
partial effect matching the pilot pattern, and three redundant causes. Results
are repeated over fixed random seeds at 8, 16, 32, and 64 samples per
intervention cell.

The values in `sanjith_pilot_values.json` are a separate transcription of the
four acceptance-rate rows reported in `results_causal.md`; they are not mixed
with synthetic scores or presented as additional API runs.

## Temporal responsibility extension

`temporal_attribution.py` separates a cause's effect on the first failure
transition (`onset`) from its effect on later failure persistence. Run:

```
python temporal_analysis.py
```

This produces `results/sanjith_temporal_profile.csv`, `.json`, and `.svg` from the
reported pilot values. In that pilot, feedback has an onset effect of `0.69`
(`1.00 - 0.31`) and a persistence effect of `0.00`; this is an analysis of
the reported table, not a new API run. The extension is intentionally
stage-aware because a cause that initiates a behavior need not maintain it.

`temporal_benchmark.py` validates this new stage-aware classification on
known initiator-only, maintainer-only, both-stage, and inactive mechanisms at
multiple sample sizes. Its output is `results/temporal_benchmark.csv`.

## Explicit temporal SCM

`scm.py` provides a declared structural causal model with context channels
(`M_t`, `F_t`, `R_t`), a persistent collusion state (`C_t`), and Alice's
verdict (`A_t`). It supports binary interventions and exports the DAG through
`graph_dot()` and `graph_mermaid()`. This is a hypothesis-driven SCM, not a
causal graph learned automatically from one pilot trajectory. The default
parameters encode the pilot-facing hypothesis that feedback can create a
persistent state; they are not additional API measurements.

Run `python export_scm.py` to write `results/temporal_scm.dot` and
`results/temporal_scm.mmd` for reports or diagram rendering.

`causal_estimands.py` adds pathway-level questions that ordinary ablation
cannot answer: total effect, direct effect with the persistent state blocked,
effect mediated through the state, and minimal interventions that prevent or
create the state. Run `python causal_report.py` to write
`results/causal_pathway_report.csv` and `.json`. These are outputs of the
declared SCM, not additional real-model measurements.

`sensitivity_report.py` varies the learning and baseline probabilities and
writes `results/scm_sensitivity.csv`, checking whether the pathway conclusion
depends on one exact parameter choice.

## Intervention policy experiment

`policy_experiment.py` compares concrete prevention and recovery actions:
remove feedback, remove peer messages, remove reflections, remove all context,
reset the persistent state, and reset the state plus feedback. It evaluates
which lowest-cost policy prevents onset and which recovers an already-active
state under several SCM parameter sets. Run:

```
python policy_experiment.py
```

This writes `results/policy_experiment.csv`, `.json`, and `.svg`. It is a
model-based intervention-planning experiment, not a new API run.
