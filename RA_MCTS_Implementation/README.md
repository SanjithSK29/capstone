# Independent RA-MCTS implementation

This folder is an independent, reproducible implementation of the core
Responsibility-Attribution Monte Carlo Tree Search (RA-MCTS) idea from
Triantafyllou and Radanovic, *Towards Computationally Efficient Responsibility
Attribution in Decentralized Partially Observable MDPs*, AAMAS 2023
([arXiv:2302.12676](https://arxiv.org/abs/2302.12676)).

It is deliberately self-contained and does not import either of the other
capstone implementations.

## What is implemented

* A finite-horizon, three-or-more-agent synthetic Dec-POMDP-like environment.
  Safe, joint, redundant, and sequential failure patterns are represented by
  the transition equations, with noisy private observations and a fixed seed.
* A structural causal model facade that evaluates factual and same-context
  counterfactual trajectories.
* Action interventions at arbitrary time/agent coordinates, including
  multi-variable interventions.
* AC1--AC5-style cause checks, minimality, witness contingencies, and
  variable/agent responsibility summaries.
* A specialized prefix tree with UCB1 selection, linear scalarization of value
  and causal score, simulation, evaluation, minimality-aware backup, and
  budget-bounded pruning.
* Brute-force and random-search baselines plus reproducible comparison and
  sensitivity CLIs.
* A small Beta posterior sampler for uncertain Bernoulli context variables.

The implementation is a practical discrete engineering adaptation. The paper
uses a more general SCM and card-game benchmark; this repository does not claim
to reproduce those games or every theorem-level detail. In particular, AC4/AC5
are exposed as explicit checks with a documented finite-context proxy, and the
posterior sampler is a reusable Bernoulli component rather than a complete
inference engine for arbitrary hidden-card histories.

## Installation and tests

From this directory:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -v
```
The package uses only the Python standard library at runtime; `pytest` is the
test dependency.

## Experiments

```powershell
python experiments\run_experiment.py --episodes 8 --budget 80 --seed 7
python experiments\run_random_search.py --budget 100 --seed 7
python experiments\run_bruteforce.py --agents 3 --horizon 3
python experiments\compare_methods.py --budget 100
python experiments\sensitivity_analysis.py --budgets 10,30,100 --exploration 0.5,1.414,2
```

All generated files remain below `results/`. The scripts compute their values
from executed simulations; no result values are checked in or fabricated.

## Architecture

`src/environment.py` defines states, actions, observations, transitions, and
outcomes. `src/dec_pomdp.py` provides the modular model interface.
`src/scm.py` and `src/interventions.py` preserve the factual trace while
changing only intervened action variables. `src/responsibility.py` enumerates
minimal cause/witness candidates. `src/search_tree.py` and `src/ra_mcts.py`
implement the responsibility-aware search. `src/posterior_sampling.py` holds
the uncertainty component.

See `docs/methodology.md` for the mathematical conventions, algorithm phases,
and limitations.

## BibTeX

```bibtex
@inproceedings{triantafyllou2023responsibility,
  title={Towards Computationally Efficient Responsibility Attribution in
         Decentralized Partially Observable MDPs},
  author={Triantafyllou, Stelios and Radanovic, Goran},
  booktitle={Proceedings of the 22nd International Conference on Autonomous
             Agents and Multiagent Systems},
  year={2023}
}
```
