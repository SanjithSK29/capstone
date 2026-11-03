# Methodology

## Synthetic causal environment

At time `t`, a joint action is a binary vector
`a_t = (a_t^1, ..., a_t^n)`. Zero is safe and one is hazardous. The state
contains the time, accumulated safe progress, and a failure flag. A transition
increments progress only for an all-safe joint action. Two or more hazards
produce a joint failure. A first-step hazard followed by an otherwise safe
second step produces a sequential failure. Repeated hazards create redundant
candidate causes. This gives small cases whose outcomes can be checked by hand
while still requiring multi-agent and multi-step reasoning.

Observations are private noisy copies of the local actions. The random seed is
explicit; causal evaluation itself replays the same factual trace, so an
intervention cannot accidentally resample unrelated noise.

## Actual causality and responsibility

For a factual failing trace `u`, a candidate `X` is a tuple of action
variables `(time, agent)`. The implementation checks:

1. **AC1:** the factual outcome is failure and every variable in `X` has its
   factual hazard value.
2. **AC2:** setting `X` to safe, while retaining the same context and all other
   actions, changes the outcome to success.
3. **AC3:** no strict non-empty subset of `X` satisfies AC2.
4. **AC4:** the counterfactual is evaluated using the same exogenous/factual
   context.
5. **AC5:** the finite synthetic environment's stability proxy holds for the
   candidate.

The size-based responsibility of a minimal cause is `1 / |X|`. Agent
responsibility is the maximum score among causes containing that agent; this
is the discrete counterpart used by the experiment scripts. The source paper
uses a formal responsibility definition over actual-cause/witness pairs; the
proxy is intentionally visible in code and README rather than presented as a
general theorem.

## RA-MCTS phases

The tree stores action prefixes. Each iteration:

1. selects a child with UCB1 plus a linearized causal score;
2. expands one unvisited joint action;
3. completes the prefix with seeded random actions;
4. evaluates the factual outcome and enumerates minimal cause/witness pairs;
5. backs up scalar reward and the best causal score;
6. marks evaluated leaves as prunable.

The budget is the exact number of simulations. The tree's statistics expose
simulation and pruning counts for comparisons with the baselines. Brute force
enumerates every finite trace and is the ground-truth method only for the small
synthetic instances; random search uses the same scorer and budget.

## Uncertainty

`BernoulliPosterior` supports seeded Beta posterior draws and updates. It is
kept separate from deterministic SCM evaluation because a full posterior over
hidden exogenous histories depends on the application model. It should not be
described as complete hidden-state inference.

## Limitations and extensions

The benchmark is not the paper's Euchre, Spades, or Team Goofspiel test bed.
The current tree branches on joint actions rather than using the paper's
agent/action node hierarchy, and AC4/AC5 are finite-context checks. Natural
extensions are an explicit agent-level tree, domain-specific SCMs, posterior
conditioning on observation histories, confidence intervals over repeated
seeds, and plot generation from the CSV outputs.
