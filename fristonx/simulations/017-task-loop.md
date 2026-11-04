# Task loop and strategy table

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_taskloop.py

## Why we ran it

To check if a strategy table can improve a task without changing weights.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5004 to 5015 in the fristonx repo.

## What we got

### 21.18 Task loop

These figures predate the settled stream and the rule that ABSTAIN doesn't count against a strategy, and they left out the floor. Section 21.22 runs the table with both.

`audit/round6_work/r6_taskloop.py`. Assumed rates: two contact stages, whose strategies succeed at 0.97 and 0.90, and at 0.55, 0.95 and 0.30, with the prior policy favouring the first of each. Success per episode by block of 8 episodes:

| Block | 1–8 | 9–16 | 17–24 | 25–32 | 33–40 |
|---|---|---|---|---|---|
| No table | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| Table | 0.69 | 0.78 | 0.82 | 0.84 | 0.86 |

The better of two strategies at 0.9 and 0.6 was the greedy pick after 60 episodes in 98.7% of runs. With every strategy at 0.3, shrunk `P_s` fell under an irreversible stage's floor after a median of 4 opportunities, and a 90th percentile of 12, so a hopeless task stops itself early.

## What we inferred

Kept the strategy table, because it needs no learning from the network.

## Why we think this happened

If a better way exists and is being proposed, counting outcomes is enough to find it.

## Limits

Assumed rates. It says nothing on whether a real policy offers two ways. Figures left out the floor.
