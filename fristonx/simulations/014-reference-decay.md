# Reference decay over 5 years

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_decay.py

## Why we ran it

To see what happens when the fixed reference grows stale.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4962 to 4975 in the fristonx repo.

## What we got

### 21.15 Reference decay over 5 years

`audit/round6_work/r6_decay.py`. This is a scenario and not a measurement. The reference falls by the stated amount per month relative to a stack that adapts, each quarterly candidate truly loses 0.02 against current, the decay estimate has a standard deviation of 0.04, and the approval table is the final rule's. Its fixed-margin column follows round 5's model of the old rule, using the final approval table.

| Decay per month | Fixed margin: mean loss, past 0.10, past 0.20, commits | Final rule: mean loss, past 0.10, past 0.20, commits |
|---|---|---|
| 0 | 0.05, 0%, 0%, 2.3 | 0.04, 0%, 0%, 1.9 |
| 0.004 | 0.21, 100%, 55%, 10.6 | 0.08, 3%, 0%, 3.8 |
| 0.008 | 0.36, 100%, 100%, 17.7 | 0.05, 0.1%, 0%, 2.7 |
| 0.015 | 0.39, 100%, 100%, 19.3 | 0.03, 0%, 0%, 1.7 |

At the higher decays, the loss is small because C3 stops once `decay` reaches 0.20, and the scenario doesn't model the Recertification that would follow. So those rows show a stop and not continued safe progress.

In this scenario the effective margin and the stale-reference stop did nearly all the work. The absolute floor matters when the current stack itself falls, which Section 21.16 covers. These loss figures aren't a guarantee, and they depend on the assumed noise.

## What we inferred

Added the effective margin and a stale-reference stop for C3.

## Why we think this happened

As a reference falls behind, tests get easier, so a fixed margin lets real loss through.

## Limits

A scenario, not a measurement. The high-decay rows show a stop, not continued safe progress.
