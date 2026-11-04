# Margin creep

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To see whether many small "acceptable" losses add up.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4839 to 4846 in the fristonx repo.

## What we got

### 21.6 Margin creep

Fifteen candidates each truly lost 0.02, under the final approval rule with 8 families.

| Comparison | Mean final loss | Worst | Runs past the 0.10 margin | Runs |
|---|---|---|---|---|
| Against parent | 0.070 | 0.160 | 10.3% | 800 |
| Against fixed reference | 0.036 | 0.080 | 0.0% | 300 |

## What we inferred

Every change is compared with a fixed reference, not its parent (invariant I9).

## Why we think this happened

Each step compared with its parent spends a fresh margin, so losses stack. A fixed reference counts them all against one line.

## Limits

Scenario with a constant loss per candidate.
