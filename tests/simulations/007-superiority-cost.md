# Cost of proving superiority

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To learn how many pairs prove a real gain, as compared with non-inferiority.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4848 to 4855 in the fristonx repo.

## What we got

### 21.7 Superiority (level 0.05, matched pairs)

| True gain | Median pairs |
|---|---|
| 0.05 | 558 |
| 0.10 | 77 |
| 0.15 | 34 |
| 0.20 | 24 |

## What we inferred

Small improvements can't be proved one by one, so only large gains are read directly.

## Why we think this happened

Pairs needed scale roughly with one over the gain squared.

## Limits

Matched pairs at an assumed disagreement rate.
