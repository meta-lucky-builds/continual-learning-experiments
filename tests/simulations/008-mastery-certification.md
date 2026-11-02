# Mastery certification

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To see what it costs to certify each task strictly.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4857 to 4866 in the fristonx repo.

## What we got

### 21.8 Mastery certification

| Target | All-success minimum | True rate | Median episodes, no limit | Certified within 60 |
|---|---|---|---|---|
| 0.70 | 9 | 0.85 | 62 | 0.50 |
| 0.80 | 14 | 0.90 | 101 | 0.24 |
| 0.80 | 14 | 0.95 | 45 | 0.75 |
| 0.80 | 14 | 0.99 | Not run | 1.00 |

The sequential bet used here isn't optimised, so a tighter procedure may lower these numbers. The order of magnitude still shows that per-task certification competes with learning cost.

## What we inferred

Two levels of mastery: cheap working mastery, and sampled certification with a cap.

## Why we think this happened

A confident claim about a high rate needs many clean successes. That is an information floor, not a tuning problem.

## Limits

The bet was not optimised, so the numbers are upper-side.
