# Evidence overlap

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To see what correlated evidence does to confidence.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4889 to 4896 in the fristonx repo.

## What we got

### 21.11 Evidence overlap

| Innovation autocorrelation | Overlap | Actual coverage of nominal 95% |
|---|---|---|
| 0 | None | 95.2% |
| 0.5 | None | 74.8% |
| 0.9 | None | 34.9% |
| 0.9 | 4× | 18.0% |

## What we inferred

Windows must not overlap, and evidence is scaled by effective sample size (invariant I11).

## Why we think this happened

Correlated windows repeat the same information, so the filter grows too confident.

## Limits

The source check confirmed these match the AR(1) design-effect formula.
