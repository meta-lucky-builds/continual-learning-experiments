# Measuring information

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To see if information gain per cycle can be measured from outcomes.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4868 to 4875 in the fristonx repo.

## What we got

### 21.9 Measuring information

| Forecast noise, old to new | True gain (bits) | Outcomes for 80% power |
|---|---|---|
| 1.0 → 0.9 | 0.024 | 9,649 |
| 0.5 → 0.3 | 0.020 | 2,385 |
| 1.0 → 0.7 | 0.056 | 1,384 |
| 1.0 → 0.4 | 0.098 | 356 |

## What we inferred

Outcome information is too noisy per cycle, so gain is read over at least 500 outcomes, with rolling dynamics measures.

## Why we think this happened

Each outcome is a single noisy bit, while the gain being measured is a tiny fraction of a bit.

## Limits

A later check could not reproduce these gains from a matched Gaussian forecaster, so the definition of forecast noise is unclear.
