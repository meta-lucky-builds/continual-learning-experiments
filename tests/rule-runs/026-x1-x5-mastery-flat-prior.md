# Mastery with a flat prior

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5049

## Why we ran it

Mastery failed on improving tasks under a fitted family prior.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5049.

## What we got

| Mastery with flat prior, block, alternatives | `x1`, `x5` | Crossing 96%, 82% and 71% at spreads 0, 0.10 and 0.20, against 18%, 34% and 51% for the family fit | 12.8 |

## What we inferred

Adopted a flat prior.

## Why we think this happened

A family fit reads early failures as the task's nature.

## Limits

Assumed rates. See the spec section named in the row.
