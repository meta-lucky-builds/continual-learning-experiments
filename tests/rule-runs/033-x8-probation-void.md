# Probation wealth over a void

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5056

## Why we ran it

To keep validity when a period is voided.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5056.

## What we got

| Probation with wealth carried over a void | `x8` | False rollback about 0.002, with a void in 0 to 30% of runs | 13.7 |

## What we inferred

Wealth carries over a void.

## Why we think this happened

A void removes data but needn't reset evidence.

## Limits

Assumed rates. See the spec section named in the row.
