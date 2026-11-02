# Row weight

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5057

## Why we ran it

To see if weighting block rows helps calibration.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5057.

## What we got

| Row weight | `x8` | Section 21.19, and the 0.07 figure in 12.7 | 12.7 |

## What we inferred

Weights kept to earlier decisions only.

## Why we think this happened

Rows selected on their own score need no correction.

## Limits

Assumed rates. See the spec section named in the row.
