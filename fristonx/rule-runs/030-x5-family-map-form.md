# Family map form

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5053

## Why we ran it

To choose a map for families with few rows.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5053.

## What we got

| Family map form | `x5` | Mean error 0.054–0.071 for an offset, 0.131–0.142 for a monotone map | 12.7 |

## What we inferred

Adopted an offset below 100 rows.

## Why we think this happened

One parameter can be fitted on few rows, and a full curve can't.

## Limits

Assumed rates. See the spec section named in the row.
