# Stuck rule

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5051

## Why we ran it

Fixed failure counts asked for too much help.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5051.

## What we got

| Stuck rule | `x3` | Help requests fell from 1.45 to 0.40 per task, with no spread, and PARKED from 7.5% to 0% on the same curve. A flat 0.70 task spends about 22 opportunities more | 7.8 |

## What we inferred

Adopted the rule with progress checks.

## Why we think this happened

Improving tasks fail early by nature, so a fixed count misreads them.

## Limits

Assumed rates. See the spec section named in the row.
