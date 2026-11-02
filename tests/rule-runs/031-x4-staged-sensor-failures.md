# Staged sensor failures

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5054

## Why we ran it

To check labels from sensors can be trusted.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5054.

## What we got

| Staged sensor failures | `x4` | 99.9% caught at 50% missed, 1.7% false void at 2% | 12.9 |

## What we inferred

Adopted staged failure checks.

## Why we think this happened

Staged failures give known truth.

## Limits

Assumed rates. See the spec section named in the row.
