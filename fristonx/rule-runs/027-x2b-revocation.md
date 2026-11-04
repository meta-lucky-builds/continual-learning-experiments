# Revocation boundary

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5050

## Why we ran it

To set a revocation line that catches real falls but rarely fires on healthy tasks.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5050.

## What we got

| Revocation at H0 0.75, `A_r` 20 | `x2b` | Within 200 opportunities: 0.01% at 0.85, 0.4% at 0.80, 28% at 0.70, 97% at 0.60. Within 40: 0.01%, 0.1%, 4%, 31% | 12.8 |

## What we inferred

Adopted H0 0.75 with A_r 20.

## Why we think this happened

A rate well below the line produces evidence quickly, and a rate near it barely does.

## Limits

Assumed rates. See the spec section named in the row.
