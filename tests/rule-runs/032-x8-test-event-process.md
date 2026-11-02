# Test event process

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5055

## Why we ran it

To catch event rises inside a test.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5055.

## What we got

| Test event process | `x8` | Neutral firing at most 0.0006 over 110 pairs. A fivefold rise fires 0.9%, 34% and 97% at base rates 0.01, 0.03 and 0.08 | 13.5 |

## What we inferred

Adopted a level-0.01 process.

## Why we think this happened

Power depends on the base rate, so rare events are hard to read.

## Limits

Assumed rates. See the spec section named in the row.
