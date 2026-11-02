# Routine, family and severe event rules

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5052

## Why we ran it

To replace single-event rollback triggers.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5052.

## What we got

| Routine rate process, family event check, severe count | `x4`, `x6`, `x6b` | Tables in 13.7 | 13.7 |

## What we inferred

Adopted rate processes.

## Why we think this happened

A rate read over many episodes separates chance from a real rise.

## Limits

Assumed rates. See the spec section named in the row.
