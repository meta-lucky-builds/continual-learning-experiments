# Power of real model controls

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5068

## Why we ran it

To size real control tests.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5068.

## What we got

| Power of the real model controls | `d12_fix.py` | 0.80 at 10 groups and one between-group standard deviation, about 0.5 at 6 | 6.8 |

## What we inferred

Set at least 10 groups.

## Why we think this happened

Group count sets the power of a t-test.

## Limits

Assumed rates. See the spec section named in the row.
