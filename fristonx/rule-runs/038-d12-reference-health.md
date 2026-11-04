# Reference health as one stream

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/spec/SPEC.md line 5066

## Why we ran it

Reading anchors one by one alarmed on healthy runs.

## What we did

A design run on one rule. Copied from `docs/spec/SPEC.md` line 5066.

## What we got

| Reference health as one stream | `d12_fix.py` | 96–100% of healthy runs alarmed under the old reading, none of 300 under this one | 13.6 |

## What we inferred

Read as one centred stream.

## Why we think this happened

Many separate streams give many chances for false alarms.

## Limits

Assumed rates. See the spec section named in the row.
