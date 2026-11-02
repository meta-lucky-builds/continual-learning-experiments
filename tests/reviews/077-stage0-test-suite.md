# Stage 0 test suite

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** tests/, scripts/ci.py, scripts/close.py

## Why we ran it

To prove steps 1 to 3.

## What we did

See the source named above.

## What we got

728 passed, rerun independently by me. 253 of 253 scoped requirements passed. Step 3's code came before its tests, which broke the rule.

## What we inferred

Steps 1 to 3 were committed.

## Why we think this happened

Tests written by the builder share the builder's blind spots.

## Limits

Hypothesis ran at 200 examples only. Some requirements are only partly proven.
