# Builder mutation run

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/stages/stage-0/BUILD_LOG.md

## Why we ran it

To see if tests catch deliberate faults.

## What we did

See the source named above.

## What we got

29 hand-made mutations: 26 caught, 3 survived. Tests were added, and all 3 are now caught.

## What we inferred

Added forged-record tests and a stricter drill.

## Why we think this happened

Passing tests don't show the tests can fail.

## Limits

Small sample chosen by the author.
