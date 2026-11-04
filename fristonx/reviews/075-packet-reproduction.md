# Stage 0 packet reproductions

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** docs/stages/stage-0/work

## Why we ran it

To check spec tables before building them.

## What we did

See the source named above.

## What we got

Approval, Probation, ratchet, mastery, revocation, Certify, monitor and K1 tables matched within noise. Script checked 2,073 quotes with 0 mismatches. State tables had no dead end except one defect, fixed.

## What we inferred

Corrected the CUSUM lead and the stuck-rule lead from the math search.

## Why we think this happened

Partial models miss rules that a full reading includes.

## Limits

Reads section by section of the spec.
