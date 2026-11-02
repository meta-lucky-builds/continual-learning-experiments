# Recheck of all findings

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/recheck

## Why we ran it

To confirm every finding was closed on the final text, not on an older one.

## What we did

See the source named above.

## What we got

All 813 rows traced on hash 00af3899. 29 rows were wrongly marked fixed. First 8 batches: 394 fixed, 18 no longer apply, 17 cannot close, 11 open.

## What we inferred

Open rows were fixed, and 105 rows moved to cannot close after the harm review.

## Why we think this happened

A fix can be undone by later edits, and a label can outlive the text it described.

## Limits

The harm review was one medium-effort pass.
