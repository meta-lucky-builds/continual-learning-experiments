# 12. Marathon T1: does the task-boundary "shock" drive forgetting?

**Date:** 2026-10-28. **Source files:** as test 05; superseded first runs in `source/notes/marathon/results/v1_superseded/`.

## Why we ran it
Old-class accuracy collapses 20–40 steps into each new task. If that shock drives final forgetting, a fix at the boundary (better init for new head rows, a warm-up) would help.

## What we did
Compared a centred mean initialisation of new head rows, and a learning-rate warm-up, against plain ER.

## What we got
- The centred init raises the dip minimum (+8.1 \* toy A, +12.1 \* MNIST) but **never improves final accuracy**.
- Warm-up hurts (−2.3 \* toy A, −1.3 \* toy B).

## What we inferred
The boundary collapse is not what drives final forgetting. Killed.

## Why we think it occurred
The dip recovers on its own; final accuracy is set by the steady prior offset accumulated over the whole task (test 05), not by the transient at the start.

## How reliable this test is
- **Medium-high**, after a fix. The first T1 runs used an uncentred mean init that put old-class accuracy at 0 at step 0. A reviewer flagged this as CRITICAL; those runs were superseded and are not used.
