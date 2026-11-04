# 11. Marathon W6: does repetition give real savings?

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
After test 10 removed the false savings, we needed to know whether *any* real savings exist when tasks recur, which bears on Lucky's "pre-skill buffer" idea (forgotten skills relearn faster).

## What we did
Cyclic stream: tasks 0–4 repeated 3 times. Measured retention at each return and end-of-cycle accuracy, raw and bias-corrected, for ER and ER-ACE.

## What we got
- ER retention at return grows from the 2nd to the 3rd occurrence: +3.1 \* toy (bias-corrected +1.7 \*), +8.2 \* MNIST (bias-corrected +5.4 \*).
- End-of-cycle accuracy: toy +3.5 \* by cycle 3 (plateaus after cycle 2); MNIST +4.5 \*; ER-ACE +5.3 \*.

## What we inferred
Repetition gives small but real savings, visible even after bias correction. Return *speed* cannot be read from ER's raw curves (test 10).

## Why we think it occurred
[interpretation] Each repetition leaves the shared features slightly better suited to the recurring classes, so less has to be relearned. This was not measured directly.

## How reliable this test is
- **Medium.** 5/5 seeds and bias-corrected, but small effects on a toy and MNIST. Not repeated on CIFAR. A recurring-context test on CIFAR is still owed.
