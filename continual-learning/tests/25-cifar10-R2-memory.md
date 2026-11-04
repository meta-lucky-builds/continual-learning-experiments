# 25. CIFAR-10 round 2, memory: statistics vs images at equal bytes

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{plan.md "Round 2", results_r2.md, cifar2.py, analyze_r2.py}`, `results/r2/`.

## Why we ran it
The two-path system stores a stable expert's statistics on top of the buffer. A fair critic asks: would spending those bytes on more buffer images do as well?

## What we did
- R1: MIX_half(ER M = 500, RC 512-d) vs ER+bc at M = 680 (same total memory).
- R2: MIX_half(ER 500, wide RC 1024-d) vs ER+bc at M = 1,200 (same memory as the wide expert, 2.1 MB).
- R3: wide RC vs RC.

## What we got
- R1: 57.3 vs 54.2, **+3.2 \***.
- R2: 59.4 vs 58.2, +1.2 (4/5), not significant; streaming-valid version +2.4 (4/5), not significant.
- R3: 57.8 vs 54.0, **+3.8 \***.
- ER+bc gains 5.8 points from M = 500 to 1,200.

## What we inferred
At small budgets, statistics beat images per byte. At larger budgets a bigger buffer catches up, and the case is open (not significant is not a tie).

## Why we think it occurred
Buffer accuracy rises steeply with buffer size at first, so 700 extra images are worth a lot; a fixed statistics expert's value does not grow with memory.

## How reliable this test is
- **Medium-high for R1 and R3**; R2 is underpowered (5 seeds).
