# 36. Audit A2: the four-version bank against its own memory-matched buffer

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{audit_plan.md, results_audit.md, analyze_audit100.py}`, `results/audit/analysis100.txt`.

## Why we ran it
T-S5 compared the bank against a buffer matched to BANK_all's memory, not BANK_4spread's. The audit asked for the fair comparison at BANK_4spread's own budget.

## What we did
BANK_4spread vs MIX_half(ER with M = 3,439, RC), M from budget.json; and vs ER+bc at the same memory.

## What we got
- BANK_4spread 31.2 vs 30.0: **+1.3 ± 0.3 \***.
- vs ER+bc at the same memory: +7.2 \*.

## What we inferred
At its own budget, a small spread bank beats spending the memory on a bigger buffer.

## Why we think it occurred
Each spread version holds full stream statistics for the classes it saw, in its own coordinates (~480 items per class), which is far more history per byte than ~20 buffer images per class.

## How reliable this test is
- **Medium-high.** 5/5. "Spread" knows the stream length. Superseded as the best online system by test 39 (X2 transport mixture).
