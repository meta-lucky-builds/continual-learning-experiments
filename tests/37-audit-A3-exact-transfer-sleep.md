# 37. Audit A3: exact-transfer sleep control

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{audit_plan.md, results_audit.md, sleep_exact.py, analyze_audit10.py}`, `results/audit/analysis10.txt`, logs `results/audit/slp_exact_*.log`.

## Why we ran it
Sleep lost 8.4 points (test 30). The audit asked which part caused it: the reset and architecture, or the approximate (distilled) transfer.

## What we did
Copied the teacher exactly into the new expert (Δ = teacher − student; max logit error 3.8e-6), then ran the same reset and continuation. CIFAR-10, 5 seeds.

## What we got
- Exact copy: 52.2 vs ER 52.4 (−0.2, not significant).
- Full-rank distillation vs the exact copy: **−8.2 \***.

## What we inferred
**All** of Sleep's loss is approximate transfer through distillation on 500 buffer images; none of it comes from the reset or the architecture.

## Why we think it occurred
An exact copy preserves the function everywhere; distillation preserves it only near the dream images (test 31).

## How reliable this test is
- **High.** Clean control with an exactness check. Not independently reviewed as code.
