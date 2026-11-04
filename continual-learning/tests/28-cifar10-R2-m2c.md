# 28. CIFAR-10 round 2, protection: M2c at real-image scale

**Date:** 2026-10-29. **Source files:** as test 25, plus `qp_check.py`, `m2c2_check.py`.

## Why we ran it
M2c with the spec's rival rule had looked promising on the toy (test 14, +12.5 over ER). It had to be tested at real-image scale before being kept or removed.

## What we did
- m2c=1 (first port): the pool drawn from the reservoir, which contains current-task items.
- m2c=2 (faithful to the toy): the pool from past-task items only. Declared before any m2c=2 result.
- Checked feasibility of every projection, the solver (NNLS vs the original L-BFGS-B), and the logit-scale trajectory.

## What we got
- m2c=1: 26.3 vs ER+bc 52.4, **−26.1 \***. Logits climbed steadily from 5 to 4e4 by step 1,550 (seed 0), up to 5.6e5.
- m2c=2: 39.7, **−12.8 \***. Logits from ~12 after task 1 to 270–1,360 by task 5 (ER: ~16). In mixtures, the online weight moves almost all onto RC.
- Projection feasible throughout (worst violation ≤ 3e-10). The NNLS solver matched the original to 2e-15 and gave an identical logit trajectory, so the solver is not the cause.

## What we inferred
M2c loses badly on CIFAR in both variants. Removed at this scale, by the audit's stopping rule.

## Why we think it occurred
- m2c=1: the reservoir holds current-task items from the first batch, so it forbade margin decreases on the task being learned. That is a porting difference from the toy.
- m2c=2: forbidding any first-order decrease of old margins while new classes are learned forces the logit scale up, and accuracy falls.
- Untested hypothesis: once logit gaps exceed ln 2²⁴ ≈ 16.6, float32 rounds the top softmax probability to 1, so some late dynamics may be precision effects on top of the real failure.
- [Lucky's audit] "M2c forces logit scale up" is not proved: the zero step is always feasible, so it could stall instead. Only "this implementation failed under these settings" is supported.

## How reliable this test is
- **High that this implementation fails**; settings were the toy's best, untuned, and SGD 0.03 replaced the toy's Adam 1e-3. The solver was swapped mid-round (validated and declared before any M2c result); the first attempt was stopped at ~40% (declared).
