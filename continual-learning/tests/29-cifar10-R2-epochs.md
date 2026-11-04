# 29. CIFAR-10 round 2: three passes per task

**Date:** 2026-10-29. **Source files:** as test 25.

## Why we ran it
All tests so far were single-pass online. Does the two-path gain survive when the plastic net sees each task three times?

## What we did
V1: ER+bc (3 epochs) vs RC-LDA. V2: MIX_half (3 epochs) vs the better single expert. V3: ER+bc 3 epochs vs 1. lr 0.03, not retuned.

## What we got
- V1: 53.2 vs 54.0, −0.9, not significant.
- V2: 58.0, **+3.3 \***.
- V3: +0.8 (4/5), not significant.

## What we inferred
The two-path gain is robust to more passes. Extra passes barely help ER here.

## Why we think it occurred
The plastic path's failure is mostly readout bias and limited history, which more passes over the current task do not fix. lr was not retuned, which may also limit V3.

## How reliable this test is
- **Medium-high for V2** (5/5). V3 may be limited by the untuned lr.
