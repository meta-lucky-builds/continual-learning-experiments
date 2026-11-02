# 32. Sleep T-S3: does dream content matter?

**Date:** 2026-10-29. **Source files:** as test 30.

## Why we ran it
If the loss comes from limited dream coverage (test 31), more diverse dreams might help, and out-of-distribution dreams should hurt.

## What we did
Compared buffer dreams with mixup dreams and with noise dreams (full rank).

## What we got
- Mixup vs buffer: reserve agreement +0.7, final +0.0 (null). Mixup closes the held-out gap only by fitting the buffer worse.
- Noise vs buffer: reserve agreement −46 \*, final **−30.6 \*** (13.3% after bc).

## What we inferred
Consolidation needs in-distribution inputs; noise is catastrophic. Mixup's extra diversity did not help. This rejects noise and mixup dreams, not all synthetic data.

## Why we think it occurred
The teacher's function on noise says nothing about its function on real images, so matching it there transfers nothing useful.

## How reliable this test is
- **High for noise; medium for mixup** (null, 5 seeds).
