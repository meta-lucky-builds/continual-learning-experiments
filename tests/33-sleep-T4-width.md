# 33. Sleep T-S4: is forgetting a capacity problem?

**Date:** 2026-10-29. **Source files:** as test 30.

## Why we ran it
The paper frames forgetting as insufficient capacity. If so, a wider network should forget less without any readout fix.

## What we did
T4: ER+bc vs 4× wider ER (raw). T4b: wide ER+bc vs ER+bc.

## What we got
- 4× width gives no raw gain (42.3 vs 42.6).
- T4: bias correction alone beats width, **+10.1 \*** (pre-declared threshold > 5: passed).
- T4b: with correction, width adds **+1.4 \*** at ~4× the FLOPs.

## What we inferred
On CIFAR-10, forgetting is a readout problem, not a capacity problem. This contradicts the paper's framing for this benchmark.

## Why we think it occurred
The newest-class offset (test 05) does not depend on width: a wider network learns the same training prior.

## How reliable this test is
- **High.** The width arm used the same lr (0.03), not retuned.
