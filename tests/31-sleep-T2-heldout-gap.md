# 31. Sleep T-S2: is the transfer lossy off the dream set?

**Date:** 2026-10-29. **Source files:** as test 30.

## Why we ran it
The paper never measures teacher–student agreement off the dream set. If distillation only copies the function where it was trained, the gap between buffer agreement and held-out agreement shows it.

## What we did
Measured student–teacher agreement on the buffer (trained on) vs the reserve (held out), and the accuracy cost of each sleep (student − teacher), for rank 4, rank 16, full rank and mixup dreams.

## What we got
- Held-out gap: rank 16 **6.6 points**, full 8.5, rank 4 6.3 (all 5/5). Buffer agreement 85–93%, reserve 78–85%. Test agreement equals reserve agreement.
- Cost of each sleep: −3.6 to −5.7 points, every arm, 5/5. The predicted ordering r4 < r16 < full was **not** confirmed (r4 = r16).

## What we inferred
The transfer is lossy: 6–9 points of agreement are lost off the dream set, and the losses accumulate over sleeps.

## Why we think it occurred
Distillation on 500 images constrains the student only near those images; elsewhere it is free to differ. This is the general "fitted on a set looks better on that set" pattern.

## How reliable this test is
- **High.** Direct measurement, 5/5; the reserve is a fair proxy for test.
