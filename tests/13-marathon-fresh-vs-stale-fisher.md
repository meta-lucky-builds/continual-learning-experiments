# 13. Marathon T4: fresh vs stale Fisher (EWC)

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
Spec hypothesis H1: importance (Fisher) estimates should be computed fresh, on the buffer at the current weights, not stored from when the task ended. This tests H1 with EWC.

## What we did
EWC with a stale Fisher (stored at task end) vs a fresh Fisher (recomputed on the buffer), λ ∈ {0.1, 1, 10, 100}, on toy A and MNIST.

## What we got
- Toy A: fresh > stale at λ = 0.1, 1, 10 (+3.1 \*, +3.9 \*, +4.9 \*); fresh − ER = +7.1 \* at λ = 1.
- MNIST: fresh > stale at λ = 0.1 (+6.5 \*), but fresh is **much worse** at λ ≥ 10 (−10.0 \*, −13.8 \*). Stale EWC at λ = 100 gives +5.2 \* over ER.

## What we inferred
H1 holds at small λ only. "Fresh beats stale" is not unconditional.

## Why we think it occurred
A fresh Fisher measured at the current weights is large exactly where the current task is learning, so a strong fresh penalty blocks plasticity. A stale Fisher puts its penalty on old-task directions.

## How reliable this test is
- **Medium.** Clear crossover, 5 seeds. MLP scale only. On CIFAR-10, EWC gave nothing (test 27).
