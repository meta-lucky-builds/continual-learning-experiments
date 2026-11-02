# 08. CIFAR-10 round 2: memory, promotion, protection, epochs

**Source:** ewt.md §8. **Code (not uploaded):** `cifar2.py` (K1–K10), `cifar2_checks.py`, `qp_check.py`, `m2c2_check.py`.

## Why we did it
With the two-path result in hand, four design questions remained:
1. **Memory:** per byte, are statistics worth more than buffer images?
2. **Promotion:** should frozen copies of the network be kept as extra experts?
3. **Protection:** do EWC or M2c help at real-image scale?
4. **Epochs:** does the two-path gain survive multiple passes?

## What we got
| Step | Comparison | Result |
|---|---|---|
| Memory | R1: MIX_half(ER500, RC) vs ER+bc at M = 680 (equal memory) | **+3.2 \*** |
| Memory | R2: MIX_half(ER500, wide RC) vs ER+bc at M = 1,200 | +1.2 (4/5), not significant |
| Memory | R3: wide RC (1024-d) vs RC | 57.8 vs 54.0, **+3.8 \*** |
| Promotion | S1: last frozen version vs RC | 50.2 vs 54.0, **−3.8 \*** |
| Promotion | S3: bank of all 4 versions vs MIX_half | 60.0, **+2.7 \*** (costs ~3.7 MB) |
| Protection | EWC (λ = 10) | +0.7 and +0.1, null; λ = 1,000 diverged |
| Protection | M2c first port | 26.3, **−26.1 \*** (logits up to 5e5) |
| Protection | M2c faithful (past-task items only) | 39.7, **−12.8 \*** |
| Epochs | 3 passes per task | ER +0.8 (n.s.); the MIX gain +3.3 \* survives |

## What we inferred
- At small memory, statistics beat images per byte. At larger memory, a bigger buffer catches up.
- Wider random features help.
- Keeping all versions helps but is expensive.
- **Protection mechanisms fail at this scale.** M2c was removed by the audit's stopping rule.

## Why we think it happened
- **M2c:** never letting old margins fall while new classes are learned pushes the logit scale up, and logits inflated to 300–1,400. (The audit later noted this is not proved: the zero step is always feasible, so it could stall instead. Supported conclusion: this implementation failed in these settings.)
- **The first M2c port was our error.** It protected the current task's items too, because the reservoir fills from the first batch. It forbade margin decreases on the task being learned. The faithful variant was added *before* its result was seen, and it still lost by 12.8.
- **Statistics vs images:** a class mean and a shared covariance summarise every item seen, at a fixed cost. A buffer image is one sample. At small budgets the summary wins; as the buffer grows, its samples start to cover the distribution.

## Caveats
- M2c used toy-best settings, untested at scale, with SGD 0.03 instead of the toy's Adam.
- An untested hypothesis: float32 rounds softmax to exactly 1 once a logit gap exceeds ~16.6, which may add precision effects to the late M2c dynamics.
- The L-BFGS-B solver was swapped for NNLS mid-round after profiling, validated to 2e-15, and declared before any M2c result.
