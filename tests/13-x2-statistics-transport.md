# 13. X2: carrying class statistics into new encoders (CIFAR-100)

**Source:** ewt.md §16. Plan `x2_plan.md`; results `results_x2.md`; independent code review found no result-changing bug.

## Why we did it
A1 ([11](11-audit-corrections.md)) showed the gap is statistics stranded in old coordinates: oracle statistics in the final encoder reach 35.3, the buffer alone ~24.9. Can a learner that keeps only class statistics plus its buffer **move** the statistics into each new encoder as it goes?

## What we did
At each task end:
1. Regress the new encoder's buffer features on the old encoder's, after within-class centring (the "control variate", CV).
2. Move every stored class mean with that map.
3. Discard the old encoder.

This was replayed offline on the saved T-S5 runs. Gates: the reconstructed buffer reproduces saved statistics exactly; the oracle matches the saved probe to 1e-6; the T-S5 headline numbers reproduce.

## What we got
| Arm | Acc (100 classes) |
|---|---|
| O_full: oracle statistics in final features | 35.30 |
| B_only: final buffer only | 24.93 |
| **CH_cv: chained transport** | **30.56** |
| CH_aff: shared affine | 30.50 |
| CH_shift: per-class mean shift (SDC-like) | 26.87 |
| BANK_4spread (4.4 MB) | 31.21 |
| **MIX3_CH_cv** (ER+bc, RC, transported means) | **32.08** |

- CH_cv − B_only = **+5.63 \***, closing 54% of the gap.
- A shared affine map ≈ CV, so the MNIST affine failure did not replicate.
- MIX3 beats the bank by +0.86 \* with 4–8× fewer bytes, but this is **fragile** (seed 1 only +0.04).
- About 92% of the gap is in the means, not the covariance.

## What we inferred
The first component that beats the version bank with fewer bytes. Stranded statistics can be carried, imperfectly, by a linear map per hop.

## Why we think it happened
Consecutive encoders are close (we later measured within-class R² ≈ 0.95 per hop in [18](18-atlas-v1-cifar100.md)), so a linear map fits well. A mean shift alone fails because the change between encoders includes rotation and scaling, not just translation.

## Caveats
- Replayed offline on saved runs, not live inside a learner.
- All recent tests (A1, X1–X3) reuse the same 5 training runs.
- **Our own fresh re-implementation did not replicate the +5.6** ([18](18-atlas-v1-cifar100.md): −2.1). The likely reason is a different (weaker) learner, but X2's size is not robust.
