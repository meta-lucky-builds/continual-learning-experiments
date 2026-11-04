# 14. X3: waking dormant class statistics with a few key items (CIFAR-100)

**Source:** ewt.md §17. Independent code review: no number-changing bug, but three readings corrected.

## Why we did it
Lucky's dormancy idea: if an old encoder is gone and only its class statistics remain, can we wake them? The method is to map today's features *back* into the old coordinates using a few stored "key" items per class. If that works, old statistics can sleep cheaply and be woken on demand.

## What we did
Map back with k = 1, 5 or all keys per class. Score 5-way accuracy within each task's classes (task-aware), averaged over tasks 1–19.

## What we got
| Arm | Acc |
|---|---|
| OLD: old encoder with its own statistics | 78.94 |
| REACT_all / 5 / 1: old statistics, features mapped back | 77.25 / 75.34 / 66.60 |
| TRANS: X2's transported means | 74.49 |
| BUF_all / 5 / 1: keys alone | 67.80 / 57.76 / 35.35 |

- Old statistics add a lot over the keys alone: +31.2 \* (k = 1), +17.6 \* (k = 5), +9.5 \* (all).
- Recovery is 98% / 95% / 84% of OLD. The oldest tasks recover least.

## What we inferred
Dormancy defined as "recoverable from a few keys" works and is measurable. As a memory design it is not cheaper than keeping encoders, except at k = 1.

## Why we think it happened, and the corrections
- **R3 corrected:** "mapping back beats transport" (+2.8) came partly from a covariance mismatch (5-class vs 100-class). Matched, the lead is +0.9 \*, and on the oldest tasks transport wins.
- **"OLD is the ceiling" was false:** the final encoder with full statistics beats it by +3.1 \*.
- **Bytes corrected:** keys cost 0.11 / 0.56 / 2.24 MB per version, against the 0.375 MB encoder body. Only k = 1 is cheaper.

## Caveats
- The measure is task-aware 5-way, so it is not comparable with 100-class numbers.
- Keys were chosen from the final buffer, using knowledge of which items survive.
- The plan file was saved 19 s before the code, so pre-declaration cannot be proven from the files.
