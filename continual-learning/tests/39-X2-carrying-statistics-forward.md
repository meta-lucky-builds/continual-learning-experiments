# 39. X2: can class statistics be carried into each new encoder online?

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{x2_plan.md, x2.py, analyze_x2.py, results_x2.md}`, `results/x2/`.

## Why we ran it
Test 35 showed the gap is statistics stranded in old coordinates: full-history statistics in final features give 35.3, the best online system 31.2. If a learner could move its stored class statistics into each new encoder, it would recover that gap without keeping old encoders.

## What we did
Replayed the T-S5 runs offline (CIFAR-100, 5 seeds) with only what a learner holds at each task end: its stored statistics, current buffer, and the previous and current encoder (the old one discarded after use). Transport maps for class means:
- **cv (control variate):** regress the new encoder's buffer features on the old encoder's after centring within each class (ridge, λ = 1e-3·tr/d), then μ_new = mean_buffer,new + A(μ_old − mean_buffer,old);
- **aff:** one shared affine map on raw buffer pairs;
- **shift:** per-class mean shift (SDC-like);
- chained (online, task by task) and direct (non-online) versions; ± transported covariance.

Gates: gR (reconstructed buffer reproduces every version's statistics exactly), gS (stream statistics match), gO (oracle arm equals saved probe to 1e-6), gB (T-S5 headline numbers reproduce).

## What we got

| Arm | Acc (100 classes) |
|---|---|
| Oracle full-stream statistics | 35.30 |
| Buffer only | 24.93 |
| **Chained cv transport** | **30.56** |
| Chained shared affine | 30.50 |
| Chained per-class shift | 26.87 |
| BANK_4spread (4.4 MB) | 31.21 |
| **MIX3 (ER+bc, RC, transported means)** | **32.08** |

- cv − buffer only = **+5.63 \*** (predicted +2 to +5); closes 54% of the gap.
- Shared affine ≈ cv (−0.06, ns): **my prediction (affine worse) was wrong**; the MNIST failure (test 09) does not replicate.
- Shift − cv = −3.68 \*.
- MIX3 − BANK_4spread = +0.86 \*, with 4–8× fewer bytes. Fragile: seed 1 is +0.04.
- Means carry ~92% of the oracle gap; covariance transport adds +0.23 \*.

## What we inferred
The stranded-statistics problem is about half solvable online and cheaply. This is the first component that beats the frozen-version bank at fewer bytes.

## Why we think it occurred
On CIFAR the drift between consecutive encoders is close to linear in the region the buffer covers, so a map fitted on ~20 paired items per class carries the stored means reasonably well. Chaining accumulates error (direct affine reaches 64% of the gap vs 54% chained). The per-class shift uses only each class's few buffer items, so it is noisy.

## How reliable this test is
- **Independently reviewed: no result-changing bug.** Corrected after review: plan timestamps (my guesses), byte figures (d = 512 not 256), and MIX3 computed for the wrong arm (not the declared Q4 arm).
- Offline replay of saved runs, not yet run live inside a learner. Ridge λ fixed in advance, not tuned. Equal mixture weights.
