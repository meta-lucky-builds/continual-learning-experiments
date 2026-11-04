# 38. X1: is the readout bias a head weight-norm (scale) problem?

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{x1_plan.md, x1.py, results_x1.md}`, `results/x1/`.

## Why we ran it
The geometry literature feed (374 papers) suggested that much of recency bias is a weight-norm profile in the head (for example equal-norm or cosine heads fix long-tail bias). If true, a scale-free head could replace our fitted bias correction.

## What we did
Post-hoc readouts on the final network of the T-S5 ER learner (CIFAR-100, 20 tasks, M = 2,000, linear head), 5 seeds. Fitting on the 20-per-class reserve, scoring on the full test set. Arms: linear head; no bias; τ-normalisation 0.5 and 1 (equal norms); cosine NMC; Euclidean NMC; LDA on the reserve; each with and without held-out bias correction. Declared primary: scale share S = (C1 − A)/(A+bc − A).

## What we got

| Arm | Plain | + bias correction |
|---|---|---|
| A linear | 17.37 | 22.70 |
| C½ τ-norm 0.5 | 12.87 | **24.60** |
| C1 τ-norm 1 | 9.08 | 23.94 |
| D cosine NMC | 21.41 | 18.00 |
| R LDA on reserve | **25.43** | – |

- S = **−1.61** (every seed −1.1 to −2.0). Equal norms make accuracy *worse* by 8.3 \*. **My prediction (0.4–0.8) was wrong.**
- Head norms: oldest classes largest (1.99 task 0 → 1.14 task 19), Spearman −0.91.
- τ-norm 0.5 + bc beats A+bc by +1.90 \*. LDA on 20 reserve items beats A+bc by +2.7 \*.

## What we inferred
Scale is not the source of the bias in our replay learner; it points the other way. A fitted held-out correction stays necessary; fixing norms as well adds a little. The best readout is a class-statistics readout.

## Why we think it occurred
Replay keeps training old classes' head rows, so their weights keep growing. The recency bias lives in the weight–feature alignment, not the norms; removing norms takes away the old classes' one advantage. In the literature's long-tail settings the norm profile runs the other way.

## How reliable this test is
- Gate g1 passed (the network reproduces its saved probe exactly). One self-test bug (a check turned into a comment) was fixed before any real result was read.
- **Not independently reviewed.** Post-hoc heads only; a trained cosine head was not tested. τ only in {0.5, 1}.
