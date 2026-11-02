# 24. CIFAR-10 E2: choosing the correction size and mixture weight online

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{plan.md §E2, results_r2.md, analyze_e2.py}`, `results/e2/`.

## Why we ran it
E1 used a fixed weight of 0.5 and found the correction size should be ~0.7, both chosen with hindsight. A real learner must choose them online. 50 reserve items were too noisy in E1; this tested 200.

## What we did
Reserve of 20 per class (200 images), never trained on. At each task end, choose α̂ (correction size) and ŵ (mixture weight) by reserve NLL.

## What we got
| | Comparison | Result |
|---|---|---|
| Q1 | ER+bc(α̂) vs ER+bc(α = 1) | 54.7 vs 52.4: **+2.3 \*** |
| Q2 | MIX(ŵ, α̂) vs MIX_half | 58.7 vs 57.3: **+1.4 \*** |
| Q3 | product of experts vs probability average | **+1.0 \*** |
| Q4 | MIX(ŵ, α̂) vs better single expert | **+4.0 \*** |

Final α̂ 0.6–0.8 (matches the oracle sweep); ŵ 0.3–0.5.

## What we inferred
The two-path system is fully online and causal, and still wins. 200 held-out items (not 50) are enough to set both choices.

## Why we think it occurred
Reserve items were never trained on, so their loss is an unbiased estimate of balanced historical accuracy, unlike the stream loss or the buffer.

## How reliable this test is
- **High** (5/5 on all four). The reserve becomes part of the algorithm, so separate test data is still needed for final claims; the test set was kept separate.
