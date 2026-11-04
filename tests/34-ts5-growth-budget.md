# 34. T-S5: growth budget on Split CIFAR-100 (keep frozen versions, or spend the memory on a bigger buffer?)

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{ts5_plan.md, results_ts5.md, ts5_check.py, analyze_ts5.py}`, budget `results/ts5/budget.json`, results `results/ts5/`.

## Why we ran it
The sleep paper's second idea is growth: add capacity over time. Our version: keep frozen copies of the learner ("versions") as experts. On a longer stream (20 tasks), does keeping versions beat spending the same memory on more buffer images, and which versions should be kept?

## What we did
Split CIFAR-100, 20 tasks of 5 classes, M = 2,000, lr 0.03 (not retuned), 5 seeds. Arms: ER, RC-LDA, MIX_half, BANK_all (+19 versions, +21 MB), BANK_4spread (+4 versions spread over the stream, +4.4 MB), BANK_4last (the 4 newest), MIX with memory-matched ER (M = 8,836), Sleep+bc.

## What we got

| Arm | Final (100 classes) |
|---|---|
| ER raw / +bc | 17.4 / 22.7 |
| RC-LDA | 28.2 |
| MIX_half(ER, RC) | 29.7 |
| BANK_all | 31.0 |
| BANK_4spread | **31.2** |
| BANK_4last | 28.0 |
| MIX_half(memory-matched ER, RC) | 30.1 |
| Sleep+bc | 14.8 |

| | Comparison | Result | Prediction |
|---|---|---|---|
| G1 | BANK_all vs MIX_half | +1.4 (4/5), ns | not shown |
| G2 | BANK_4last vs BANK_all | **−3.1 \*** | **wrong** |
| G3 | BANK_4spread vs BANK_all | +0.2, null | as predicted |
| G4 | BANK_all vs MIX(memory-matched ER, RC) | +0.9 (4/5), ns | not contradicted |
| G5 | Sleep+bc vs ER+bc | **−7.9 \*** | confirmed |

Each version's own LDA accuracy, by task frozen: 26.0, 27.8, 29.4, 29.1, 26.3, 24.9 (tasks 1, 4, 8, 12, 16, 19); RC 28.2.

## What we inferred (first), then what survived
- Which versions are kept matters more than how many: 4 spread versions ≈ all 19; the 4 newest are harmful.
- First reading of the version curve: "online feature quality peaks mid-stream, then falls below random". **Withdrawn (test 35):** each number mixed encoder quality with an unmatched readout. Matched probes show the features keep improving.

## Why we think it occurred
A version frozen late gets statistics for earlier classes only from the buffer (~20 items each), while an early version saw ~480 per class on the stream. Late versions are therefore judged with far fewer samples, which looked like decaying features.

## How reliable this test is
- **Numbers reliable** (they were regenerated exactly after the files vanished). **The version-curve reading was wrong**, found by Lucky's audit. "Spread" uses knowledge of the stream length; a streaming rule for choosing versions is untested.
