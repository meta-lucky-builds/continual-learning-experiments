# T-S5: growth budget on Split CIFAR-100, 20 tasks: results (2026-10-29)

Plan: ts5_plan.md, written before any CIFAR-100 run, with the memory budget fixed in results/ts5/budget.json.

**Gates.** After the change that makes the harness configurable for 100 classes, the CIFAR-10 self-tests still pass at the defaults: C1–C13, and cifar2 K1–K10. The CIFAR-10 streams and batches are bit-identical to before. ts5_check.py passes 11/11:
- data counts;
- 20 disjoint tasks of 5 classes on every seed;
- 20 reserve items per class, none of them in any batch;
- a smoke run of every arm.

In the analysis, every config matched its declaration, every arm saw the same stream, and the sleep arm was bit-identical to ER through task 2. There was no restart: the boot time is still 04:33 UTC. Analysis: analyze_ts5.py, output in results/ts5/analysis.txt.

**Setting.** 20 tasks of 5 classes, online, lr 0.03 (not retuned), M = 2,000, RESV = 20, 5 seeds. Final accuracy is over all 100 classes (chance is 1%).

| Arm | Final (%) |
|---|---|
| ER raw | 17.4 ± 0.4 |
| ER+bc | 22.7 ± 0.2 |
| RC-LDA (random-conv statistics) | 28.2 ± 0.4 |
| MIX_half(ER, RC) | 29.7 ± 0.2 |
| BANK_all (+19 versions, +21.0 MB) | 31.0 ± 0.4 |
| BANK_4last (+4 newest versions, +4.4 MB) | 28.0 ± 0.4 |
| **BANK_4spread** (versions from tasks 4/9/14/19, +4.4 MB) | **31.2 ± 0.3** |
| ER+bc at M = 8,836 (same total memory as BANK_all) | 25.9 ± 0.4 |
| MIX_half(ER M = 8,836, RC) | 30.1 ± 0.1 |
| Sleep (r16, buffer dreams)+bc | 14.8 ± 0.4 |
| MIX_half(Sleep, RC) | 28.2 ± 0.3 |

| | Comparison | Result | Prediction |
|---|---|---|---|
| G1 | BANK_all vs MIX_half | +1.4 ± 0.5 (4/5), **not significant** | Positive: not shown |
| G2 | BANK_4last vs BANK_all | **−3.1 ± 0.2 \*** (below MIX_half) | Keeps at least ⅔ of the gain: **wrong** |
| G3 | BANK_4spread vs BANK_all | +0.2 ± 0.2, null: a 4-version budget keeps all of it | Keeps at least ⅔: confirmed |
| G4 | BANK_all vs MIX_half(ER at matched memory, RC) | +0.9 ± 0.4 (4/5), **not significant** | Bigger buffer wins or ties: tie, confirmed |
| G5 | Sleep+bc vs ER+bc | **−7.9 ± 0.3 \*** | Negative: confirmed |

**Accuracy by number of versions kept (newest first):** 29.7 with 0, 28.5 with 1, 27.5 with 2, 28.0 with 4, 29.3 with 8, 30.3 with 12, 31.0 with all 19.

**Each frozen version on its own (LDA):** 26.0 for the task-1 version, rising to a peak of 29.5 around tasks 7–9, then falling to 24.9 for the task-19 version. RC is 28.2.

## What it means

1. **Growth does not clearly pay over 20 tasks.** Keeping all 19 frozen versions adds +1.4 (not significant). At the same memory, a bigger buffer in the same mixture does about as well: 30.1 vs 31.0, not significant.
2. **What you keep matters more than how much.**
   - Four spread-out versions do as well as all 19, at a fifth of the memory.
   - The four newest versions are *harmful*: below no versions at all.
3. **Learned features peak mid-stream and then decay.** A frozen encoder from around task 8 gives better statistics than random features (29.5 vs 28.2). By task 19 the encoder has specialised and is worse than random (24.9). This is the first direct sign that the plastic network's *features* degrade over a long stream, not only its readout. It matches round-2 S1 on CIFAR-10, where the last version was worse than random.
4. **Sleep-style consolidation fails again** (−7.9, 5/5), now over 19 sleeps. This replicates T1 at 4× the tasks.
5. **The statistics expert matters more as tasks grow.** RC alone (28.2) beats bias-corrected ER (22.7) by 5.5 points here; on CIFAR-10 it was +1.6.

## Caveats

- **Accuracies are low in absolute terms** (about 30% over 100 classes), with a small CNN, one pass and an untuned lr.
- **The budgeted subsets** (newest 4, spread 4) were pre-declared, but they are only two of many possible selection rules.
- **"Spread" picks versions using knowledge of the stream length** (tasks 4/9/14/19). A streaming rule would need to decide which versions to keep online.
- **Compute.** The bank runs took about 51 minutes and the sleep runs about 47, against about 9 for ER at M = 8,836, all under 4-way parallel load.
