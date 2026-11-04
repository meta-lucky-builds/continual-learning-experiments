# 35. Audit A1: matched-data probes (do the features actually decay?)

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{audit_plan.md, results_audit.md, ts5_probe.py, ts5_probe_run.py, probe_selftest.py, analyze_audit100.py}`, outputs `results/ts5_probe/` and `results/audit/analysis100.txt`.

## Why we ran it
Lucky's external audit of ewt.md pointed out that T-S5's version readouts were not matched (test 34): a version frozen at task t had its old classes from the buffer only. The "features decay below random" claim might be a readout artefact. This test fitted every encoder's readout on the same data.

## What we did
For every saved encoder (v0–v18 and final, 5 seeds), fitted LDA on the same 480 stream items per class (and, separately, on 20 per class), and scored on the test set. Also the task-0 classes alone, 5-way. Gates: g1, g2 (exact), g3 (exact), g4 (the probe reproduces RC's online accuracy exactly).

## What we got

| Encoder frozen at task | 0 | 4 | 8 | 12 | 16 | 18 | final | RC |
|---|---|---|---|---|---|---|---|---|
| Online readout (T-S5) | 26.0 | 28.5 | 29.5 | 28.7 | 26.3 | 24.9 | – | 28.2 |
| Matched, 480/class | 26.0 | 29.3 | 31.9 | 33.4 | 34.5 | 34.9 | 35.3 | 28.2 |
| Matched, 20/class | 17.7 | 20.6 | 22.8 | 23.8 | 24.6 | 24.7 | 25.1 | 18.2 |
| Task-0 classes only, 5-way | 78.4 | 80.5 | 81.4 | 82.6 | 82.3 | 82.3 | 82.3 | 78.0 |

- v18 − v8 (matched) = **+3.0 \***; v18 − RC = **+6.7 \***; 20/class: +2.0 \*; tasks 0–3 classes: +2.5 \*; task-0 classes: +0.9 (4/5, ns). My pre-declared prediction (v18 − v8 ≥ −1) held.

## What we inferred
- **The online features never got worse.** They improved for all 20 tasks, including on the first task's classes.
- The whole online decline was the readout: −10.0 points at v18, −2.4 at v8.
- **What decays is statistics stranded in old coordinates**: a new encoder has better features but almost no history. Full-history statistics in final features reach 35.3, vs 31.2 for the best online system and 22.7 for ER+bc.
- On CIFAR-100, 20 trained-on buffer items gave about the same LDA as 20 fresh items (24.9 vs 24.7), so here the gap is sample count, not replay bias (contrast test 21).

## Why we think it occurred
Replay keeps training the shared features on all classes, so they keep improving. But statistics collected in an old encoder's coordinates cannot be used in the new one, and the new one has only the buffer's ~20 items per class.

## How reliable this test is
- **High computationally** (exact gates). Not independently reviewed as code. Same 5 training runs as tests 34, 38–40.
