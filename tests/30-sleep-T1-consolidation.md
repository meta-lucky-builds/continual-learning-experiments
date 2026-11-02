# 30. Sleep T-S1: does sleep-style consolidation help?

**Date:** 2026-10-29. **Source paper:** Behrouz et al., "Language Models Need Sleep" (arXiv 2606.03979). **Source files:** `source/notes/cifar/{sleep_plan.md, results_sleep.md, sleep.py, sleep_selftest.py, sleep_z4_control.py, analyze_sleep.py}`, `results/sleep/`.

**Shared setup (tests 30–33):** an analogue, not a reproduction. At each task end, compress what the network learned since the last sleep into a new expert (rank-r or full update) by distilling the network's own outputs on "dreams", then reset the fast weights and continue. No language model, no self-generated inputs, no RL reward, no SEAL dreaming. Split CIFAR-10, RESV = 20, seeds 0–4, online, lr 0.03, M = 500. Distillation: SGD + momentum, dlr 0.03, 30 epochs (chosen on seed 100 reserve). Self-tests: 20 checks; two review rounds; every sleep arm bit-identical to ER through task 2.

## Why we ran it
Lucky asked whether the paper's sleep mechanism helps continual learning. Its core claim is that consolidating learned changes into a new expert during sleep fights forgetting.

## What we did
T1: Sleep (rank 16, buffer dreams)+bc vs ER+bc. T1b: full rank. T1c: MIX(Sleep, RC) vs the version bank.

## What we got
- T1: 44.0 vs 52.4, **−8.4 \***.
- T1b: **−8.4 \*** (full rank no better than rank 16).
- T1c: **−6.0 \***.

## What we inferred
Sleep-style self-distillation is not a forgetting remedy here; it loses accuracy. The loss is not compression (full rank loses equally).

## Why we think it occurred
Each sleep transfers the network's function only on what the dreams cover (500 buffer images); off those images the student disagrees with the teacher (test 31). Test 37 (A3) later proved this: copying the teacher *exactly* and then doing the same reset gives no loss at all.

## How reliable this test is
- **High that this analogue fails**; it does not refute the paper's main claim, which uses upward distillation and RL dreaming we did not test. Distillation hyperparameters came from one seed's 200 reserve items.
