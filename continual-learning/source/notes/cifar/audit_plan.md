# Tests from the ewt.md audit (pre-declared 2026-10-29 13:45 UTC, before any of these runs)

Source: Lucky's external audit, continual_learner_ewt_audit/EVIDENCE_AUDIT_AND_SYNTHESIS.md. The tests below answer the three claims in it that can be settled by experiment on our harness.

**Code facts checked before writing this plan:**
- In cifar2.py `promote=1`, a version frozen at the end of task t starts its streaming LDA from the **buffer's** features at promotion, then adds every later stream batch.
  - So the version from task 18 has its 95 old classes from about 21 buffer items each. Those items were also trained on.
  - The version from task 0 has 95 of 100 classes from the full stream (480 per class).
  - The audit's coverage confound is therefore real: the T-S5 "versions alone" curve mixes encoder quality with readout data.
- The CIFAR head is `nn.Linear`, not the spec's cosine head (10·cos). The M2c port was faithful in its pool only.
- The paired rule in analyze_cifar.py is |mean| > **2**·SE and 5/5 signs. ewt.md described it as "mean ± SE excludes 0", which is wrong. The 2·SE rule is not redundant with 5/5; the 1·SE rule would be.
- Sleep's student is S + Δ. At rank 0 (full) Δ covers every parameter, and the run has no other reset: ER uses plain SGD with no optimiser state. So an exact transfer (Δ = teacher − S) should reproduce ER up to float rounding.
- **The T-S5 erp_{0..4}.npz files are no longer on disk.** analysis.txt (12:28 UTC) was computed from them. The container also restarted around 13:35. I don't know why the files are gone. They are regenerated below, with gates against analysis.txt.

## A1. Matched-data snapshot probes (CIFAR-100, T-S5 setting)

**Re-run.** The erp learner (MEM = 2000, mem = 2000, promote = 1) is re-run for seeds 0–4 with the T-S5 command. This regenerates results/ts5/erp_s.npz. Each frozen encoder, the LDA state of each version and the final learner are saved to results/ts5_probe/enc_s.pt.

**Gates. Any failure means no results.**
- g1: the new erp preq and ylearn equal the slp arm's bit for bit through task 2, and ylearn equals rc's.
- g2: recomputing the analysis.txt numbers from the regenerated npz gives them to 0.05 points: ER, ER+bc, MIX_half, BANK_all, BANK_4spread and versions alone.
- g3: the saved LDA states give exactly the logprobs in the npz for the final task.

**Probes.** These are offline diagnostics, not online results. Each is applied to every encoder: versions v0..v18, the final learner, and the RC encoder from the rc arm. Every probe fits LDA with the same estimator (C.StreamLDA, same shrinkage) on **the same items for every encoder**:
- **PF.** All non-reserve training items of all 100 classes (480 per class). Evaluated 100-way on the test set.
- **PS.** A fixed random 20 per class from the non-reserve training items (rng 777 + seed), identical for every encoder. Evaluated 100-way on test.
- **PO.** PF restricted to task-0's 5 classes: fit and evaluate 5-way on those classes' test items. Every version had already learned these classes, so a drop here is forgetting, not missing forward transfer.
- **PO4.** The same for tasks 0–3 (20 classes). Only versions v3 and later count for the forgetting reading.

**Pre-declared comparisons.** 5 seeds, using the rule as coded (|mean| > 2·SE and 5/5 signs).
- v8 is used because it was the observed online peak. That is a selection made after seeing the data, and it is declared here.

| | Comparison | Decision |
|---|---|---|
| D1 | PF(v18) − PF(v8) | Significantly negative: the late decline survives matched data, so feature decay is supported. Otherwise the online decline is **not** evidence of feature decay. |
| D2 | PF(v18) − PF(RC) | Significantly negative: below random under matched data. |
| D3 | PO(v18) − PO(v8) | The forgetting test on classes both versions learned. |
| D4 | PS(v18) − PS(v8) | The same question at a small, realistic sample size. |
| D5 | (online − PF) for v18 vs for v8 | Descriptive: how much of each version's online number is lost to its readout data. |

**My prediction, written down honestly:** D1 ≥ −1. That is, most of the online decline is readout coverage, not features. Confidence is low.

## A2. The four-version bank against its own memory-matched buffer

From budget.json, 4 versions cost 4 × 1,105,296 = 4,421,184 bytes, which is 1,439.2 images at 3,072 bytes each. So M = 2000 + 1439 = **3439**.
- Arm: er_m3439, ER at mem = 3439 on the T-S5 stream, seeds 0–4.
- **E1.** BANK_4spread vs MIX_half(ER M = 3439, RC).
- **E1b.** BANK_4spread vs ER+bc(M = 3439).
- Caveat: "spread" uses knowledge of the stream length, so E1 compares a horizon-advantaged bank. A causal retention rule is the next test, not this one.
- No prediction on direction.

## A3. Exact-transfer Sleep control (CIFAR-10, the T-S1 setting)

**Arm sleep_exact.** At every sleep, set Δ = teacher − S exactly (full rank, no optimisation), merge as usual, and log the maximum logit difference between merged net and teacher on buffer, reserve and test. Seeds 0–4, CIFAR-10, RESV = 20, M = 500.
- **Gate.** The maximum logit difference at each sleep is ≤ 1e-4.
- **H1.** Final ER+bc of sleep_exact vs ER+bc (E2). Predicted |diff| < 1 point and not significant. If so, the −8.4 of T1b comes entirely from approximate transfer (distillation on dreams), not from the reset or the architecture.
- Float rounding means S + (teacher − S) need not equal teacher bit for bit, and SGD is chaotic, so trajectories will diverge slightly. That is why the test compares final accuracy across seeds.

## Not run now (to be done next, or needs a decision)

- A causal version-retention rule (marginal ensemble benefit on the reserve), after A1 and A2.
- Margin-crossing diagnostics for M2c: low value, since M2c is out.
- Wording corrections to ewt.md from the audit's statistical and logical points. These are made directly.
