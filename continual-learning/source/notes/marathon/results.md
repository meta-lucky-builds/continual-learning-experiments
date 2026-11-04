# Sharp-test marathon: results

2026-10-28. Settings:
- **toy A:** the v1.3 toy, SEP 0.6, 5 items per class, 4 replay;
- **toy B:** SEP 0.8, 20 items per class, 8 replay;
- **MNIST:** Split-MNIST, MLP 784-256-256-10, 5 items per class, 4 replay.

All numbers are 5 seeds, mean ± SE, paired by seed. **\*** marks the pre-registered rule: the paired mean ± SE excludes 0 **and** all 5 seeds share the sign. With n = 5, five same-sign pairs give an exact sign-test p of 1/32 one-sided and 1/16 two-sided [wording corrected 2026-10-29], so \* means "consistent", not "proven". Mean ± SE is not a 95% interval. After this many adaptive tests, all of it is development evidence.

Plans: [plan.md](plan.md) (wave 1) and [plan2.md](plan2.md) (wave 2). Code: `mar.py` and `mar2.py`. Summaries are printed by `analyze.py`, `analyze2.py` and `analyze_fu.py`.

## 0. Validity

**Harnesses.**
- `mar.py` passes S1–S11. These include exact reproduction of toy_cl3 ER and diag.py ER-ACE, null switches, planted readouts, snapshot bit-identity, and MNIST bounds.
- `mar2.py` passes S1–S11 plus N1–N13. These cover measurement-only checks (all wave-2 flags leave training bit-identical), planted affine and LOO maps, and logit-adjustment unit cases.
- Both harnesses had independent code reviews, and every finding was fixed before the runs counted.

**Invalid or superseded runs (not used anywhere below):**

| Runs | Status |
|---|---|
| `results/v1_superseded/` | T1 with an uncentred mean init, which put old-class accuracy at 0 at step 0 (reviewer CRITICAL); old MNIST T2 |
| `results/w2_v1_superseded/` | first wave-2 launch, before the review fixes |
| T6 runs relaunched after the container restart | launched by me **without the toy A settings** (they ran the saturated SEP 1.6 toy); archived in `v13/invalid_sep16/` and re-run correctly. A guard now checks that the ER seeds reproduce S1 |

**Known interpretive limits.**
- The toy is Gaussian classes with a shared covariance, so raw-input LDA is Bayes-optimal there. The toy cannot say anything about feature *learning* (see W3).
- All of this is MLP scale. Nothing here licenses a claim about CIFAR-scale continual learning.

## 0b. Revisions after the v1.3 deep audit (2026-10-29)

These revise findings 1–3 below. The originals are struck through in place. The new runs are in `results/A_*`, the code is `readout_checks.py` and `buffer_bias_check.py`, and the checks are in `mar2.py` N14–N16.

**R1. The bias is not a closed-form function of the prior (revises finding 1).**
- The spec's loss takes separate equal-weight means. In my harness that is `w = 4`, which induces a new/old per-class prior ratio of 4 (measured 3.9–4.0), not 16.
- Under that loss, logit adjustment by the loss-induced prior at τ = 1 **under-corrects badly**:
  - toy A: 53.2 vs biascorr 59.1 (−5.9 \*);
  - MNIST: 58.9 vs 71.4 (−12.5 \*).
- The oracle-best τ for that prior is 2.7–3.4 on toy A and ≥ 4 on MNIST (4 is the grid cap).
- Even under the plain loss, the best τ is 1.4–1.7 (toy) and 2.1–2.6 (MNIST), not 1.
- The audit offered a global logit-scale explanation for τ > 1. It is **not supported**: the 2-way calibration temperature among the newest classes is T_new ≈ 0.9–1.3.
- **Reading.** With a tiny buffer, up-weighting replay mostly fits the 40 stored items. It does not raise old-class logits on old-class *test* inputs, so the bias is larger than any loss-induced prior predicts.
- **Revised finding.** The bias's direction and order of magnitude follow from the stream prior, but its size does not. A data-driven correction (biascorr) is required, and the "zero-parameter mechanism" claim is withdrawn.
- Replay weighting `w = 4` raises raw accuracy (+4.3 \* toy, +1.9 \* MNIST), but not bias-corrected accuracy (+1.4 and +0.3, not significant).

**R2. Learned features are far more sample-efficient; fixed features win only with full-history statistics (revises finding 2).**

Same frozen ER checkpoints (guard: these reproduce T2_er exactly). LDA from k random training items per class, 10 draws per k, MNIST:

| k per class | trained (256-d) | untrained net (256-d) | raw | random ReLU 256 | random ReLU 1024 |
|---|---|---|---|---|---|
| 5 | **74.0** | 55.6 | 63.7 | 60.8 | 65.2 |
| 20 | **83.5** | 70.6 | 70.5 | 73.8 | 79.0 |
| 50 | 85.4 | 80.0 | 77.5 | 82.3 | **85.9** |
| all | 86.7 | 87.2 | 87.5 | 89.1 | **94.0** |

- At 5 per class, trained features beat every fixed alternative by 9–18 points (5/5 seeds). The gap closes by about 50 per class and reverses with full data.
- Toy A shows the same pattern: 54.4 trained vs 36–53 at k = 5, 68.4 vs 78.4 raw with full data.
- Width matters: random ReLU features give 89.1 at 256-d, 94.0 at 1024-d and 96.4 at 4096-d. At matched width 256, random features still beat trained ones with full data (+2.3, 5/5).
- **Revised finding.** The fixed-feature win comes from **exact full-history statistics**, not from better features. With few samples per class, the learned representation is much better. This is the audit's "historical sample size" thesis, confirmed directly.
- "Non-learning continual learner" was the wrong name: it is a *fixed-feature* continual learner. "These benchmarks cannot test continual feature learning" is also withdrawn. These settings *do* show a feature-learning benefit, in sample efficiency.

**R3. The replay buffer is a biased source of class statistics (revises finding 3).**
- Same checkpoints, MNIST, 5 items per class. Old-class LDA accuracy is **49.4 from the replayed buffer vs 73.8 from 5 fresh, never-replayed items** (+24.3, 5/5). On toy A it is 46.6 vs 53.8 (+7.2, 5/5).
- The buffer means of the old classes are about 2.3× farther from the true current mean than fresh-sample means: 0.38–0.52 vs 0.20–0.23 relative error on MNIST.
- The network has fitted the replayed items, so their features are atypical. Finding 3's "sample-size ceiling" was therefore partly **buffer memorisation**, not pure sample size. Raising the memory (F8) reduces both.
- **Design test** at equal memory (5 per class): keep r items per class stored but never replayed, and use their means.
  - On MNIST this lifts LDA over buffer-LDA by +5.8 to +7.3 \*.
  - But it **loses to biascorr on the full replay buffer** (−5.2 to −6.8 \*), because less replay hurts the network.
  - On the toy, no gain.
  - **Verdict:** the effect is real, but a reserve split is not a win at a 5-per-class budget.
- The implication is for the stable path: statistics should be accumulated **at arrival**, from un-replayed data, in fixed coordinates.

## 1. Headline findings

1. **Forgetting here is mostly a readout bias.** ~~Its size follows, to first order, from the training-stream class prior.~~ [revised by R1: the direction and order of magnitude do, the size does not]
   - The same ER weights, read out with per-class logit bias correction, go 41.5 → 57.7 (toy A), 79.9 → 83.4 (toy B) and 50.7 → 71.2 (MNIST), all \*.
   - A **closed-form logit adjustment with no fitted parameter** (subtract log π, π = class frequencies in the last task's training batches, τ = 1) recovers the bias-correction gain:
     - toy A: 83–120% (vs biascorr −0.5 ± 1.2, not significant);
     - toy B: 130–160%, i.e. it beats biascorr (+1.6 \*);
     - MNIST, 5/class: 66–80% (τ = 2 overshoots, +2.6 \*);
     - MNIST, 50/class: 89.4 vs ER 83.2.
   - Under ER-ACE, the theory predicts that only replay items create cross-class competition, and they are class-balanced, so there should be no bias. Observed: replay-only π gives +0.1 ± 0.1, and ER-ACE's biascorr gain is ≈ 0. The **wrong** π (all labels) hurts ACE by −4.9 \*, so the test can discriminate.
   - **First-principles reading.** Softmax cross-entropy fits the log-odds of the *training* distribution. In the last task, new classes are 16/20 of every batch and the old classes share 4/20, so the model learns logits offset by about log(π_new/π_old) ≈ log(0.4/0.025) ≈ 2.8 nats. That offset sits on the shared (common-mode) feature direction: F4 decomposes ER's MNIST gap into a common term of 4.3 out of 4.56. At test time the classes are balanced, so the offset is pure bias.
   - ~~τ_eff > 1 on MNIST means part of the gap is item-specific, not prior.~~ [withdrawn: τ > 1 has several possible causes. R1 rules out a global logit scale but does not identify the cause.]
   - *Known in the literature* (logit adjustment, Menon et al. 2021; LAS for online CL, Huang et al. 2023). What these tests add is that the mechanism is confirmed with zero free parameters, including ER-ACE's prediction.

2. **Floor control (W3).** ~~On Split-MNIST the network's learned features are no better than untrained ones, and a non-learning continual learner wins by more than 20 points.~~ [revised by R2: true only with full-data statistics; at 5 per class, learned features win by 9–18 points]

   Oracle LDA on all training data. The fixed-feature rows (untrained net, raw, random ReLU) are streaming-valid zero-forgetting continual learners. The **trained-feature rows are diagnostics only**: they need the old inputs re-encoded [corrected after the audit].

   | Features | MNIST | toy A |
   |---|---|---|
   | ER features after the stream | 86.7 ± 0.5 | 68.4 |
   | Initial random network | 87.2 ± 0.4 | 64.1 |
   | Raw pixels | 87.5 | 78.4 |
   | Features after task 1 | **75.2** (−11.5 \* vs now) | — |
   | Random ReLU expansion of the input (1024-d) | **94.0 ± 0.0** | 76.6 |

   Trained methods on MNIST, for comparison: ER 50.7, best post-hoc fix 71.2, ER-ACE 70.6, sleep 72.2.

   - The wave-1 "features keep the information" result (T3: probe_full 89% while the model gets 39% on old classes) is real, but it says the **input** keeps the information. On these tasks, learning adds little to linear separability.
   - Training on task 1 alone *destroys* separability for the future classes, from 87 (init) to 75: the first task's features collapse onto "0 vs 1". Later tasks partly restore it.
   - The toy is worse still: trained features (68) are below the raw input (78).
   - *Known in spirit* (SLDA, Hayes & Kanan 2020; RanPAC, McDonnell et al. 2023, both with pretrained backbones; GDumb-style baseline critiques). ~~For the spec it is decisive: these benchmarks cannot test continual feature learning.~~ [withdrawn, R2] Any claim of *full-data* superiority for learned features must be made on data where fixed features fail.

3. **Readout sample size, not features, sets the post-hoc ceiling (F8, W1).** [revised by R3: part of it is buffer memorisation]
   - Buffer readouts plateau at about 71 (MNIST, 5/class) vs an oracle of 89. At 50 per class, refit_buf reaches 89.5 vs the oracle's 92.3.
   - LDA mixing (W1) puts the gap in the **class means**, not the covariance. Full means + buffer covariance give 78.3; buffer means + full covariance give 55.5.
   - Five samples in 256 dimensions make an error-dominated mean, since the squared error scales as σ²d/n.

4. **Stored class statistics go stale catastrophically, and the drift is not a shared transformation on MNIST (F7, W2).**
   - Stale stored means give about 20% (chance is 10%) at every memory size, on every setting.
   - The old-class mean drift is 1.0–1.9× the mean's own norm.
   - Correcting with a single shared affine map fitted leave-class-out (oracle, all classes) gets close on the toy (65.8 vs oracle 68.4), but **not on MNIST** (68.3 vs 86.7, −18 \*).
   - Fitted on the buffer alone, the affine map is worse than plain buffer means at 5/class. It is marginally better at 50/class (90.5 vs about 89.6).
   - Per-class translation (SDC) gives 25.9 on MNIST.
   - **Reading.** Feature drift in a trained network is class-specific. Statistics only stay valid if (a) the features are frozen, or (b) they are re-estimated from current-weight data. This is H1 at its strongest.
   - The in-sample affine fit "explaining" 100% is an artefact of parameter count, which is why the leave-class-out arms exist.

5. **"Snapshot restore saves time on recurrence" was a readout artefact (T5 + FU-d).**
   - Under the raw readout, restoring the task-1 head rows looks like strong savings: start accuracy 67 vs 17 (toy), 87 vs 51 (MNIST).
   - Under a bias-corrected readout, the restored rows are **worse than keeping the current ones**: start 17.0 vs 42.3 (toy, −25.3 \*) and 20.0 vs 76.7 (MNIST, −56.7 \*). They are only slightly better than restoring the *initial* rows (+9.4 \* toy, +1.8 \* MNIST).
   - Old rows no longer fit drifted features (finding 4). Their apparent benefit came from their large "newest class" logits, which is finding 1 again.
   - A **full** snapshot restores task 1 perfectly but wipes everything else: all-seen bc accuracy 21% at return start.
   - ER's fast raw recovery (MNIST ttt ≈ 4 steps) is also mostly the bias re-flipping toward the current task.

6. **Repetition gives small but real savings (W6, cyclic 0–4 × 3).**
   - ER retention at return grows from the 2nd to the 3rd occurrence: +3.1 \* toy (bias-corrected +1.7 \*), +8.2 \* MNIST (bias-corrected +5.4 \*).
   - End-of-cycle accuracy: toy +3.5 \* by cycle 3, then plateaus after cycle 2. MNIST +4.5 \*, ER-ACE +5.3 \*.
   - The return speed cannot be read off ER's raw curves (finding 5).

## 2. The rest, test by test

**T1, boundary shock.** The centred mean init raises the dip minimum (+8.1 \* toy A, +12.1 \* MNIST) but never improves final accuracy. Warm-up hurts (−2.3 \* toy A, −1.3 \* toy B). **Verdict:** the step-20–40 collapse is not what drives final forgetting. Killed.

**T2, which part of the bias.**
- WA (row norms) does nothing (WA gamma ≈ 1). Bias correction, NCM and buffer refit all fix most of it.
- Every method plus its best buffer readout converges on the same ceiling, because the ceiling is the readout's sample size (finding 3).
- **Verdict:** the bias is a common-mode offset, not norm or direction.

**T3, features vs readout.** Confirmed as measured. Reinterpreted by W3 (finding 2).

**T4, fresh vs stale Fisher (H1).**
- Toy A: fresh > stale at λ = 0.1, 1 and 10 (+3.1 \*, +3.9 \*, +4.9 \*), and fresh − ER = +7.1 \* at λ = 1.
- MNIST: fresh > stale at λ = 0.1 (+6.5 \*), but fresh is **much worse** at λ ≥ 10 (−10.0 \*, −13.8 \*). Stale EWC at λ = 100 gives +5.2 \* over ER.
- **Reading.** A fresh Fisher measured on the buffer at the current weights is large exactly where the current task is also learning, so a strong fresh penalty blocks plasticity. A stale Fisher spreads its penalty on old-task directions.
- **Verdict:** H1 is supported at small λ only. "Fresh beats stale" is not unconditional.

**T6, M2c rival coverage (toy A, zero-slack M2c).** The ER seeds reproduce S1 exactly, so the setting is verified.

| Arm | Test accuracy | Held-out old-class accuracy at task end | Flips per task |
|---|---|---|---|
| ER | 41.5 ± 1.9 | 33.1 | 73 |
| M2c, top-2 rivals | 44.6 ± 1.5 (+3.1 \* vs ER) | 38.5 | 67 |
| **M2c + the spec's "labels born since the last refresh" rivals** | **54.1 ± 2.2** (+9.5 \* vs top-2, +12.5 \* vs ER) | 52.5 | 54 |

- **Why it works.** At refresh times the new classes are already top-2 rivals 90% of the time. But the collapse starts 20–40 steps into a task, before the first refresh (every 25 steps), and in that window only the born-label rule constrains the new rivals.
- **Verdict:** the spec's rival rule is necessary for M2c, and the kill condition is not met.
- **Honest comparison.** M2c + new rivals costs about 30× ER compute, and it is still **below** the zero-cost readout fixes on the same toy: biascorr 57.7, logit adjustment 57.2, ER-ACE 57.8. Whether M2c stacks with logit adjustment is untested.

**T7, context detection (M3).**
- With real (LDA) label evidence, detection takes 1–4 batches with no false switches. The null control (q = 0) never switches.
- In the noisy-evidence scan, detection succeeds below the two-state L_min: 11/20 at mean log L 0.126 with 2 states, 17/20 with 3 states.
- **Verdict:** L_min is a constant-evidence boundary, **not a necessary condition** under noisy evidence. The spec wording must change.

**W7, is interference predictable before training.**
- The centred feature-mean similarity predicts which of the two new classes absorbs an old class: 63–72% (chance 50) \* in most settings.
- But **raw-input similarity is as good or better**: 60–74%, and 74 vs 66 on MNIST.
- **Verdict:** interference is moderately predictable, and the learned features add nothing. Similarity-chosen rivals (for M2) are worth at most a modest gain. Not a big insight.

**R13, end-of-stream "sleep" (200 steps on the buffer alone, full network).**
- ≈ biascorr on ER (toy A +1.1, MNIST +1.1, both ns).
- Beats it under ER-ACE (+2.8 \* MNIST) and on toy B (+1.4 \*).
- Full beats head-only (+4.6 \* toy A, +11.3 \* MNIST).
- **Verdict:** a modest win, mostly the same debiasing.

## 3. What this means for the spec

1. **The readout should be closed-form or debiased.** Use either logit adjustment by the stream prior (zero parameters; it is the right model for ER) or class-balanced statistics. Readout protection machinery (M2c margins on head rows) is aimed at a symptom that a one-line correction removes.
2. **Stored statistics (M1) are valid only with frozen or re-estimated features.** With learned, drifting features, they must be re-estimated at current weights. That is H1, and the gap is sample size (F8).
3. **M4 snapshots of head rows are invalid under feature drift.** Snapshots must be paired with the feature state they were taken in, or not used. The earlier "restore saves time" evidence is withdrawn.
4. **T7:** rewrite L_min as a constant-evidence boundary. Under noisy evidence it is neither necessary nor sufficient. Multi-state helps.
5. **The real question is now sharp:** can a continual learner beat *frozen features + exact streaming statistics* on data where frozen features are poor? On MNIST and the toy, that baseline wins outright (94.0 vs at most 72.2). Anything the spec adds must be shown where feature learning is needed, against this baseline and against a pretrained-feature version.

## 4. Next tests, by value

1. **The decisive one:** split CIFAR-100 (or CIFAR-10 on CPU at reduced size). Compare (a) random-feature streaming LDA, (b) ER + logit adjustment, (c) ER-ACE, (d) a two-timescale design: a slow feature path whose statistics stay valid, plus a learned part. Learning must beat (a) there or the spec has no case.
2. **Drift-aware statistics:** re-estimate class means at current weights from the buffer and from a cheap generative or feature cache. Measure how many items per class reach the oracle; F8 suggests about 50.
3. **Why τ_eff > 1 on MNIST:** measure the item-specific part of the gap under logit adjustment (F6-style decomposition after LA).
