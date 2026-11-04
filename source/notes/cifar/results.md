# CIFAR-10 CPU tests: results (2026-10-29)

The plan, settings and comparisons were pre-declared in plan.md, with amendments made before any evaluation run. The harness is cifar.py; analysis is in analyze_cifar.py, fu_mix.py and fu_bc.py. Raw outputs are in results/e1/*.npz, results/e0/*.log and results/e1/analysis.json.

## 0. Validity

- **Self-tests C1–C13 all pass** on the code that produced every number below (results/selftest_v2.log). They cover:
  - data and the stream partition, with the reserve excluded;
  - streaming LDA equal to batch two-pass statistics, and order-invariant;
  - LDA equal to the marathon lda_predict;
  - a planted separable case, probabilities and the ε for unseen classes;
  - causality via a label-flip test for ER, ACE, the frozen learner and the rc expert;
  - determinism;
  - the Bayes mixture against a direct loop, including a leak detector;
  - reservoir uniformity, the ACE gradient mask, balanced bias correction, and the accuracy metric.
- **An independent code review** found one real bug before any evaluation run. Probabilities were stored as float32 and underflowed to 0, which made the stream NLL infinite and the stream-mixture weights arbitrary. The fix is a float64 log-probability per item. The review also found weak checks, now strengthened; see plan.md.
- **Cross-check.** The E1 streaming RC-LDA expert must equal the E0 full-data LDA on the same random encoder, minus 50 reserve items. It does, to within 0.1 point on all three overlapping seeds: 55.2/52.0/54.1 vs 55.3/52.0/54.0. RP-LDA matches too: 43.3/43.8/43.5 vs 43.3/43.8/43.4.
- **Learning rate.** It was chosen on seed 100, a separate stream, using its test set. This is mild test reuse and is acknowledged. The value is 0.03 for both ER and ACE.
- **Scale and limits.** A small 3-block CNN, online single pass, M = 500, and 5 seeds on CPU. This is a real image benchmark, but far from a ResNet-18 multi-epoch setup.

## 1. E0: CIFAR-10 is a valid test of feature learning (MNIST was not)

Full-data LDA, test accuracy (%), seeds 0/1/2:

| Feature map | k=5 | k=100 | all |
|---|---|---|---|
| joint10 (trained CNN) | 49/48/46 | 69/69/69 | 75.7/75.5/75.4 |
| init (random CNN, 512-d) | 25/21/24 | 47/45/46 | 55.3/52.0/54.0 |
| init_wide (random CNN, 1024-d) | 24/22/24 | 47/47/47 | 56.7/57.8/58.4 |
| rp4096 | 22/20/21 | 32/32/32 | 50.3/50.2/50.4 |
| raw | 20/18/19 | 25/25/25 | 41.7/41.7/41.7 |

**The pre-declared criterion passes on every seed.** Learned features beat the best random map by **17–19 points** with full data, against a required 5. The joint head reaches 76.5 ± 0.2. The MNIST result that random features beat learned ones (94 vs 87) does not carry over to real images.

## 2. E1: Split CIFAR-10, 5 seeds

Final accuracy over all 10 classes and average incremental accuracy:

| Arm | Final | Avg incr |
|---|---|---|
| ER | 42.7 ± 1.2 | 61.7 |
| ER+bc_buf (spec default) | 52.6 ± 0.5 | 68.8 |
| ER+bc_resv | 51.7 ± 0.5 | 68.0 |
| ER bufLDA | 51.9 ± 0.5 | 68.7 |
| ACE | 50.8 ± 0.6 | 66.2 |
| ACE+bc_buf | 48.0 ± 0.9 | 65.3 |
| **RC-LDA** (stable, random conv) | 54.0 ± 0.5 | 68.7 |
| RP-LDA (stable, random pixels) | 43.6 ± 0.2 | 59.7 |
| FZ-LDA (learned on half of task 1, then frozen) | 54.3 ± 0.5 | 69.4 |
| MIX_stream (Bayes, stream loss) | 42.7 ± 1.2 | 61.7 |
| MIX_resv (Bayes, reserve) | 53.6 ± 0.6 | 69.4 |
| **MIX_half** (0.5 ER+bc_buf + 0.5 RC-LDA) | **57.9 ± 0.4** | **71.9** |
| JOINT1 (iid, one pass, half of ER's steps) | 49.6 ± 1.0 | – |

A star (\*) marks a pre-declared comparison that meets the significance rule.

| | Comparison | Result | Read |
|---|---|---|---|
| P1 | RC-LDA − ER+bc_buf | **+1.4 ± 0.9 (4/5), not significant** | A random-feature statistics expert ties tuned plastic ER here |
| P2 | MIX_half − best single expert | **+3.5 ± 0.4 (5/5) \*** | The two paths add |
| P2b | MIX_resv − best single | −0.7 ± 0.4 (0 up, 3 down, 2 ties) | Choosing weights from 50 reserve items fails |
| P3 | ACE − ER+bc_buf | **−1.8 ± 0.5 (5/5) \*** | |
| P4 | FZ-LDA − RC-LDA | +0.3 ± 0.8 (2 up, 3 down), null | |
| P5 | ER+bc_resv − ER+bc_buf | −0.9 ± 0.2 (5/5) \* | |
| P6 | ER bufLDA − ER+bc_buf | −0.7 ± 0.9, null | |
| – | ER+bc_buf − ER | +9.9 ± 1.6 (5/5) \* | |
| – | RC-LDA − RP-LDA | +10.5 ± 0.6 (5/5) \* | |

**The prediction made in advance is confirmed.** MIX_stream is exactly ER: the posterior weight on ER is 1.0000 at every final, on every seed.
- The log 2 regret bound holds on the stream on all seeds. But the stream loss measures the current task, where plastic ER wins: its stream NLL is 0.53 against 1.09 for RC-LDA.
- So the Bayes guarantee is real but about the wrong target. Weighted that way, the audit's mixture proposal throws away the stable expert entirely.

## 3. Mechanism follow-ups (saved outputs only, no new training)

**Where the mixture gain comes from (fu_mix.py).**
- On the final test set, ER+bc and RC-LDA are both right on 37.6% of items. ER+bc alone is right on 15.0%, RC alone on 16.4%. Either one is right on 69.1%.
- The error sets are strongly complementary.
- By task of origin: ER+bc runs from 62 (oldest task) down to 38 (newest), while RC-LDA is flat at 53–56. RC-LDA has no order effect, as theory requires. MIX scores 64/64/57/55/50.
- The oracle-best fixed weight is 0.4–0.5 on every seed, so 0.5 is close to optimal and not tuned. A post-hoc product of experts (logit average) gives 58.9. That was not pre-declared, so it is hypothesis-generating.

**Bias correction over-corrects the newest task on CIFAR (fu_bc.py).**
- After mean-logit subtraction, ER predicts the newest task for 10.9% of test items (the truth is 20%). Newest-task accuracy is 38, against 56 on old classes.
- This is **not** buffer memorisation. b_buf ≈ b_test (−0.20 on the newest classes), and correcting with the oracle test-set mean logit is no better (52.0).
- The form of the correction is wrong in size. Scaling it by α ≈ 0.7 gives 55.0 (an oracle sweep on test, diagnostic only).
- On MNIST and the toy, logit adjustment under-corrected relative to biascorr. Here biascorr itself over-corrects. Together these say that no fixed formula sets the correction size; it must be estimated from data.
- ACE+bc being worse than ACE fits the same story: ACE has less bias, so subtracting the mean logit over-corrects more.

## 4. What this changes

1. **The two-path design is supported on real images** (P2, 5/5). A stable statistics expert plus a plastic net, combined by a plain probability average, beats both: +3.5 over the better one and +5.3 over the spec's ER+bc default.
2. **The weighting rule must not come from stream log-loss.** This is now shown, not just argued: the exact Bayes mixture reduces to ER. Reserve-based weighting with 50 items is too noisy. A fixed prior weight works. A streaming-valid estimator of the weight (and of the bias-correction size α) with enough held-out items is the next thing to test.
3. **Random features are not a substitute for learning on CIFAR.** Full data gives a 17–19 point gap. Yet no online method here gets within 17 points of joint10 (75.5). The headroom is in features, and neither path captures it in a single online pass.
4. **Freezing after task 1 buys nothing** over random features: 2 classes and 5k images (P4).
5. **ER-ACE does not beat ER+bc_buf** in this setting (P3). That is consistent with the marathon's view that bias correction subsumes most of ACE's benefit, though here bc over-corrects (section 3).

## 5. Caveats (honest)

- **One architecture, one memory size, online only.** Multi-epoch-per-task regimes may change P1 and P3.
- **JOINT1 is not a clean upper bound.** It used half the gradient steps of ER, which is why it scores below ER+bc.
- **ER and ACE learning rates** were chosen on the seed-100 test set.
- **Five seeds** give at best p = 1/32 one-sided per comparison. Several comparisons are reported, so treat single borderline ones (P5) cautiously.
- **The α and w sweeps in section 3** use the test set and are diagnostic only.
