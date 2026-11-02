# Results of the first Atlas v2 tests: what they mean

Source: Lucky's Colab T4 run of `atlas_v2_tests.ipynb`, 2026-10-30. 2 configs × 5 seeds. CIFAR-100, 20 tasks, online.
These are fresh runs of my re-implementation, not ewt's T-S5 runs.

## 1. The headline nobody predicted: the random chart beats everything learned

| | Config A | ewt (T-S5) |
|---|---|---|
| RC_LDA (fixed random features, exact statistics) | **35.1** | 28.2 |
| O_full (learned final encoder, oracle means) | 25.1 | 35.3 |
| ER head (the learner's own softmax) | 8.1 | 17.4 |

- In ewt the learned encoder was 7 points **better** than random features. Here it is 10 points **worse**, even with perfect class means.
- So in this run, migration is not the bottleneck. The whole learned path is capped at 25.1, while the "stable reference" alone gets 35.1.
- **Most likely cause: my learner is weak, not that learning is useless.** The ER head reaches 8.1 where ewt's reached 17.4.
  - The setup is plain SGD, lr 0.03, no momentum, one gradient step per batch of 10, a single pass.
  - ewt's exact recipe is unknown, because its code was never uploaded.
- My random chart is also stronger than ewt's (35.1 vs 28.2). It uses 256 maps, 5×5 kernels and 2×2 pooling.
- **Consequence:** every estimator comparison below is about migrating a *weak* encoder. With a stronger learner, drift, R² and the size of the gaps could all change. None of the rankings below should be treated as final until the learner is at least at ewt's level (ER head ≥ 15, O_full > RC_LDA).
- This is not a bug that I can find. The asserts passed, O_full has zero error by construction, and RC_LDA and O_full use the same pipeline. It is a property of the learner I wrote.

## 2. The pre-registered tests, read honestly

| Test | Result | What it really says |
|---|---|---|
| **H1** GR_multi_res > CH_cv_buf | +2.40, 5/5, supported | True, but for the wrong reason. See H2: the gain comes from per-class regression, despite the reserve bridge, not because of it. |
| **H2** reserve bridge > buffer bridge | **−2.38, 0/5, reversed** | ARCH-v2's "reserve is the unbiased bridge" choice fails at this size. |
| **H3** per-class GREG > global map, same bridge | +2.76, 5/5, supported | The per-class correction is real. |
| **H4** equal bytes, B > A | **−3.82, 0/5, reversed** | Spending half the memory on a reserve costs far more than it buys. |
| **S1** CH_cv_buf > buffer-only (ewt X2: +5.6) | **−2.07, 0/5, reversed** | X2's result did not replicate with this learner. |
| **S2** r helps over reserve-only means | +1.66, 5/5, supported | The stable chart carries usable information. |

### The clearest positive finding: per-class regression beats a global map
- Same bridge (buffer), config A: GR_prev_buf 22.3 vs CH_cv_buf 18.0, about **+4.3**.
- The best arm, GR_multi_buf at 22.8, closes about 55% of the gap between buffer-only (20.1) and oracle (25.1).
- The per-class bridge mean anchors each class. A single global map applied to all carried means pulls them toward each other: ridge shrinkage compresses between-class spread.

### Why the reserve bridge lost: variance, not bias
- The reserve gives 10 items per class (5 in config B). The buffer gives about 20 per class at the end, and hundreds per class in early hops.
- Buffer-only error 5.86 vs reserve-only 10.99 is almost exactly the 2× you expect from half the sample size.
- The bias from the learner having trained on buffer items is not visible at all at this scale. Variance dominates.
- **Caveat:** this confounds n with bias. A matched-n test (10 buffer items vs 10 reserve items) would separate them. The practical answer for this budget is already clear: use the buffer.

### The stable chart as an auxiliary: real but small
- R²(f_T | r) = 0.67 in config A. That is higher than my 0.2–0.5 prediction, so I was wrong there.
- But [r, f_{T-1}] reaches 0.956 vs 0.953 for f_{T-1} alone. Once you have the previous chart, r adds almost nothing.
- In accuracy, GR_multi_buf beats GR_prev_buf by only about +0.5 (22.8 vs 22.3). That difference was not formally tested.
- In config B (smaller bridge), adding r **lowered** R² (0.910 vs 0.947), and GR_multi_res was worse than GR_prev_res (14.2 vs 16.0).
  - Choosing r's penalty by cross-validation is noisy with 500 to 1000 items, and it overfits.
  - That is a flaw in my estimator at small n.

### Drift is close to linear between neighbours
- Hop R² is 0.955 on average, with a minimum of 0.92.
- Across all 19 tasks it falls to 0.675.
- **Consequence:** small hops are nearly linear, so chaining works. A single long jump does not. This supports the "publish before R² drops too low" cadence idea.

### A measurement warning
- Mean error (err_all) and accuracy disagree. CH_cv_buf has lower error than buffer-only (3.8 vs 5.9) but lower accuracy (18.0 vs 20.1).
- Average Mahalanobis error does not capture whether class means keep their relative separation.
- **Use accuracy, or a margin-aware error, for decisions, not err_all.**

## 3. Scorecard of my predictions

| Prediction | Outcome |
|---|---|
| Hop R² ≥ 0.8 | right |
| H3 supported | right |
| H1 supported, +0.5 to +3 | right in size, wrong in mechanism |
| R²(f_T\|r) 0.2–0.5 | wrong (0.67) |
| S1 replicates X2 | wrong |
| H2 coin flip | reversed, decisively |
| H4 unlikely | reversed |

## 4. What this changes in ARCH-v2

1. **Keep:** Part D's per-class GREG, chained on the previous serving chart. This is the strongest effect measured.
2. **Change:** the bridge is the replay buffer, not the reserve. The reserve shrinks to a small control set for fitting scalars, or is dropped. Re-test bias with matched n.
3. **Demote:** r as a migration auxiliary. It adds about 0.5 at best and hurts at small n. Keep it only if it earns its place in a stronger-learner rerun.
4. **Promote, pending the stronger learner:** r as a readout expert. Right now it is the best component in the system. The ε-mixture / PoE of RC_LDA with the migrated learned readout was not measured, and it is the obvious next arm.
5. **Do not conclude yet** that learned features are worse than random ones. That contradicts ewt and is most likely my learner.

## 5. Next run (one notebook edit)

1. **Strengthen the learner.** Use momentum 0.9 and several gradient steps per incoming batch, with a small lr sweep on seed 0. Accept the setup only when ER head ≥ 15 and O_full > RC_LDA.
2. Rerun configs A and B with: GR_prev_buf, GR_multi_buf, CH_cv_buf, buffer-only, oracle, RC_LDA, plus a **PoE of RC_LDA + GR_prev_buf** with weights fitted on the control split.
3. Add a matched-n bridge test: 10 buffer items vs 10 reserve items per class.
4. Fix r's penalty choice for small n: a fixed κ, or nested CV.

## Appendix: raw numbers as reported

```
Config A acc: RC_LDA 35.110±0.159 | O_full 25.056±0.265 | GR_multi_buf 22.794±0.274 | GR_prev_buf 22.302±0.233 | GR_stable_buf 21.656±0.231
  GR_multi_res 20.410±0.270 | B_only_buf 20.080±0.193 | GR_prev_res 19.932±0.169 | GR_stable_res 18.996±0.262 | CH_cv_buf 18.014±0.298
  CH_cv_res 17.650±0.275 | B_only_res 17.334±0.276 | ER_head 8.110±0.734
Config A err_all: GR_multi_buf 3.225 | GR_prev_buf 3.566 | CH_cv_buf 3.825 | CH_cv_res 4.311 | GR_stable_buf 4.750 | B_only_buf 5.857
  GR_multi_res 7.439 | GR_prev_res 7.775 | GR_stable_res 10.225 | B_only_res 10.988
Config A R2: fT|r 0.673 | fT|f(T-1) 0.953 | fT|f(T-5) 0.940 | fT|f(T-19) 0.675 | fT|[r,f(T-1)] 0.956 | hop_mean 0.955 | hop_min 0.920
Config B acc: RC_LDA 35.188 | O_full 24.232 | GR_multi_buf 20.020 | GR_prev_buf 19.758 | GR_stable_buf 18.378 | CH_cv_buf 17.548
  CH_cv_res 17.184 | B_only_buf 16.782 | GR_prev_res 16.016 | GR_multi_res 14.190 | B_only_res 13.364 | GR_stable_res 12.530 | ER_head 8.954
Config B err_all: CH_cv_buf 4.075 | CH_cv_res 5.020 | GR_multi_buf 7.404 | GR_prev_buf 7.793 | GR_stable_buf 11.274 | B_only_buf 13.010
  GR_prev_res 15.184 | B_only_res 21.174 | GR_multi_res 23.362 | GR_stable_res 34.117
Config B R2: fT|r 0.380 | fT|f(T-1) 0.947 | fT|f(T-5) 0.921 | fT|f(T-19) 0.589 | fT|[r,f(T-1)] 0.910 | hop_mean 0.941 | hop_min 0.874
H1 +2.40±0.33 5/5 | H2 -2.38±0.08 0/5 | H3 +2.76±0.25 5/5 | H4 -3.82±0.42 0/5 | S1 -2.07±0.27 0/5 | S2 +1.66±0.10 5/5
```
