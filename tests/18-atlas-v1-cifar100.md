# 18. Atlas first tests on CIFAR-100 (notebook v1, run by Lucky on a Colab T4)

**When:** 2026-10-30. **Files:** `notebooks/atlas_v2_tests.ipynb`, `src/v1/`. The full write-up is `RESULTS-v1.md` (project files).

## Why we did it
ARCH-v2 rests on two untested ideas:
- the migration estimator: per-class GREG with auxiliaries [stable chart r, previous chart];
- using a held-out **reserve** as the bridge, because buffer items were trained on and might be biased.

Its first tests were:
- **Test 1:** measure within-class R² of learned features on r and on the previous version.
- **Test 2:** an estimator shoot-out against X2's chained map, buffer-only means and the oracle.

The original T-S5 runs were not available, so the notebook regenerates fresh runs.

## What we did
- Re-implemented learner: ~107k-parameter 3-block GroupNorm CNN; ER with reservoir buffer; SGD lr 0.03, no momentum; batch 10 new + 10 replay; one online pass; 20 tasks × 5 classes; 5 seeds.
- Stable chart r: fixed random 5×5 conv (256 maps) + ReLU + 2×2 pooling = 1024-d.
- **Config A:** buffer 2000 + reserve 20/class (10 bridge, 10 control).
- **Config B (equal images):** buffer 1000 + reserve 10/class.
- Every arm scored by the **same** LDA with the **same** pooled covariance, so differences come only from the means.
- Hypotheses, the significance rule (|mean| > 2·SE and 5/5 seeds same sign) and predictions were written into the notebook before any result.

## What we got (config A unless noted)
| Arm | Acc |
|---|---|
| RC_LDA (random chart, exact statistics) | **35.1** |
| O_full (learned final encoder, oracle means) | 25.1 |
| GR_multi_buf | 22.8 |
| GR_prev_buf | 22.3 |
| B_only_buf | 20.1 |
| CH_cv_buf (X2-style) | 18.0 |
| ER head | 8.1 |

R²: f_T on r 0.673; f_T on f_{T-1} 0.953; [r, f_{T-1}] 0.956; per-hop mean 0.955; f_T on f_1 0.675.

| Pre-registered | Result |
|---|---|
| H1 GR_multi_res > CH_cv_buf | +2.40, 5/5, supported |
| H2 reserve bridge > buffer bridge | **−2.38, 0/5, reversed** |
| H3 per-class GREG > global map, same bridge | +2.76, 5/5, supported |
| H4 equal bytes, B > A | **−3.82, 0/5, reversed** |
| S1 X2 replication: CH_cv_buf > buffer-only | **−2.07, 0/5, reversed** |
| S2 r helps over reserve-only | +1.66, 5/5, supported |

## What we inferred
1. **Random features beat everything learned** (35.1 vs 25.1 even with perfect means). In ewt it was the reverse (28.2 vs 35.3). Migration is not the bottleneck in this run; the learner is.
2. **Per-class regression beats a global map** by ~4.3 points (22.3 vs 18.0, same buffer bridge). This is the clearest positive.
3. **The reserve-as-bridge choice failed.** The buffer, with more items per class, was better, and no training bias was visible.
4. **r as an auxiliary adds little** once the previous chart is used (+0.5), and hurts at small n (config B).
5. **Drift between neighbouring tasks is close to linear** (R² ≈ 0.95), but not across 19 tasks (0.68). Many small hops beat one big jump.
6. X2's +5.6 did not replicate here.

Our prediction scorecard: 3 right, 1 right in size but wrong in mechanism, 3 wrong.

## Why we think it happened
- **Weak learner:** the ER head reached 8.1% where ewt's reached 17.4%. Plain SGD at lr 0.03 with no momentum and one step per batch of 10 under-trains a CNN in a single pass. ewt's exact recipe is unknown. Our random chart is also stronger than ewt's (35.1 vs 28.2).
- **Reserve lost on variance, not bias:** reserve 10/class vs buffer ~20/class (hundreds early on). Buffer-only error 5.86 vs reserve-only 10.99 is almost exactly the 2× expected from half the sample size.
- **Global map loses to per-class GREG:** a single ridge map applied to all carried means shrinks between-class spread, pulling means together. Per-class bridge means anchor each class.
- **r adds little:** the previous chart already explains ~95% of within-class variance. Choosing r's penalty by CV with 500–1000 items is noisy and overfits.

## A measurement warning found here
Mean Mahalanobis error and accuracy disagreed: CH_cv_buf had lower error than buffer-only but lower accuracy. Average error does not capture whether class means keep their separation. Decisions should use accuracy or a margin-aware error.

## Caveats
- Fresh runs of our re-implementation, not ewt's.
- Every learned-arm ranking is conditional on a weak encoder.
- Config B's comparisons confound reserve size with buffer size.
- The random chart's exact covariance is itself ~4.6 MB, so "equal bytes" is only equal in images.
