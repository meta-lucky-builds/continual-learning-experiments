# X1: how much of the readout bias is weight scale? (declared 2026-10-29 17:10 UTC, before any run)

Origin: geometry feed, SYNTHESIS.md D2. Sezer's equinorm result, Łapacz's NMC result, and Moreira's fixed-radius result all suggest that a large share of class-incremental readout bias is a norm profile. We have never tested how much of E1's bias-correction gain is just scale.

**Material.**
- The final ER-trained network of the T-S5 erp learner, from results/ts5_probe/enc_s.pt, seeds 0–4. Setup: CIFAR-100, 20 tasks, M = 2000, linear head.
- Readout-side only. No retraining.
- Fitting data: the reserve, i.e. the first 20 arrivals of each class, which were never trained on. Score: the CIFAR-100 test set, all 100 classes.

**Gate g1.** The loaded final network must reproduce its saved matched probe. The PF value for 'final' in results/ts5_probe/probe_s.json has to match to within 1e-6. A mismatch means a wrong network, and the run stops.

Note: the erp_*.npz files have vanished from results/ts5 a second time. They were present at about 16:00, when g2 passed. That is why g1 uses the probe JSON and not the stored logits.

## Arms (scores on test features F of the final network; W, b = the head)

| Arm | Scores |
|---|---|
| A raw | F Wᵀ + b |
| B no-bias | F Wᵀ |
| C½ τ-norm, τ = 0.5 | F (W_c / ‖W_c‖^0.5)ᵀ |
| C1 τ-norm, τ = 1 (cosine to head directions) | F (W_c / ‖W_c‖)ᵀ |
| D cosine-NMC | cos(F, reserve class means of L2-normalised features) |
| E Euclid-NMC | −‖F − reserve class mean‖² |
| R LDA on reserve | C.StreamLDA on 20 per class (reference) |

Each of A, B, C½, C1 and D is also scored "+bc". That subtracts the balanced mean score vector over the reserve, computed in that arm's own score space. This is the same estimator as bc_resv in cifar.summarize.

**Diagnostics.**
- max/min column norm of W;
- Spearman correlation of ‖W_c‖ and of b_c with the class's task index;
- spread of pairwise cosines of W.

## Declared readings (5 seeds; star = |mean| > 2·SE and 5/5 signs)

**Primary: the scale share.** S = (C1 − A) / (A+bc − A), seed mean.
- S ≥ 0.7, with C1 not starred-lower than A+bc by more than 2 points → "most of the bias is scale". A scale-free head is then the default.
- S ≤ 0.3 → "scale is a minor share". The fitted correction stays essential.
- In between → mixed.

**Secondary.**
- (C1+bc) − C1: does a fitted correction still add on top of the scale fix?
- B − A: how much of the bias lives in b.
- D vs E: does normalising features matter for NMC?
- Best scale-free arm vs A+bc: is anything better than the fitted correction?

**Prediction (mine, written before the run).** S lies between 0.4 and 0.8. The bias b carries part of the bias. C1+bc stays starred above C1, because recency also shows up in angles. If I am wrong about any of this, I report it as stated.

## Planted self-test (must pass before the real run)

The synthetic data are Gaussian classes in d = 64, with means of equal norm. True head W = μ_c · g_c, where the gain g_c grows with the "task", and b = 0.

Checks:
- P1: C1 equals argmax of the cosine to μ_c, exactly.
- P2: on this data, C1's accuracy is not below A's, and A's accuracy falls as the gain spread grows.
- P3: bc matches C.balanced_mean_logit to 1e-12.
- P4: accuracy matches C.acc_seen.
- P5: with all g_c = 1, A = B = C1 exactly.

**Output.** results/x1/x1_s.json, with a summary in results/x1/summary.txt. Results go into notes/cifar/results_x1.md.


**Timestamp correction (17:37 UTC).** The "17:10 UTC" in the header was my estimate, and it was wrong. The file's modification time is 17:05:44 UTC. The first result was written at 17:07:16, after the plan, so the order was plan, then code (17:06:42), then results.
