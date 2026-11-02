# Sharp-test marathon, wave 2

2026-10-28. Code: `notes/marathon/mar2.py`. It is mar.py with measurement-only additions; `python mar2.py selftest` runs S1–S11 and N1–N10. Results go to `results/W*`, `results/C*`.

**Rule.** Every wave-2 measurement uses its own RNG. N1 checks that switching all of them on leaves the training outputs bit-identical. Same significance rule as wave 1: the paired mean ± SE excludes 0, and 5/5 seeds share the sign.

## What wave 1 left open

| Finding | Evidence |
|---|---|
| Buffer readouts plateau well below the oracle | MNIST 71 vs 89 |
| The plateau is mostly sample size | F8: at 50 items/class, refit_buf is 89.5 vs oracle 92.3 |
| Stale stored statistics are catastrophic | about 20% at every memory size |
| Translation drift compensation only reaches buffer level | |

Three questions follow:
- Is the drift structured?
- Is the bias predictable from first principles?
- Do "features retain information" claims survive a floor control?

## Tests

Each test lists the hypothesis, the arms, and the kill condition. Tags: [N] novel, [R] radical, [A] asymmetric bet (cheap, large payoff if true).

**W1. What carries the buffer-vs-oracle gap: means or covariance?**
- Arms: LDA with
  - full means + buffer covariance;
  - buffer means + full covariance;
  - Euclidean full or buffer means;
  - cosine NCM on full means.
- H: the class means (5 samples in 64–256 dimensions) carry the gap.
- Kill: buffer means + full covariance ≈ the oracle.

**W2. Is old-class feature drift a shared affine map? [N]**
- At each task end, store the class means plus the features of the buffer (and of a 4000-item oracle subsample).
- At the end, predict each old class's current mean with:
  - the stale mean;
  - per-class translation (SDC);
  - global translation;
  - a buffer ridge-affine map (α ∈ {0.1, 1, 10}, pre-registered, all reported);
  - an oracle affine map, plus a **leave-class-out** oracle. An affine map can move ~10 means anywhere, so only the leave-class-out version is evidence of a shared transformation.
- Metrics:
  - drift explained, E = 1 − ‖pred − now‖² / ‖then − now‖²;
  - LDA accuracy with the predicted means and the current covariance (isolates the means).
- H: the leave-class-out affine map explains most of the drift (E > 0.9). If so, "store statistics + correct with a shared map" (spec M1) is possible in principle, and the bottleneck is estimating the map.
- Kill: leave-class-out E is no better than global translation.

**W3 / R1. Floor controls for "features retain information". [R][A]**
- The same oracle readouts (LDA and logistic refit on all data) on:
  - the current features;
  - the **initial random network's** features;
  - features frozen after task 1;
  - **raw inputs**;
  - a random ReLU expansion of the raw input (1024 dimensions on MNIST, 512 on the toy).
- The streaming-LDA statistics are exact and order-independent, so LDA on frozen features is itself a zero-forgetting continual learner (stores d² + Kd numbers, no buffer).
- H: trained features beat every floor.
- Asymmetric: if a floor matches or beats ER-trained features, then the wave-1 "features keep the information" result is about the input, not the learning. A frozen random expansion + streaming LDA would then be the baseline the spec must beat.
- Toy seed 0 already shows raw-input LDA 78% > trained features 65%. That is plausible: the toy is Gaussian with shared covariance, so raw LDA is Bayes-optimal.

**W7. Is interference predictable before training? [N]**
- At the start of each task, before any update, compute the similarity of each new class to each old class:
  - centred class-mean cosine in feature space (200 items, and the first batch only);
  - feature distance;
  - a raw-input cosine baseline.
- At task end: confusion rate old → new on the test set.
- H: feature similarity predicts which old class gets absorbed by which new class.
- Measures: within-task rank correlation, plus a binary "which of the two new classes takes more of old class y". The chance level is 50%.
- Spec use: rivals for the M2 margin constraints chosen by similarity to incoming classes. T6 showed that top-2 logit rivals miss 83–100% of the flips.
- Kill: no better than chance, or no better than raw-input similarity.

**R12. Is the readout bias exactly the log prior of the recent gradient stream? [R][A]**
- Post-hoc logit adjustment: logits − τ log π, where π is the class frequency in the training batches (whole history, or the last task only), τ ∈ {0.5, 1, 2}. τ = 1 is the pre-registered prediction; it has no fitted parameter.
- H: last-task π with τ = 1 recovers most of the bias-correction gain. If so, the bias has a closed-form first-principles explanation.
- Kill: LA(last, τ = 1) recovers less than half of the biascorr gain over ER.

**R13. End-of-stream consolidation ("sleep") on the buffer alone.**
- On a copy of the model: 200 Adam steps on the 50-item buffer, head-only vs full network.
- H: full-network consolidation beats head-only and the post-hoc readouts. That would mean the features can be re-aligned cheaply.
- Kill: full ≤ head-only, or both are below biascorr.

**W6. Repeated recurrence: cyclic stream 0–4 × 3 (15 tasks). [N]**
- Arms: ER and ER-ACE (start-insert), on toy A and MNIST.
- Per return:
  - retention at return start;
  - time to 90% of first-occurrence accuracy;
  - accuracy on all seen classes, raw and bias-corrected, at every task end.
- H (M4, savings): retention rises and the time to threshold falls with each cycle.
- Kill: no trend across cycles.

## Jobs

| Arm | Bundle and settings |
|---|---|
| ER | W1 + W2 + W3 + W7 + R12 + R13 + readouts, on toy A, toy B and MNIST |
| ER-ACE start-insert | same bundle, on toy A and MNIST |
| ER, 50 items/class, MNIST | W1 + W2 + R12 |
| ER and ER-ACE start-insert | W6, on toy A and MNIST |

## Review fixes (independent reviewer, 2026-10-28)

All wave-2 jobs were re-run on the fixed code; the first launch is archived in `results/w2_v1_superseded/` and is not used.

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | Critical | stats/drift crash or mislabel on streams that revisit a task | refused by assert (N12); W6 runs do not use them |
| 2 | Major | LOO for task-1 classes was fitted on one class | added `aff_oracle_allcls_loo`, fitted at time t on a subsample of all 10 classes minus k; E reported by task |
| 3 | Major | E is a squared ratio and an unweighted mean | added pooled E and the residual norm fraction; **LDA accuracy is the decisive W2 number** |
| 3b | Major | under ACE, the logit-adjustment π counted masked new items | added `last_rep` (replay items only); the τ = 1 prediction is tested on ER only |
| 4 | Minor | `stats2` alone produced nothing | it now implies `stats` |
| 5 | Minor | one RNG was shared across measurements | separate RNGs per measurement |
| 6 | Minor | floors assumed task 0 | it is now asserted |
| 7 | Minor | wrong comment | corrected |

Self-test additions:
- N11, planted: LOO really excludes class k, and the non-LOO fit absorbs a class-specific offset.
- N1c: ACE start-insert with every flag leaves accuracy and the full curve unchanged.
- N4 is relabelled "definitional", since it cannot fail.
