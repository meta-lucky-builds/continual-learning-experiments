# CIFAR-10 CPU tests: plan and pre-declared criteria (written 2026-10-29, before any CIFAR result)

These tests answer the v1.3 deep audit's experiment plan (E0 and E1) at real-image scale. They run on a 4-core CPU with no GPU.

## Setting (fixed before running)

- **Data.** CIFAR-10 from the fast.ai S3 mirror (md5 7d7b892d003476893d94d030e4f7c0f0). Train 50,000 and test 10,000, at 5,000 and 1,000 per class. Pixels map to (x/255 − 0.5)/0.25, a fixed constant, so the scaling uses no data statistics.
- **Network.** A small CNN of three conv blocks (32, 64, 128 channels) with GroupNorm(4), ReLU, max-pool after blocks 1 and 2, adaptive average pool to 2×2, and a linear head. The features are 512-dimensional.
  - GroupNorm is chosen over BatchNorm so that no running statistics drift, and so that each sample's output does not depend on the rest of its batch. That makes the training forward pass a valid prequential prediction.
- **Stream.**
  - The class order is a per-seed permutation split into 5 tasks of 2 classes, with items shuffled within each task. It is a single pass (online).
  - Each step takes 10 stream items plus 10 replay items. Loss = mean CE(stream) + mean CE(replay), for ER and ER-ACE alike, so the two differ only in ACE's masking.
  - The reservoir buffer holds M = 500 images.
- **Reserve.**
  - The first 5 arriving items of every class are set aside on arrival. No learner trains on them and no statistics include them. All methods within a seed lose the same 50 items.
  - They serve as an unbiased, class-balanced held-out set for bias correction and for mixture weights. They count as extra memory: 50 images, 154 KB.
- **Statistics expert.**
  - A fixed encoder feeds streaming LDA: exact class sums, counts, and the raw second moment in float64. Covariance shrinkage is 0.1 toward (tr/d)·I, plus an absolute floor of 1e-6·I.
  - Prediction probabilities use equal priors over seen classes. Unseen classes share ε = 1e-3 of the mass.
- **Evaluation.**
  - At each task end, accuracy is measured on the test items of the classes seen so far. The argmax is restricted to the seen classes for every method.
  - Final accuracy covers all 10 classes. Average incremental accuracy is the mean over the 5 task ends.
- **Seeds.** Seeds 0–4 are the evaluation seeds. Seed 100 is used only for choosing the learning rate, from a grid of {0.01, 0.03, 0.1} picked separately for ER and ER-ACE by final ER+biascorr / ACE accuracy.
- **Significance.** The paired mean ± SE must exclude 0, with 5/5 seeds sharing the sign. With 5 seeds, that is the strongest a sign test can give: p = 1/32 one-sided.

## E0: is CIFAR-10 a testbed where learned features matter? (gap check)

LDA with full training statistics (the oracle, diagnostic only), plus k-shot sweeps (k = 5, 20, 100, 500), on these feature maps:

- **joint10.** The CNN trained jointly on all 50k for 10 epochs (SGD with momentum 0.9, lr 0.05 cosine, flip and crop).
- **init.** A random-init CNN, which is the same encoder as the E1 random-conv expert.
- **init-wide.** A random CNN with 64/128/256 channels and 1,024 features.
- **raw.** The 3,072 pixels.
- **rp.** A random ReLU expansion of the pixels, at widths 512, 2,048 and 4,096.

**Criterion, pre-declared.** CIFAR-10 counts as a valid test of continual *feature* learning at this scale only if LDA_all(joint10) − max over the random maps of LDA_all ≥ 5 points on every seed. If it fails, the E1 comparison says nothing about feature learning. Three seeds (0, 1, 2) is enough for a yes/no gap check.

## E1: Split CIFAR-10, the two-path question

| Arm | What it is |
|---|---|
| ER | Plastic reservoir ER |
| ER+bc_buf | Subtract the class-balanced mean logit on the buffer (spec B4.1 default) |
| ER+bc_resv | The same, with the mean logit taken on the reserve |
| ER bufLDA | LDA from the buffer items' current features |
| ACE, ACE+bc_buf | ER-ACE; incoming CE is masked to the classes present in the incoming batch |
| RC-LDA | Stable expert: random-conv encoder plus streaming LDA, with statistics taken at arrival |
| RP-LDA | Stable expert: rp512 plus streaming LDA |
| FZ-LDA | Learned-then-frozen: train with ER on the first half of task 1, freeze, then stream LDA from that point on |
| JOINT1 | IID single pass over the same items, batch 20, no replay (upper reference) |
| MIX_stream | Bayes mixture of ER and RC-LDA, with weights from the prequential stream log-loss |
| MIX_resv | Bayes mixture of ER+bc_buf and RC-LDA, with weights from the reserve log-likelihood at each task end |
| MIX_half | A fixed 0.5/0.5 probability average of ER+bc_buf and RC-LDA |

**Pre-declared comparisons.** All are paired over 5 seeds, on final accuracy (avg incremental accuracy reported alongside).

- **P1.** RC-LDA vs ER+bc_buf: does a stable random-feature expert beat plastic ER on real images, as it did on MNIST?
- **P2.** The best of MIX_resv/MIX_half vs the better single expert per seed. Do the two paths add?
- **P3.** ACE vs ER+bc_buf.
- **P4.** FZ-LDA vs RC-LDA: do features learned from 5k images beat random ones when both are frozen?
- **P5.** ER+bc_resv vs ER+bc_buf: buffer bias on CIFAR.
- **P6.** ER bufLDA vs ER+bc_buf.

**Prediction stated in advance.** MIX_stream will be dominated by ER, because the stream loss measures the *current-task* distribution and not balanced historical risk. Its log 2 guarantee is about the wrong target. This tests the audit's mixture proposal.

## Reliability gates (all must pass before any number counts)

| Gate | Check |
|---|---|
| C1 | Data shapes, counts and label range |
| C2 | Stream: each non-reserve item appears exactly once, tasks hold exactly their classes, the reserve is the first 5 per class, and the stream is deterministic per seed |
| C3 | Streaming LDA equals batch statistics; the result is order-invariant |
| C4 | Cholesky LDA argmax equals the marathon's inverse-based lda_predict |
| C5 | Planted separable Gaussians reach about 100%; probabilities sum to 1; the ε for unseen classes is exact |
| C6 | Causality: changing the labels of batch t leaves every prequential prediction up to and including t unchanged (ER, ACE and LDA expert), and does change later ones |
| C7 | Determinism: the same seed gives identical outputs |
| C8 | Mixture: the cumulative log-loss is ≤ the best expert + log 2; a mixture of identical experts equals the expert; weight 1 equals expert A |
| C9 | Reservoir inclusion probability is uniform (statistical test) |
| C10 | ACE: zero gradient on the logits of absent classes in the stream term |
| C11 | biascorr equals the marathon definition on a balanced buffer |
| C12 | The accuracy metric on a hand-built example |

An independent code review must also be done before results are reported.

## Amendments before any E0/E1 evaluation run (2026-10-29)

These came from the independent review of cifar.py. The lr-tuning runs on seed 100 were already done, and they are affected only in their NLL fields.

1. **Bug fixed: float32 underflow.** Prequential probabilities were stored as float32 after exp(). LDA's early, overconfident predictions underflowed to exactly 0, so the NLL came out infinite and MIX_stream weights were arbitrary. Now each item's true-label log-probability is stored in float64 (`preq_lp`), and the mixture runs in log space (logaddexp). New check C13 covers this.
2. **C8 strengthened.** It now compares against a direct per-item loop, includes a leak detector (C8b), and tests the dominated-expert case through the function itself. C9's tautological sub-check was removed. C11 now has an imbalanced-buffer case.
3. **E0 augmentation.** The crop offset is now drawn per sample, and padding is black (−2 in normalised space), as torchvision does. Declared: SGD with momentum 0.9, weight decay 5e-4, batch 64, cosine schedule over 10 epochs.
4. **E0 rp512** is now the same map as the E1 RP-LDA expert.
5. **LR has no default** for learner methods.
6. **LR choice** (seed 100, test-set accuracy of a different stream: mild test reuse, acknowledged). ER: 0.01 / 0.03 / 0.1 gave final ER+bc_buf 52.2 / 53.9 / 51.8, so 0.03. ACE: 50.0 / 51.7 / 49.8, so 0.03. Both use 0.03.

## E2: streaming-valid choice of the mixture weight and the correction size (pre-declared 2026-10-29, before running)

This E2 is a different experiment from the "E2" (the tested interpretations) referred to in the audit file.

**Setup.** Same as E1, with RESV = 20 per class (200 images, 614 KB extra memory, charged). Only ER (lr 0.03) and RC-LDA are rerun, on seeds 0–4. All comparisons are made inside E2, since its learning stream has 150 fewer items than E1's.

**Rules.** At each task end, everything is chosen using only the reserve items of seen classes. Those are held out of all training and statistics, and they are available causally.
- **α̂.** Chosen from {0, 0.1, …, 1.5} to minimise the reserve NLL of softmax(L − α·b_buf).
- **ŵ.** Chosen from {0, 0.1, …, 1} to minimise the reserve NLL of ŵ·P_ER(α̂) + (1 − ŵ)·P_RC.
- **POE_half.** A product of experts, argmax of 0.5·log P_ER+bc + 0.5·log P_RC, with no fitting. This was found post hoc in E1 and is pre-declared here.

**Comparisons** (final accuracy, paired, same significance rule):

| | Comparison | Question |
|---|---|---|
| Q1 | ER+bc(α̂) vs ER+bc_buf (α = 1) | Is a data-chosen correction size better than the formula? |
| Q2 | MIX(ŵ, α̂) vs MIX_half (α = 1) | Does fitting the weights help? |
| Q3 | POE_half vs MIX_half | Does a product of experts beat a probability average? |
| Q4 | MIX(ŵ, α̂) vs the better single expert per seed | Pre-declared headline: is a fully streaming-valid two-path system better? |

**Reported alongside:** α̂ and ŵ at each task end, and the oracle (test-set) α and w for reference.

## Round 2, steps 2–5 (pre-declared 2026-10-29, before running)

**Shared setup.** All arms use RESV = 20 and the same streams as E2, and reuse E2's ER and RC runs. The code is cifar2.py, a generalised learner that is self-tested to equal cifar.py's ER exactly at its defaults. Each arm also reports its streaming-valid MIX(ŵ, α̂) and POE_half, using the E2 rules.

**Memory accounting.** A buffer image is 3,072 bytes. Expert statistics are stored as float32 symmetric matrices plus means: 512-d is 0.55 MB and 1024-d is 2.1 MB. A random encoder costs nothing persistent, since it is regenerated from its seed. A promoted version costs its parameters (0.37 MB) plus its statistics.

### Step 2: memory matching

- **ER at M = 680** (the 512-d stats' memory) and **ER at M = 1,200** (the 1024-d stats' memory). Single runs each.
- **RCW-LDA:** a random CNN with 64/128/256 channels and 1,024 features, the same encoder as E0 init_wide.

| | Comparison |
|---|---|
| R1 | MIX_half(ER500, RC) vs ER+bc(M = 680) |
| R2 | MIX_half(ER500, RCW) vs ER+bc(M = 1,200) |
| R3 | RCW-LDA vs RC-LDA |

### Step 3: promotion

ER with promote = 1. Versions are taken at the ends of tasks 1–4; each version's statistics come from buffer items at promotion, then from the stream.

| | Comparison |
|---|---|
| S1 | P_last (version from end of task 4) vs RC-LDA |
| S2 | MIX3 = mean(ER+bc, RC, P_last) vs MIX_half |
| S3 | BANK = mean(ER+bc, RC, all 4 versions) vs MIX_half |

Memory is reported: BANK costs about 3.7 MB.

### Step 4: protection on the corrected two-path base

**EWC.** λ is picked from {1, 10, 100, 1,000} on seed 100, by MIX_half final accuracy with that seed's RC. The Fisher is the empirical diagonal Fisher on the buffer, taken at task ends.

**M2c.** Zero slack, with born-label rivals, refreshed every 25 steps on a pool of 32. These are the toy-best settings, with no tuning.

| | Comparison |
|---|---|
| U1 | EWC+bc vs ER+bc |
| U2 | MIX_half(EWC, RC) vs MIX_half(ER, RC) |
| U3 | M2c+bc vs ER+bc |
| U4 | MIX_half(M2c, RC) vs MIX_half(ER, RC) |

Compute (wall time, images through fwd/bwd) is reported. Per the audit's stopping rule, a method that loses at matched resources is removed at this scale.

### Step 5: robustness, 3 epochs per task

ER with epochs = 3, at the same LR of 0.03 (not retuned; this is declared). RC is unchanged, since its statistics are order-invariant and come from the first pass.

| | Comparison |
|---|---|
| V1 | ER+bc(E3) vs RC-LDA |
| V2 | MIX_half(E3) vs the better single expert |
| V3 | ER+bc(E3) vs ER+bc(E1) |

**Amendment, 2026-10-29 06:45 UTC (no M2c result existed yet).** Profiling showed that 100% of M2c's wall time was going into the dual QP solver (L-BFGS-B with tolerance 1e-15). That reflects my implementation, not M2c itself.

I replaced it with an exact small NNLS: mu = nnls(R, −R^{−T} r), with G = RᵀR. qp_check.py validates it, and it matches the original correction to a relative error of 2e-15, on 200 random instances (including dependent columns) and on 110 real CIFAR projection steps. Both solvers return feasible steps. The QP being solved is unchanged.

The four M2c runs in progress (about 40% done after 2 hours) were stopped, and all 5 seeds were rerun with the new solver. The wall-time comparison for M2c uses the new solver.

**Amendment, 2026-10-29 07:41 UTC (after the m2c=1 results, before any m2c=2 run).** The m2c=1 runs collapsed to 26% (ER+bc) with test logits reaching 1e4 to 5e5. A diagnostic run (seed 0, the NNLS solver) shows the logit scale growing steadily from the start of task 1: 5 at the first projected step, 365 by step 600 and 4e4 by step 1,550. The projection is feasible throughout (violations around 1e-10).

The cause is a porting difference, not the solver. The toy filled its buffer only at task ends, so M2c protected past-task items only. On CIFAR the reservoir holds current-task items from the first batch, so m2c=1 forbids any first-order margin decrease on the current task's own items too. That includes task 1, where it should be inactive.

Added **m2c=2**, with the pool drawn from past-task buffer items only. Rivals and every other setting are unchanged. m2c2_check.py checks that it equals ER through task 1 (M1), is active and feasible in task 2 (M2), draws past-task labels only (M3), and leaves m2c=1 unchanged (M4). All four pass. U3 and U4 are rerun as m2c=2 on seeds 0–4, and both versions are reported. The m2c=1 result stands as a result about that variant.
