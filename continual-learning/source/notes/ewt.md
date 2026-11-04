# ewt.md: every test we ran, what it showed, and what it means

Written 2026-10-29 at Lucky's request; updated 2026-10-30 with X1–X3 (sections 15–17) and a section on how reliable the tests themselves are (section 18). It covers every test from the v1.2 critique checks through X3, rebuilt from the saved results files, logs and scripts, not from memory. Where a number is quoted, its source file is named.

**Honesty rules for this file:**
- Negative and null results are reported with the same weight as positive ones.
- Withdrawn claims are marked as withdrawn, not quietly dropped.
- Bugs, mistakes and process errors (mine included) are listed.
- "\*" always means the pre-registered rule **as coded** (analyze_cifar.paired): |mean| > **2**·SE **and** all 5 seeds share the sign. [Corrected after Lucky's audit: an earlier version of this line said "mean ± SE excludes 0", which is wrong; that 1·SE rule would be implied by 5/5 alone.] 5/5 in a direction chosen in advance has sign-test p = 1/32 (1/16 for either direction). Mean ± 2·SE is not a 95% interval at n = 5. It means "consistent", not "proven".
- **Non-significant results are not ties.** Where a row below says "tie: confirmed" or "null", read it as "no difference detected", not as shown equivalence.
- After this many adaptive tests, everything here is development evidence, not confirmatory evidence.

**Scale warning that applies to everything.** CPU only, no GPU:
- the toy is a 20-d Gaussian MLP;
- MNIST uses a 784-256-256-10 MLP;
- CIFAR uses a 3-block, 98k-parameter CNN.

All CIFAR runs are online (one pass) with M = 500 (CIFAR-10) or M = 2,000 (CIFAR-100), and 5 seeds. None of this is ResNet-18 multi-epoch scale. Nothing here shows a full system winning at publication scale.

---

## Contents

0. [Summary: what we actually know now](#0-summary-what-we-actually-know-now)
1. [Spec math checks (v1.2 and v1.3 audits)](#1-spec-math-checks-v12-and-v13-audits)
2. [Toy tests of M2c (v1.3, A3)](#2-toy-tests-of-m2c-v13-a3)
3. [Why M2c tied: the diagnosis (A4)](#3-why-m2c-tied-the-diagnosis-a4)
4. [The sharp-test marathon (toy + Split-MNIST, A5)](#4-the-sharp-test-marathon-toy--split-mnist-a5)
5. [Marathon revisions after the v1.3 deep audit (A6)](#5-marathon-revisions-after-the-v13-deep-audit-a6)
6. [CIFAR-10 round 1: E0 gap check and E1 two-path test (A7)](#6-cifar-10-round-1-e0-and-e1)
7. [CIFAR-10 E2: choosing the weight and correction online](#7-cifar-10-e2-streaming-valid-weight-and-correction)
8. [CIFAR-10 round 2: memory, promotion, protection, epochs](#8-cifar-10-round-2)
9. [Sleep-paper tests T-S1 to T-S4 (CIFAR-10)](#9-sleep-paper-tests-t-s1-to-t-s4)
10. [T-S5: growth budget on Split CIFAR-100](#10-t-s5-growth-budget-on-split-cifar-100)
11. [Cross-test insights (new, from reading everything together)](#11-cross-test-insights)
12. [Every bug, mistake and process error](#12-every-bug-mistake-and-process-error)
13. [What is still open](#13-what-is-still-open)
14. [Corrections after Lucky's audit, with three new tests](#14-corrections-after-luckys-audit)
15. [X1: is the readout bias weight scale?](#15-x1-is-the-readout-bias-weight-scale-cifar-100)
16. [X2: carrying class statistics into new encoders](#16-x2-carrying-class-statistics-into-new-encoders-cifar-100)
17. [X3: waking dormant class statistics with a few key items](#17-x3-waking-dormant-class-statistics-with-a-few-key-items-cifar-100)
18. [How reliable are the tests themselves?](#18-how-reliable-are-the-tests-themselves)

---

## 0. Summary: what we actually know now

**Supported in the tested development settings** (replicated, pre-registered, gates passed; "established" was too strong):
1. **In class-incremental learning on these benchmarks, a large share of forgetting is an output-layer bias toward the newest classes, not lost knowledge.**

   | Setting | ER raw → ER + bias correction | Gain |
   |---|---|---|
   | Toy A | 41.5 → 57.7 | +16.2 |
   | Split-MNIST | 50.7 → 71.2 | +20.5 |
   | Split CIFAR-10 | 42.6 → 52.4 | +9.8 |
   | Split CIFAR-100 | 17.4 → 22.7 | +5.3 |

   Making the CIFAR-10 network 4× wider gives no raw gain (42.3 vs 42.6).
2. **The size of that bias cannot be computed from a formula. It must be estimated from held-out data.**
   - On the toy and MNIST, prior-based logit adjustment *under*-corrects.
   - On CIFAR-10, mean-logit correction *over*-corrects; the best scale is α ≈ 0.7–0.8.
   - A scale chosen online from 200 reserve items gains +2.3 \*.
3. **A stable statistics expert plus the plastic network, mixed, beats either alone on real images.**
   - CIFAR-10: +4.0 \* over the better single expert with streaming-valid weights, and +3.2 \* at matched memory.
   - CIFAR-100: +1.5 over RC alone and +7.0 over ER+bc (these two differences were not run through the significance rule).
4. **Every "protect the parameters" or "compress the learner" mechanism tested either failed or did nothing:**
   - M2c (all variants, at real-image scale);
   - EWC;
   - head-row snapshots;
   - Sleep-style consolidation;
   - frozen copies of late networks.
5. **Anything fitted on the replay buffer looks better on the buffer than on the population.** This holds for LDA statistics, distillation and margin constraints, and it is why everything must be judged on held-out items.

**New this week** (single setting, so less certain):
- ~~On CIFAR-100, the quality of the online network's features peaks mid-stream and then decays below random features.~~ **Withdrawn (section 14).** With every encoder's readout fitted on the same data, the learned features improve monotonically through all 20 tasks (26.0 → 35.3) and stay 6.7 points above random. The mid-stream peak came from readout coverage, not from the features.
- Growth (keeping frozen versions) doesn't clearly beat spending the same memory on a bigger buffer. *Which* versions are kept matters more than how many.

**New since the audit (sections 15–17; one learner, offline replay of saved runs):**
- **Head weight norms are not the bias (X1).** Equalising norms costs 8.3 \*; old classes have the larger norms under replay. A class-statistics readout (LDA on 20 reserve items per class) is the best readout.
- **Stranded statistics can be carried forward online (X2).** A ridge map fitted on the buffer at each task end moves stored class means into the new encoder: +5.6 \* over buffer-only statistics, 54% of the gap to the oracle. Mixed with the other two paths it beats the version bank (+0.9 \*, fragile) with 4–8× fewer bytes.
- **Dormant statistics can be woken by a few key items (X3).** 84–98% of the old readout's within-task accuracy is recovered with 1–all keys per class. After review: not better than X2's transport once covariances are matched, and only cheaper than keeping the encoder at 1 key per class.
- **How far to trust the tests themselves:** section 18.

**Not achieved.** On CIFAR-10, joint training's features reach 75.5. Our best online system reaches about 59–61. **None of the online methods captures that 15–17 point feature headroom.** That is the main open problem.

---

## 1. Spec math checks (v1.2 and v1.3 audits)

**What.** These are numerical checks of the claims in two external critiques of the spec, and of my fixes. They are algebra and toy-scale checks, not learning experiments.

Code: notes/v12-checks/{checks.py, fix2.py, review_checks.py}, notes/v13-checks/{verify13.py, frontier*.py, heldout_frontier.py, stale_slack_counterexample.py}. Write-up: formal-spec.md Part A, with v1.2 in archive/.

**v1.2 critique (16 points, all substantially correct).** I re-ran the scripts today, and these outputs reproduce:
- **An Adam-composed preconditioner can go uphill:** g·P⁻¹U = −3.04. The fix, solving (D⁻¹ + wH)d = g, gives descent (+0.665).
- **K-FAC protects directions the outputs can't see:** exact 0.0 vs K-FAC 0.5 on a constructed Δ.
- **The Woodbury exact solve** matches a dense solve to ≤ 4.7e-6 relative, up to w = 1e4, and equals Adam exactly at w = 0.
- **PCG** has 0 non-descent iterates in 2,000. Unpreconditioned CG at ω = 0 did *not* reproduce Adam (65% error); this was a reviewer finding, now fixed.
- **The token-coupled sketch is unbiased:** exact 106.6 vs estimate 107.1.
- **Power calculation:** 80% power at a 12.5-point effect needs n ≈ 53.
- An independent reviewer found 21 defects in the v1.2 draft (0 critical, 12 major, 9 minor), and all were fixed.

**v1.3 audit (15 patch areas, all valid).** These counterexamples reproduced exactly:
- 3.04 for group-rate post-scaling;
- −977 vs 27.6 nats for conditional novelty;
- 0.843 for the never-switching HMM;
- powers of 0.808, 0.363 and 0.204;
- the (6, 5) certificate false alarm;
- the (1, 0, 2) new-class steal;
- the H = 11ᵀ null-space case;
- the 0.25 → 0.953 usage jump.

One error was mine: v1.2's delay formula was off by a factor of κn = 4, and its premise was wrong anyway.

**New structure found and checked (N1–N6):**
- **N1** [proved]. Every protection step is one B⁻¹-proximal map, which gives surrogate descent and non-expansion.
- **N2.** The centred certificate pairs with the centred metric, with 0 violations in 5,748 random cases.
- **N3** [**corrected**]. My first frontier said margin constraints "dominate" the quadratic brake. **That was an in-sample tautology:** damage was measured on the constrained items themselves. On 200 held-out items, the quadratic brake did better: damage 64 at progress 171, vs 77 at 160. The follow-up slogan "constraints memorise, penalties generalise" was later **withdrawn** too (sections 3 and 4 contradict it).
- **N4.** Tracked slack gives a cumulative guarantee on the constrained pairs. With stale slack, a linear counterexample falls to −5.5.
- **N5.** An exact detectability threshold for the sticky filter: log L_min = 0.166. It was **later shown not to be necessary** under noisy evidence (marathon T7).
- **N6.** Inferential reactivation needs n ≈ 170 probes for robust power, so it runs in control mode.

**Meaning.** These checks fixed the spec's math. They say nothing about whether the learner works. A second reviewer of v1.3 found 20 more defects (including errors in N3 and N4), and all were fixed.

---

## 2. Toy tests of M2c (v1.3, A3)

**What.** The first learning test of the spec's lead protection mechanism, M2c: a per-step projection that forbids first-order decreases of stored items' margins against their top rivals.

**Setup.** notes/v13-checks/toy_cl3.py, with earlier versions toy_cl.py and toy_cl2.py:
- 10 Gaussian classes in 20-d with separation 0.6, split into 5 tasks of 2;
- an MLP 20-64-64-10 trained with Adam at lr 1e-3;
- 300 steps per task, batch 16 + 4 replay;
- 5 memory items per class, 5 seeds;
- ER and M2c share identical streams and replay draws (paired).

| Method | Final acc | vs ER |
|---|---|---|
| fine-tune | 18.2 | |
| GEM (one constraint per task) | 39.1 | |
| ER | 41.5 | |
| M2c as specified (R_pre = 1 or 25) | 41.2 | −0.3 ± 0.5 |
| compute-matched ER (36 replay items) | 44.9 | +3.4 ± 0.9 |
| joint (upper bound) | 73.5 | |

**Result.** M2c raised buffer accuracy (94.8 → 96.8%) but **did not beat ER on the test set**. Spending its compute on replay instead won by 3.7 points.

**Harness issues caught by review:**
- the default separation of 1.6 saturates (ER 99.4%);
- the first version had unpaired streams;
- there was no compute matching;
- a docstring falsely claimed a test-never-trained assert.

All were fixed in toy_cl3.

---

## 3. Why M2c tied: the diagnosis (A4)

notes/v13-checks/why-m2c-failed.md, diag.py.

| Method (start-insert = buffer filled at task start) | Acc | Old-class errors predicted as a newest class |
|---|---|---|
| ER | 41.5 | 87% |
| M2c, zero slack | 44.6 (+3.1 ± 0.8, 5/5) | 84% |
| ER, replay loss ×9 (same items, 1× compute) | 47.8 | 79% |
| ER, replay loss ×20 | 50.0 | 76% |
| ER-ACE, start-insert | **57.8** (vs ER start-insert 39.9) | 25% |

- **The main finding: the knowledge survives in the features, and the readout fails.** Refitting only the last layer on ER's frozen features gives 70.6%, against 73.5% for joint training.
- **Mechanism, from first principles.** The softmax gradient p_j − 1[j = y] pushes every old logit down on every new item. With 16 new and 4 replay items per step, that is a ~16:1 pull per class against old classes. Both predictions of this account held: reweighting replay helps at no extra compute, and masking the new items' loss to new logits (ER-ACE) helps most.
- **Why compute-matched ER "won" in section 2:** loss balance, not compute. The same loss share obtained by weighting does better at 1× compute.
- **Why M2c as specified did nothing:** replay keeps buffer items at large margins, so its slack let them lose a lot before any constraint bound. It removed only 19% of first-order damage on its pool and 13% held out.
- **Zero-slack M2c does act, and its protection transfers:** 86% of held-out first-order damage removed. That **contradicted** my earlier "constraints memorise" claim. But it bought only +3.1 points.
- **Open from this test** (never resolved): removing 86% of first-order damage bought only ~4 points of old-class accuracy, while readout fixes bought 8–30. ~~Most of the accuracy loss is not step-wise first-order margin damage.~~ [Withdrawn after the audit: removing 86% of *summed* damage can leave every boundary crossing in place, for example if the protected items had large margins anyway. This only shows that summed damage is a poor stand-in for accuracy.]
- **Harness defect found:** diag.py's "finetune" silently replays, so it is excluded.
- **Protocol trap found:** with end-of-task insertion, ER-ACE never replays the last task.

---

## 4. The sharp-test marathon (toy + Split-MNIST, A5)

notes/marathon/results.md, mar.py and mar2.py (self-tests S1–S11 and N1–N16, two independent reviews).

**Settings:**
- toy A: separation 0.6, 5 items per class;
- toy B: separation 0.8, 20 items per class;
- Split-MNIST: MLP, 5 items per class.

All use 5 seeds, paired.

| Test | Result | Status |
|---|---|---|
| **Readout bias (T2 and others).** Per-class bias correction on the same ER weights. | 41.5 → 57.7 (toy A), 79.9 → 83.4 (toy B), 50.7 → 71.2 (MNIST), all \* | Holds |
| **Logit adjustment by the stream prior, τ = 1, no fitted parameter** | Recovered 66–160% of the correction gain. Predicted *no* bias under ER-ACE, and that was observed. The wrong prior hurts ACE by −4.9 \*, so the test discriminates. | **Partly withdrawn** (section 5, R1): the direction is right, the size is not |
| **W3 floor control.** Oracle LDA on fixed vs learned features (MNIST) | ER features 86.7, initial random net 87.2, raw 87.5, random ReLU 1024-d **94.0**. Every trained method was at most 72.2. Features after task 1 alone: 75.2 (the first task *destroys* separability for future classes). | **Revised** (R2): true only with full-history statistics |
| **F8/W1: readout sample size.** | Buffer readouts plateau at ~71 vs an oracle of 89 (5 per class). The gap is in the class means (full means + buffer covariance 78.3; the reverse 55.5). | **Revised** (R3): partly buffer memorisation |
| **F7/W2: stale statistics.** | Stored class means go stale catastrophically, to ~20% everywhere. MNIST drift is class-specific: shared-affine correction gives 68.3 vs 86.7. | Holds |
| **T5: head-row snapshots** | "Restore saves time" was a **readout artefact**. Under bias-corrected readout, restored rows are worse than current ones (−25.3 \* toy, −56.7 \* MNIST). | Holds (the claim is withdrawn) |
| **W6: repetition** | Small real savings on recurrence: +3.1 \* toy, +8.2 \* MNIST at the 3rd occurrence | Holds |
| **T1: boundary shock** | The centred mean init raises the dip minimum, but never improves final accuracy. Warm-up hurts. | Killed |
| **T4: fresh vs stale Fisher (EWC)** | Fresh beats stale at small λ (toy +3.1–4.9 \*, MNIST +6.5 \*). Fresh is **much worse** at λ ≥ 10 on MNIST (−10.0 \*, −13.8 \*). | "Fresh beats stale" holds only at small λ |
| **T6: M2c rival coverage** | Adding "labels born since the last refresh" as rivals lifts zero-slack M2c from 44.6 to **54.1** (+9.5 \*). That is still below the free fixes (biascorr 57.7, LA 57.2, ER-ACE 57.8), at ~30× ER compute. | The rule is necessary; M2c is still not justified |
| **T7: context detection** | Detection in 1–4 batches with no false switches. It succeeded below L_min under noisy evidence (11/20 with 2 states, 17/20 with 3). | L_min rewritten as a constant-evidence boundary |
| **W7: predicting interference** | Feature similarity predicts which new class absorbs an old one 63–72% of the time, but raw-input similarity is as good (60–74%). | Not a big insight |
| **R13: end-of-stream "sleep"** (200 steps on the buffer) | ≈ biascorr on ER. Wins under ER-ACE (+2.8 \* MNIST). | A modest win, mostly the same debiasing |
| **Centring the head** | The bias moves from the common term into the item-specific term | Failed |

**Process error (mine).** After a container restart I relaunched T6 without the toy-A env settings, so those runs used the saturated toy. They were caught, archived as invalid and re-run. A guard now checks that ER reproduces S1.

---

## 5. Marathon revisions after the v1.3 deep audit (A6)

Lucky's deep audit questioned three marathon conclusions. I tested each one, and each was revised:
- **R1. The bias size is not a closed-form function of the prior.**
  - Under the spec's loss, the induced prior ratio is 4, not 16, and logit adjustment by it under-corrects: −5.9 \* on the toy, −12.5 \* on MNIST.
  - The oracle-best τ is 2.7–3.4 (toy) and ≥ 4 (MNIST).
  - The audit's global-logit-scale explanation is **not supported** (newest-class temperature ≈ 1).
  - **Conclusion:** a data-driven correction is required, and the "zero-parameter mechanism" claim is withdrawn.
- **R2. Learned features are far more sample-efficient.** At 5 per class, trained features beat every fixed alternative by 9–18 points (5/5). Fixed features win only with full-history statistics. Width matters: random ReLU gives 89.1 at 256-d, 94.0 at 1024-d and 96.4 at 4096-d.
- **R3. The replay buffer is a biased statistic source.**
  - MNIST old-class LDA is **49.4 from the replayed buffer vs 73.8 from 5 fresh items** (+24.3, 5/5). Buffer means are ~2.3× farther from the truth.
  - Holding memory out of replay helps LDA (+6–7 \*), but loses to biascorr at this budget.
  - **Conclusion:** stable statistics must be accumulated *at arrival*, not from replayed items.

These three revisions produced the two-path design that the CIFAR tests then examined.

---

## 6. CIFAR-10 round 1: E0 and E1

notes/cifar/plan.md (pre-declared), results.md, cifar.py (self-tests C1–C13), plus an independent review.

**E0: is CIFAR a valid test of feature learning?** Full-data LDA on these feature maps, seeds 0/1/2:

| Feature map | Accuracy |
|---|---|
| joint-trained CNN | 75.7/75.5/75.4 |
| random CNN 512-d | 55.3/52.0/54.0 |
| random CNN 1024-d | 56.7/57.8/58.4 |
| random pixel projection | ~50 |
| raw pixels | 41.7 |

**Pre-declared criterion (≥ 5-point gap): passed on every seed, by 17–19 points.** MNIST's "random beats learned" does not carry over to real images. This is what made CIFAR a meaningful test.

**E1: Split CIFAR-10, 5 seeds** (M = 500, online, lr 0.03 chosen on seed 100):

| Arm | Final |
|---|---|
| ER | 42.7 |
| ER + bias correction (bc_buf) | 52.6 |
| ACE | 50.8 |
| RC-LDA (random-conv statistics) | 54.0 |
| FZ-LDA (frozen after half of task 1) | 54.3 |
| MIX_stream (Bayes, stream loss) | 42.7 |
| MIX_resv (50 reserve items) | 53.6 |
| **MIX_half** | **57.9** |

- **P2: the two paths add, +3.5 ± 0.4 \*.** Their errors are complementary: either expert alone is right on 69.1% of items, both on 37.6%.
- **The exact Bayes mixture on the stream's log loss reduces to ER**, with weight 1.0000 on every seed. Its log 2 regret guarantee holds, but it targets the *current-task* risk, where the plastic net wins. The audit's mixture proposal, weighted that way, discards the stable expert. This was predicted in advance and confirmed.
- **Bias correction over-corrects the newest task on CIFAR.** The newest task gets 10.9% of predictions vs a true 20%. It is not buffer memorisation: b_buf ≈ b_test. An oracle scale α ≈ 0.7 is best.
- **Nulls:** FZ-LDA ≈ RC. ACE is −1.8 \* vs ER+bc.
- **RC-LDA vs ER+bc: +1.4 (4/5), not significant.** A random-feature statistics expert ties tuned ER.

**Bugs caught before results:**
- float32 probabilities underflowed to 0, which made the stream NLL infinite; fixed with float64 log-probs (C13);
- vacuous checks C8, C9 and C11 were strengthened;
- the C8 threshold 1 − 1e-20 rounds to 1 in float64.

**Caveat:** the lr was chosen on the seed-100 test set (mild test reuse, acknowledged).

---

## 7. CIFAR-10 E2: streaming-valid weight and correction

Plan: plan.md §E2. Results: results/e2/analysis_e2.json and results_r2.md.

RESV = 20 per class (200 reserve images). At each task end, α̂ and ŵ are chosen by reserve NLL.

| | Comparison | Result |
|---|---|---|
| Q1 | ER+bc(α̂) vs ER+bc(α = 1) | 54.7 vs 52.4: **+2.3 ± 0.4 \*** |
| Q2 | MIX(ŵ, α̂) vs MIX_half | 58.7 vs 57.3: **+1.4 ± 0.4 \*** |
| Q3 | POE_half vs MIX_half | **+1.0 ± 0.2 \*** |
| Q4 | MIX(ŵ, α̂) vs the better single expert | **+4.0 ± 0.3 \*** |

- **Meaning: the two-path system is now fully online and causal, and it still wins.** With 200 held-out items (but not 50), both the correction size and the mixture weight can be chosen online.
- The final α̂ values are 0.6–0.8, matching the oracle sweep from round 1. The final ŵ values are 0.3–0.5.

---

## 8. CIFAR-10 round 2

Plan: plan.md "Round 2", with amendments. Results: results_r2.md and results/r2/analysis*.txt. Code: cifar2.py (K1–K10), cifar2_checks.py (X1–X5), qp_check.py (Q1–Q2), m2c2_check.py (M1–M4).

| Step | Comparison | Result | Meaning |
|---|---|---|---|
| Memory | R1: MIX_half(ER500, RC) vs ER+bc at M = 680 (equal memory) | **+3.2 \*** | At small budgets, statistics beat images per byte |
| Memory | R2: MIX_half(ER500, wide RC) vs ER+bc at M = 1,200 | +1.2 (4/5), not significant. Streaming-valid version +2.4, not significant | At larger budgets a bigger buffer catches up (ER+bc gains 5.8 from M 500 → 1,200) |
| Memory | R3: wide RC (1024-d) vs RC | 57.8 vs 54.0, **+3.8 \*** | Wider random features help, as on MNIST |
| Promotion | S1: last frozen version vs RC | 50.2 vs 54.0, **−3.8 \*** | A late copy is worse than random features |
| Promotion | S3: BANK (all 4 versions) vs MIX_half | 60.0, **+2.7 \*** | It helps, but costs ~3.7 MB, more than R2's buffer |
| Protection | U1/U2: EWC (λ = 10) | +0.7 and +0.1, null | No measurable benefit. λ = 1,000 diverged |
| Protection | U3: M2c, first port | 26.3, **−26.1 \*** | Collapse: logits up to 5e5 |
| Protection | U3′: M2c, faithful (past-task items only) | 39.7, **−12.8 \*** | Still a large loss: logits inflate to 300–1,400 |
| Epochs | V1–V3: 3 passes per task | ER +0.8 (not significant); MIX gain +3.3 \* survives | The two-path gain is robust to more passes |

**M2c story in full, including my error:**
- My first CIFAR port protected the buffer's *current-task* items too. The reservoir fills from the first batch, while the toy filled at task ends. So it forbade margin decreases on the task being learned, even during task 1. The logits climbed steadily, from 5 to 4e4 by step 1,550 on seed 0.
- Not the solver: the NNLS solver matched the original to 2e-15, and a direct comparison gave an identical logit trajectory.
- I added the faithful variant before seeing its result, and it still lost by 12.8 points.
- **Real mechanism:** never letting old margins fall while new classes are learned forces the logit scale up.
- Caveats:
  - the toy-best settings were used untuned;
  - SGD at 0.03 was used, where the toy used Adam at 1e-3;
  - an **untested hypothesis**, prompted by Barron et al. (arXiv 2610.31642) on softmax/cross-entropy precision: once a logit gap exceeds ln 2²⁴ ≈ 16.6, float32 rounds the top softmax probability to exactly 1, so some of the late M2c dynamics (gaps in the hundreds) may be precision effects layered on the real failure. Nothing here measured this, and the paper itself does not test M2c.
- **M2c is removed at this scale, by the audit's stopping rule.**

**Other process events:**
- the container restarted at 04:33 UTC (queues were resumable, so no partial results);
- I swapped in the NNLS solver mid-round after profiling, validated it and declared it before any M2c result;
- I stopped the first M2c attempt at about 40%, and declared that.

---

## 9. Sleep-paper tests T-S1 to T-S4

Source paper: Behrouz et al., "Language Models Need Sleep" (arXiv 2606.03979), reviewed in the Claude Doc. Plan: sleep_plan.md, with three dated amendments, all made before any 5-seed run. Code: sleep.py. Self-tests: sleep_selftest.py, 20 checks, plus sleep_z4_control.py. Two independent review rounds. Results: results_sleep.md.

**Analogue tested.** At each task end, compress what the network learned since the last "sleep" into a new expert: a rank-r or full update. This is done by distilling the network's own outputs on "dreams", then resetting the fast weights and continuing. **It is not a reproduction.** There is no language model, no self-generated inputs, no RL reward and no SEAL dreaming.

| | Comparison | Result |
|---|---|---|
| T1 | Sleep (rank 16, buffer dreams)+bc vs ER+bc | 44.0 vs 52.4, **−8.4 \*** |
| T1b | Sleep (full rank)+bc vs ER+bc | **−8.4 \*** (so compression is not the cause) |
| T1c | MIX(Sleep, RC) vs BANK | **−6.0 \*** |
| T2 | Held-out gap: buffer agreement − reserve agreement | rank 16: **6.6 points**; full: 8.5; rank 4: 6.3 (all 5/5) = **lossy** |
| T2b | Cost of each sleep (student − teacher) | −3.6 to −5.7 points per sleep, 5/5, every arm. The predicted ordering r4 < r16 < full was **not** confirmed (r4 = r16) |
| T3 | Mixup vs buffer dreams | Null (final +0.0). Mixup closes the held-out gap only by fitting the buffer worse |
| T3 | Noise vs buffer dreams | **−30.6 \***: consolidation needs in-distribution inputs |
| T4 | ER+bc vs 4× wider ER, raw | **+10.1 \*** (pre-declared > 5: yes) |
| T4b | Wide ER+bc vs ER+bc | +1.4 \* at ~4× the FLOPs |

**Meaning:**
- **Sleep-style self-distillation is not a forgetting remedy here, and it actively loses accuracy.** The loss the paper never measured (teacher vs student off the dream set) is real: 6–9 points of agreement.
- **Forgetting on CIFAR-10 is a readout problem, not a capacity problem.** That contradicts the paper's core framing for this benchmark.

**Bugs the self-tests caught before any result:**
1. **Adam drifted away from an exact optimum.** With nothing to consolidate, the KL went 0 → 0.033. This is the paper-independent "softmax cancellation + Adam" problem that Barron et al. describe. I switched to SGD with momentum (drift ~1e-9).
2. The planted tests were too weak and were redesigned.
3. One test itself had a bug (3-class vs 10-class mask).
4. Review round 1 found 12 more items (untested task-end path, a non-identifiable mixup test, a missing leak guard, and others), all fixed.
5. The rank test's tolerance was too tight for SGD-sized updates. It was fixed, and a control proved it can fail.

**Hyperparameters.**
- The first grid's best point was at its edge. I extended the grid on seed 100 (declared) to give the method more room. The choice stayed at dlr 0.03 and 30 epochs, now an interior point.
- 100 epochs fit the buffer perfectly but overfit: reserve agreement 0.80–0.91.

---

## 10. T-S5: growth budget on Split CIFAR-100

Plan: ts5_plan.md. Budget fixed in advance in results/ts5/budget.json. Results: results_ts5.md.

**Setting.** 20 tasks of 5 classes, M = 2,000, lr 0.03 (not retuned).

**Gates:**
- the C1–C13 and K1–K10 self-tests still pass at the CIFAR-10 defaults after the 100-class change;
- the CIFAR-10 streams are bit-identical;
- the CIFAR-100 stream checks pass (11/11);
- the sleep arm is bit-identical to ER through task 2.

| Arm | Final (100 classes) |
|---|---|
| ER raw / +bc | 17.4 / 22.7 |
| RC-LDA | 28.2 |
| MIX_half(ER, RC) | 29.7 |
| BANK_all (+19 versions, +21 MB) | 31.0 |
| BANK_4spread (+4.4 MB) | **31.2** |
| BANK_4last (+4.4 MB) | 28.0 |
| MIX_half(ER at matched memory M = 8,836, RC) | 30.1 |
| Sleep+bc | 14.8 |

| | Comparison | Result | Prediction |
|---|---|---|---|
| G1 | BANK_all vs MIX_half | +1.4 (4/5), not significant | Positive: not shown |
| G2 | BANK_4last vs BANK_all | **−3.1 \*** | **Wrong** |
| G3 | BANK_4spread vs BANK_all | +0.2, null | Confirmed |
| G4 | BANK_all vs MIX(memory-matched ER, RC) | +0.9 (4/5), not significant | Predicted "bigger buffer wins or ties": not contradicted (a non-significant difference is not a shown tie) |
| G5 | Sleep+bc vs ER+bc | **−7.9 \*** | Confirmed |

**The new finding (not pre-declared as a comparison, but computed from saved outputs).** Each frozen version's own LDA accuracy, by the task where it was frozen:

| Task frozen | 1 | 4 | 8 | 12 | 16 | 19 |
|---|---|---|---|---|---|---|
| Accuracy | 26.0 | 27.8 | 29.4 | 29.1 | 26.3 | 24.9 |

RC (random) is 28.2. ~~Online feature quality rises, peaks around tasks 7–9, then falls below random.~~ **Wrong reading, withdrawn (section 14).** Each number is encoder plus online readout, and a version frozen at task t had its old classes only from the buffer. Matched-data probes show the features keep improving.

**Caveat:** "spread" uses knowledge of the stream length. A streaming selection rule is untested.

---

## 11. Cross-test insights

These come from reading all the results together. Each is labelled with how strongly the data supports it.

**I1. ~~The readout-bias share of forgetting shrinks as the problem gets harder, and feature degradation takes over.~~** [**withdrawn**, see section 14: gain ÷ raw accuracy is not a share of forgetting, and the feature-decay evidence was confounded. What survives: on longer streams the *readout statistics* are the bottleneck, and they get worse the later the coordinates they live in were created.]
- The bias-correction gain falls in relative terms: +39% of raw accuracy (toy), +40% (MNIST), +23% (CIFAR-10), +30% of a much lower base (CIFAR-100).
- More tellingly, the random statistics expert's lead over corrected ER grows: tie on CIFAR-10 (+1.6), +5.5 on CIFAR-100.
- The CIFAR-100 version curve shows the learned features themselves decaying late in the stream.
- So "forgetting is just readout bias" was right for 5-task benchmarks and becomes less true with 20 tasks. The spec should treat feature decay as the second failure, not a footnote.

**I2. Every result that looked good on the items it was fitted to was worse on held-out items. It is the same phenomenon each time.** [strong: 5 independent instances]
- M2c "dominates" in-sample but not held out (N3).
- Buffer LDA is 49 vs 74 from fresh items (R3).
- Head snapshots looked like savings until bias-corrected (T5).
- The Bayes stream mixture picks the expert that wins on the current stream (E1).
- Sleep distillation agrees 85–93% on its dreams vs 78–85% held out (T2).
- **Design rule this forces** [narrowed after the audit]: separate the roles of memory. Replay adapts parameters. Statistics in a fixed encoder's coordinates may use any items, including training items, because nothing was fitted to them in those coordinates. Choices (correction size, mixture weights, which version to keep) use a reserve that never trained the learner, and that reserve is then part of the algorithm, not untouched evaluation data. Final claims need separate evaluation data. 200 reserve items worked here; that is not a general threshold.

**I3. Methods that constrain or compress the learner lost. Methods that add an independent readout path won.** [strong at our scale]
- Lost or null: M2c (toy +3.1 at high cost, CIFAR −12.8), EWC (null), Sleep (−8), snapshot restore (artefact), late frozen versions (worse than random).
- Won: bias correction (+10 to +20), α̂ (+2.3), the statistics expert mixture (+3.5 to +4.0), POE (+1.0), the version bank with spread versions.
- **Interpretation:** the winning mechanisms don't fight the plastic network's drift. They read out around it, from coordinates that don't drift. A spec built around protecting parameters is built around the wrong lever, at least at this scale.

**I4. The tested formulas are wrong in different directions, so the size must be fitted.** [narrowed after the audit: these are two different correction families, prior adjustment and mean-logit subtraction, each wrong by a different amount; the required correction never changed sign. This rejects these formulas in these runs, not every possible formula.]
- The prior-based correction under-corrects on the toy and MNIST (τ needs to be about 1.5–4).
- Mean-logit correction over-corrects on CIFAR (α ≈ 0.7).
- Only the data-chosen α̂ was right everywhere it was tested.

**I5. Learned features have real but transient value online.** [moderate]
- E0: learned beats random by 17 points with joint training.
- MNIST: learned features are more sample-efficient (R2).
- But on CIFAR-10 a version frozen early (FZ) ≈ random, the last version < random (S1), and on CIFAR-100 versions peak mid-stream.
- The first task alone destroys separability for future classes (MNIST 87 → 75).
- ~~**Reading:** online class-incremental training first learns general features, then over-specialises them. Capturing the peak is a concrete target.~~ **Withdrawn (section 14):** the features did not peak. The CIFAR-10 S1 result (last version < random) used the same unmatched online readout, so it has the same confound and should not be read as feature decay either.

**I6. Memory economics flip with budget size.** [moderate]
- At small budgets, statistics beat images (R1 +3.2 \*).
- At larger budgets, a bigger buffer ties (R2, G4).
- So the stable statistics path is most valuable when memory is tight, which is the regime continual learning claims to care about.

**I7. Rigour paid for itself repeatedly.** [fact]
- At least eight real bugs or design errors were caught by self-tests or review before they produced results (section 12).
- Two of my own claims were withdrawn after better tests: "constraints dominate" and "constraints memorise, penalties generalise".
- Three marathon findings were revised after the audit.
- Without the gates, the M2c port error, the Adam drift and the float32 underflow would each have produced a wrong headline number.

---

## 12. Every bug, mistake and process error

| # | Where | What | Caught by | Effect on results |
|---|---|---|---|---|
| 1 | spec v1.2 | Delay formula off by a factor of κn = 4, with a wrong premise | audit | Formula replaced |
| 2 | spec v1.3 N3 | "Constraints dominate" was in-sample | reviewer | Withdrawn |
| 3 | toy_cl | Saturated default separation; unpaired streams; false docstring | reviewer | Fixed before A3 |
| 4 | diag.py | "finetune" silently replays | me | Excluded |
| 5 | toy protocol | End-of-task insertion breaks ER-ACE | me | Start-insert comparison added |
| 6 | marathon T1 | Uncentred mean init put old-class accuracy at 0 | reviewer (CRITICAL) | Runs superseded |
| 7 | marathon T6 | Relaunched without the env after a restart, so ran the saturated toy | me (guard added) | Runs archived as invalid |
| 8 | cifar.py | float32 probability underflow made the NLL infinite | reviewer | Fixed before E1 |
| 9 | cifar.py | Vacuous checks C8, C9, C11; a C8 threshold that rounds to 1 | reviewer / me | Strengthened |
| 10 | cifar2.py | Vacuous checks K2, K4 (EWC part) and K9; K7 finite differences dominated by curvature | reviewer / me | X1–X5 added; K7 fixed |
| 11 | cifar2 M2c | L-BFGS-B solver was 100% of the wall time | profiling | NNLS swap, validated |
| 12 | cifar2 M2c | Port protected current-task items (differs from the toy) | diagnostic run | Faithful variant added; both reported |
| 13 | sleep.py | Adam drift from an exact optimum | self-test Z5 | SGD |
| 14 | sleep tests | Weak planted tests; a test with the wrong mask size; a too-tight rank tolerance; an untested save path; a tautological mixup test | self-tests and 2 reviews | All fixed before runs |
| 15 | my messages | Guessed timestamps (07:50/07:52 instead of 07:41) written into a plan amendment and a chat message | me | Corrected, with an edit note |
| 16 | my messages | First M2c status claim (before the solver profile) was wrong | me | Corrected by edit |
| 17 | results_r2.md draft | Claimed EWC cost "1.1× ER" without checking. Wall times were measured under different loads. | me | Corrected |
| 18 | ops | `pgrep -f` matched itself; `pkill -f` killed my shell; the noexec mount | me | Workarounds noted in memory |
| 19 | timing | My T-S5 estimate of 2–3 h was too low. A later estimate was corrected. | me | None on results |
| 20 | x1.py self-test | An edit turned check P4 into a comment, so it was neither printed nor counted; also a float32 tolerance too tight | me, before reading results | Fixed and re-run before any real result |
| 21 | x2.py self-test | T1 expectations wrong (shift recovers under rotation; ridge biases aff); thresholds amended twice | self-test failure | Amended before any real run, logged |
| 22 | x2 write-up | Plan timestamps were my guesses; bytes used d = 256 instead of 512; MIX3 computed for the worst arm, not the best | independent review | Corrected; headline unchanged |
| 23 | x3 design | Covariance mismatch between REACT/OLD (5-class) and TRANS/BUF (100-class) | independent review | **R3 conclusion changed** |
| 24 | x3 write-up | "OLD is the ceiling" false; bytes compared against a whole version (1.1 MB) instead of the encoder body (0.375 MB) | independent review | **Cost conclusion changed** |
| 25 | x3 plan record | Plan saved 19 s before code; one item duplicated, so pre-declaration cannot be proved from files | independent review | Noted |
| 26 | my messages | X3 cost first reported to Lucky as "about the size of the encoder" | review | Corrected by edit with a note |

**Known weaknesses that are not bugs:**
- lr was chosen on the seed-100 test set (CIFAR);
- EWC λ came from a 3-value grid on one seed;
- the distillation hyperparameters came from one seed's 200 reserve items;
- many comparisons were run with only 5 seeds;
- CIFAR-100 used an lr that wasn't retuned;
- "spread" version selection uses knowledge of the stream length.

---

## 13. What is still open

1. **The headroom, re-diagnosed (section 14).** On CIFAR-100, the final learner's own features with full-history statistics reach 35.3. Our best online system reaches 31.2 and ER+bc reaches 22.7. The gap is mostly **statistics we never collected in the current coordinates**, not missing features. The concrete next target is carrying class statistics forward into each new encoder. [Update 2026-10-30: X2 (section 16) closes 54% of that gap online.] (Feature-drift compensation exists in the literature, and our MNIST shared-affine transport failed, so this is not trivially new.) On CIFAR-10, joint training reaches 75.5 against our 59–61 and is still unexplained; the matched probes have not been run there.
2. **Why removing first-order margin damage barely helps** (section 3, open since the toy). Untested candidates: curvature accumulated over many steps, items already wrong, rival-set timing.
3. **A streaming rule for which versions to keep** (T-S5 used stream-length knowledge).
4. **Scale.** Nothing here has been checked on a ResNet-scale, multi-epoch benchmark, or on a language model. The Sleep conclusions are about an image-classifier analogue only.
5. **Spec updates still to make:**
   - A7 should get the round-2, Sleep and T-S5 results;
   - M2c and EWC should be marked removed at this scale;
   - ~~feature decay should be added as the second failure mode.~~ The second failure mode is **statistics stranded in old coordinates** (section 14).

---

## 14. Corrections after Lucky's audit

Lucky's external audit (continual_learner_ewt_audit/EVIDENCE_AUDIT_AND_SYNTHESIS.md, 2026-10-29) was checked point by point against the code. Three of its proposed tests were pre-declared in notes/cifar/audit_plan.md and run. Results: results/audit/analysis100.txt and analysis10.txt. Full verdict: notes/cifar/results_audit.md.

**Code facts the audit exposed:**
- **T-S5 version readouts were not matched.** A version frozen at task t gets its old classes only from the buffer. The version from task 18 has about 21 items for each of 95 classes; the version from task 0 has 480 for each of 95.
- **Every toy and CIFAR head is `nn.Linear`, not the spec's 10·cos head.** No M2c or bias-correction result here used the spec's head. "Faithful M2c" means only that its pool was fixed.
- **The significance rule is 2·SE plus 5/5 signs, as coded.** This file had misdescribed it (fixed above).
- **The toy "start-insert" methods add a task's memory items at task start, before those items arrive.** That is non-causal, so the ER-ACE 57.8 figure (sections 3 and 4) is a diagnostic, not an online result.
- **The T-S5 learner output files had disappeared from disk** (cause unknown, around a container restart). They were regenerated, and they reproduced analysis.txt exactly (gate g2), so every earlier T-S5 number stands.

**A1. Matched-data probes** (CIFAR-100, 5 seeds). Every encoder's LDA is fitted on the same 480 items per class. All gates passed: g1, g2 exact, g3 exact, and g4 (the probe reproduces RC's own online accuracy exactly).

| Encoder frozen at task | 0 | 4 | 8 | 12 | 16 | 18 | final | RC |
|---|---|---|---|---|---|---|---|---|
| Online readout (T-S5) | 26.0 | 28.5 | 29.5 | 28.7 | 26.3 | 24.9 | – | 28.2 |
| Matched, 480 per class | 26.0 | 29.3 | 31.9 | 33.4 | 34.5 | 34.9 | 35.3 | 28.2 |
| Matched, 20 per class | 17.7 | 20.6 | 22.8 | 23.8 | 24.6 | 24.7 | 25.1 | 18.2 |
| Task-0 classes only, 5-way | 78.4 | 80.5 | 81.4 | 82.6 | 82.3 | 82.3 | 82.3 | 78.0 |

- D1: matched v18 − v8 = **+3.0 ± 0.1 \***.
- D2: v18 − RC = **+6.7 \***.
- D3, task-0 classes: +0.9 (4/5), not significant.
- D4, 20 per class: +2.0 \*.
- Tasks 0–3 classes: +2.5 \*.
- My pre-declared prediction (D1 ≥ −1) held.

**Meaning:**
- **The online features never got worse.** They kept improving for all 20 tasks, including on the first task's classes.
- The whole online decline is the readout. At v18 the online readout loses 10.0 points against matched data; at v8 it loses 2.4.
- **What decays is the statistics stranded in old coordinates.** A new encoder is better, but it has almost no history.
- The buffer's 20 trained-on items per class gave about the same LDA as 20 fresh items (online 24.9 vs matched-20 24.7 at v18). So on CIFAR-100 the gap is sample count, not the replay bias seen on MNIST.

**A2. The four-version bank against its own memory-matched buffer** (M = 3,439, from budget.json):
- BANK_4spread is 31.2 vs MIX_half(ER M = 3,439, RC) at 30.0: **+1.3 ± 0.3 \***.
- Against ER+bc at the same memory: +7.2 \*.
- At its own budget, a small spread bank does beat spending the memory on a bigger buffer.
- Caveat: "spread" knows the stream length.

**A3. Exact-transfer Sleep control** (CIFAR-10, 5 seeds). Copying the teacher exactly, with a maximum logit error of 3.8e-6, then running the same reset and continuation gave 52.2 against ER's 52.4 (−0.2, not significant). Full-rank distillation was **−8.2 \*** against the exact copy. So **all** of Sleep's loss is approximate transfer through distillation on 500 buffer images; none of it comes from the reset or the architecture.

**Wording corrections accepted from the audit** (applied above or noted here):
- "Established" became "supported in the tested development settings".
- Non-significant results are not ties: G3, G4 and R2 show no detected difference, not equivalence, and no crossover is shown.
- Bias-correction gain ÷ raw accuracy is not a "share of forgetting" (I1 withdrawn).
- The correction never changed sign; two formula families were wrong by different amounts (I4).
- 10.9% newest-class predictions alone does not prove bias; the held-out gain from α̂ is the real evidence.
- The 69.1% "either expert right" figure only bounds picking one expert's top-1 answer per item. Averaging can beat it (the audit gives a 3-class example).
- The Bayes stream mixture's collapse to ER is not a failure of its theorem. It optimises a different objective (current-stream log loss). A class-balanced reserve objective is the right one for balanced historical accuracy.
- The 86%-damage inference is withdrawn (section 3).
- "M2c forces logit scale up" is not proved: the zero step is always feasible, so it could stall instead. The supported conclusion is only that this implementation failed under these settings.
- Sleep: noise dreams failing rejects those dreams, not all synthetic data. Our analogue does not test the paper's upward distillation or RL dreaming, so it does not refute the paper's main claim.
- "Accumulate at arrival" is not enough while the encoder drifts. Statistics must be attached to an immutable encoder version, with provenance and counts, or carried over by a validated transport.

**Not accepted, or only partly:** none of the audit's points was wrong on the facts it had. Where it said "if all readouts were already refitted on identical data, the objection disappears", they were not, and the objection turned out to be decisive.

---

## 15. X1: is the readout bias weight scale? (CIFAR-100)

Plan: cifar/x1_plan.md. Results: cifar/results_x1.md, results/x1/. Origin: the geometry feed's claim that much of recency bias is a head weight-norm profile (feed/geometry/SYNTHESIS.md D2).

**Setting.** Post-hoc readouts on the final network of the T-S5 ER learner, 5 seeds. Fitting uses the 20-per-class reserve (never trained on); scoring is the full test set.

**Gates.** Self-test P1–P5 pass (after one test bug, row 20 of section 12, was fixed before any real result was read). g1: the loaded network reproduces its saved matched probe exactly on every seed. A+bc reproduces T-S5's ER+bc (22.7).

| Arm | Plain | + held-out bias correction |
|---|---|---|
| A linear head | 17.37 | 22.70 |
| C½ τ-norm 0.5 | 12.87 | **24.60** |
| C1 τ-norm 1 (equal norms) | 9.08 | 23.94 |
| D cosine NMC, reserve means | 21.41 | 18.00 |
| R LDA on reserve (20 per class) | **25.43** | – |

- **Primary:** equalising head norms makes accuracy *worse* (C1 − A = −8.3 \*). The declared scale share is −1.61. **My prediction (0.4 to 0.8) was wrong.**
- **Why:** in this replay learner the oldest classes have the *largest* norms (1.99 for task 0 down to 1.14 for task 19, Spearman −0.91), because replay keeps training them. The bias sits in weight–feature alignment, not norms.
- τ-norm 0.5 + bc beats A+bc by +1.90 \*. LDA on 20 reserve items per class beats A+bc by +2.7 \*.
- **Meaning:** the feed's scale story does not hold for our learner. A fitted held-out correction stays necessary (summary point 2 stands), and the best readout is again a class-statistics readout.
- **Not done:** no independent code review of X1. A trained cosine head was not tested (post-hoc only).

## 16. X2: carrying class statistics into new encoders (CIFAR-100)

Plan: cifar/x2_plan.md (two test-threshold amendments before any real run, a timestamp correction, one deviation, one byte correction, all logged there). Results: cifar/results_x2.md, results/x2/, analyze_x2.py. **Independent code review done: no result-changing bug.**

**Question.** Section 14 found the gap is statistics stranded in old coordinates. Can a learner that keeps only class statistics plus its buffer carry them into each new encoder?

**Method (online, "chained").** At each task end, regress the new encoder's buffer features on the old encoder's (after within-class centring: the "control variate", cv), move every stored class mean with that map, and discard the old encoder.

**Gates, all passed.** gR: the reconstructed buffer reproduces every version's saved statistics exactly. gS: the recomputed stream statistics match saved ones. gO: the oracle arm equals the saved probe to 1e-6. gB: ER+bc 22.70, BANK_4spread 31.21, RC 28.17 reproduce T-S5.

| Arm | Acc (100 classes) |
|---|---|
| O_full: oracle full-stream statistics in final features | 35.30 |
| B_only: final buffer only | 24.93 |
| **CH_cv: online chained transport** | **30.56** |
| CH_aff: shared affine map | 30.50 |
| CH_shift: per-class mean shift (SDC-like) | 26.87 |
| BANK_4spread (4.4 MB of versions) | 31.21 |
| **MIX3_CH_cv (ER+bc, RC, transported means)** | **32.08** |

- Q1: CH_cv − B_only = **+5.63 \*** (predicted +2 to +5). It closes 54% of the gap to the oracle.
- Q2: shared affine ≈ cv (−0.06), so **the MNIST failure of a shared affine map (section 4) does not replicate on CIFAR**; my prediction was wrong. The SDC-style shift is clearly worse (−3.68 \*).
- Q4: MIX3_CH_cv − BANK_4spread = +0.86 \*, with 4–8× fewer bytes. **Fragile:** seed 1 is only +0.04.
- Q5: about 92% of the oracle gap is in the means, not the covariance.
- **Meaning:** the first component here that beats the frozen-version bank at fewer bytes. It is replayed offline on saved runs, not yet run live inside the learner.

## 17. X3: waking dormant class statistics with a few key items (CIFAR-100)

Plan: cifar/x3_plan.md. Results: cifar/results_x3.md, results/x3/, analyze_x3.py. **Independent code review done** (results/x3/review.md, check script x3_review_check.py): no bug that changes a number, no leakage, **but three of my readings were wrong and are corrected below.**

**Question (Lucky's dormancy idea).** If the old encoder is gone and only its class statistics remain, can a map from today's features back into the old coordinates, fitted on k stored key items per class, reawaken the old readout?

**Measure.** 5-way accuracy *within* each task's own classes (task-aware), averaged over tasks 1–19. It tests whether old statistics still separate their classes, not the cross-task bias.

**Gates.** gR, gS, gO passed on all seeds; self-test S1–S3 pass.

| Arm | Acc |
|---|---|
| OLD: the old encoder with its own statistics | 78.94 |
| REACT_all / REACT_5 / REACT_1: old statistics, features mapped back with all / 5 / 1 keys per class | 77.25 / 75.34 / 66.60 |
| TRANS: X2's transported means | 74.49 |
| BUF_all / 5 / 1: keys alone, no old statistics | 67.80 / 57.76 / 35.35 |

- R1: old statistics add a lot over the keys alone: +31.2 \* (k = 1), +17.6 \* (k = 5), +9.5 \* (all). About 1.8 points of each comes from a covariance difference (review).
- R2: recovery is 98% of OLD with all keys, 95% with 5, 84% with 1; the oldest tasks recover least (95% with all keys).
- **R3 corrected after review.** As run, mapping back beat forward transport by +2.8 \*, but OLD and REACT used a 5-class covariance and TRANS a 100-class one. With matched covariance the lead is +0.9 \*, and on the oldest tasks transport wins by 3.1 \*.
- **"OLD is the ceiling" was false.** The final encoder with full statistics beats it by +3.1 \*.
- **Bytes corrected after review.** Keys' old-coordinate features cost 0.11 / 0.56 / 2.24 MB per version on average (k = 1 / 5 / all), against the 0.375 MB encoder body they replace. Only k = 1 is cheaper.
- **Meaning:** dormancy defined as recoverability works and is measured: a few key items wake old statistics to 84–98% of the old readout. As a memory design it is not yet cheaper than keeping encoders, except at k = 1.
- **Process caveats:** the plan file was last modified 19 s before the code, so the files cannot prove pre-declaration; keys were chosen from the final buffer, which uses knowledge of which items survive.

---

## 18. How reliable are the tests themselves?

Lucky asked for this separately from "are the results right". There are two layers:
- **Computational reliability:** given the design, did the code compute what it claims? This is strong.
- **Design reliability:** does the test actually measure the question, with fair comparisons? This is weaker, and it is where our errors have been.

### 18.1 What the evidence says about computational reliability (strong)

- **Exact reproduction gates.** Every recent experiment reloads saved runs and must reproduce earlier saved numbers exactly before reporting: T-S5 analysis (g2), matched probes (g1, g4), buffer reconstruction (gR), stream statistics (gS), the oracle arm (gO), and T-S5 headline numbers (gB). All passed. When result files vanished from disk twice, regenerated runs reproduced the old numbers exactly.
- **Planted self-tests with known answers** before each real run (C1–C13, K1–K10, sleep Z-tests, X1 P1–P5, X2 T1–T4, X3 S1–S3).
- **Independent reviewers recomputed numbers by hand** from the saved outputs for X2 and X3 and matched them.
- **Literature numbers** used in the feeds were re-fetched by an independent agent: 10/10 (geometry) and 11/11 (memory-and-drift) matched.
- **Track record:** section 12 lists 26 bugs, mistakes and process errors. Most were caught *before* they produced a reported number. The exceptions, where something already reported had to be withdrawn or superseded, are rows 2 (an in-sample claim), 6 (marathon T1 runs), 16 (an M2c status message), the T-S5 readout design (section 14) and rows 23–24 (X3 readings).

**My estimate:** a reported number in this file is very likely the number its code computes (I would put it above 95%). I have not found a case where a reported number was arithmetically wrong; the withdrawals above were wrong *readings* or wrong *designs*, not wrong arithmetic. This is a judgement, not a measured rate.

### 18.2 What the evidence says about design reliability (moderate; this is the real weakness)

**Track record.** Of the 7 experiments since the audit (A1, A2, A3, X1, X2, X3, and the T-S5 reading they corrected), design-level problems found *after* the run:

| Test | Design problem found after the run | Found by | Changed a conclusion? |
|---|---|---|---|
| T-S5 version curve | Readouts not matched: late versions had about 20 items per class | Lucky's audit | **Yes:** "features decay" withdrawn |
| X2 | Byte figures wrong (d = 512, not 256); MIX3 computed for the wrong arm; plan timestamps were my guesses | independent review | No (headline stands; byte ratio 6× became 4–8×) |
| X3 | Covariance mismatch between arms; "OLD is the ceiling" false; bytes compared against the wrong encoder size | independent review | **Yes:** R3 reversed for the oldest tasks; cost claim reversed |
| X1 | none found, but **X1 was not independently reviewed** | – | unknown |
| A1–A3 | none found, but **not independently reviewed as code** (they answered an external audit) | – | unknown |

So in the two experiments that got a full independent review, both had a design-level flaw, and one changed a conclusion. Self-tests did not catch any of these, because self-tests check the code against my design, not the design against the question.

**My estimate:** for a conclusion that has passed gates but *not* independent review, I would expect roughly a 1-in-3 chance that a design flaw changes part of the reading (not usually the headline number). This figure rests on very few cases (2 of 7 conclusions changed; 2 of 2 reviewed experiments had a flaw), so treat it as a rough judgement, not a rate. After independent review, the remaining risk is lower but not zero: reviews have so far found something every time.

### 18.3 Specific weaknesses of the test setup itself

1. **All recent tests reuse the same 5 training runs.** A1, X1, X2 and X3 replay the T-S5 ER runs. They are not independent replications. A flaw in those runs would propagate everywhere: the gates check consistency, not correctness of the training. The training code (cifar.py) has self-tests C1–C13 and an earlier review.
2. **No published number has been reproduced on this harness.** Lucky's test-reliability rule asks for that first. Our absolute accuracies are internally consistent but not calibrated against the literature. Relative comparisons are what we can trust.
3. **One learner, one scale.** Online ER, small GroupNorm CNN, CIFAR-10/100, 5 seeds, CPU. Every claim is conditional on that.
4. **The significance rule is weak and is applied many times.** |mean| > 2·SE plus 5/5 signs has sign-test p = 1/32 per comparison. Across dozens of comparisons, a few starred results are expected by chance. Small starred effects (for example X2 Q4, +0.86 with one seed at +0.04) should be treated as provisional.
5. **Pre-declaration is self-reported.** File modification times show order but not that a plan was unchanged; X2's plan times were my guesses, and X3's plan was saved only 19 s before the code. Test thresholds were amended twice in X2 before real runs, and those amendments were set by my expectation, not derived.
6. **Some tests are post-hoc or offline.** X1 tests post-hoc heads, not a trained cosine head. X2 and X3 replay transport offline; they have not been run live inside a learner.
7. **Hyperparameters.** lr was chosen on the seed-100 test set and not retuned for CIFAR-100; EWC λ came from a 3-value grid; ridge λ in X2/X3 was fixed in advance (conservative, but untuned).
8. **Measures differ between tests.** X3 uses a task-aware 5-way measure; X1, X2 and T-S5 use 100-class accuracy. They should not be compared directly.
9. **Infrastructure.** Result files vanished from the network mount twice (cause unknown). Gates were rebuilt on small JSON outputs that survived; no result was lost, but it is a fragility.

### 18.4 Bottom line

- **The numbers are reliable** for the code that produced them: gates reproduce exactly, self-tests pass, reviewers recompute the same values.
- **The tests are moderately reliable as tests of the questions they claim to answer.** Every independently reviewed experiment has had a design flaw; two of the last seven conclusions changed after outside checking. Treat any conclusion without an independent review as provisional.
- **Fix going forward:** independent review *before* reading results (of the design, not just the code); matched comparisons by construction (same covariance, same data, same readout); reproduce one published number on this harness; and replicate the key findings (X2, X3) on fresh training runs, not only the saved ones.
