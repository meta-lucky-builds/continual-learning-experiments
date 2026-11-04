# Formal specification of a continual learner (v1.3)

2026-10-28. Written for Lucky. v1.3 replaces v1.2, which is kept at notes/archive/formal-spec-v1.2.md.

**Why v1.3 exists.** Lucky shared a second external audit of v1.2. It came as four files in continual_learner_v12_audit:
- AUDIT_AND_RECOMMENDED_PATCHES.md (P01–P15);
- EDGE_CASE_LEDGER.md;
- check_counterexamples.py;
- counterexample_results.json.

I read all of it. I re-ran its script, and checked every claim myself with independent code (notes/v13-checks/). The audit is right, and v1.3 fixes everything it found.

**Beyond the audit.** I looked for structure that makes the design simpler rather than adding more guards. That produced five shared contracts (Part C) and a new protection mechanism, M2c. M2c looked clearly better in my first-order check, but that check was flawed: measured on held-out items, the quadratic brake does better (B5.0). On a nonlinear toy, M2c **did not beat replay**, and compute-matched replay beat both. A3 explains why, and what that implies.

**What this document is.** A specification of a family of continual-learning algorithms. Whether they beat tuned replay is **an empirical question**.

**Labels:**

| Label | Meaning |
|---|---|
| [src: P#] | the claim comes from that paper |
| [proved] | complete argument, under the stated assumptions, in exact arithmetic |
| [approx] | stated approximation |
| [design] or [guess] | my choice |
| [control] or [evidence] | see C5 |
| [corrected] | changed after review |

---

## Part A. Response to the second audit

### A1. Verdict

**All 15 patch areas are valid.** Their counterexamples reproduce exactly:
- 3.04 for group-rate post-scaling;
- −977 vs 27.6 nats for conditional novelty;
- 0.843 for the never-switching HMM;
- 0.808, 0.363 and 0.204 for power;
- the (6, 5) certificate false alarm;
- the (1, 0, 2) new-class steal;
- the H = 11ᵀ null-space case;
- the 0.25 → 0.953 usage jump.

**Every row's status in the 62-row ledger is correct.**

One error was purely mine. v1.2's delay formula said 6/I_y; with its own numbers it evaluates to 1.52/I_y, because I forgot to divide by κn = 4. The formula's whole premise was also wrong for a sticky filter (B8.2).

| # | Audit point | Verdict | Where fixed |
|---|---|---|---|
| P01 | Learner state, prediction path, evaluation target, A3 vs implemented scores, L-a overstated, "label shift" misnamed | Valid | C1, B0 (A3′, Bayes form of L-a, conditional shift) |
| P02 | Group rates break the SPD guarantee; momentum ≠ gradient; decay outside proof; separate counters | Valid (3.04 reproduced) | C2 (rates inside B, surrogate-descent label, separate k_m/k_v, step clip) |
| P03 | Floor domain, stable root, natural metric, HΔ_0 = 0, retention/floor conflict, numerical contract | Valid | C2 facts 4–6 and the numerical contract. I add a proof that ρ̂ ≤ 1 in B⁻¹ for **both** (Q) and (K) |
| P04 | Sketch over- and under-protects; the chosen step exploits it; head nonlinearity; Θ_P scope | Valid. I measured exploitation at a median **12×** on a random toy | B5.3 (M2b demoted to a variant with independent validation), B5.2 (head excluded from M2a) |
| P05 | Conditional novelty suppressed; causal filter order; label floor not normalised; quarantine; one router | Valid (−977 reproduced) | B8.1 (new_cond, smoothing), C1, B7.1, B8.4 |
| P06 | Delay arithmetic; HMM is not a static test; first-batch theorem | Valid | B8.2 (exact threshold L_min, simulation), B0.3 |
| P07 | Certificate failure ≠ forgetting; new classes; reference timing; quantisation; evaluation vs control references | Valid | B7.2 (F⁻ as the trigger, r(x) headroom, versioned registry, observed-event references) |
| P08 | Usage jump; brake restore not safe; exposure "cap" misnamed | Valid | B5.4, B6.2, B7.3 (boost-only default) |
| P09 | Snapshot schema, vectors, migration, invalidation, ring limits | Valid | C3, B9.1–B9.2 |
| P10 | Power depends on discordance; adaptive acceptance | Valid. My envelope check: 80% power needs n ≈ 170 if discordance ≤ 0.4 | B9.3 (control mode by default) |
| P11 | Budget feasibility, peak memory, quotas, metadata | Valid | C4, B6.3 |
| P12 | Transactions S0 → S1 → S2 | Valid | C4, B10.4 |
| P13 | Loop defects (no insertion, warm-up order) and empty-state policies | Valid | C1, B14, B12 rows 73–90 |
| P14 | D1–D4 not additive; refit decoders; censored savings; claim families | Valid | B2.2, B2.1, B15 |
| P15 | Verification-suite corrections and new tests | Valid | B3 (T1–T17, V1 parity, V6 conditions) |

**One qualification of the audit.** Its matrix-free exact operator Hv = Σ JᵀJv is correct, but I estimate it costs about 20× a training step at n_H = 64. So it is a diagnostic, not a training path (B5.3).

### A2. What I found by going further (the "elegance" part)

The audit's own conclusion was to use fewer, shared contracts instead of more special-case guards. I agree, and v1.3 is organised around **five contracts** (Part C). Every mechanism must obey them:
1. one causal event order;
2. one update geometry;
3. one versioned schema;
4. one transaction and budget policy;
5. a strict evidence/control split.

Searching for deeper structure, I found six things. Each is checked in notes/v13-checks/. A second, independent reviewer then checked v1.3 itself and found 20 defects, including errors of mine in N3 and N4; all are fixed below and marked [corrected].

**N1. Every protection method is one proximal map** [proved]. Write the Adam step as Δ_0 = −Ba. Then every protected step is the B⁻¹-proximal map of Δ_0 under a penalty 𝓟 with 𝓟(0) = 0 and 𝓟 ≥ 0: a quadratic (Q), or the indicator of a constraint set (K), in which case it is a projection. Two facts follow for both:
- surrogate descent, ⟨a, Δ⟩ ≤ −‖Δ‖²_{B⁻¹};
- non-expansion, ‖Δ‖_{B⁻¹} ≤ ‖Δ_0‖_{B⁻¹}.

So the floor's ratio ρ̂ really is a fraction in [0, 1] in that metric. The floor, group rates and protection become one object (C2).

**N2. The raw logit metric charges for harmless logit shifts** [my analysis]. The audit found that the certificate is broken by a common shift of all logits. The *protection* metric ‖JΔ‖² has the same flaw. Centring with P_⊥ fixes both together. The centred certificate then pairs with the metric through max_j(d_j − d_y) ≤ √2‖P_⊥d‖, with 0 violations in 5,748 certified random cases (B5.0).

**N3. Protection must be judged on held-out items; there, penalties beat constraints** [toy evidence; corrected after review].
- My first frontier said per-item margin constraints **dominate** the quadratic brake (zero damage at 79.6% of the step's progress, vs 13.4% for the brake). **That was an in-sample tautology**: damage was measured on exactly the constrained set.
- Measured on 200 held-out items, with Jacobians that share structure so protection can transfer, the quadratic brake does better: held-out damage 64 at progress 171 (λ = 3), vs 77 at 160 for all-rival constraints (B5.0).
- ~~Constraints memorise; penalties generalise.~~ **Withdrawn (A6):** zero-slack constraints did protect held-out items (A4) and the born-label rival rule added +9.5 (A5). The defensible statement: this constraint implementation has not earned its compute against corrected readouts.

M2c is GEM-family (per-item GEM, mGEM arXiv 2410.00868). mGEM's Proposition 1 bounds memory-vs-population gradient agreement through a search-class complexity term; it motivates checking held-out transfer but does **not** predict this specific outcome or rank constraints below penalties [corrected, A6]. M2c stays a gated variant, and **every protection claim in v1.3 is now measured on items not used to build the protection.**

**N4. Constraints give a cumulative guarantee on the pairs they keep constraining** [proved, first order; scope corrected after review]. With slack computed from **tracked** margins, s = κ_s·max(0, m̃ − m_floor) and m̃ updated by cᵀΔ every step, no pair kept continuously in the constraint set crosses min(m_0, m_floor) over any number of projected steps.
- With slack only refreshed every R_pre = 25 steps, a linear counterexample falls to −5.5, so tracking is required (B5.1).
- Pool resampling, nonlinearity and baseline steps all break it in the real algorithm. It is a statement about the constrained pairs, not about retention.

**Conflict detection is per class** [corrected after review]: a global passed-fraction ratio hides a minority conflict (one blocked coordinate in 16 still gives ρ̂ = 0.94). Persistent per-class blocking is the conflict signal for both (Q) and (K), and it drives adapter creation from M5, or a logged release below M5 (B5.1, I4).

**N5. Exact detectability threshold for the sticky filter** [proved]. For two symmetric states with constant evidence, the posterior **settles** above π if and only if L ≥ O_π(hO_π + 1 − h)/((1 − h)O_π + h), which is log L_min = 0.166 at h = 0.02, π = 0.9. At or below it, the filter never crosses π (B8.2). With more states, thresholds and delays are found by simulation; no general "more states raise the threshold" rule holds (in marathon T7, 3 states made detection easier). This replaces a wrong delay formula with a constant-evidence boundary. **[A5 correction]** It is *not* a necessary condition when per-batch evidence is noisy: in marathon T7b, detection happened below L_min (11/20 runs with 2 states, 17/20 with 3).

**N6. Inferential reactivation is impractical at probe scale** [computed]. Power depends on discordance (n ≈ 170 for robustness). So reactivation runs in **control mode**, and its claims come only from confirmation streams.

### A3. First nonlinear test of M2c (toy; not evidence for real networks)

**Setup** (notes/v13-checks/toy_cl3.py; earlier versions toy_cl.py, toy_cl2.py). Synthetic class-incremental task:
- 10 Gaussian classes in 20 dimensions with a shared linear mixing, separation 0.6, so ER is far below ceiling (the reviewer showed the default separation 1.6 saturates at ER 99.4%);
- 5 tasks of 2 classes;
- an MLP 20-64-64-10 trained with Adam (lr 1e-3);
- 300 steps per task, batch 16, with 4 replay items per step;
- memory of n_mem = 5 items per class.

**M2c as specified** (final harness, toy_cl3.py): k_r = 2 rivals, a pool of 32 buffer items that are correct now, κ_s = 0.2, m_floor = 0.5, slack from **tracked** margins, constraints refreshed every R_pre = 1 or 25 steps. The pool draws from its own RNG, so ER and M2c see **identical streams and replay draws** per seed, and differences are paired. The QP is solved on the dual.

**Compute-matched ER (er_cm)** spends M2c's extra per-step item budget (32 pool items) on extra replay instead. It draws more replay from the same RNG, so it is compared per seed, not item-paired.

**Results** (test accuracy over all 10 classes at the end, 5 seeds; paired differences are mean ± standard error):

| Method | Accuracy | Replay-buffer accuracy | Difference vs ER | Difference vs er_cm |
|---|---|---|---|---|
| fine-tune (toy_cl2) | 18.2 | — | | |
| GEM, one constraint per task (toy_cl2) | 39.1 | — | | |
| ER | 41.5 | 94.8% | — | |
| **M2c, R_pre = 1** | **41.2** | 96.8% | −0.3 ± 0.5 | −3.7 ± 1.3 |
| **M2c, R_pre = 25, tracked margins** | **41.2** | 96.8% | −0.4 ± 0.4 | −3.7 ± 1.2 |
| **compute-matched ER** | **44.9** | 99.6% | **+3.4 ± 0.9** | — |
| joint, upper bound (toy_cl2) | 73.5 | — | | |

An earlier, less faithful harness (toy_cl2.py: unpaired streams, k_r = 3, constraints on every pool item) gave M2c 41.1 at n_mem = 5, and at n_mem = 6 (3 seeds) ER 45.1, M2c 45.8, ER on half the memory 34.0, M2c with half the memory used only as constraints 42.4. All are consistent with the table.

**What this shows, honestly:**

1. **M2c protects what it constrains, a little.** Buffer accuracy rose from 94.8% to 96.8%.
2. **On the test population it does not beat ER**, and **compute-matched ER beats it by 3.7 points**, in 4 of 5 seeds by more than 3 points. Spending the same compute on replay is clearly better here.
3. **Tracked margins make refresh staleness harmless on this toy.** R_pre = 25 matched R_pre = 1 to within 0.1 points.
4. **This matches the held-out first-order check (B5.0)** for this configuration: the protection transferred poorly. (A4 and A5 later showed zero-slack and born-label rivals do transfer; the general claim is withdrawn, A6.) It is the coverage limitation (B16 bet 1), and GEM's known generalisation gap (mGEM, Proposition 1).
5. **Constraint-only items are worth something, but less than replay** (toy_cl2, n_mem = 6: 42.4 vs 34.0 for ER on half the memory, vs 45.1 for ER on all of it).

**Consequences for the design:**
- M2c is a **gated variant**, not the lead. The toy prior for E4c is now **negative** against compute-matched ER.
- ~~The bar for every mechanism is compute-matched ER.~~ **Corrected by A4:** compute-matched ER's gain was loss balance, not compute. The bar is a debiased M1 at the same compute.
- **The bottleneck is coverage**, i.e. generalisation from finite memory, **not** per-step geometry. A better solver cannot fix it.
- **Next to test:** M2a (the (Q) form that generalised better held out), and coverage-widening protection (augmented or neighbouring versions of stored items, feature-space neighbourhoods). These are hypotheses. This toy cannot test augmentation fairly, because its Gaussian noise *is* the data distribution.
- **None of this is evidence about real networks**, in either direction.

**Harness checks run** (and what is not checked):
- train and test are separate draws (shape and no-overlap asserts);
- surrogate descent asserted on every projected step;
- constraint-only items never enter replay (assert);
- ER reproduced exactly (41.52) across harness versions with the same seeds, a determinism check;
- fine-tune lower bound and joint upper bound reported;
- the toy has no group rates, no step clip and no registry growth; those are covered by the spec's tests (B3), not by this toy;
- "test never trained" is true by construction (test items have no path into training), not asserted. The earlier docstring claimed an assertion; it was corrected.
- The harness was reviewed independently; its defects (saturated default, unpaired streams, no compute matching, docstring) are all fixed in toy_cl3.py.

### A4. Diagnosis: why M2c tied and why compute-matched ER won [new; notes/v13-checks/why-m2c-failed.md]

Five seeds, identical streams; the harness reproduces A3's numbers exactly.
- **The dominant failure is readout bias, not lost features.** Under ER, 87% of old-class errors are predicted as a newest-task class. Refitting only the last layer on the frozen features gives 70.6%, against 73.5% for joint training.
- **Mechanism.** The softmax gradient p_j − 1[j = y] pushes every old logit down on every new item. With 16 new and 4 replay items per step, the pull per class is about 16:1 against old classes.
- **The cheap fixes win at 1× compute.**
  - Weighting the same 4 replay items ×9 gives 47.8; ×20 gives 50.0.
  - ER-ACE gives 57.8 vs 39.9 under the same buffer policy, at the cost of the newest task (53% vs 90%).
- **M2c as specified hardly acted.** Its slack lets well-fitted buffer items lose a lot of margin before any constraint binds. It removed 19% of first-order damage on its pool and 13% held out.
- **With zero slack it acts, and its protection transfers.** It removes 86% of held-out first-order damage, which corrects A3's "constraints memorise". But the gain is only +3.1 ± 0.8, to 44.6, at a large compute cost.
- **Open.** Most of the accuracy loss is not captured by step-wise first-order margin damage; that needs per-item nonlinear logging.
- **Consequence.** Implement the spec's M1 faithfully first, including its balanced, refreshed head. Test protection only on top of a debiased M1.

### A5. Sharp-test marathon: corrections forced by the evidence [new, 2026-10-28; notes/marathon/results.md]

These are MLP-scale tests on the toy and on Split-MNIST, with 5 seeds, paired, under a pre-registered rule. They isolate mechanisms; they do not show that any full system wins.

1. **The readout bias tracks the stream prior** [revised in A6: direction and order of magnitude only]. Post-hoc logit adjustment by the last task's training-batch class frequencies (τ = 1, no fitted parameter) recovers 66–160% of the bias-correction gain across settings.
   - It predicts *no* bias under ER-ACE, and that is observed.
   - Consequence for B4 and B10: the M1 readout must be prior-corrected or class-balanced by construction. Margin protection of head rows (M2c) targets a symptom.
2. **Floor control** [revised in A6: the fixed-feature win holds only with full-history statistics]. On Split-MNIST, ER-trained features are no more linearly separable than the untrained network or raw pixels: oracle LDA 86.7 vs 87.2 vs 87.5.
   - Frozen random ReLU features with exact streaming class statistics reach **94.0**, against at most 72 for every trained method tested.
   - Consequence: claims of continual *feature* learning must be made where frozen features fail, against this baseline (B2 and B15 gates).
3. **Stored statistics go stale catastrophically** (≈20% everywhere). On MNIST the drift is class-specific: a shared affine correction fitted leave-class-out gets 68 vs 87.
   - Consequence for B4 and B6: statistics are valid only for frozen features, or when re-estimated at the current weights. The re-estimation gap is a sample-size gap (about 50 items per class reach the oracle within 3 points).
4. **M4 head-row snapshots are withdrawn as evidence.** Their recovery benefit vanishes under a bias-corrected readout: they are worse than keeping the current rows (−25 toy, −57 MNIST). Old rows do not fit drifted features.
   - Consequence for B9: a snapshot is valid only together with the feature state it was taken in.
5. **H1 (fresh vs stale estimates) holds only at a small penalty.** Fresh Fisher beats stale at λ ≤ 1, but loses badly at λ ≥ 10 on MNIST, where it blocks plasticity.
6. **Centring the head is not a general fix.** The bias moves from the common term into the item-specific term.
7. **T7.** L_min is a constant-evidence boundary, not a necessary condition (corrected in N5, B8 and B12).
8. **The M2c rival rule is necessary, but not yet sufficient to justify M2c.** Adding the rivals for "labels born since the last refresh" lifts zero-slack M2c from 44.6 to 54.1 (+9.5, 5/5 seeds).
   - That is still below free readout fixes: logit adjustment 57.2, ER-ACE 57.8.
   - Consequence for B5: M2c must be evaluated *on top of* a prior-corrected readout.

### A6. Second v1.3 audit (DEEP_AUDIT_AND_REDESIGN, 2026-10-29): what I verified, what I fixed, what the new tests changed

**Math defects: all the definite ones were real and are fixed in place.** I checked each against this text, and re-ran the audit's `check_math.py`; it reproduces to 1e-16.
- B14: margins are now tracked from the **applied** (clipped) step. Pre-clip tracking can report +6.9 while the actual margin is −0.5.
- Numerical contract: dual damping relaxes hard constraints, so the original feasibility is now re-checked after the solve.
- B5.5: the M2a rank uses the full operator, d_out·rank(A_1).
- B5.0: the sketch counterexample is now centred.
- N5, A3′, B8.2: no L_min necessity claim remains, and no "more states raise the threshold" rule.
- B8.2: the Bayes first-batch bound is max_y Σ_c P(c|I) P(y|x,I,c).
- B10.1: adapters are triggered per class (β_j), not by the global ρ̂.
- B4.5: the covariance ridge has an absolute floor.
- "Constraints memorise; penalties generalise" is withdrawn, and so is the mGEM over-reading.
- **Still open** (listed, not yet rewritten): head rebuilds as dirty events for C and m̃ (7.10); reference/certificate scope (7.12); the novelty-state transitions and mixed-context batches (7.13); per-class gradient cost and the small-pool split (7.14).

**The audit's empirical interpretations, tested on the marathon harness** (5 seeds; notes/marathon/results.md §0b):
1. *"Use the loss-induced prior q_eff."* Correct in principle, since the spec's loss (separate equal means) induces a new/old ratio of 4, not 16. **But it fails quantitatively.** Logit adjustment by q_eff at τ = 1 under-corrects (−5.9 toy, −12.5 MNIST vs biascorr). With a 40-item buffer, up-weighting replay fits the stored items instead of moving old-class test logits.
   - **M1 default (B4):** a data-driven class-balanced correction, re-estimated at each head refresh. Closed-form logit adjustment is an ablation.
2. *"τ > 1 may be a global logit scale."* Not supported: the newest classes calibrate at T ≈ 1.
3. *"The 94% fixed-feature result is not a clean single factor."* Confirmed and sharpened.
   - On the same frozen checkpoints, learned features beat every fixed alternative by 9–18 points at 5 samples per class, and lose only with full-history statistics.
   - Width also matters: random ReLU features give 89.1 at 256-d, 94.0 at 1024-d and 96.4 at 4096-d.
4. *"Historical sample size is the mechanism."* Confirmed, plus a new effect: **replayed items are biased statistic sources.** Old-class LDA from the 5 replayed items is 49.4, vs 73.8 from 5 fresh items (MNIST).
   - Holding some memory out of replay helps LDA (+6 to +7), but does not beat biascorr at a 5-per-class budget.

**Direction adopted for the next build (a hypothesis, not a result):** the audit's two-path design.
- A **stable expert**: an immutable encoder with exact streaming class statistics accumulated at arrival, never from replayed items.
- A **plastic candidate**: ER with a data-driven balanced readout.
- A causal probability mixture of the two. Its regret is ≤ log 2 against the best fixed expert in cumulative log loss; this is not a forgetting guarantee.
- Promotion of a learned version only together with its compatible decoder and a recorded data-coverage count.
- M2/M4 machinery is tested only for the residual failure, at matched resources.
- The decisive test needs a setting with a verified gap between joint learning and the best frozen features (CIFAR is a candidate).

---

# Part C. The five contracts (read before Part B)


Every mechanism in Part B obeys these five contracts. Most v1.2 defects were mechanisms that were each correct in isolation but broke one of these when combined. A mechanism that cannot satisfy a contract is not built.

### C1. One causal event order [corrected after audit P05, P13]

**Learner state.** The state is

$$S_t=(\theta,\phi,\text{optimiser moments and counters},\text{heads},\text{adapters},\text{filter},\mathcal L_v,\text{stores},\text{references},\text{caches},\text{RNG states},\text{versions}),$$

where 𝓛_v is the versioned label registry (C3).

**Prediction.** The deployed prediction is ŷ_t = F(S_{t⁻}, X_t). Here S_{t⁻} contains the current inputs' evidence but **no current labels**. Every risk and certificate names the path it is computed on:
- **deployed:** filter plus routing;
- **shared:** the global head;
- **forced-context:** a given c.

**Per batch, in this order and no other** (each line names its owner section):

1. Receive X. Preflight the resource ledgers (C4). Freeze the pre-batch predictive state, i.e. context heads and statistics.
2. Propagate the filter prior through T. Multiply by input evidence to get p_t⁻ (B8).
3. Predict on the deployed path over the *current* registry 𝓛_v, and save the predictions. A label not yet in 𝓛_v cannot be predicted (C3).
4. Reveal Y. Log the saved predictions (prequential risk). Score the **frozen** context label models to get p_t⁺. A label not in a context's registry gets the **open-set probability** p_open: a small constant ε_open/|𝓛_v| for existing contexts, and a larger one for the novelty states, so an unregistered label is evidence for novelty rather than a −∞ score [corrected].
5. Process filter decisions once, in this precedence: switch > new_cond > new_in > stay (B8.3). A switch **chooses** its snapshot transaction here; it is **executed** at step 17.
6. Register newly revealed labels (C3). Migrate dependent schemas and caches.
7. Assign each new item an **immutable role** at first sight: train, or probe-reserved (B7.1).
8. Update usage timestamps from **all** observed items, probes included (B5.4).
9. Sample replay from the **pre-existing** buffer. An empty replay set contributes loss 0.
10. Compute the training loss on train-role items only. An empty set contributes loss 0.
11. Update optimiser moments and counters (C2).
12. If protection is not initialised, or its cache is dirty, take the baseline step and log it. Otherwise take the protected step (C2, B5).
13. Apply the remaining mutations: decay (B4.2), then φ.
14. Insert eligible train-role items into the buffer (B6).
15. Update context heads from permitted train-role items only, after step 4's scoring (B8).
16. Write references at their causal timestamps (B7.2).
17. Run scheduled or triggered transactions within budget (C4): refresh, monitor, reactivation, consolidation.
18. Rebuild dirty caches. A protected step can only use a cache built in a previous batch.
19. Verify the persistent and peak ledgers. Log uncovered or evicted contexts.

**Warm-up** is counted in **gradient-used items**, n_warm = 1,000. Protection, monitoring and filter decisions begin only after their own initialisation transactions have committed.

**Level isolation.** M1 and M2 never run M3 filter, commit or reset logic.

### C2. One update geometry [corrected after audit P02, P03]

**Step matrix.** Learning rates live **inside** the step matrix. They are never applied afterwards. Over trainable coordinates i, with group rate η_{g(i)} and Adam's second moment v̂,

$$B_{ii}=\frac{\eta_{g(i)}}{\sqrt{\hat v_i}+\epsilon},\qquad \Delta_0=-Ba,\qquad \lVert w\rVert_{B^{-1}}^2=w^\top B^{-1}w.$$

Frozen coordinates are removed from the system, not given rate 0.

**Direction a.** a = m̂ (Adam). Every descent statement below is **surrogate descent**: ⟨a, Δ⟩ < 0. It is not descent of the current loss. Example: g = 1 and m̂ = −1 gives an uphill step even with no protection [audit, verified]. The harness logs the fraction of steps where the current batch gradient has gᵀΔ ≥ 0. A variant that recomputes with a = g when that happens is gated (E3).

**Every protected step solves one problem**, the B⁻¹-proximal map of the Adam step:

$$\Delta^\star=\arg\min_\Delta\ \tfrac12\lVert\Delta-\Delta_0\rVert^2_{B^{-1}}+\mathcal P(\Delta),$$

where 𝓟 is one of:
- **(Q) a quadratic penalty**: 𝓟(Δ) = (λ/2)ΔᵀHΔ, with H ⪰ 0 (M2a, M2b);
- **(K) a constraint indicator**: 𝓟(Δ) = 0 if CᵀΔ ≥ −s, else +∞, with s ≥ 0 (M2c).

**Facts** [proved; both solvers checked numerically in notes/v13-checks/verify13.py]:
1. **(Q)** Δ* = −(B⁻¹ + λH)⁻¹a. For H = CCᵀ,

   $$\Delta^\star=-Ba+BC(\lambda^{-1}I+C^\top BC)^{-1}C^\top Ba.$$

   - Check: maximum relative error vs a dense solve 4·10⁻¹¹, with B spanning 10 decades.
   - λ = 0 is a separate code branch that returns Δ_0 exactly.
2. **(K)** has a solution, and it is unique, because 0 is feasible and the objective is strictly convex. Writing it as Δ* = Δ_0 + BCμ*, μ* solves the dual

   $$\min_{\mu\ge0}\ \tfrac12\mu^\top(C^\top BC)\mu+\mu^\top(C^\top\Delta_0+s).$$

   - Δ* = Δ_0 exactly when Δ_0 is feasible.
3. **Surrogate descent and non-expansion, for both forms** [corrected after review]. Δ* = prox_𝓟(Δ_0) in the B⁻¹ metric, with 𝓟 convex, 𝓟(0) = 0 and 𝓟 ≥ 0. For (K) this is the B⁻¹-projection onto the feasible polyhedron, which contains 0. For (Q) it is a proximal map, **not** a projection. Hence

   $$\langle a,\Delta^\star\rangle\le-\lVert\Delta^\star\rVert^2_{B^{-1}},\qquad \lVert\Delta^\star\rVert_{B^{-1}}\le\lVert\Delta_0\rVert_{B^{-1}}.$$

   - **Proof for (K).** Write ⟨·,·⟩ for the B⁻¹ inner product. The projection inequality ⟨Δ_0 − Δ*, 0 − Δ*⟩ ≤ 0 gives ‖Δ*‖² ≤ ⟨Δ_0, Δ*⟩ = −⟨a, Δ*⟩ (since Δ_0 = −Ba). Non-expansion of the projection, with P(0) = 0, gives the norm bound.
   - **Proof for (Q).** With M = B⁻¹ + λH ⪰ B⁻¹: aᵀΔ* = −aᵀM⁻¹a = −Δ*ᵀMΔ* ≤ −‖Δ*‖²_{B⁻¹}. Non-expansion follows from M⁻¹ ⪯ B, i.e. B^{1/2}MB^{1/2} ⪰ I. (Equivalently, the prox inequality ⟨Δ_0 − Δ*, 0 − Δ*⟩ ≤ 𝓟(0) − 𝓟(Δ*) ≤ 0 gives both, for any such 𝓟.)
   - **Check.** 0 failures out of 200 (K) and 500 (Q) random cases.
4. **Floor, in the step's own geometry.** Define ρ̂ = ‖Δ*‖²_{B⁻¹} / ‖Δ_0‖²_{B⁻¹}. By fact 3, ρ̂ ∈ [0, 1], so it **is** a passed fraction in this metric [corrected: the Euclidean ratio could exceed 1].
   - If ρ̂ < ρ_min, the blend Δ = Δ* + λ_b(Δ_0 − Δ*) solves α_qλ_b² + β_qλ_b + γ_q = 0 with every inner product taken in B⁻¹: α_q = ‖Δ_0 − Δ*‖², β_q = 2⟨Δ*, Δ_0 − Δ*⟩, γ_q = ‖Δ*‖² − ρ_min‖Δ_0‖² < 0. By fact 3, β_q ≥ 0 always, so the root is computed in the cancellation-free form

     $$\lambda_b=\frac{-2\gamma_q}{\beta_q+\sqrt{\beta_q^2-4\alpha_q\gamma_q}}.$$

     [corrected: the draft's b < 0 branch was dead, and its a, b, c clashed with the Adam direction a.]
   - The root exists in (0, 1] and preserves surrogate descent, valid for 0 < ρ_min ≤ 1 and Δ_0 ≠ 0.
5. **Priority when retention and plasticity conflict** [new; audit P03]. No rule can guarantee both: a single parameter with a protected output and a nonzero new gradient is a counterexample.
   - **(Q)** applies the floor: it is a soft brake, so floor first.
   - **(K)** is **retention first**: no floor. Conflicts are detected **per class** of new items (β_j, B5.1), not by the global ρ̂, which hides minority conflicts [corrected]. A conflict event is consumed by M5's adapter split, or below M5 by a logged, time-limited release of the blocking pairs.
   - Either way the event is logged, never hidden.
6. **Exact no-effect condition** [corrected]. For (Q), Δ* = Δ_0 iff HΔ_0 = 0, **not** Ha = 0. Counterexample: H = 11ᵀ, D = diag(1, 2), a = (1, −1).

**Numerical contract**, for every solve:
- fp64 accumulation of every Gram matrix;
- symmetric (Cholesky) factorisation with diagonal damping 10⁻¹⁰·tr/n. **Damping relaxes the hard constraints** (one constraint Δ ≥ 0, Δ_0 = −1, Gram 1 + ε gives Δ = −ε/(1 + ε) < 0) [A6]. So after the solve, check the **original** constraints Cᵀ Δ ≥ −s; on a violation beyond tolerance, use rank-revealing (QR/SVD) active-set solves or the feasible scaled step below (K only; Q has no slack vector and falls back to Δ_0 with the event logged);
- finite-value checks;
- KKT residual < 10⁻⁸ relative for (K);
- the check aᵀΔ < 0, unless Δ = 0.

**On failure**, take the **feasible scaled step** Δ = α Δ_0, with α = min(1, min_{i: c_iᵀΔ_0 < −s_i} s_i/(−c_iᵀΔ_0)). It is feasible and surrogate-descent for α > 0, and Δ = 0 when α = 0. Log the failure.

**Stored G or C in fp16 is a quantised surrogate.** Exactness and unbiasedness statements refer to the unquantised quantities.

**Mutations after the solve.** Decay, clipping and retraction are outside these theorems. Any whole-step claim must be checked on the final displacement.

**Optimiser state.**
- m and v have **separate** bias-correction counters k_m and k_v.
- A switch resets m and k_m only.
- After a reset, carried v on long-inactive coordinates can give large steps. The step is scaled **uniformly** by α = min(1, min_i 10η_{g(i)}/|Δ_i|). Uniform scaling preserves feasibility (s ≥ 0) and both fact-3 inequalities; a coordinatewise clip preserves neither [corrected].

### C3. One versioned schema [new; audit P07, P09, P13]

**Label registry 𝓛_v.**
- Labels get stable IDs, are never renumbered, and each addition bumps the version.
- A label first revealed at step t gets a head row **after** step t's prediction.
  - The row is initialised from a dedicated RNG, with zero moments and its own counters.
  - Its decay anchor is its birth value. It has no θ_0 row.
  - Matching rows are created in φ and in each context head.
- Prequential prediction of a never-seen label is always wrong, and is logged as "unknown label".

**Reference record per monitored item** (B7). The record holds:
- item ID;
- source ID, which survives augmentation and context reassignment;
- reference time;
- registry version;
- reference prediction;
- reference correct bit.

Full reference logits are kept only for a drift-diagnostic subset of 4 items per class.

**Snapshot record** (B9). It holds:
- target checkpoint time;
- registry version and the label-ID → row map;
- topology version;
- **every** parameter tensor: matrices as base + low-rank difference, vectors exact;
- normalisation buffers;
- head and adapter identities;
- base reference;
- compression metadata and validation count and result;
- a provenance label: exact, inferred-boundary, or contaminated.

**Schema migration.** An operation that combines states with different registry or topology versions works as follows:
1. Map rows by label ID.
2. Take historical rows for labels that both states contain.
3. Keep current rows for labels absent from the old state.
4. Revalidate on the full current label set.

The result is a **migrated proposal**, not exact restoration.

### C4. One transaction and budget policy [new; audit P11, P12]

**Transactions.** Every multi-state mutation runs as a versioned transaction: reactivation, consolidation, adapter creation and merge, head growth, base creation.

```
begin: freeze validation sets and their denominators; reserve peak memory (else abort)
checkpoint S0; step A → checkpoint S1; step B → S2
validate S2 (measured accuracy/risk + coverage) → commit, or restore S1 and validate → commit, or restore S0
on commit: mark dependents dirty (dependency graph below); rebuild before their next use
```

**What a restore covers.** It restores:
- θ, φ;
- optimiser moments and counters;
- heads, adapters, filter statistics;
- the pool and the B/G/C caches;
- references, if they were changed;
- buffer counters;
- RNG states.

**What a restore never undoes.** It never undoes physical stream events, observed labels, or the evaluation ledger.

**Offline steps** (consolidation replay) advance optimisation counters, but **not** stream time or recurrence timestamps.

**Dependency graph.** Each cache is marked dirty, and rebuilt before its next use, when the things it depends on change:

Ordinary gradient steps are **not** dirty events: caches built from θ are refreshed on their own schedules (R_pre, R_mon), and their staleness is the stated approximation. Only **structural** events dirty caches [corrected: the draft dirtied the pool on every θ change, which made every step a baseline step].

| Cache | Marked dirty by these structural events |
|---|---|
| pool G/C, tracked margins, A_u, λ | reactivation or consolidation commit; topology change; adapter creation in θ_P |
| C under head growth | **migrated exactly** by zero-padding, plus new-rival columns (B5.1); not dirtied |
| class statistics and head rebuild | reactivation or consolidation commit; topology change |
| filter statistics | context commit, merge or dissolve; reactivation commit |
| snapshot ring | topology change (ring entries are re-schemed, C3) |
| references | registry growth adds the new-label check only (B7.2); not dirtied |

**Two budgets.** 𝔅_pers is the persistent bytes beyond θ and standard Adam state (B6.3). Its split must include **every** persistent component: buffer, probes **and quarantine**, pool columns, references, snapshots and bases, adapters, head-row growth with its optimiser state, φ and θ_0 where used, filter and context metadata [corrected]. 𝔅_peak is the working memory: rollback copies, solver and compression workspaces, and candidate snapshots.

**Allocator rules.**
- Every allocation is **preflighted**.
- Eviction happens *before* allocation.
- An optional event that cannot reserve its peak is aborted and logged.
- Quotas are **desired**, not mandatory. A context below its coverage minimum is marked ineligible for the operation that needs it.
- Context metadata and filter states are evictable.
- Bases are garbage-collected at reference count 0.
- The end-of-step assertion is a test, not the policy.

Head-row growth and its optimiser state are charged to 𝔅_pers.

### C5. Evidence versus control [audit P10, P14]

Every quantity carries one tag:
- **[control]**: used by the learner to decide something; reused adaptively; no statistical claim;
- **[evidence]**: computed once, on data the design never touched (confirmation or final streams, B2.3).

The tag is shown in every report.

**What is [control]:**
- McNemar acceptance in reactivation, by default;
- certificates;
- alarms;
- 𝒬^sel and 𝒬^acc results;
- the sketch-validation pool.

An *inferential* reactivation mode exists only with fresh acceptance blocks and a prespecified error budget α_k = α/[k(k+1)] (B9.3).


## A7. Split CIFAR-10 evidence (2026-10-29; notes/cifar/results.md)

The setting is a small CNN trained online, M = 500, with 5 seeds, CPU only. The plan was pre-declared, the self-tests passed, and the code was independently reviewed.

1. **CIFAR is a valid feature-learning test.** Jointly trained features beat the best random features by 17–19 points under full-data LDA. That revises A5.2, whose claim held on MNIST only.
2. **The two paths add.**
   - A fixed 0.5/0.5 probability average of ER+bc and a random-conv streaming-LDA expert beats both: +3.5 ± 0.4 over the better one, 5/5 seeds. Their errors are complementary: either one is right on 69% of test items, against 53–54% for each alone.
   - The stable expert alone ties ER+bc: +1.4, not significant.
3. **Mixture weights must not come from stream log-loss.**
   - The exact Bayes mixture puts weight 1.0 on ER, because the stream measures the current task. Its log 2 guarantee holds but targets the wrong risk.
   - Weights from a 50-item reserve are too noisy.
   - The spec's mixture rule needs a balanced held-out estimator, or a fixed prior weight.
4. **Bias correction over-corrects on CIFAR.**
   - Mean-logit subtraction leaves the newest task under-predicted: 10.9% of predictions against a true 20%.
   - Buffer memorisation is not the cause, since the oracle test mean is no better.
   - An oracle sweep finds α ≈ 0.7 best. MNIST showed the opposite direction for logit adjustment.
   - So B4.1's correction magnitude is an estimated quantity, not a formula. This is open: a streaming-valid α estimator.
5. **Two nulls on CIFAR.** A learned-then-frozen encoder (half of task 1) gives no gain over random features. ER-ACE scores below ER+bc (−1.8, 5/5).

---

# Part B. The specification

## B0. Problem definition [revised after audit P01]

### B0.1 Stream model

**Contexts.** A context c has a joint distribution P_c(x, y). Its true identity at time t is c_t, which is **latent**. The learner's inferred context is ĉ_t. True episode boundaries are distinct from the boundaries the learner declares.

**The stream.** It is a sequence of episodes (c_1, T_1), (c_2, T_2), …, where each batch X_t ∼ P_{c_t}. For fixed-distribution recurring contexts, P_c is the same on every return. **Smooth drift** is a separate regime: P_t varies within an episode, a variation measure is reported, and the stationary-episode claims (B8) do not apply to it unchanged.

**Test-then-train.** The order is: predict on X_t, then reveal Y_t, then update. The prediction path is defined in C1. It depends on the whole learner state, not on θ alone.

### B0.2 Assumptions

- **A1. Consistency within a context.** P_c(y|x) is fixed during an episode.
- **A2. Persistence.** Episodes last T_min ≥ 2 batches. This only *permits* delayed label evidence. It does not imply that the evidence is enough.
- **A3. Identifiability, distributional.** For c ≠ c′, either:
  - **(a)** their input distributions differ, d(P_c(x), P_{c′}(x)) ≥ Δ_x > 0; or
  - **(b)** their conditionals differ on a set of positive mass.
- **A3′. Identifiability, by the implemented scores** [new]. The per-batch log score difference between the true context and each rival has expected value above log L_min (B8.2) while the context persists.
  - A3 does not imply A3′. Two distributions with equal feature means but different higher moments are invisible to a shared-covariance mean score. The feature map can also erase the difference.
  - **A3′ can fail.** Then detection is not guaranteed; noisy evidence can still produce switches, true or false (A5). This is stated, not guarded.
- **A4. Budgets.** A persistent byte budget 𝔅_pers, a peak working-memory budget 𝔅_peak, and a compute budget per item 𝒞_max (C4).
- **A5. Evaluation objective.** Evaluator-fixed weights w_c and horizon, fixed **before** comparing algorithms:

  $$\mathcal R=\sum_c w_cR_c.$$

  It is reported together with prequential risk on the deployed path, per-context risks, and oracle-context risks.
  - The learner's own recurrence estimates may steer resources. They never redefine the headline objective [corrected: otherwise ignoring a context lowers its weight].
  - Context-conditional evaluation uses a fixed history and warm-up policy, because shuffled held-out data through a persistent filter is order-dependent.
- **A6. Permitted error.** R_c(t) − R_c(ref_c) ≤ ε_c, with ref_c the **observed** reference event of c (B7.2).

### B0.3 Limits (what no algorithm can do)

**L-a. Conflict without information** [proved; corrected]. Let I_t be everything available at prediction time. At an input x_0 where contexts disagree, the Bayes-optimal accuracy is max_c Pr(c_t = c | I_t). It equals ½ only for two equally likely contexts with no distinguishing history. History, unequal priors or a predictable schedule can beat ½ [corrected: v1.2 claimed no method could beat the shared head].

**L-b. Bounded state** [proved, counting argument]. Total finite persistent state, model included, distinguishes at most 2^{8·bytes} states. So unboundedly many independent past bits cannot all be kept, and A6 is a trade-off target.

**L-c. Recognition delay** [src: P128; B8.2]. Under constant evidence, detection needs per-batch evidence above log L_min, and near overlap it never happens. Under noisy evidence it can happen below L_min (A5).

### B0.4 Hypotheses (not laws)

- **H1. Anchor.** Estimates from real stored data at the current weights beat stale or self-generated ones. Gate E9.
- **H2. Single knob.** A pure magnitude brake trades plasticity for retention. One-sided per-item constraints escape that trade-off **only on the pool they constrain**. On held-out items (B5.0) and on the nonlinear toy's population (A3), they did not beat a quadratic brake or replay. Gate E4c, against compute-matched ER.
- **H3. Access before storage.** A contingent pattern, with counter-cases P197 and P124. Gate E1.

**Naming** [corrected, audit]. The "fixed P(x), changing P(y|x)" stream is **conditional (concept) shift**, not label shift. Label shift changes P(y) while keeping P(x|y) [Lipton et al. 2018].

---

## B1. Notation

| Symbol | Meaning |
|---|---|
| t | step; batch X_t of n_new items |
| c_t, ĉ_t | true (latent) and inferred context |
| κ | cell: class (M2), or (class, context) from M3 |
| θ = (θ^st, θ^pl) | stable layers; plastic layers plus head. θ_P is the protected set |
| θ_0 | pre-trained weights; head rows anchor at their birth value |
| φ | slow EMA of θ; slow view only |
| B | Adam step matrix with group rates (C2) |
| a, Δ_0 | Adam direction m̂; baseline step −Ba |
| C, s | constraint columns and slacks (M2c), or factor columns (Q) |
| 𝓛_v | versioned label registry (C3) |
| ℬ | buffer: real items with ID, source ID, cell, role, exposure count and reference record |
| 𝒬 | probes: sel, acc and mon, fixed at intake, never trained |
| 𝔅_pers, 𝔅_peak | persistent and peak budgets beyond θ and standard Adam (C4) |
| P_⊥ | centring projector I − 11ᵀ/K over the current registry |

---

## B2. Measurements [revised after audit P14]

### B2.1 Quantities

**Prequential risk** on the deployed path is the primary online measure. The other measures, each defined below:

- **Accuracy and NLL** per context, on its evaluation split, on the deployed, shared and oracle-context paths.
- **Paired forgetting F⁻ and repair counts** (B7.2).
- **Plasticity** Π_c = a_{ref(c),c} / a^fresh_c, where a^fresh is θ_0 trained on c alone with the same budget [src: P124, P197].
- **Savings** are measured as **censored time-to-threshold** [corrected].
  - The threshold is a target accuracy fixed **before** confirmation.
  - The summary is the restricted mean time to reach it within a fixed horizon H, measured in items and compute.
  - Runs that never reach it count as H.
  - The pass condition is a paired comparison with both the fresh run and the wrong-snapshot control Ψ_{c′}.
  - There are no ratios and no division by zero.
- **Displacement** ‖θ_t − θ_{t−Δ}‖ [src: P208].
- **In/Forg/Out AUROC** [src: P160].
- **Resource ledgers:** persistent bytes by component, peak memory, FLOPs, latency quantiles, and measured event frequencies.
- **M2c coverage diagnostic** (B11 I5).

### B2.2 Controlled readout table [replaces v1.2's D1–D4 sum]

**The table.** A(r, d, p, π) is accuracy as a function of four factors:
- r: representation version;
- d: training-data source (buffer, or full history);
- p: fitting procedure, fixed and prespecified;
- π: routing (actual, or oracle).

**How differences are read.** Each comparison changes one factor and holds the rest fixed.
- **D1:** head, actual vs rebuilt, same r, d, π.
- **D2:** d, buffer vs full, **same p** [corrected: the Gaussian-refit cosine head and a logistic probe are different procedures].
- **D3:** π.
- **D4:** r at t vs at ref, each with a **freshly refit** decoder [corrected: reusing a decoder mislabels an invertible rotation as storage loss].

**What the differences are not.**
- **They do not add up to a unique decomposition.** If a telescoping sum is reported, its path through the table is stated, and the contributions depend on that path.
- **Linear probes measure linear recoverability only.**
- **Layer and regularisation choices** are made on validation data only.

### B2.3 Data tiers

The tiers are as in v1.2:
- 𝒬^sel, 𝒬^acc, 𝒬^mon: control, used inside the learner;
- development streams: adaptive, by us;
- confirmation streams: frozen, one per decision;
- final stream: untouched until the end.

**Tags.** Every quantity carries a C5 tag.

### B2.4 Protocol

**Baselines, all tuned with reported selection budgets:**
- fine-tuning with swept group rates (P158);
- a frozen extractor with a rebuilt head (P197);
- tuned ER (P178);
- **compute-matched ER**;
- **GEM and mGEM** [new: M2c belongs to their family, so they are the direct baselines];
- a **rolling-window** baseline for drift streams.

**Streams:**
- ordered class-incremental;
- smooth drift;
- recurring pool with heavy-tailed gaps;
- an open stream with ≥ 100 contexts;
- **conditional shift**, recurring *and* novel;
- multiple prespecified orders.

**Memory and compute:**
- results on the 𝔅_pers frontier, and at equal buffer item count;
- peak memory reported.

**Statistics:**
- sampling unit: the (stream, order, seed) triple, with ≥ 5 units;
- intervals: paired bootstrap;
- claim families as in B15.

---

## B3. Harness verification suite (Lucky's reliability rule) [revised after audit P15]

No result counts until every check passes. The checks are automated tests that rerun on every code change.

Each test states **which object it checks**. Exact-arithmetic identities are tested separately from finite-precision tolerances. A passing randomised test is evidence, not proof.

**V1. Parity, not one interval** [corrected].
- **Primary test: cross-framework paired parity.** On a fully specified fixture (data protocol, preprocessing, initialisation, order, memory, optimiser, metric), our ER and fine-tuning must match Mammoth or Avalanche **per seed** within a stated tolerance.
- **Secondary:** reproduce P178's ER (28.2 ± 1.2) and P158 (88.86 ± 0.83) under their full protocols. Landing inside a published interval is neither necessary nor sufficient for correctness.

**V2. Known-answer metrics.**
- A hand-built accuracy matrix gives exact forgetting, F⁻, repair counts and censored savings.
- The Wilson interval, the exact McNemar test, and the exact multinomial power with regressions match scipy/statsmodels to 1e-12. They also reproduce the audit's 0.808, 0.363 and 0.204.
- A probe on separable synthetic data gives 100%.

**V3. Planted effects.**
- An oracle model scores 100%; a random model scores chance.
- Scrambling the head while keeping features moves D1, not D4.
- Damaging one context's **isolated fixture weights** moves only that context [corrected: with shared weights this is false by construction].
- A planted leak is caught by V5.

**V4. Toy cases with known answers.**

| Case | Object checked | Expected result |
|---|---|---|
| T1 correlated (z, b) | exact pool logit metric (M2b/M2c objects) | K-FAC protects the harmless off-diagonal Δ, the exact metric does not; M2a is a different object and is not tested here |
| T2 conditional (concept) shift | the exact filter recursion (B8) | (a) a return to an accurate stored head; (b) a **novel** conditional rule on familiar inputs: detected only through new_cond; without it, rejected. Report first-passage delays, **non-detections and false switches**; under constant evidence below L_min, no switch ever occurs |
| T3 long gap, moving weights | snapshot record | reconstruction vs **the saved ring checkpoint**; every vector parameter exact; every registry version handled; "no eligible probes" handled by exact storage or "no snapshot" |
| T4 null space | (Q) solver | Δ* = Δ_0 iff HΔ_0 = 0 (audit's example: H = 11ᵀ, D = diag(1, 2), a = (1, −1) **does** change) |
| T5 rotating statistics | design | no stored eigen-coordinates exist (asserted) |
| T6 budget | allocator (C4) | preflight before and **during** events; quota infeasibility, metadata growth, peak rollback copies; persistent and peak ledgers never exceeded |
| T7 linear regression | whole loop | measured forgetting matches the closed forms of P140 and P201 |
| T8 solvers | C2 | (Q) Woodbury vs dense; (K) active set vs a primal QP solver, plus KKT residuals; group-rate post-scaling ascent is reproduced (3.04), and the in-B form never ascends; warm-started CG ascent reproduced as a regression test; λ = 0 returns Δ_0 exactly; blend root and non-expansion in B⁻¹ |
| T9 certificates | B7.2 | new output classes (the audit's (1, 0) → (1, 0, 2)), common-logit shifts ((1, 0) → (6, 5): prediction kept, the old certificate fails, r(x) passes), wrong-at-reference items, empty S^y (N/A), ties, fp16 tolerance |
| T10 token-shared weights | M2b/M2c columns | true gradients through tokens are unbiased; a per-token sum is not |
| T11 shared class set, permuted labels | label models | renormalising a shared head gives log-ratio 0; per-context heads separate |
| T12 sketch exploitation | M2b | the step chosen with the sketch has exact-metric damage measured by an independent check; ratio reported (my toy: median 12×) |
| T12b held-out protection | M2a, M2b, M2c | every protection's damage is measured on items not used to build it; the in-sample number is reported only next to it (notes/v13-checks/heldout_frontier.py) |
| T13 momentum vs gradient | C2 | g = 1, m̂ = −1: ascent is reported and logged, not hidden |
| T14 usage | B5.4 | continuous at T_act; alarms never write t_last |
| T15 transactions | C4 | consolidation rolls back to S1 with S1's state exactly; reactivation marks caches dirty; offline steps do not move stream time |
| T16 causal order | C1 | predictions are written before labels exist in memory (assert); a first-occurrence label is logged as unknown; at warm-up end, no protected step runs before the pool commits |
| T17 margin guarantee | M2c | on a linear model with tracked margins and every step projected, no continuously constrained pair crosses min(m_0, m_floor); with refresh-only slack (R_pre = 25) the stale-slack counterexample (notes/v13-checks/stale_slack_counterexample.py) must reproduce its violation; uniform clip keeps feasibility; on a nonlinear model the violation is reported per refresh, **on the pool and on an independent held-out pool** |

**V5. Leak assertions** [extended].
- No ID of 𝒬, quarantine, or evaluation splits appears in **any** gradient: main loss, context heads, detectors, adapters, Gaussian head-feature fitting, candidate fine-tuning.
- Source IDs survive augmentation and context reassignment.
- Semantic near-duplicates are tracked by source ID where the dataset provides one. Otherwise the limitation is stated.

**V6. Limits** [corrected].
- **ω̃ = 0 (Q) / no constraints (K)** [notation corrected]. M2 reproduces M1 bit for bit only when **every** non-baseline control is off: refresh schedules, decay, 𝒬 intake, resets, cached-B semantics, all M2 RNG streams, and framework RNG including dropout.
- **Single fixed context.** M3 reproduces M2 **predictions and weights** only if label-head training and its compute are excluded from the comparison. State and resources differ, and this is said.

**V7. Pre-registration and review.**
- Predictions, analysis plans and claim families are written before confirmation.
- Every run logs its config, seed, commit and ledgers.
- The harness gets an independent code review.

---

## B4. Core learner M1

### B4.1 Architecture and readout [src: P205, P208, P186, P197, P120, P143, P158]

- **Backbone and normalisation.** A pre-trained backbone with GroupNorm or LayerNorm, or BatchNorm with frozen statistics (P148).
- **Stable and plastic split.** Default ℓ_s = 2/3 of depth. Gate E2 sets it.
- **Head.** A **global** cosine head over all seen classes:

  $$f_j=\varsigma\cos(w_j,h(x)),\qquad \varsigma=10.$$

  This is the routing-free shared path in every level. Two-level readout is only added in M5, and only if it beats this head.
- **Prediction-time correction (default) [A6].** Predict with f_j − b̂_j, where b̂_j is class j's mean logit over the class-balanced buffer (re-estimated at each head refresh and on every registry growth). Closed-form logit adjustment by the loss-induced prior q_eff (B4.3) is a named ablation: the marathon shows it under-corrects with a small buffer. Report historical-balanced and current-stream accuracy separately, since the correction trades one for the other.

### B4.2 Rates, optimiser and decay

**Learning rates.** η_b = 1e-5, η_p = 1e-4, η_h = 1e-2. They sit **inside** B (C2) and are never applied after the solve.

**Choosing η_p** [corrected after audit]. Take the best validated rate. The high-side cliff η^cliff is the first **higher** tested rate that is at least 2 points below the best. Set η_max = η^cliff/3. If no upper cliff is observed, keep the best rate and do not invent a cliff [src: P158 doses].

**Adam, with v never reset** [design, simplified from P182's blended rule]. From M3 on, m is reset on a declared switch, with its **own** bias-correction counter k_m (C2) [guess]. Gate E3 compares this with P182's rule.

**Decay toward the pre-trained weights** [design; L2-SP idea, Li et al. 2018; corrected after review]. Decoupled, per group:

θ ← θ − η_group λ_wd (θ − θ_0),  λ_wd ∈ {0, 0.01, 0.1}, default 0.01. New head rows anchor at their birth value (C3).

- At η_p = 1e-4 the pull is 1e-6 per step, about 1% per 10⁴ steps. It is weak by design, and E2 sweeps it.
- It bounds drift from θ_0 only in balance with the gradient. It is **not** a control on separation from φ. v1.2's draft claimed that, but its pull was 1e-8 per step and inert.
- Norms ‖θ − θ_0‖ per group are logged.
- Decay is applied after the solve (C1 step 13). It is outside the C2 theorems, and "zero step" in case 62 excludes it.

**The slow trace.**

$$\phi\leftarrow\phi+\alpha_\phi(\theta-\phi),\qquad \alpha_\phi=10^{-3},$$

over all layers, used **only** for the slow view (B8). It does **not** store contexts; that is the snapshots' job (B9).

### B4.3 Batches and loss

**Batch composition.**
- Each batch has n_new = 16 new items and n_rep replay items, with r ∈ [0.25, 0.75] and default 0.5.
- n_rep is tuned [src: P178].
- Probe-reserved and quarantined items never enter L_new.
- If the buffer holds fewer than n_rep items, replay samples with replacement.

$$\mathcal L=w_{new}\overline{\mathrm{CE}}_{new}+w_{rep}\overline{\mathrm{CE}}_{rep},\qquad w_{new}=w_{rep}=1.$$

An empty mean is 0, never a division by 0. The weights are stated explicitly, because changing n_rep changes variance and compute, not the coefficient (case 24).

**Startup** is governed by C1: warm-up is counted in 1,000 gradient-used items, and nothing protected runs before its initialisation transaction commits.

### B4.4 Buffer (M1)

- **Fill:** train-role items only (C1 step 14), as a reservoir balanced across cells. In M1–M2 there is a single context.
- **Sampling:** uniform in M1; B6.1 from M2.
- **Exposure:** e_i counts uses (B6.2).

### B4.5 Head rebuild and drift trigger [src: P143, P158, P174, P146]

**Rebuild.** At each refresh:
1. Compute fresh class means μ_j and a **pooled within-class** covariance Σ̂_w, with Ledoit–Wolf shrinkage a_LW plus a ridge of 10⁻⁴·tr/d **plus an absolute floor ε_abs·I** tied to the documented feature scale (a trace-proportional ridge is 0 when every class has one sample) [A6]. Classes missing from the subsample keep their previous statistics and are flagged.
2. Sample 256 features per class from 𝒩(μ_j, Σ). This is a distributional assumption, stated (case 23).
3. Retrain the head for 5 epochs with the backbone frozen.

**Drift trigger.** The refresh fires when either condition holds:
- R_head = 2,000 steps have passed;
- δ_drift = max_j ‖μ_j − μ̄_j‖ / n̄_h > 0.05, checked every R_check = 200 steps on 10% of ℬ. Here n̄_h is the mean feature norm.

---

## B5. Protection (M2)

All protection variants solve C2's single problem and differ only in 𝓟. **The bar for every variant is a debiased M1 at the same compute: loss-balanced replay plus a recency-bias fix (balanced head refresh, or ER-ACE), not plain or compute-matched ER.** The diagnosis in A4 shows that, on the toy, compute-matched ER's 3.4-point gain came from loss balance, and that weighting replay ×9 (47.8) or ER-ACE (57.8) beat it at 1× compute.

| Variant | 𝓟 | Object it protects | Status after the toys |
|---|---|---|---|
| M2a | (Q), H = A_u ⊗ I per linear map | the layer-output change on pool inputs | exact for its object; cheap; not yet run |
| M2b | (Q), centred logit sketch | the linearised centred logit change on the pool | unbiased only for fixed directions; the chosen step exploits the sketch (12× on a toy) |
| M2c | (K), per-item margin constraints | the first-order margin decrease of correct pool items against their rivals | protects its pool exactly, but **generalised worse than (Q) to held-out items** on the first-order toy, and did **not** beat ER or compute-matched ER on the nonlinear toy |

### B5.0 What the checks showed (this section changed most after review)

**Finding 1: centre the logit metric** [my analysis].
- Softmax is invariant to adding the same constant to every logit, but ‖JΔ‖² charges for it. So the metric is centred: ‖P_⊥JΔ‖², and in the sketch ξ = P_⊥ε.
- It pairs with the translation-invariant check: max_j(d_j − d_y) ≤ √2‖P_⊥d‖.
  - **Proof:** d_j − d_y = (P_⊥d)_j − (P_⊥d)_y, and |v_j − v_y| ≤ √2‖v‖.
  - **Check:** 5,748 certified random cases with large common shifts, 0 violations.
- The gain measured in the toy was small. Centring is adopted because it removes a wrong penalty.

**Finding 2: protection must be judged on held-out items, and then hard constraints lose** [corrected after review].
- **My first frontier was an in-sample tautology.** Its damage metric was exactly the constrained set, so "all-rival constraints give damage 0" held by construction. I had reported that constraints "dominate" the quadratic brake. That claim is withdrawn.
- **The reviewer's held-out version** (notes/v13-checks/heldout_frontier.py) reran it with structured Jacobians that share a low-rank part, so protection *can* transfer, and measured damage on 200 non-pool items. Adam alone: 371 damage, 198 progress.

  | Step | Held-out damage | Progress |
  |---|---|---|
  | (Q), λ = 3 | 64 | 171 |
  | (Q), λ = 30 | 49 | 135 |
  | (K), all rivals, s = 0 | 77 | 160 |
  | (K), with headroom slack | 89 | 170 |

- **Reading.** Constraints zero the pool's damage exactly, but transfer less **in this configuration** (top-2 rivals, slack). The general slogan "constraints memorise, penalties generalise" is withdrawn (A6). The nonlinear toy agrees: M2c lifted replay-buffer accuracy to 99% without moving test accuracy.
- **Rule adopted for every protection claim:** damage is measured on items **not** used to build the protection (V4 T12b, T17).
- **Prior art.** M2c is GEM-family. mGEM [arXiv 2410.00868] reports that finer constraints help on Split CIFAR-100. Their own Proposition 1, a generalisation gap that grows with the constraint set's complexity, predicts exactly the held-out loss seen here. Whether that trade-off favours constraints at scale is an open question for E4c.

**Finding 3: the bottleneck is coverage.** Neither per-step geometry nor a better solver changes what happens on items the method never sees. Mechanisms that widen coverage are the next hypothesis, untested:
- protecting augmented or neighbouring versions of stored items;
- feature-space neighbourhoods;
- (Q) on a larger, cheaper pool.

### B5.1 M2c: margin-constrained step [design; properties proved; kept as a gated variant]

**Pool.** At each refresh (every R_pre = 25 steps), take n_H = 32 items uniformly from buffer items that are **correct now**, in cells with u ≥ u_min = 0.5 (B5.4).

**Rivals.** The rival set of x is the top k_r = 2 rivals under the current registry, **plus every label born since the last refresh** [corrected, review: a new row starts low and would otherwise be unconstrained].

**Constraints and slack.**

$$c_{x,j}=\nabla_{\theta_P}(f_y(x)-f_j(x)),\qquad s_{x,j}(t)=\kappa_s\max(0,\tilde m_{x,j}(t)-m_{floor}).$$

**Tracked margins** [corrected, review]. At a refresh, m̃ is set to the measured margin. After every projected step, m̃ ← m̃ + c_{x,j}ᵀΔ. That product is already computed by the solver, so tracking costs nothing.
- **Why it matters.** With slack computed only at refresh, a 1-D linear counterexample falls from 2 to −5.5 over 100 steps at R_pre = 25.

**Defaults.** κ_s = 0.2 and m_floor = 0.5ς/10, both doses (ς is the cosine-head scale).

**The step.** Δ* = argmin ½‖Δ − Δ_0‖²_{B⁻¹} subject to c_{x,j}ᵀΔ ≥ −s_{x,j}. This is C2 (K). The dual has n_H·k_r = 64 variables (more after head growth), solved by an exact active set in fp64.

**Head growth** [corrected, review]. Old margins f_y − f_j do not depend on a new row's weights. So on head growth, C is migrated **exactly**, by zero-padding the new coordinates. Columns for the new rival are added: the head part is closed-form for the cosine head, and the backbone part costs one VJP per pool item. Registry growth does **not** mark the whole pool dirty.

**Cumulative first-order guarantee** [proved, with its full assumptions]. The statement holds for a pair (x, j) **continuously in the constraint set over [t_0, t]**, under all of:
- linear margins, m_{t+1} = m_t + c_tᵀΔ_t;
- exact current c_t;
- every step in [t_0, t] a projected step, with no baseline steps;
- decay applied outside θ_P, or zero.

Then

$$m_{x,j}(t)\ge\min(m_{x,j}(t_0),m_{floor}).$$

**Proof.**
- If m̃ ≥ m_floor: m̃′ − m_floor ≥ (1 − κ_s)(m̃ − m_floor) ≥ 0.
- If m̃ < m_floor: s = 0, so m̃′ ≥ m̃.

**What breaks it in the real algorithm** [stated]:
- **Pool resampling.** A buffer item is constrained for only about a fraction n_H/|ℬ| of refresh windows.
- **Stale c within R_pre**, and **nonlinearity** (second-order error ≤ ½L_m‖Δ‖² per step, L_m unknown). At each refresh m̃ is reset to the measured margin, so these errors do not compound across refreshes.
- **Baseline steps** on a dirty structural cache.
- **The rival set.** A rival outside the top k_r can overtake; it enters at the next refresh.

**Logged per refresh** (C5, control): the actual nonlinear margin change on the pool, and the same quantity on an **independent validation pool** of 16 buffer items never used to build C.

**The step clip** is applied as a **uniform** scale, α = min(1, min_i 10η_{g(i)}/|Δ_i|). Since s ≥ 0, uniform scaling keeps the step feasible and keeps both C2 inequalities. A coordinatewise clip does not: with c = (1, −1), the step (5, 5) is feasible, but clipping its first coordinate to 1 gives cᵀΔ = −4.

**Plasticity.** Improving a margin is free, and constraints of the class being learned are mostly inactive. True conflicts (same input, new label, shared head) are blocked by design and raise conflict events (below).

**Conflict with replay repair** [corrected, review]. A repair gradient for a forgotten class y′ raises f_{y′} near its inputs. That lowers the margins of pool items of other classes against y′. Where those margins are at or below m_floor, s = 0 and repair is blocked.
- **Rule:** during a unit's replay-boost window (B7.3), every pair whose rival is that unit's class gets slack κ_s·max(m̃, 0) + s_rep, with s_rep = 0.1.
- Logged.

**Conflict detection** [corrected, review]. The global ratio ρ̂ hides a minority conflict: one blocked coordinate in 16 gives ρ̂ = 0.94. So conflicts are detected **per class j of new items**:

$$\beta_j=\frac{-g_j^\top\Delta}{-g_j^\top\Delta_0},$$

where g_j is the loss gradient of class j's new items in the batch (β_j is skipped when the denominator ≤ 0). If β_j < 0.2 for P_conf = 20 consecutive batches containing j, a **conflict event on j** fires.
- **Below M5**, the event releases (s = ∞) every pair whose rival is j, for T_rel = 500 steps, and logs the release as accepted conflict loss. Otherwise conditional shift would deadlock M2c at every level below M5.
- **From M5**, the event triggers an adapter (B10.1).
- **The same β_j test is used under (Q)**, so (Q) also produces conflict events.

**Cost** (ViT-B plastic third, N_P ≈ 2.8·10⁷; estimates, logged when run).
- **Refresh:** 64 margin gradients per 25 steps, about 2.6 backward passes per step.
- **Storage:** C is 64 columns × N_P in fp16, about 3.6 GB. Protecting only the head and last block halves it.
- **Per step:** CᵀBC, about 1.1·10¹¹ multiply-adds, plus a 64-variable QP.

### B5.2 M2a: layer-output metric [proved for its object]

- **Object.** 𝓜_a(Δ_ℓ) = tr(Δ_ℓA_uΔ_ℓᵀ), with A_u = (1/n_H)Σ_x u_x Σ_p z_{x,p}z_{x,p}ᵀ. This is exact for the summed squared change of the layer's **pre-activation** outputs on pool inputs, with inputs held fixed.
- **Scope** [corrected, audit P04]. Plain linear maps only. The cosine head is excluded, because its logits are normalised and M2a's identity does not hold for them.
- **Solve.** PCG from 0 with preconditioner B, never warm-started. Every iterate is a surrogate-descent direction in exact arithmetic for a **fixed** operator; finite precision is checked by C2's numerical contract.
- **Next test.** The held-out first-order evidence (Finding 2) favours (Q)-type protection. M2a is the next variant to run, against compute-matched ER.

### B5.3 M2b: centred logit sketch [qualified]

- **Columns.** g_{x,k} = ∇_{θ_P}(ξ_{x,k}ᵀf(x)) with ξ = P_⊥ε and ε ∼ 𝒩(0, I).
- **Unbiased, but only for a direction fixed independently of the draw:** E_ξ 𝓜̂(Δ) = (1/n_H)Σ u_x‖P_⊥J_xΔ‖².
- **Variance:** (2/(n_H²J_ξ))Σ u_x²‖P_⊥J_xΔ‖⁴ [audit, Gaussian].
- **Both errors occur** [corrected]. The sampled operator is not below the exact one in PSD order. Example [corrected, A6: the draw must itself be centred]: K = 3, J = I₃, ξ = (2, −1, −1) = P_⊥ξ gives Ĥ − P_⊥ with eigenvalues −1, 0, 5.
- **The solver's choice exploits the sketch.** The step is chosen *using* Ĥ, so directions in its null space are free. On my random toy, the sketch-chosen step damaged the exact metric a median of **12×** as much as the exact-metric step (notes/v13-checks/verify13.py).
- **Required safeguard.** Validate every refresh's steps on an independent pool with **exact** JVPs. Log sketch cost, independent linearised cost, and actual output change.
- **Status:** gated variant only.

**Exact alternative** [audit; cost note mine]. The matrix-free Hv = (1/n_H)Σu_xJ_xᵀP_⊥J_xv is exact on the pool. At n_H = 64 and 10 CG iterations it costs about 1,300 extra passes per step, about 20× training. Diagnostic only.

### B5.4 Usage and release [corrected after audit P08 and review]

**Active set.** 𝒜_t is the set of cells observed in the stream within the last T_act = 500 steps.

**Age.** a = t − t_last(κ). The last observation counts probes. Alarms never write t_last.

**Usage, for all variants:**

$$u(a)=u_{act}\Big(1+\frac{\max(0,a-T_{act})}{\tau_u}\Big)^{-1/2},$$

continuous and non-increasing. For (Q), u_act ∈ {0.25, 1} is a dose. For M2c, u_act = 1, because one-sidedness handles the active cell, and u is used only for pool eligibility, u ≥ u_min = 0.5.
- **Release horizon** [proved]: τ_u(u_min⁻² − 1) after T_act, i.e. 3τ_u, about 1.5·10⁴ steps at τ_u = 5·10³. From then on the cell is protected only by replay.
- The v1.3 draft used u_min = 0.1, i.e. 99τ_u, so release never effectively happened [corrected].

**Settings.** τ_u = clip(2·median observed gap, [10³, 10⁴]), and 5·10³ before any recurrence. The context usage u_c (B6.3) is the maximum over c's cells.

**Repair default:** replay boost only [audit P08]. Restoring the brake at a damaged function slows repair too. Restore-on-alarm is variant **E7b**, with its own override multiplier (on C-column weights or s), expiry after 5·R_mon, and a forced cache rebuild.

### B5.5 Strength normalisation for (Q) [corrected after review]

$$\lambda=\tilde\omega\cdot r/\mathrm{tr}(C_1^\top BC_1),$$

where C_1 is the column matrix with u ≡ 1.

- **What ω̃ means.** It is the mean eigenvalue of λB^{1/2}HB^{1/2} over its range **if** C_1 has full column rank r and B is held at its refresh value; each protected direction is then shrunk by about 1/(1 + ω̃) on average.
  - λ is frozen between refreshes while B moves, so ω̃ is approximate in between. λ is invalidated on any C4 structural dirty mark.
  - For M2a, H = A_1 ⊗ I_out, so r is the rank of the **full operator**, r = d_out·rank(A_1) [corrected, A6: using rank(A_1) with the full-operator trace makes λ d_out times too small]. Mean eigenvalue ω̃ does not imply mean shrinkage exactly 1/(1 + ω̃); the shrinkage is nonlinear in the spectrum.
- **Degenerate trace.** If tr ≤ 10⁻¹²·r, protection is skipped and the step is baseline.
- M2c needs no λ.

---

## B6. Memory manager (M2)

### B6.1 Replay sampling [src: P199, P200, P202, P181, P146; mixing rule design]

$$q_{\kappa}\propto n_{\kappa}^{\alpha_f}\,(1+\beta_a\,\mathbb 1[\mathrm{alarm}(\kappa)]),$$

where κ is a cell: a class in M2, a (class, context) pair from M3.

- α_f is a dose (default 0.5, gate E6). β_a = 1.
- Sampling is uniform **within** a cell. There is no error-based selection [src: P178, P162].

### B6.2 Replacement-eligibility threshold [design, inspired by P133; renamed after audit: it is not a cap]

**The rule.** An item with e_i ≥ E_max = 8 becomes **eligible for replacement**.
- At **every** arrival of a train-role item in the same cell, check for eligible items. If one exists, the oldest eligible item is replaced.
- Otherwise the capped item stays at full weight. Old classes in class-incremental streams have no newer items, so they are never under-replayed.

**Rate.** Let n_κ^buf be the cell's **current buffer occupancy**. This is distinct from the lifetime stream count used in q_κ. An item in cell κ is drawn about n_rep·q_κ/n_κ^buf times per step, so eligibility starts after about E_max·n_κ^buf/(n_rep q_κ) steps.

When no fresh item exists, e_i may exceed E_max indefinitely. Reservoir and replacement together change the sample distribution toward recent items in cells that are still arriving. This is stated, not hidden.

**Source** [src: P133]: there, exposure is the per-block inverse effective sample size, with an absolute cap B_exp = 4.

### B6.3 Byte budget [corrected after review; enforcement in C4]

**What 𝔅_pers covers.** 𝔅_pers is the persistent memory **beyond** what every method has: the initial model θ and its standard Adam state (m, v). Head-row growth and its optimiser state count against 𝔅_pers (C4). It covers:
- buffer;
- probes;
- reference logits;
- the protection columns C (M2c, M2b) or A_u (M2a), plus the validation pool;
- φ;
- the θ_0 decay anchor;
- snapshot bases and differences, plus the snapshot ring (B9.1);
- statistics, heads, detectors, filter state and context metadata;
- adapters and their optimiser state;
- the probe quarantine.

**𝔅_peak** separately covers rollback checkpoints, solver and compression workspaces, candidate snapshots and the synthetic head-training features (C4).

**Allocation** [design]. Fixed-size components **of the levels in use** are allocated first, at their stated minimums. θ_0 is the decay anchor from M1; φ is needed only from M3 [corrected].
- θ_0 for the decayed layers (M1+);
- φ (M3+);
- the protection columns;
- the ring (M4).

The **remainder** is shared as: buffer 0.7, snapshots 0.2, probes and statistics 0.1.

If 𝔅 is below the fixed minimum, the configuration is **infeasible** and is reported as such, not silently shrunk.

**Example,** ViT-B in fp16: φ 0.17 GB, θ_0 0.17 GB, M2c columns 3.6 GB, ring of 2 at 0.34 GB. That is about 4.3 GB fixed. M2a instead brings it to about 0.8 GB: φ, θ_0 and the ring (0.68 GB) plus M2a's per-layer input covariances [corrected: v1.3's draft said 1.6 GB].

**Eviction**, by lowest value per byte:
- buffer: the cell with the most items relative to its share;
- snapshots: lowest u_c, then oldest;
- probes: by quota excess first. Quotas are desired, not mandatory, and a unit below its minimum is marked ineligible (C4);
- context metadata and filter states: lowest u_c, after their snapshots and probes are gone;
- ties are broken deterministically by ID.

Evictions are atomic.

**Fairness and the frontier.** Baselines spend the same 𝔅, mostly on buffer. At equal 𝔅, ER may store very large buffers. The whole CIFAR-100 training set is about 150 MB, against our fixed costs of several GB. **That is the real trade-off**: our method spends bytes on model state, while ER spends them on data.

So results are reported:
1. on the 𝔅 frontier;
2. **at equal buffer item count** as a secondary comparison, with the extra bytes stated.

Neither alone is a fair summary.

**Limit** [B0.3 L-b]. For unrelated contexts, loss beyond the budget is unavoidable.

---

## B7. Monitor: measured forgetting first, certificates second (M2)

### B7.1 Probes and item roles [corrected, audit P05, P11]

**Role at arrival.** Each arriving item gets an **immutable role** at first sight: train, or probe-reserved. It is reserved with probability p_q = 0.25 while its class (and, from M3, its context) is below the desired quota q_cls = 30. Otherwise it is train.

**Reserved items of a provisional context** wait in a budgeted quarantine and are never trained.
- On commit, they become that context's probes.
- On merge into the parent, they are **discarded**: never trained, never probes, kept only in the log [corrected after review: a new_cond child's items may carry the conflicting rule, so making them the parent's probes would corrupt F⁻, alarms and reactivation acceptance, and training on them would inject the conflict].
- A provisional context that neither commits nor merges within T_prov = 2,000 steps **dissolves** the same way, logged. This closes the state "neither committed nor log BF < 2".
- If the quarantine is full, the item is train **from the start**.

A trained item never becomes a probe [corrected].

**Splits.** Reserved items are split at intake into 𝒬^sel, 𝒬^acc and 𝒬^mon, 10 each per class.

**Quotas are desired, not mandatory** (C4).
- Below 10 𝒬^mon items, a unit's monitor reports "insufficient coverage" and skips alarms.
- Below the B9.3 minimum, a context is ineligible for reactivation.

**Context heads, detectors and candidates never train on 𝒬** (V5).

### B7.2 What is monitored [corrected, audit P07]

**References** [corrected after review]. There are two reference sets, written by different parties:
- **evaluation references** are written **by the evaluator**, at the **true** episode boundaries, from the stream's ground truth. They are immutable and never reset, and only they feed [evidence] F⁻;
- **control references** are written by the learner at **observed** deactivation events: the class leaves 𝒜_t (M2), or the inferred episode ends (M3+). They drive alarms only, and may be reset after accepted loss.

New buffer and probe items get control references at insertion, if their unit is already inactive. The monitor runs every R_mon = 500 steps.

**Primary signal: paired forgetting** [evidence-grade only on confirmation data; control otherwise]:

$$F^-_U=\frac{\#\{x\in U:\ \text{correct at ref, wrong now}\}}{\#\{x\in U:\ \text{correct at ref}\}}.$$

It is reported together with the repair count (wrong at ref, correct now), absolute accuracy and NLL. An empty denominator gives N/A, never 0.

**Diagnostic: translation-invariant robustness headroom** for items correct now. With d = f_now − f_ref over the reference registry,

$$r(x)=m^y_{ref}(x)-\max_{j\in\mathcal L_{ref},\,j\ne y}(d_j-d_y),$$

plus the new-label check f_now,y > max_{j ∈ 𝓛_now∖𝓛_ref} f_now,j.
- r(x) > 0 together with the new-label check is a sufficient condition for "still correct". It is never necessary.
- Counterexample to necessity: f_ref = (1, 0), f_now = (6, 5) is unchanged in prediction, yet the v1.2 certificate failed it.
- With labels in hand, actual correctness is measured directly. So r is used only as a **headroom diagnostic**: how close retained items are to flipping. It is never treated as loss.
- The population Markov bound is kept, in centred form, with ε > 0:

  $$\Pr(\text{wrong now})\le\Pr(\text{wrong at ref})+\Pr(0<m_{ref}\le\sqrt2\epsilon)+\mathbb E\lVert P_\perp\Delta f\rVert^2/\epsilon^2+\Pr(\text{new-label check fails}).$$

  The last term is needed because Δf is taken over the reference registry, and a label born later can overtake without moving any reference-registry logit [corrected].

  Here "Pr(wrong at ref)" is a mixture over each item's own reference time. It is not the risk of one model.

**Numerical honesty.** Stored logits are fp16, so comparisons use a tolerance of 2⁻¹⁰·max|f|. Evaluation runs in eval mode with no augmentation.

### B7.3 Alarms [control; corrected]

**Fire** when, on a unit's 𝒬^mon, the forgotten count ≥ max(3, ⌈0.1·n_ref-correct⌉) **and** the buffer F⁻ ≥ 0.1.

| Rule | Value |
|---|---|
| Refractory period | 5·R_mon |
| Recovery | buffer F⁻ ≤ 0.05 |
| After 3 alarms without recovery | log accepted loss (measured), reset only the **control** references |

**Effect:**
1. Replay boost β_a = 1 for that unit, for 5·R_mon steps.
2. From M4, attempt reactivation.

Restore-on-alarm is variant E7b (B5.4).

---

## B8. Context inference (M3) [corrected, audit P05, P06]

### B8.1 Hypotheses and the transition model

**States.** Existing contexts c ∈ 𝒞, plus two novelty types:
- **new_in:** new input regime. Input model 𝒩(m̄, s_new²S); label model π_base.
- **new_cond(ĉ):** a new conditional regime **sharing the input model of the current MAP context ĉ**, p_in(X | new_cond(ĉ)) = p_in(X | ĉ). Its label model starts at π_base.

Without new_cond, a changed label rule on familiar inputs is crushed by the broad input model's penalty. At d = 768, s_new = 2, n_eff = 4 that penalty is −977 nats, against a maximum label advantage of 27.6 [audit; reproduced].

**Transitions,** normalised; h is clipped to [10⁻³, 0.5]:
- T(c → c) = 1 − h;
- T(c → c′ ≠ c) = h(1 − π_new)·w_{c′}/Σw, with smoothed recurrence counts w = n_rec + 1;
- with no other context, that mass goes to the novelty states;
- T(c → new_in) = T(c → new_cond(c)) = hπ_new/2.

h = 1/T̄ is a geometric-duration approximation. Heavy-tailed episodes are mis-modelled, and that is stated, not hidden.

**Label models.** Each context has its own cosine head on h̄, smoothed so it normalises:

$$p_\epsilon(y\mid x,c)=(1-\epsilon)p_{H_c}(y\mid x)+\epsilon\,\pi_{base}(y),\qquad \epsilon=0.05,$$

where π_base is uniform over the registry, with positive support. The v1.2 floor max(p, 10⁻³) did not normalise and is removed.

**Temperatures.** κ_in = n_eff/|X| and κ_lab = n_eff/|X| are separate settings. They make this a generalised-Bayes score, not the posterior under the true stream.

**Order** is C1 steps 2–5. Label models are frozen while scoring, and trained afterwards on train-role items only.

### B8.2 What detection needs [proved for the deterministic symmetric case; corrected]

**v1.2's delay formula was wrong twice:**
1. It mis-evaluated: it should be 1.52/I_y, not 6/I_y.
2. It treated a sticky HMM as a static test.

**The correct statement.** For two symmetric states with constant per-batch likelihood ratio L, the odds evolve as

$$O_t=L\,\frac{(1-h)O_{t-1}+h}{hO_{t-1}+1-h}.$$

The posterior **settles above** threshold π if and only if

$$L\ >\ L_{min}=O_\pi\frac{hO_\pi+1-h}{(1-h)O_\pi+h},\qquad O_\pi=\frac{\pi}{1-\pi}.$$

- At h = 0.02 and π = 0.9, log L_min = 0.166 per batch.
- At or below it, the posterior converges to at most π and **never** crosses it. For example, log L = 0.1 converges to 0.843.
- **Scope** [corrected twice; A5, A6]. This is exact only for two symmetric states, constant L, the stated transition rule and an initial posterior below π. It is **not** a necessary condition under noisy evidence (marathon T7b). With more states, novelty mass or variable evidence, detection is characterised by simulated first-passage distributions, false switches and non-detections.
- Checked: just below L_min the posterior settles at 0.8994; just above, at 0.9006.

**Consequence.** Under constant evidence, detection needs per-batch evidence κ·n·(information per item) above about log L_min. With noisy evidence the boundary is soft, and detection below it occurs (A5, marathon T7b). The actual first-passage delays, including non-detections and false switches, are **measured by simulation** (V4 T2). They are not predicted by formula.

**First-batch limit** [corrected; A6]. The Bayes-optimal accuracy at an input is max_y Σ_c Pr(c_t = c | I_t)·P(y | x, I_t, c), where I_t is everything available. It reduces to max_c Pr(c_t = c | I_t) only when every context has a distinct deterministic label at that input (contexts 0.4, 0.35, 0.25 with labels 0, 0, 1 give 0.75, not 0.4). It equals ½ only in the symmetric, history-free case. History or a predictable schedule can beat it.

### B8.3 Decisions

| Condition | Action |
|---|---|
| p⁺(MAP) > 0.9 and the MAP changed | switch (snapshot from the ring, B9) |
| p⁺(new_cond) > 0.5 | create a provisional child context that shares ĉ's input model |
| p⁺(new_in) > 0.5 | create a provisional context with a new input model |

At most one decision is taken per batch, by C1 precedence.

**Provisional contexts** commit after ≥ 50 train items **and** ≥ T_min batches with mass > 0.5. Otherwise they merge into the parent when the log Bayes factor is < 2.

**One router.** If M5 detectors are kept, they enter only as a calibrated term in the input evidence. They never gate routing separately.

### B8.4 Slow view and routing

**Slow view.** h̄(x) is the penultimate feature of φ, the slow model. It costs one extra forward pass per batch, which is counted [src: P156, P203].

**M3 makes no routed predictions.** The global head predicts. The filter changes only bookkeeping, usage, the m reset and snapshots.

**Routed prediction (M5)** sends an input to ĉ only if p_t⁻(ĉ) ≥ τ_route = 0.9; otherwise it uses the shared head. A posterior of 0.9 is **not** a calibrated 90% routing accuracy. Calibration is measured, with true-context diagnostics, and

$$\mathrm{acc}=\Pr(\text{routed})\,\mathrm{acc}_{routed}+\Pr(\text{shared})\,\mathrm{acc}_{shared}$$

is reported against oracle routing (D3).

---

## B9. Snapshots and reactivation (M4) [corrected, audit P09, P10]

### B9.1 Snapshots

**Snapshot record** (C3). Every parameter tensor is stored:
- matrices as base + rank-16 difference;
- **vectors, biases, LayerNorm parameters and scales exactly**;
- the relative-change cutoff is removed.

**Target.** The target is the ring checkpoint θ_τ chosen at the switch (C1 step 5; the transaction runs at step 17):
- L_ring = 2, budgeted;
- provenance labelled exact, inferred-boundary, or contaminated;
- contamination and lag rates are reported.

**Compression check.** The reconstruction is compared with **θ_τ**, not with the current θ.
- If |𝒬^sel_c| ≥ 20, accept when top-1 disagreement ≤ max(1, 2%) of items.
- If |𝒬^sel_c| < 20, store **exact** if the budget allows. Otherwise store nothing and mark "no snapshot".
- There is no automatic acceptance.

**Durability.** A snapshot lasts only while its base and schema are retained. Eviction removes savings, and is logged.

### B9.2 Reactivation (a C4 transaction)

**Candidates** are **migrated proposals** (C3):
- historical rows for labels that both states contain;
- current rows for newer labels;
- all non-head state from Ψ_c (β ∈ {0.5, 1}), or plastic layers only.

**Select** on 𝒬^sel_c. **Accept** on 𝒬^acc_c if all three hold:
- the exact **one-sided** McNemar score p < 0.05 (improvement direction) **and** ≥ 5 net improvements, the second rule applied separately [control];
- pooled other-context harm ≤ max(2, 1%) of items;
- every other context with ≥ 10 sel items loses ≤ max(1, 10%). Contexts with fewer items are flagged, not guarded.

**On commit:**
- invalidate the pool, statistics, filter statistics and reference caches (C4);
- reset m and k_m;
- keep v (resetting it is variant E8b), with C2's step clip;
- leave φ unchanged as the historical slow view.

### B9.3 Statistics of acceptance [corrected, audit P10]

The no-regression power table is correct, and was reproduced by both of us: 80% power at Δ = 0.125 needs n ≥ 53.

**Net improvement does not determine power.** With p₊ = Pr(old wrong, new right) and p₋ = Pr(old right, new wrong), at n = 53 and the same net 0.125, power is 0.81, 0.36 or 0.20 for p₊ = 0.125, 0.25 or 0.5.

Over discordance up to 0.4, 80% power needs n ≈ **170**. So per-context inferential acceptance is **impractical** at our probe sizes.

**Default: control mode.**
- McNemar is an acceptance score.
- Eligibility needs |𝒬^acc_c| ≥ 53. At 10 𝒬^acc items per class, that means a context needs **≥ 6 classes** to be eligible [stated].
- Claims about reactivation come only from confirmation streams (C5).

**Inferential mode** (optional) requires all of:
- fresh acceptance blocks, never used before;
- prespecified p₊ and p₋, or an envelope;
- α_k = 0.05/[k(k+1)] across attempts.

---

## B10. Adapters, readout, controllers, consolidation (M5) [revised; each one is added alone and gated]

### B10.1 Adapters [src: P195, P176]

**Form.** W^{(c)} = W + U_c diag(g_c) V_cᵀ, with U_c, V_c orthonormal (kept so by retraction), and k = 4.

**Creation** [revised; A6]. The primary trigger is a **per-class conflict event** (B5.1): persistent blocked progress β_j for some class j of new items for P_conf consecutive steps, not the global ρ̂, which hides minority conflicts. Computing β_j needs per-class gradients, which cost extra passes; that cost is charged. Classes absent from a batch skip, and a non-positive denominator resets the counter.
- This is a direct signal that the retention constraints block the current learning, which is exactly the situation an adapter is for.
- The v1.2 gradient-cosine EMA is kept as a secondary check. It is skipped when either gradient norm is 0.
- Creation is a C4 transaction. The adapter and its optimiser state are charged to 𝔅_pers. At the cap, creation is blocked and logged.

**Merge** (a C4 transaction). The screening test uses a symmetric scale:

$$\frac{\mathbb E\lVert(A_1-A_2)z\rVert^2}{\tfrac12(\mathbb E\lVert A_1z\rVert^2+\mathbb E\lVert A_2z\rVert^2)}<0.05,\qquad A_i=U_iG_iV_i^\top.$$

- If both adapters are zero on the samples, they count as identical.
- If one is zero, they count as identical only if E‖(A_1 − A_2)z‖² is below an absolute tolerance.
- **End-to-end validation decides.** Measured top-1 disagreement on both contexts' 𝒬^sel must be ≤ max(1, 2%), with ≥ 20 items.

**Replay items** use their own context's adapter.

### B10.2 Routed readout (gated)

Context detectors, if used, feed one posterior (B8.4). Routed prediction is confidence-gated. It is kept only if it beats the shared head on confirmation streams, with oracle routing reported.

### B10.3 Learning-rate controller [src: P148; the raise rule is a guess]

**Signal.** σ_t = acc(φ) − acc(θ) on incoming batches before training, averaged over windows of R_lr = 1,000 steps.
- Plateau means the window mean changed by < 1 standard error.
- The slow model's logits use matching registry rows (C3).

**Actions:**
- η_p ← η_p/2 when σ > 3% for 3 windows.
- η_p ← min(1.25η_p, η_max) when σ < 0.5% for 3 windows with no alarms.

**Bounds.** η_p is kept within [η_p⁰/8, η_max]. η_max is the high-side cliff rate (B4.2).

### B10.4 Consolidation (a C4 transaction)

**Triggers:**
- every T_cons = 5,000 steps;
- an alarm outside its refractory period;
- 2,000 items of cells first seen since the last consolidation.

**Transaction:**
1. Freeze the validation sets and their denominators. Reserve peak memory, or abort.
2. Checkpoint **S0**.
3. Refresh everything that reads features → checkpoint **S1**.
4. Optional offline replay: n_off = 500 steps on ℬ only, under the active protection. These steps advance optimiser counters only, not stream time or recurrence timestamps.
5. Adapter merges → **S2**.
6. Validate S2 on **measured** pooled accuracy and NLL over committed contexts' 𝒬^sel, plus coverage. Certificate counts are not the test.
   - If accuracy drops by more than max(2 items, 1%), restore **S1** and validate it.
   - If S1 also fails, restore **S0**.
7. On commit, mark dependents dirty.
8. Log the newest context separately [src: P122, P185].

**Cost** is reported as measured frequency × measured cost.

---

## B11. Interactions [rewritten for v1.3]

**I1. One geometry.** Learning rates, Adam scaling, protection and the floor all live in B (C2).
- Decay (B4.2) is a separate, weak, decoupled pull toward θ_0. It is outside the descent theorems, and it is ablated (λ_wd = 0).

**I2. Surrogate descent only.**
- Both (Q) and (K) give ⟨m̂, Δ⟩ ≤ −‖Δ‖²_{B⁻¹}, with ‖Δ‖_{B⁻¹} ≤ ‖Δ_0‖_{B⁻¹}.
- Neither proves that the current loss decreases, or that old contexts are retained, beyond M2c's first-order statement for continuously constrained pairs.

**I3. Protection and replay do conflict** [corrected after review].
- Replay supplies *repair*: gradients toward correct labels on stored items. M2c supplies *non-damage*: no first-order margin decrease on correct pool items.
- Repairing a forgotten class y′ raises f_{y′}, which lowers other pool items' margins against y′. At s = 0 the projection blocks exactly that repair.
- Rule: during a boost window, pairs whose rival is the alarmed class get extra slack (B5.1). Repair is "boost replay", not "restore brake" (B5.4).

**I4. Protection and the floor.**
- (Q): floor first.
- (K): retention first, with conflict events.
- Conflict events come from the **per-class** blocked-progress ratio β_j (B5.1), for both (Q) and (K).
- **The conflict-event stream is the interface to M5**: adapters are created where protection persistently blocks learning. Below M5, an event releases the blocking pairs for T_rel = 500 steps, logged, so conditional shift cannot deadlock the learner. The gradient-cosine signal is kept as a secondary check.

**I5. Protection and the monitor share one quantity: margins.**

| Component | What it does with margins | Scope |
|---|---|---|
| M2c | constrains the first-order margin change | pool |
| the monitor | measures actual flips (F⁻) and headroom r(x) | 𝒬^mon and buffer |
| the population bound | links mean centred logit drift to risk | population |

**Disagreement** between M2c's prediction (no first-order flips on the pool) and measured F⁻ on *non-pool* items is the main diagnostic of coverage failure.

**I6. Filter and everything else.**
- Filter errors change cells, and so usage, the pool, references and snapshots. In M3 they do not change predictions (no routing).
- One posterior drives all of these, in C1 order. Label models are frozen while scoring.

**I7. Transactions and caches.** Committed **structural** mutations dirty their dependents (C4); ordinary steps do not. The next protected step after a dirty mark is a **baseline** step, logged, until the rebuild commits. Baseline steps break M2c's cumulative statement, so they are counted.

**I8. Timescales.**

$$R_{pre}=25\ (M2c),\ 50\ (Q)\ \ll\ T_{act}=R_{mon}=500\ \ll\ \tau_u\in[10^3,10^4],\qquad 1/\alpha_\phi=10^3.$$

- Release from the M2c pool (u < 0.5) takes 3τ_u after T_act.
- Snapshots are durable, so trace lifetime and gap length are unrelated.

**I9. Compute per incoming item**, in forward-backward units, each term **measured**:

$$\mathcal C=\tfrac{1}{1-r}+c_{solve}+\tfrac{n_Hk_r}{R_{pre}n_{new}}+c_{\bar h}+c_{heads}+\tfrac{\text{measured event cost}}{\text{measured interval}\cdot n_{new}}.$$

Events are throttled to keep 𝒞 ≤ 𝒞_max (C4). Latency quantiles are reported. Every level is compared with compute-matched ER.

**I10. Self-reference audit.**

| Estimator | Anchored to |
|---|---|
| pool constraints | real stored items at the current θ |
| statistics and heads | real features |
| filter | real inputs and real labels, scored before training |
| monitor | real labels against fixed reference correctness |
| reactivation | disjoint real probes, in control mode |

**Self-referential parts:**
- context identities;
- φ;
- the rival choice: the model's own logits pick which margins to protect.

That last one is a known blind spot: a rival the model does not yet rank highly is unprotected until it rises. That is case 98.

---

## B12. Edge-case ledger (all 62 inherited cases, plus 37 added: 34 from the audit's list, some merged, and 3 new)

**Status codes:**
- **F** fixed by a stated mechanism, within its stated object;
- **Q** narrower claim kept;
- **E** an empirical question with a named gate;
- **L** a fundamental limit (no guard can fix it);
- **U** unresolved, with the limitation stated.

The audit's ledger was checked row by row. I agree with every row's status. Where v1.3's mechanism differs from the audit's suggestion, the section says so.

| # | Case | Status | v1.3 treatment |
|---|---|---|---|
| 1 | Low forgetting because nothing was learned | Q | Report Π and absolute accuracy; censored acquisition (B2); a step floor is not learning |
| 2 | Protection makes the new task unlearnable | Q | C2 priority: (Q) floor; (K) retention first with conflict events |
| 3 | Lock without release saturates | Q | Continuous release (B5.4); cells leave the M2c pool at u < 0.5 (3τ_u); byte admission (C4) |
| 4 | Tightening helps a recurring pool | E | Dose sweep at matched resources (E7a) |
| 5 | Importance vanishes at fit | Q | Margin gradients do not vanish when CE gradients do; zero-trace branch (B5.5) |
| 6 | Stale statistics | Q | Age bound R_pre; immediate invalidation on large edits (C4) |
| 7 | Collapsed buffer, head underdetermined | U | Ridge, and missing cells reported; no information is created |
| 8 | Score drift vs crowding | Q | Controlled table A(·), with refit decoders (B2.2) |
| 9 | Confident wrong recall | U | Calibration measured; shared fallback; no abstention guarantee |
| 10 | Forgotten context looks novel | Q | Label evidence, new_cond; misleading forgotten heads remain possible |
| 11 | Overlap too high to detect | L | L_min condition (B8.2); non-detections reported |
| 12 | A shortcut gets protected | U | Counterfactual challenge sets where available; otherwise stated |
| 13 | Ambiguous assignment becomes irreversible | Q | Immutable roles, provisional state, merge rules (B7.1, B8.3) |
| 14 | Negative savings | Q | Control-mode acceptance; harm guards; censored savings |
| 15 | No trace survives a long gap | F/Q | Immutable bases and a complete schema; not for evicted, contaminated or unvalidated snapshots |
| 16 | Decay erases traces | Q | Decay toward θ_0 can erase deviations; snapshots survive; λ_wd = 0 ablation |
| 17 | Old first moments obstruct | Q | Separate k_m; step clip; carry vs reset gated |
| 18 | Annealing fails under drift | E | Controller vs fixed doses (M5) |
| 19 | Large online batches | E | Exposure and compute reported; mixed-context batches allowed; the filter scores them as they are |
| 20 | Compute binds before storage | Q | Admission and throttling of events; latency quantiles |
| 21 | Self-trained router collapses | Q | Prequential frozen scoring; provisional competition; true-context diagnostics |
| 22 | Detector and memory share a degrading substrate | U | Acknowledged; external tests only |
| 23 | Generated replay compounds error | F | Input generation is off. Gaussian head features are an assumption, stated |
| 24 | Replay lowers plasticity per shot | E | Loss weights explicit; ratio sweep |
| 25 | Replay depends on architecture | E | Matched setups; no extrapolation |
| 26 | Full replay still forgets | Q | Paired F⁻ logged; no guarantee |
| 27 | Recency-only replay harms | Q | Coverage and evicted-risk tracking |
| 28 | A window beats retention under drift | E | Explicit rolling-window baseline |
| 29 | Behaviour stays while representation drifts | F | Functional measures; D4 refits its decoder |
| 30 | Random order flatters | F | Multiple prespecified orders, paired |
| 31 | Untuned baselines | F | Equal-resource tuning; selection budgets reported |
| 32 | Hidden capacity | F | All persistent and peak state charged (C4) |
| 33 | Overshoot transient | Q | Per-batch prequential loss; event-adjacent validation |
| 34 | Rise then fall under a trend | Q | Nonstationary objective defined (B0); full curves |
| 35 | Order asymmetry | E | Order-conditioned and worst-order results |
| 36 | Similarity effects | Q | Conflict-cosine rules defined (zero norm means skip) |
| 37 | Domain shift makes detection easy | F | Same-input conditional streams, novel and recurring (B2.4) |
| 38 | Peak ≠ final | F | Both reported; target fixed before confirmation |
| 39 | Test stream holds historical classes | F | Test-then-train separated from untouched evaluation |
| 40 | Raw input storage forbidden | U | **Unsupported configuration** in v1.3: buffer, probes and pool need raw inputs |
| 41 | Adaptive probe overfitting | Q | Control labels; fresh blocks for inference (C5) |
| 42 | Alarm storms | F/Q | Triggered by measured F⁻, not uncertifiability; immutable evaluation references |
| 43 | K-FAC protects harmless directions | F | Removed. Sketch errors in both directions (M2b) are stated |
| 44 | Adam composition goes uphill | F | One B-geometry problem; surrogate descent only |
| 45 | Floor has no root | F | B⁻¹-metric floor; stable root; stated domain |
| 46 | Snapshot base drifts | F | Immutable bases |
| 47 | Pure conditional shift | Q | new_cond, smoothing, exact recursion; under constant evidence, detection needs L ≥ L_min |
| 48 | Routing mistakes | Q | Single posterior; calibration measured |
| 49 | Monitor reused | F | C5 tags everywhere |
| 50 | Total memory grows | F | Allocator with preflight and eviction (C4) |
| 51 | Small steps accumulate drift | Q | M2c's cumulative first-order floor holds only for continuously constrained pairs with tracked margins (B5.1); otherwise measured F⁻ against fixed evaluation references [corrected] |
| 52 | Cross-layer interactions | Q | Covered within θ_P at the pool's weight version |
| 53 | Same direction, different size | Q | Functional precheck; zero-denominator rule (B10) |
| 54 | Warm-started CG | F | PCG from 0 only; numerical contract |
| 55 | Token-shared weights | F | True gradients; masks fixed during a solve |
| 56 | Normalisation cancels release | F | u ≡ 1 normalisation; M2c needs none |
| 57 | Current class most protected | F | M2c one-sidedness; continuous u for (Q) |
| 58 | Classes without probes | Q | Desired quotas; insufficient coverage marked |
| 59 | Low reference accuracy causes permanent alarm | F | F⁻ over reference-correct items; N/A when empty |
| 60 | Snapshot after the switch | Q | Ring with provenance label; lag reported |
| 61 | m reset shrinks the step | F | Separate k_m |
| 62 | m̂ = 0 | F | Zero step; "zero step" excludes decay, which is applied separately |
| 63 | Group rates applied after the solve | F | Rates inside B (C2) |
| 64 | Zero trace / all u = 0 | F | Skip branch; empty C means baseline |
| 65 | Momentum ≠ gradient | Q | Surrogate label; gᵀΔ ≥ 0 rate logged |
| 66 | Sketch-dependent step | Q | Independent validation pool; M2b only as a variant |
| 67 | fp16 G or C | Q | Quantised surrogate, stated |
| 68 | Large edit between refreshes | F | Dirty marks (C4) |
| 69 | Common logit translation | F | Centred metric; r(x); F⁻ as the trigger |
| 70 | New class outranks old | Q | New-label check; versioned registry; C migrated exactly by zero-padding, and every label born since the last refresh is always a rival (B5.1). Still first-order and pool-only [corrected from F] |
| 71 | Reference quantisation | F | fp16 tolerance |
| 72 | Mixed reference times | Q | Timestamps kept; no single-model claim |
| 73 | No reference-correct items | F | N/A, with absolute accuracy shown |
| 74 | Warm-up off-by-one | F | C1 step 12/18 |
| 75 | Empty buffer / empty new loss | F | Loss 0; sampling with replacement when the buffer is smaller than n_rep |
| 76 | New items never inserted | F | C1 step 14 |
| 77 | Provisional item later committed | F | Immutable roles and quarantine |
| 78 | Label on first occurrence | F | Logged unknown; row created after reveal |
| 79 | Novel rule without input novelty | F/Q | new_cond |
| 80 | First-batch timing | F | C1 steps 2–4 |
| 81 | Two routers | F | One posterior |
| 82 | Weak constant evidence | Q | L_min is the constant-evidence boundary (two symmetric states), not necessary under noisy evidence (A5); multi-state thresholds and delays by simulation [corrected from F] |
| 83 | Quota exceeds budget | F | Desired quota, ineligibility |
| 84 | Exposed item with no fresh replacement | F | Renamed "replacement-eligibility threshold"; checked at every eligible arrival |
| 85 | Non-matrix parameters in snapshots | F | Stored exactly |
| 86 | Snapshot shape mismatch | F | Migration (C3) |
| 87 | No compression probes | F | Store exact, or "no snapshot" |
| 88 | Candidate exceeds peak memory | F | Preflight, abort |
| 89 | Rollback needs the post-refresh state | F | S0 → S1 → S2 |
| 90 | Offline replay moves the clock | F | Forbidden (C4) |
| 91 | Same net improvement, different power | Q | Envelope; control mode |
| 92 | Tiny context harm guard | F | Absolute and fractional limits, minimum count |
| 93 | Feature rotation | F | Refit probe |
| 94 | Undefined savings | F | Censored time-to-threshold |
| 95 | Low-rate "cliff" chosen | F | High-side cliff rule (B4.2) |
| 96 | Confirmation reused after redesign | F | New confirmation set per claim; final untouched |
| 97 | **True conflict under (K) or (Q)** [new] | Q | Per-class blocked-progress β_j detects it (ρ̂ hides minority conflicts); adapter from M5; below M5, a logged release for T_rel steps so the learner cannot deadlock [corrected] |
| 98 | **Rival outside the top k_r overtakes** [new] | Q | Enters at the next refresh; nonlinear margin change measured |
| 99 | **Constraint infeasibility from stale c** [new] | F | 0 is always feasible (s ≥ 0); the scaled fallback step |

---

## B13. Hyperparameters

v1.2 values are kept except for those below. **(dose)** means the value is swept, not fixed.

| Setting | Value |
|---|---|
| Group rates | η_b = 1e-5, η_p = 1e-4, η_h = 1e-2, inside B |
| Step clip | uniform scale so that every |Δ_i| ≤ 10·η_{g(i)} |
| ε | 1e-8 |
| M2c | n_H = 32, k_r = 2 plus labels born since the last refresh, R_pre = 25, κ_s = 0.2 (dose), m_floor = 0.5ς/10 (dose), u_min = 0.5, s_rep = 0.1, independent validation pool 16 |
| Conflict | β_j < 0.2 for P_conf = 20 batches; release T_rel = 500 below M5 |
| (Q) | ω̃ ∈ {0.1, 0.3, 1, 3, 10} (mean protected-subspace strength); n_H = 64; R_pre = 50 |
| Floor | ρ_min = 0.2 in the B⁻¹ metric [guess], (Q) only |
| Usage | T_act = 500; τ_u ∈ [10³, 10⁴], default 5·10³; u_act ∈ {0.25, 1} for (Q) only |
| Monitor | R_mon = 500 |
| Provisional contexts | dissolve after T_prov = 2,000 steps; open-set probability ε_open = 0.01 |
| Probes | q_cls = 30 (10/10/10), p_q = 0.25; quarantine 5% of 𝔅_pers |
| Alarms | ≥ max(3, 10%) forgotten **and** buffer F⁻ ≥ 0.1; recovery ≤ 0.05; boost 5·R_mon |
| Filter | ε_lab = 0.05; n_eff = 4 (both temperatures); s_new = 2; π_new = 0.1; h clipped to [10⁻³, 0.5]; switch 0.9, novelty 0.5 |
| Snapshots | r_Ψ = 16, vectors exact; L_ring = 2; compression disagreement ≤ max(1, 2%) with ≥ 20 items |
| Reactivation | control mode; |𝒬^acc| ≥ 53; harm max(2, 1%) pooled and max(1, 10%) per context with ≥ 10 items |
| Warm-up | 1,000 gradient-used items |

This is about 50 settings. They are set on development streams only, and confirmed once.

---

## B14. Main loop (C1, concretely)

```
init S; registry L_0; ledgers; rng_main, rng_M2, rng_rows, rng_filter separate
for each batch X:
  preflight(ledgers); freeze(context models)                                  # C1.1, C4
  p_minus ← filter.predict_prior(); p_minus ← p_minus · p_in(X | ·)  [M3+]     # C1.2
  ŷ ← predict(X, path = shared [M1–M4] | gated routed [M5]); save(ŷ)          # C1.3
  Y ← reveal(); log_prequential(ŷ, Y); p_plus ← p_minus · p_lab(Y | X, ·) [M3+]# C1.4
  decide_once(p_plus, precedence)  → maybe switch/new_cond/new_in [M3+]       # C1.5, B8.3
     on switch: Ψ ← ring.pick(boundary); schedule snapshot Tx for C1.17 [M4]; reset m, k_m
  register_new_labels(Y) → new rows; zero-pad C, add new-rival columns        # C1.6, C3
  roles ← assign_immutable_roles(X, Y)  (train | probe-reserved | quarantine) # C1.7, B7.1
  t_last[cells(X)] ← t  (all items)                                           # C1.8
  R ← sample_replay(buffer_before_this_batch, q_κ)  (∅ → loss 0)              # C1.9, B6
  a ← Adam moments on CE(train ∪ R)  (k_m, k_v separate)                      # C1.10–11
  Δ0 ← −B a
  if not protection_ready or structurally dirty: Δ ← Δ0 (log)                 # C1.12
  else: Δ ← solve_Q(B, C, λ) | solve_K(B, C, s(m̃)) with numerical contract; floor (Q)
        β_j per class → conflict counters                                    # B5.1
  Δ_app ← α_clip·Δ ; θ ← θ + Δ_app                                            # C1.13
  m̃ ← m̃ + CᵀΔ_app  (the APPLIED step; tracking the pre-clip Δ overstates margins) [A6]
  decay toward anchors (its margin effect is outside the tracked guarantee); φ ← φ + αφ(θ − φ) [M3+]
  insert(train-role items) with replacement-eligibility rule                  # C1.14, B6
  update context heads on train-role items [M3+]                              # C1.15
  write references at observed deactivation events                            # C1.16, B7.2
  run due transactions within budget: refresh (pool, C, s, λ), monitor, reactivation [M4],
      conflict → adapter [M5], consolidation S0→S1→S2 [M5]                    # C1.17, C4
  rebuild dirty caches (ready from next batch)                                # C1.18
  verify ledgers; log coverage/evictions                                      # C1.19
```

---

## B15. Levels, build order and gates [revised; build order follows audit's recommendation, which I agree with]

**Levels.** Each depends only on lower levels.

| Level | Contents | Gate before the next level |
|---|---|---|
| M0 | harness, C1–C5 infrastructure, V1–V7 including T1–T17 | all tests pass; independent code review |
| M1 | global cosine head, group rates in B, carried v, refreshed head, replay, ledgers | V1 parity |
| M2a | layer-output (Q) on plain linear maps | E4a vs M1 and compute-matched ER, damage measured held out |
| M2c | margin constraints (variant) | E4c vs M2a, M1 and compute-matched ER, held out; the coverage diagnostic (I5) is reported |
| M2b | sketch (Q) with independent validation | only if both M2a and M2c fail |
| M3 | filter with **known fixed heads**, then learned heads, then novel conditional regimes; no routing | T2 delays and non-detections; oracle-filter comparison |
| M4 | versioned snapshots + control-mode reactivation | E8: savings vs the wrong-snapshot control, with censoring |
| M5 | routed heads/adapters (driven by conflict events), then controllers, then consolidation, **one at a time** | each against the level below at matched resources |

**Experiment gates** [new: v1.2 and the v1.3 draft referred to E-numbers without a table]. Each compares at matched resources, against compute-matched ER where it concerns retention.

| Gate | Question | Section |
|---|---|---|
| E1 | Access vs storage: how much forgetting is readout and routing (H3) | B2.2 |
| E2 | Stable/plastic split depth and decay strength | B4 |
| E3 | Optimiser rule: carried v and m reset vs P182's blend; a = g recompute on uphill steps | C2 |
| E4a | M2a vs M1 (held-out damage) | B5.2 |
| E4c | M2c vs M2a and M1 (held-out damage, coverage diagnostic) | B5.1 |
| E4 | Protection type × replay ratio | B5 |
| E6 | α_f dose | B6 |
| E7a | Protection dose and release × floor | B5.4 |
| E7b | Restore-on-alarm vs boost-only repair | B5.4, B7.3 |
| E8 | Snapshot savings vs the wrong-snapshot control, censored | B9 |
| E8b | Resetting v at reactivation | B9.2 |
| E9 | Anchoring to real data at current weights (H1) | B0.4 |

**Gate rules.**
- **The claim family is defined before confirmation.**
- **Superiority:** a paired 95% interval over ≥ 5 independent stream-order-seed units, with error allocated across the family's preplanned superiority claims (Holm).
- **Non-inferiority:** an intersection-union test on every prespecified stream, with margin −1.0 point.
- **Each decision** uses its own frozen confirmation set. A redesign needs a new confirmation set. One final set stays untouched.

**Factorial tests:**
- protection type × replay ratio;
- κ_s × m_floor;
- release × floor;
- snapshots × filter quality.

**What a negative result means.** It rejects the mechanism in that setting. It does not reject H1–H3 in general.

---

## B16. Audit of v1.3's own claims

**Proved (with exact arithmetic and the stated assumptions):**
- L-a in its Bayes form; L-b.
- C2:
  - the Woodbury form;
  - (K) existence and uniqueness;
  - surrogate descent and non-expansion in B⁻¹ for both (Q) (a proximal map) and (K) (a projection);
  - ρ̂ ∈ [0, 1];
  - the blend root;
  - the no-effect condition HΔ_0 = 0;
  - the feasible fallback step.
- M2a's identity for linear maps.
- M2b's pointwise unbiasedness for fixed directions.
- M2c's cumulative first-order margin floor, for pairs continuously constrained under projected steps, with tracked margins (linearised, exact gradients).
- The centred-certificate inequality and the centred Markov bound.
- The sticky-HMM threshold L_min (two symmetric states; exact under constant evidence; **not** necessary under noisy evidence, A5).
- The McNemar threshold and the exact power formulas.
- The release horizon.
- The routing accuracy decomposition.

**Measured on toys only (not evidence for real networks):**
- the held-out first-order frontier: the quadratic brake beats per-item constraints on items not used to build them [my in-sample "dominance" claim was a tautology and is withdrawn];
- **the nonlinear toy: M2c as specified matched ER; compute-matched ER beat both, but A4 shows that was loss balance (weighted replay and ER-ACE beat it at 1× compute), and that forgetting there is mostly readout bias.** Zero-slack M2c protected held-out items at first order (86%) for only +3.1 points;
- the 12× sketch-exploitation ratio;
- the nonlinear class-incremental toy (Part A).

**Approximations:**
- linearisation;
- staleness within R_pre;
- pool coverage;
- the rival set;
- the generalised-Bayes filter temperatures;
- geometric durations;
- Gaussian head features;
- fp16 storage.

**Largest bets:**
1. **Coverage.** Protecting stored items does not protect the population: the first toy already shows this for M2c. The next mechanism to try is coverage-widening constraints (augmented or neighbouring items), which is a hypothesis.
2. Conflict events are a good adapter trigger.
3. Release by u_min.
4. Snapshot savings survive migration.

**Risks:**
- about 50 settings;
- M2c memory (about 3.6 GB at ViT-B plastic third) and compute (estimates);
- at scale, pre-training plus a slow rate plus random replay may already capture most of the benefit (P124, P205).

---

## Changelog v1.2 → v1.3

The v1.1 → v1.2 changelog and the v1.2 review round are in notes/archive/formal-spec-v1.2.md.

**New structure.** Five contracts (Part C): causal order, update geometry, versioned schema, transactions and budget, and evidence vs control.

**Geometry:**
- Group rates are inside B.
- The descent claim is relabelled surrogate descent.
- Both protection forms are a single B⁻¹-proximal map ((K) is a projection, (Q) is not), with proved non-expansion.
- The floor is measured in B⁻¹, with a stable root, explicit priority rules, and a numerical contract with a feasible fallback.
- m and v have separate counters, plus a uniform (not coordinatewise) step clip.

**Protection:**
- M2c (margin constraints) added, then demoted to a gated variant after the held-out and nonlinear checks.
- Centred logit metric.
- M2b demoted to a variant with independent validation.
- The cosine head is excluded from M2a.
- Strength is normalised as the mean protected-subspace eigenvalue.
- Usage is continuous, and alarms no longer write t_last.
- Repair is boost-only by default.

**Monitor:**
- Paired forgetting F⁻ is the trigger.
- The translation-invariant headroom r(x) and the new-label check are diagnostics.
- Evaluation references are evaluator-side, at true boundaries; control references at observed events.
- Evaluation and control references are separate.
- fp16 tolerance.
- Probe quotas are desired, not mandatory.
- Item roles are immutable, with a quarantine.

**Filter:**
- New hypothesis new_cond.
- Smoothed label models.
- Normalised transitions.
- Causal scoring order.
- Exact threshold L_min replaces the delay formula.
- Bayes form of L-a.
- "Conditional shift" naming.
- One router.

**Snapshots and reactivation:**
- Complete schema, with vectors stored exactly.
- Migration on label or topology change.
- Compression checked against the saved checkpoint.
- Provenance labels.
- Transactional reactivation with cache invalidation.
- Control-mode acceptance, with the envelope power stated.

**Budget and transactions:**
- Persistent and peak budgets.
- Preflight allocator.
- S0 → S1 → S2 consolidation.
- Offline steps do not advance stream time.

**Measurement:**
- Controlled readout table, with refit decoders.
- Censored savings.
- GEM, mGEM and rolling-window baselines.
- Claim families.
- V1 parity.
- Tests T12–T17, plus T12b: every protection claim is measured on held-out items.
- Edge ledger extended to 99 rows.

**Loop.** The main loop is rewritten in C1 order: items are inserted, warm-up is fixed, and labels are registered after prediction.

**Build order** follows the audit: M2a, then M2c, then (only if needed) M2b, before any context inference.


**Second review round (an independent reviewer checked v1.3 itself; 20 defects, all fixed):**
- **My held-out error.** "Constraints dominate the quadratic brake" was measured on the constrained set itself. Held out, the brake wins. Withdrawn; T12b added.
- **M2c guarantee.** Slack now uses tracked margins; the theorem is restated for continuously constrained, projected pairs; stale-slack counterexample (−5.5) kept as a test.
- **Caches.** Only structural events mark caches dirty; read literally, the draft disabled protection on every step.
- **Class growth.** C is migrated exactly by zero-padding; labels born since the last refresh are always rivals; the Markov bound gains a new-label term.
- **Filter.** L_min is the constant-evidence boundary (A5: not necessary under noisy evidence); multi-state thresholds by simulation; open-set probability for unregistered labels; quarantine of a merged or dissolved provisional child is discarded.
- **Conflicts.** Protection and replay repair do conflict; repair slack during boost windows; per-class conflict detection for (Q) and (K); a logged release below M5 prevents deadlock.
- **Smaller fixes.** u_min = 0.5 (release in 3τ_u, not 99τ_u); dead root branch removed and coefficients renamed; one-sided McNemar stated; ≥ 6 classes for reactivation eligibility; E-gates table; M2a fixed cost 0.8 GB; corrected toy numbers (79.6%, 13.4%, 5,748 cases, n ≈ 170).
- **Toy harness.** Separate RNGs so ER and M2c see identical streams; compute-matched ER added; spec-faithful M2c (R_pre, tracked margins, k_r = 2, correct-only pool); docstring corrected.
