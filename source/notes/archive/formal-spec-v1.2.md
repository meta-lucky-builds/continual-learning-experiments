# Formal specification of a continual learner (v1.2)

2026-10-28. Written for Lucky.

v1.2 replaces v1.1. It answers an external critique that Lucky pasted into the thread. Every point in that critique was checked: the counterexamples were reproduced numerically, and the proposed fixes were tested on small cases before being written in. An independent reviewer then found 21 defects in the first draft of v1.2; all are fixed (see the last section). The numerical checks are in v12/checks.py and v12/fix2.py.

**What this document is.** It specifies a *family of continual-learning algorithms*. They combine four ingredients:
- replay of real stored items;
- function-preserving updates, approximate and local;
- context inference and retrieval;
- adaptive control.

Whether these improve the stability–plasticity trade-off is **an empirical question**. Nothing here is a general solution to forgetting. The guarantees that *are* proved are local and are stated with their assumptions.

**Labels.**

| Label | Meaning |
|---|---|
| **[src: P#]** | From that paper, as the paper states it. |
| **[proved]** | A complete argument is given here, under the stated assumptions. |
| **[approx]** | A stated approximation. Its error is not bounded. |
| **[design]** | My choice. Motivated by the sources, but tested by none of them. |
| **[guess]** | Weak or no support. It must pass a gate before anyone relies on it. |

---

## Part A. Response to the critique

**Verdict.** All 16 points are substantially correct. I reproduced the four checkable counterexamples numerically:

- **Adam composition can go uphill.** With the critique's 2×2 example, g·P⁻¹Dg = −3.04, so the composed step is an ascent direction.
- **Kronecker (K-FAC) factorisation protects harmless directions.** For an off-diagonal Δ, the exact old-output change is 0, while the K-FAC estimate is 0.5.
- **The McNemar p-values are right:** 0.125 and 0.031.
- **The margin bound holds** in a simulation.

I have two small quibbles, neither of which changes any conclusion:

1. **§8, EMAs.** The critique is right that knowledge can persist in later weights. But the savings mechanism exists for the case where later states *overwrite* a context, and that is the case where an EMA can't recover it. I narrow the claim. The mechanism stays.
2. **§13, R_head.** v1.1 did state the drift-trigger exception. The real error was that I3 still *listed* R_head under the condition "≪ 1/α_1". The fix is to remove it from the condition.

Where each fix lives:

| # | Critique point | Verdict | Fix (section) |
|---|---|---|---|
| 1 | No precise problem definition; "laws" overstated | Valid | B0: formal problem, assumptions A1–A6, identifiability; L1–L3 renamed hypotheses H1–H3 |
| 2 | Derivation valid but local; layerwise drops cross-layer terms; tensor order | Valid | B5.1: M2a is stated as layer-local; M2b is a gradient sketch through the whole network over Θ_P, so it includes cross-layer and token terms within Θ_P; order fixed |
| 3 | K-FAC protects directions the outputs can't see | Valid (reproduced) | B5.1: M2a uses the exact layer-output metric; M2b uses per-item projected-logit gradients (unbiased for the exact metric; under-protects by sampling, stated) |
| 4 | Weighting both factors gives u² and cross-context terms | Valid | B5.1: per-sample weights inside the exact metric; linear in u |
| 5 | Floor may have no root; historical ≠ current energy; Ē basis rotation; norm ≠ plasticity | Valid | B5.3: floor enforced on the **current** step by a blend with a closed-form λ; root exists in [0,1]; descent preserved [proved]; no stored eigen-energy |
| 6 | V-all with Adam can go uphill | Valid (reproduced) | B5.2: one joint SPD system (D⁻¹ + ω𝓗)Δ = −ηm̂, solved exactly (Woodbury, M2b) or by PCG from 0 (M2a); descent w.r.t. m̂ [proved] |
| 7 | Snapshots drift with a moving base; plastic-only restore | Valid | B9: immutable base checkpoints of the full model, plus low-rank differences; reconstruction is exact up to compression error, which is measured |
| 8 | EMA argument overinterpreted; decay toward a moving anchor doesn't bound the norm | Valid | B9.1 narrowed; B4.2 states that it controls separation, not norm; norm is monitored |
| 9 | Recognition affects correctness; input-only detection fails under pure label shift | Valid | B8: sticky Bayesian context filter using input **and** delayed label evidence; confidence-gated routing with a shared fallback; correctness dependence stated |
| 10 | Probe split isn't an independent monitor; adaptive reuse | Valid | B2.3 data tiers; B7: monitor signals are control heuristics with no statistical guarantee; claims only on untouched confirmation streams |
| 11 | "Power requirement" wrong; arithmetic error; single-class contexts never eligible | Valid | B9.3: explicit selection/acceptance split at intake; eligibility from an exact McNemar power condition; per-context quotas |
| 12 | Access gap doesn't isolate access; max over test layers | Valid | B2.2: renamed "diagnostic probe advantage"; 4-way decomposition; layer chosen on validation |
| 13 | Ten inconsistencies (table) | Valid | Each fixed: see the changelog |
| 14 | Fixed buffer ≠ fixed total memory | Valid | B6.3: a total byte budget with allocation and eviction; resource–performance frontier; information limit stated |
| 15 | No link from output movement to retention | Valid; very useful | B7.2: an exact per-example retention **certificate** from actual stored logits; Markov population bound |
| 16 | Gates need correcting | Valid | B15: savings criterion fixed; non-inferiority margins; factorial interaction tests; compute-matched replay; development/confirmation streams |

---

# Part B. The specification

## B0. Problem definition

### B0.1 Stream model

**Contexts.** A context c has a joint distribution P_c(x, y) and a risk

$$R_c(\theta)=\mathbb E_{P_c}\,\ell(f_\theta(x),y).$$

**The stream.** It is a sequence of episodes (c_1, T_1), (c_2, T_2), …: context c_k is active for T_k steps, and at each step a batch X_t ∼ P_{c_k} arrives.

**Test-then-train.** At step t the learner predicts on X_t. Then labels Y_t are revealed, and only after that may it update.

### B0.2 Assumptions

Every guarantee or design argument names the assumptions it uses.

- **A1. Consistency within a context.** P_c(y|x) is fixed during an episode of c. It may differ between contexts.
- **A2. Persistence.** Episodes last at least T_min ≥ 2 batches. This is what makes delayed label evidence usable for context inference (B8).
- **A3. Identifiability.** For any two contexts c ≠ c′, at least one of these holds:
  - **(a)** their input distributions differ, d(P_c(x), P_c′(x)) ≥ Δ_x > 0;
  - **(b)** their conditionals differ on a set of positive mass, and the labels are revealed (test-then-train).

  If neither holds, the two contexts are the same context for every purpose.
- **A4. Budgets.** A fixed total byte budget 𝔅 and a compute budget per item 𝒞_max (B6.3, B11 I5).
- **A5. Evaluation objective.** Recurrence-weighted risk. Let w_c(t) be the probability that c is active at a future time, estimated from the stream's own recurrence statistics, or set by the evaluator. Then

  $$\mathcal R_t=\sum_c w_c(t)\,R_c(\theta_t),$$

  reported **together with** current-context risk R_{c_now} and per-context risks. No single number is the target.
- **A6. Permitted error.** Retention is judged against the per-context tolerance ε_c, measured from the end of c's last episode:

  $$R_c(\theta_t)-R_c(\theta_{end(c)})\le\epsilon_c.$$

### B0.3 Limits (what no algorithm can do)

**L-a. Conflicting contexts without context information** [proved]. Suppose P_A(y=0|x_0) = 1 and P_B(y=1|x_0) = 1, and the predictor sees only x_0. Then on the equal mixture, accuracy at x_0 is at most ½. Replay, preconditioning and snapshots cannot change this.

*Consequence:* conflicting contexts need context evidence. Under A3(b), that evidence can only come from labels, with at least one batch of delay (B8).

**L-b. Bounded state** [proved, counting argument]. With 𝔅 bytes of state, a learner can distinguish at most 2^{8𝔅} internal states. So it cannot exactly preserve an unbounded number of independent bits about past contexts. Beyond what structure allows to be compressed, some loss is unavoidable. A6 is therefore a trade-off target, not a guarantee.

**L-c. Recognition delay** [src: P128]. Separating two contexts with sufficiently small error needs a number of samples of order (log of the inverse error rate) / I, where I is the information gap between them, which grows with Δ². Near overlap the delay is unbounded.

### B0.4 Hypotheses (formerly called "laws")

These are hypotheses with supporting evidence. They are not established laws, and each is tested by a gate.

- **H1. Anchor.** Protection estimates computed from real stored data at the current weights beat both estimates frozen at write time and estimates computed from the model's own outputs. Evidence: P209, P129, P160, P155, P143, P154. Gate E9.
- **H2. Single knob.** A pure magnitude brake trades plasticity against retention. Escaping that needs direction selectivity, a second timescale, release, or growth. Evidence: P189, P153, P200. Gate E4/E7.
- **H3. Access before storage.** A large part of measured forgetting is in the readout or routing, not the representation. Evidence: P177, P136, P146, P196. This is a contingent pattern: P197 and P124 are counter-cases. Gate E1.

---

## B1. Notation

**Time, contexts and episodes**

| Symbol | Meaning |
|---|---|
| t | step index; batch X_t of n_new items |
| c_now | MAP context of the context filter (B8); in M1–M2 a single context |
| episode, gap | maximal run with the same c_now; end of one episode to the next start of the same context |

**Parameters**

| Symbol | Meaning |
|---|---|
| θ = (θ^st, θ^pl) | fast parameters: stable layers ℓ ≤ ℓ_s; plastic layers plus head |
| θ_0 | pre-trained initial weights |
| φ | one slow EMA of θ (B4.2); used as the slow view and the decay anchor |
| 𝔅 | total byte budget (A4) |
| ℬ | replay buffer of real items (x, y, c, e, f_ref, m_ref). e is the exposure count; f_ref and m_ref are the reference logits and margin (B7.2) |
| 𝒬_c = 𝒬^sel_c ∪ 𝒬^acc_c ∪ 𝒬^mon_c | probe items of c, disjoint, fixed at intake. Never trained on and never in ℬ |
| W_ℓ ∈ ℝ^{d_out×d_in} | layer map, y = Wz. Convolutions and attention apply W at many positions p; M2a sums over positions (exact for its object), M2b differentiates through them (B5.1) |

**Vectorisation.** vec(·) stacks columns, so vec(GΔA) = (A⊗G)vec(Δ). (Used only in Part A's discussion of K-FAC; v1.2 stores no Kronecker eigenbasis.)

---

## B2. Measurements

### B2.1 Per-context quantities

**Accuracy.** a_{t,c} is the accuracy of the system's actual output path on c's evaluation split (B2.3).

| Name | Definition | Notes |
|---|---|---|
| Retention gap | R_c(θ_t) − R_c(θ_end(c)), compared against ε_c | A6 |
| Plasticity | Π_c = a_{end(c),c} / a^fresh_c | a^fresh: θ_0 trained on c alone with the same budget [src: P124, P197] |
| Savings | Σ_c = N^fresh_c / N^relearn_c to reach 0.9·a^best_c | [src: P127, P144] |
| Savings control | Σ^wrong_c: reactivate another context's snapshot | Pass: Σ_c > 1 **and** Σ_c > Σ^wrong_c, with a paired 95% confidence interval over seeds excluding 0 [corrected per critique] |
| Forward cost | N_k for the k-th new context | [src: P127] |
| Displacement | D_t = ‖θ_t − θ_{t−Δ}‖ over the whole network, absolute | [src: P208 Eq. 4.2] |
| Self-knowledge | In/Forg/Out AUROC on the evaluation split | [src: P160] |
| Certified retention | CR_c (B7.2) | [proved] per example |
| Resources | bytes by component; FLOPs per item; *measured* event frequencies | B6.3, I5 |

### B2.2 Diagnostic probe advantage (renamed; it is not a "ceiling")

A fitted probe is not a mathematical ceiling. It can score below another readout, so Γ can be negative.

$$\Gamma_{t,c}=p_{t,c}-a_{t,c},$$

where p is an L2 logistic probe trained on frozen features. The **layer is chosen on a validation split** and the probe is then scored on the evaluation split [corrected].

Γ mixes several causes. The **decomposition** is mandatory:

| Comparison | What changes | What the difference isolates |
|---|---|---|
| D1 | the actual head vs a head rebuilt from the **same buffer** on the same features | head optimisation and staleness |
| D2 | the rebuilt head (buffer) vs a probe trained on the **full historical training data** | missing data, i.e. buffer limits |
| D3 | actual routing vs **oracle** routing, with the same heads | routing error |
| D4 | features at time t vs features at end(c), with the same full-data probe | representation loss, i.e. storage |

Only D4 measures storage loss. "Access failure" means D1 + D3.

### B2.3 Data tiers (adaptive-reuse control) [corrected per critique; src: Dwork et al. 2015, on adaptive reuse of holdout data]

| Tier | Used by | Allowed adaptivity |
|---|---|---|
| 𝒬^sel, 𝒬^acc | the learner: calibration, candidate selection, acceptance | adaptive (inside the learner) |
| 𝒬^mon | the learner's alarms and controllers | adaptive; **no statistical claim** is made from it |
| development streams | choosing settings and making gate decisions | adaptive (by us) |
| confirmation streams | one run per gate decision, after settings are frozen | non-adaptive |
| final stream | once, at the end of the project | untouched until then |

Every reported result names its tier. Only confirmation and final numbers are claims.

### B2.4 Protocol

1. **Baselines, all tuned:**
   - sequential fine-tuning with backbone and head learning rates swept separately (P158);
   - a frozen first-stage extractor with a rebuilt head (P197);
   - ER with tuned n_rep and iterations (P178);
   - **compute-matched ER**, which spends the protection compute on extra replay steps [new, per critique].
2. **Streams:**
   - ordered class-incremental;
   - smooth drift;
   - recurring pool with heavy-tailed gaps;
   - open stream with ≥ 100 contexts;
   - **pure label shift**, where P(x) is shared and P(y|x) differs [new].
3. **Equal total bytes 𝔅** across methods, reported as a frontier over 𝔅 values.
4. **Seeds:** at least 5 for gates. Paired comparisons use the same seeds and orders.
5. **Breakdowns:** per class and per context, reported as trajectories.

---

## B3. Harness verification suite (Lucky's reliability rule)

No result counts until every check passes. The checks are automated tests in the repository and are re-run on every code change.

**V1. Reproduce published numbers.**
- ER on split CIFAR-100 matches P178's tuned ER (28.2 ± 1.2) within its interval.
- Sequential fine-tuning of a ViT with a slow learning rate matches P158 (88.86 ± 0.83) within its interval.
- The same run is also executed in Mammoth or Avalanche and must agree within the seed interval.

**V2. Known-answer metric tests.**
- A hand-built accuracy matrix gives exact forgetting, savings and Π.
- The Wilson interval and the exact McNemar test match scipy or statsmodels to 1e-12.
- The probe on separable synthetic data gives 100%.

**V3. Planted effects.**
- An oracle model scores 100%; a random model scores chance.
- Scrambling the head while keeping the features moves D1, not D4.
- Damaging one context's weights moves only that context.
- A leak (a test item injected into training) must be caught by V5.

**V4. Toy cases with known answers.** Six come from the critique; the rest are mine.

| Case | Expected result |
|---|---|
| T1 correlated (z, b) | The exact per-sample metric leaves the off-diagonal Δ free; K-FAC does not (the example in Part A). M2b's sampled metric converges to the exact one as J_ξ grows |
| T2 pure label shift | An input-only detector is at chance; the per-context label filter (B8.2) detects within the delay predicted by B8.2 for the planted I_y (permuted labels with accurate heads: about 1–2 batches) |
| T3 long gap, moving weights | A snapshot reconstructs the stored model exactly up to its measured compression error, regardless of later drift |
| T4 gradient energy in the null space | Δ_ω = Δ_0 exactly, ρ̂ = 1, no blend |
| T5 rotating statistics | No stored eigen-coordinates exist in M2 (the design removes this case); a test asserts it |
| T6 many contexts, fixed 𝔅 | Bytes never exceed 𝔅 (assertion), and eviction follows its rule |
| T7 linear regression | Measured forgetting matches the closed forms of P140 and P201 |
| T8 solver | Woodbury (M2b) and PCG (M2a) match a dense solve (scratch: ≤ 5e-6 relative at ω = 10⁴); every PCG iterate from 0 is descent (scratch: 0 of 2,000 failures); a warm-started CG is shown to fail (reviewer: 74 of 200), as a regression test; ω = 0 returns Δ_0 exactly; the blend reaches ρ_min exactly |
| T10 token-shared weights | On a 3-token toy model, the sampled M2b metric is unbiased for the exact logit change (scratch: 106.6 vs 107.1), and a per-token sum is not (reviewer: 1.55 vs 33.4) |
| T11 permuted labels, shared class set | Renormalising a shared head gives a likelihood ratio of 0; per-context heads separate the contexts |
| T9 certificates | For random logits and perturbations, every item certified by the retention certificate is still correct, and every item certified by the agreement certificate keeps its prediction (deterministic checks) |

**V5. Leak assertions.** These live in the training loop:
- no item ID of 𝒬 or of the evaluation splits ever appears in a gradient batch;
- the splits are disjoint by ID;
- labels are used only after prediction (test-then-train).

**V6. Limits** [corrected after review].
- With ω = 0 **and** the M2 replay changes off (exposure swaps, alarm boosts), M2 reproduces M1 bit for bit with the same seed. This holds because every M2-only random draw (pool, ξ) uses a separate generator, and the ω = 0 step is exactly Δ_0 (B5.2).
- With the monitor on but acting on nothing, logs may differ; weights may not.
- With the context filter fixed at a single context, M3 reproduces M2.

**V7. Pre-registration and review.**
- Predictions and analysis plans are written before confirmation runs.
- Every run logs its config, seed and commit.
- The harness gets an independent code review, as the spec did.

---

## B4. Core learner M1

### B4.1 Architecture and readout [src: P205, P208, P186, P197, P120, P143, P158]

- **Backbone and normalisation.** A pre-trained backbone with GroupNorm or LayerNorm, or BatchNorm with frozen statistics (P148).
- **Stable and plastic split.** Default ℓ_s = 2/3 of depth. Gate E2 sets it.
- **Head.** A **global** cosine head over all seen classes:

  $$f_j=\varsigma\cos(w_j,h(x)),\qquad \varsigma=10.$$

  This is the routing-free shared path in every level. Two-level readout is only added in M5, and only if it beats this head.

### B4.2 Rates, optimiser and decay

**Learning rates.** η_b = 1e-5, η_p = 1e-4, η_h = 1e-2. Choose η_p from the cliff side: η_p = η^cliff/3, where η^cliff is the smallest rate on the development stream that is at least 2 points below the best [src: P158 doses].

**AdamW, with v never reset** [design, simplified from P182's blended rule]. m is reset on a declared switch, from M3 on [guess]. Gate E3 compares this with P182's rule.

**Decay toward the pre-trained weights** [design; L2-SP idea, Li et al. 2018; corrected after review]. Decoupled, per group:

θ ← θ − η_group λ_wd (θ − θ_0),  λ_wd ∈ {0, 0.01, 0.1}, default 0.01.

- At η_p = 1e-4 the pull is 1e-6 per step, about 1% per 10⁴ steps. It is weak by design, and E2 sweeps it.
- It bounds drift from θ_0 only in balance with the gradient. It is **not** a control on separation from φ. v1.2's draft claimed that, but its pull was 1e-8 per step and inert.
- Norms ‖θ − θ_0‖ per group are logged.

**The slow trace.**

$$\phi\leftarrow\phi+\alpha_\phi(\theta-\phi),\qquad \alpha_\phi=10^{-3},$$

over all layers, used **only** for the slow view (B8). It does **not** store contexts; that is the snapshots' job (B9).

### B4.3 Batches and loss

**Batch composition.**
- Each batch has n_new = 16 new items and n_rep replay items, with r ∈ [0.25, 0.75] and default 0.5.
- n_rep is tuned [src: P178].
- Items routed to 𝒬 are removed before L_new.

$$\mathcal L=\overline{\mathrm{CE}}_{new}+\overline{\mathrm{CE}}_{rep}.$$

**Startup policy** [corrected after review]. For the first n_warm = 1,000 items, i.e. t_warm = n_warm/n_new ≈ 63 steps:
- plain training;
- the buffer fills;
- φ, t_last and exposure counts **are** updated;
- no protection or monitor.

At t_warm, in this order:
1. head and statistics refresh;
2. pool build and ω;
3. reference logits for all items so far.

Only then does the first protected step run.

### B4.4 Buffer (M1)

- **Fill:** a reservoir balanced across (class, context) cells. In M1–M2 the context is a single context.
- **Sampling:** uniform.
- **Exposure:** e_i counts uses (B6.2).

### B4.5 Head rebuild and drift trigger [src: P143, P158, P174, P146]

**Rebuild.** At each refresh:
1. Compute fresh class means μ_j and a **pooled within-class** covariance Σ̂_w, with Ledoit–Wolf shrinkage a_LW.
2. Sample 256 features per class from 𝒩(μ_j, Σ).
3. Retrain the head for 5 epochs with the backbone frozen.

**Drift trigger.** The refresh fires when either condition holds:
- R_head = 2,000 steps have passed;
- δ_drift = max_j ‖μ_j − μ̄_j‖ / n̄_h > 0.05, checked every R_check = 200 steps on 10% of ℬ. Here n̄_h is the mean feature norm.

---

## B5. Protection (M2)

Two variants are specified. M2a is cheap and exact for a narrow object. M2b is the main design: per-item, cross-layer within the protected set, correct for token-shared weights, and solved exactly.

### B5.1 The protected objects

**Protected parameter set.** Θ_P = the plastic layers plus head (B4.1), with N_P parameters. Stable layers are slowed only by their learning rate. Protecting the stable layers too is a gated extension (E4c), limited by memory (B5.5).

**Usage weights.** u_x ≥ 0 per pool item (B5.4).

**M2a: layer-output metric** [proved exact for its object]. For layer ℓ with input tokens or positions z_{x,p} (p = 1…P_x; P_x = 1 for an MLP layer),

$$\mathcal M_a(\Delta_\ell)=\frac1{n_H}\sum_x u_x\sum_p\lVert\Delta_\ell z_{x,p}\rVert^2=\mathrm{tr}(\Delta_\ell A_u\Delta_\ell^\top),\qquad A_u=\frac1{n_H}\sum_xu_x\sum_pz_{x,p}z_{x,p}^\top.$$

- This is an identity: the summed squared change of the layer's outputs on pool items, under a weight change Δ_ℓ with the inputs held fixed. It is LPR's object [src: P180], weighted per item.
- The token sum is correct here, because ‖ΔZ‖²_F sums over tokens.
- The limitation: it ignores which output directions matter downstream, and it ignores input changes caused by earlier layers.
- The operator is 𝓗_a(Δ_ℓ) = Δ_ℓA_u, i.e. A_u ⊗ I. A_u is d_in×d_in, precomputed at refresh. The product costs d_out·d_in² per layer, independent of n_H.

**M2b: projected-logit metric over Θ_P** [proved unbiased; approx otherwise]. For pool item x, draw J_ξ independent ξ_{x,k} ∼ 𝒩(0, I_K) and form the gradient of a random logit projection

$$g_{x,k}=\nabla_{\theta_P}\big(\xi_{x,k}^\top f_\theta(x)\big)\in\mathbb R^{N_P}.$$

The metric and its operator are

$$\mathcal M_b(\Delta)=\frac{1}{n_HJ_\xi}\sum_{x,k}u_x\,\langle g_{x,k},\Delta\rangle^2,\qquad \mathcal H_b=\frac{1}{n_HJ_\xi}\,G U G^\top,$$

where G = [g_{x,k}] is N_P × (n_H J_ξ) and U = diag(u_x repeated).

Facts:
1. **Unbiased** [proved]. E_ξ⟨g_{x,k}, Δ⟩² = E_ξ(ξᵀJ_xΔ)² = ‖J_xΔ‖², where J_x = ∂f(x)/∂θ_P is the full Jacobian. So E_ξ 𝓜_b(Δ) = (1/n_H)Σ_x u_x‖J_xΔ‖², the linearised summed squared logit change on pool items. My numerical check: exact 106.6 vs estimate 107.1.
2. **Correct for shared weights and cross-layer terms** [proved]. g is a true gradient through the whole network restricted to Θ_P. So it includes token coupling, Σ_p J_{x,p}Δz_{x,p}, and cross-layer terms within Θ_P automatically. The reviewer's counterexample to a per-token sum (1.55 exact vs 33.4) does not apply to this form.
3. **Under-protection, stated** [corrected]. The realised operator has rank ≤ n_H J_ξ. Its null space is a **superset** of the null space of the exact metric. Directions that change old logits but are orthogonal to every sampled g are left unprotected. This is Monte-Carlo projection error, the mirror image of K-FAC's over-protection. It shrinks as J_ξ and n_H grow, and gate E4 measures it by comparing 𝓜_b with the exact ‖J_xΔ‖² on held-out pool items.
4. **Not exact.** Linearisation and finite sampling are both [approx].

**Pool.** n_H = 64 items, sampled **uniformly** from ℬ (weighting enters only through u_x), resampled every R_pre = 50 steps. G is recomputed at the current θ. Statistics are at most R_pre steps old [approx]. The pool is drawn from a **separate random generator**, so turning protection off does not change any other random draw (V6).

### B5.2 The update [proved]

Let D = diag(1/(√v̂ + ϵ)) be Adam's diagonal scaling, with ϵ = 1e-8, so D⁻¹ ⪰ ϵI ≻ 0. Let m̂ be the bias-corrected first moment. The step solves

$$K\Delta^\star=-\eta\hat m,\qquad K=D^{-1}+\omega\mathcal H,$$

the stationarity condition of min_Δ ⟨m̂, Δ⟩ + (1/2η)⟨Δ, KΔ⟩. K is symmetric positive definite.

**M2b: exact solve by Woodbury** [proved]. With S = (n_HJ_ξ/ω)U⁻¹ + GᵀDG, an (n_HJ_ξ)-square SPD matrix (cells with u_x = 0 are dropped from G first),

$$\Delta^\star=-\eta\big(D\hat m-DG\,S^{-1}G^\top D\hat m\big).$$

- This is exact, with no iteration and no warm start. Descent holds: ⟨m̂, Δ*⟩ = −η m̂ᵀK⁻¹m̂ < 0 for m̂ ≠ 0.
- **At ω = 0 the step is exactly the Adam step** Δ_0 = −ηDm̂, computed by the same code path with the correction term skipped. So V6 can hold bit for bit.
- Numerical check: error vs a dense solve ≤ 5e-6 relative, at ω up to 10⁴ and D spanning four decades; descent in all cases.
- **Cost per step:** GᵀDm̂ and DG(·) are 2·N_P·n_HJ_ξ multiply-adds. G, U and ω change only at refresh, but D changes every step, so S is re-formed (N_P(n_HJ_ξ)² multiply-adds, the dominant term) and Cholesky-factorised ((n_HJ_ξ)³/3) each step, unless r_D > 1 (B5.5).

**M2a: preconditioned conjugate gradients from zero** [proved; corrected after review]. For a layer, solve D⁻¹⊙Δ + ωΔA_u = −ηm̂ by PCG with preconditioner D, starting at **Δ = 0**, with k_CG = 10 iterations or relative residual < 1e-3. Never warm-start.
- **Every iterate is a descent direction.** The k-th iterate minimises q(Δ) = ½⟨Δ, KΔ⟩ + η⟨m̂, Δ⟩ over a linear subspace containing 0. Optimality along its own ray gives ⟨Δ_k, KΔ_k⟩ = −η⟨m̂, Δ_k⟩, so ⟨m̂, Δ_k⟩ < 0 whenever Δ_k ≠ 0.
- A warm start breaks this. The reviewer found 74 of 200 uphill steps with warm-started CG.
- **At ω = 0, the first PCG iterate is exactly Δ_0.**
- Numerical check: 0 non-descent iterates in 2,000.

**Strength normalisation** [design; corrected after review]. At each refresh,

ω = ω̃ · tr(D⁻¹) / tr(𝓗^{(1)}),

where 𝓗^{(1)} is the operator with **all u_x = 1**:
- tr(𝓗_a^{(1)}) = tr(A_1)·d_out;
- tr(𝓗_b^{(1)}) = ‖G‖²_F / (n_HJ_ξ). This is the exact trace of the sampled operator, and an unbiased estimate of the exact one.

ω is then frozen until the next refresh. Normalising with u ≡ 1 means **release lowers the absolute brake**. v1.2's draft normalised with u, which cancelled any uniform release and let one cell's restore lower protection on all others [corrected].

### B5.3 The current-step plasticity floor [proved]

Let Δ_0 = −ηDm̂ and Δ_ω = Δ* (B5.2).

**Guard.** If ‖Δ_0‖ = 0 (m̂ = 0), the step is 0 and nothing else is computed.

**Ratio.** Let ρ̂ = ‖Δ_ω‖² / ‖Δ_0‖². Note that ρ̂ can exceed 1: K⁻¹ ⪯ D does not imply ‖K⁻¹m̂‖ ≤ ‖Dm̂‖. It is a step-norm ratio, not a passed fraction.

**Blend.** If ρ̂ < ρ_min, use

$$\Delta=\Delta_\omega+\lambda(\Delta_0-\Delta_\omega),\qquad\lambda=\frac{-b+\sqrt{b^2-4ac}}{2a},$$

with a = ‖Δ_0 − Δ_ω‖², b = 2⟨Δ_ω, Δ_0 − Δ_ω⟩, c = ‖Δ_ω‖² − ρ_min‖Δ_0‖² < 0.

Facts [proved; the reviewer checked them]:
1. **The root exists and is unique.** a > 0 and c < 0, so the roots have opposite signs. The positive root lies in (0, 1], because f(1) = ‖Δ_0‖² ≥ ρ_min‖Δ_0‖².
2. **Descent is preserved.** This needs Δ_ω to be a descent direction, which B5.2 now guarantees for both variants.
3. **No stored eigen-energy, and no missing-root case.** A gradient lying entirely in the protected null space gives KΔ_0 = D⁻¹Δ_0 exactly, so Δ_ω = Δ_0, ρ̂ = 1, and there is no blend (toy case T4).

**Not a plasticity guarantee** [stated]. The floor guarantees a fraction of step norm plus first-order descent. It does not guarantee useful learning or Π_c ≥ 0.9; those are measured.

### B5.4 Usage weights [design; release from P149, P150, P163; power law from P189]

Cells are (class) in M2 and (class, context) from M3. For a pool item x in cell κ(x):

$$u_x=\begin{cases}u_{act} & \kappa(x)\in\mathcal A_t\\[2pt]\Big(1+\frac{t-t^{last}_{\kappa}}{\tau_u}\Big)^{-1/2} & \text{otherwise.}\end{cases}$$

- **Active set** 𝒜_t: the cells present in the stream within the last T_act = 500 steps.
- **u_act ∈ {0, 0.25, 1}**, a dose with default 0.25 [corrected after review]. Without it, the item being learned right now had u = 1 and was the most braked, so the brake resisted the learning signal.
  - u_act = 0 leaves active cells to the new-data gradient.
  - u_act = 1 is v1.2's draft behaviour.
- **t_last_κ** is the last step cell κ appeared in the stream, updated during warm-up too.

**Honest limitation.** In domain-incremental M2 streams, every class is always active. Protection then reduces to u_act·(uniform), and there is no release. Per-context cells fix this from M3 on.

**Release horizon** [proved]. u reaches u* after τ_u(u*⁻² − 1) steps: 3τ_u for u* = ½, and 99τ_u for u* = 0.1.

**τ_u** = clip(2·median gap, [10³, 10⁴]); before any recurrence, 5·10³.

**Brake vs anchor.** Release lowers the absolute brake (B5.2 normalisation). Replay of released cells continues. Alarms restore u (B7.3).

**Context usage** (used by B6.3 and B10.1): u_c = max over c's cells.

### B5.5 Cost and memory [estimates; the harness logs measured values]

**ViT-B, plastic third** (24 of 72 maps, plus the head, N_P ≈ 2.8·10⁷), with n_H = 64 and J_ξ = 1.

| Item | M2b | M2a |
|---|---|---|
| Memory | G is 64 × 2.8·10⁷ fp16 values, about 3.6 GB | A_u for 24 maps, about 24 × 3072² floats = 0.9 GB fp32 (or 0.2 GB for the 768-input maps only) |
| Per refresh | 64 backward passes (one per item) every 50 steps, about 1.3 extra item forward-backward passes per step | — |
| Per step | 2N_P·64 ≈ 3.6·10⁹ for the products, plus N_P·64² ≈ 1.1·10¹¹ for GᵀDG | 10 PCG iterations × 24 × d_out d_in² ≈ 10 × 24 × 2.2·10⁹ ≈ 5·10¹¹ |

**M2b per-step compute.** About 1.1·10¹¹ multiply-adds. For comparison, a batch-32 ViT-B forward-backward is about 32 × 3 × 1.75·10¹⁰ ≈ 1.7·10¹² multiply-adds, so about 7%.

**Reducing GᵀDG.** Refresh D (and hence S) every r_D steps (default r_D = 1). If r_D > 1, the **whole** step, including Δ_0 and the blend, uses that same stale D. The step is then the exact solve for an SPD K with a slightly old diagonal, so descent still holds with respect to m̂.

**v1.2's token figures were wrong** [corrected after review]. The draft's 0.3 GB and 9·10¹¹ figures assumed one (z, b) per item. For token models that is wrong, as the reviewer showed: about 56 GB and about 100× a batch. The gradient-sketch form above avoids storing per-token quantities.

**Bytes count in 𝔅** (B6.3).

---

## B6. Memory manager (M2)

### B6.1 Replay sampling [src: P199, P200, P202, P181, P146; mixing rule design]

$$q_{\kappa}\propto n_{\kappa}^{\alpha_f}\,(1+\beta_a\,\mathbb 1[\mathrm{alarm}(\kappa)]),$$

where κ is a cell: a class in M2, a (class, context) pair from M3.

- α_f is a dose (default 0.5, gate E6). β_a = 1.
- Sampling is uniform **within** a cell. There is no error-based selection [src: P178, P162].

### B6.2 Exposure cap [design, inspired by P133]

**The rule.** An absolute lifetime cap E_max = 8.
- When an item with e_i ≥ E_max is drawn and a **newer item of the same cell arrives in the stream**, that item replaces it on arrival.
- Otherwise the capped item stays at full weight. Old classes in class-incremental streams have no newer items, so they are never under-replayed.

**Rate.** An item in cell κ is drawn about n_rep·q_κ/n_κ times per step, so the cap binds after about E_max·n_κ/(n_rep q_κ) steps [corrected: v1.2's draft assumed uniform sampling].

**Source** [src: P133]: there, exposure is the per-block inverse effective sample size, with an absolute cap B_exp = 4.

### B6.3 Byte budget [corrected after review]

**What 𝔅 covers.** 𝔅 is the memory **beyond** what every method has: the model θ and standard Adam state (m, v). It covers:
- buffer;
- probes;
- reference logits;
- the protection pool G or A_u;
- φ;
- the θ_0 decay anchor;
- snapshot bases and differences, plus the snapshot ring (B8.3);
- statistics, heads, detectors and filter state;
- adapters.

**Allocation** [design]. Fixed-size components are allocated first, at their stated minimums:
- φ;
- θ_0 for the decayed layers;
- the pool;
- the ring.

The **remainder** is shared as: buffer 0.7, snapshots 0.2, probes and statistics 0.1.

If 𝔅 is below the fixed minimum, the configuration is **infeasible** and is reported as such, not silently shrunk.

Example, ViT-B in fp16: φ 0.17 GB, θ_0 0.17 GB, pool 3.6 GB, ring of 2 at 0.34 GB, about 4.3 GB fixed. M2a instead of M2b brings this to about 1.6 GB.

**Eviction**, by lowest value per byte:
- buffer: the cell with the most items relative to its share;
- snapshots: lowest u_c, then oldest;
- probes: never below the quotas of B7.1 for cells with u > 0.1.

**Fairness and the frontier.** Baselines spend the same 𝔅, mostly on buffer. At equal 𝔅, ER may store very large buffers. The whole CIFAR-100 training set is about 150 MB, against our fixed costs of several GB. **That is the real trade-off**: our method spends bytes on model state, while ER spends them on data.

So results are reported:
1. on the 𝔅 frontier;
2. **at equal buffer item count** as a secondary comparison, with the extra bytes stated.

Neither alone is a fair summary.

**Limit** [B0.3 L-b]. For unrelated contexts, loss beyond the budget is unavoidable.

---

## B7. Monitor and retention certificate (M2)

### B7.1 Probes [design; corrected after review]

**Quota per class**, q_cls = 30, split at intake and then fixed into:
- 𝒬^sel_j: 10 items;
- 𝒬^acc_j: 10 items;
- 𝒬^mon_j: 10 items.

**Intake.** While class j's quota is unmet, each arriving item of j enters 𝒬_j with probability p_q = 0.25, so the quota fills after about 120 items of j.
- In M2 the single context counts as committed.
- From M3, intake is only for committed contexts. A class item arriving under a provisional context waits until commit.
- Context-level probe sets are unions: 𝒬^·_c = ∪_{j ∈ c} 𝒬^·_{(j,c)}.

**Cost.** About 6% of a CIFAR-100 class's 500 training items are held out. Stated.

**Reference logits.** For every probe and buffer item, store f_ref(x), the logits at the end of its cell's latest episode. In M2, that is the last step its class appeared before leaving 𝒜_t. Then store two margins:
- the **true-label margin** m^y_ref(x) = f_ref,y(x) − max_{j≠y} f_ref,j(x);
- the **agreement margin** m^a_ref(x) = f_ref,ŷ(x) − max_{j≠ŷ} f_ref,j(x), where ŷ = argmax f_ref(x).

Cost: K floats per item, counted in 𝔅.

### B7.2 Certificates [proved]

Let δ(x) = ‖f_t(x) − f_ref(x)‖_∞ be the actual logit change, not linearised. Every logit moves by at most δ, so any pairwise logit gap shrinks by at most 2δ.

**Retention certificate.** For S^y = {x : m^y_ref(x) > 0}, the items correct at reference:

$$m^y_{ref}(x)>2\delta(x)\ \Longrightarrow\ x\ \text{is still classified correctly.}$$

$$CR(S)=\frac{|\{x\in S^y:\ m^y_{ref}(x)>2\delta(x)\}|}{|S^y|}.$$

- CR is a **lower bound on the fraction of S^y still correct**, exact on the sample.
- Items wrong at reference are excluded by definition. So a context with low reference accuracy is not permanently in alarm [corrected].

**Agreement certificate.** For any x, m^a_ref(x) > 2δ(x) implies argmax f_t(x) = ŷ. This is used to compare two models, for compression and merges, where the true label is irrelevant.

**Population bound** [proved; Markov's inequality]. For each item, let ref denote its own reference time. Then for every ε > 0,

$$\Pr(\text{wrong now})\le\Pr(\text{wrong at ref})+\Pr(0<m^y_{ref}\le2\epsilon)+\frac{\mathbb E\lVert f_t-f_{ref}\rVert_2^2}{\epsilon^2}.$$

Each probability is over the item population, each with its own reference time. So "Pr(wrong at ref)" is a mixture over reference times, not the risk of a single model [stated after review]. Estimated on small samples, it is an estimate.

**Why it matters.** The certificates connect output change on stored inputs to retention through the margin. They also measure **cumulative** drift against fixed references, which per-step protection cannot bound.

### B7.3 Alarms (control heuristics, no statistical claim)

**Unit and check.** Alarms are per class in M2 and per context from M3. Every R_mon = 500 steps, for each unit, compute on the unit's 𝒬^mon and buffer items in S^y:
- n_fail^mon = |{x : m^y_ref(x) ≤ 2δ(x)}|, the uncertified items;
- CR_buf.

**Fire** when n_fail^mon ≥ max(3, ⌈0.1·|S^y ∩ 𝒬^mon|⌉) **and** CR_buf < 0.9. Using counts means a single item cannot trigger it [corrected].

**Refractory period:** 5·R_mon.

**Recovery** means CR_buf ≥ 0.95 at a later check.

**After 3 alarms without recovery:** refresh that unit's references, and log the drop as accepted loss.

**Effect of an alarm:**
1. Restore u for the unit's cells (t_last ← t).
2. Boost replay by β_a.
3. From M4, attempt reactivation.

If CR_buf keeps falling across two alarms, halve ω̃ for that unit's items (ω_unit). This is applied after the u ≡ 1 normalisation (B5.2), so it is not undone by renormalising.

No false-alarm rate is claimed. The monitor is reused adaptively (B2.3).

**Why restoring u is safe.** Replay targets are true labels, and m̂ contains the replay gradient. So protection slows changes to the *current* outputs on replay items, including repairs. That is why the ω_unit back-off exists. Gate E7 tests restore-on-alarm against boost-only.

---

## B8. Context inference and routing (M3)

### B8.1 Slow view [src: P156, P203]

h̄(x) is the penultimate feature of the slow model φ, all layers. It costs one extra forward pass per batch, counted.

### B8.2 Sticky Bayesian context filter [design; uses A2, A3; corrected after review]

**State.** s_t ∈ {1, …, C} ∪ {new}, with a sticky transition:
- stay with probability 1 − h, where h = 1/T̄ and T̄ is the running mean episode length (initially 50);
- otherwise move to an existing context with probability proportional to recurrence counts, or to *new* with probability π_new.

**Input evidence.** S is a shared whitened covariance, so the constants cancel.

$$\log p_{in}(X\mid c)=-\tfrac{\kappa}{2}\textstyle\sum_{x\in X}\lVert W_S(\bar h(x)-m_c)\rVert^2.$$

- κ = n_eff/|X|, with n_eff = 4. This is a temperature for within-batch correlation.
- For *new*: the same form with mean m̄, the global mean, and covariance s_new²·S, s_new = 2, adding the normalising term −κ·|X|·d·log s_new.
- A provisional context's m_c is its first batch's mean, updated by a running mean until the next refresh.

**Label evidence (per-context predictive models)** [corrected after review]. Each committed context c has its own linear cosine head H_c on the slow features h̄, over **all** labels. It is trained at refresh on c's buffer items (head-only, cheap), and incrementally on c's batches while c is MAP.

$$\log p_{lab}(Y\mid X,c)=\kappa\textstyle\sum\log\max\{P_{H_c}(y\mid x),10^{-3}\}.$$

- For *new*, P is the global head's marginal label distribution.
- Two contexts with the same class set but different P(y|x), such as permuted labels, now get different predictive models. v1.2's draft renormalised the shared head, which gave a likelihood ratio of exactly 0.

**Filtering** (the forward algorithm):

p_t(s) ∝ p_in(X_t | s) · Σ_{s'} T(s' → s) · p_{t−1}(s' | labels up to t−1).

After Y_t arrives, multiply by p_lab(Y_t | X_t, s).

**Detection delay** [corrected after review: it is not a one-batch guarantee]. Under A3(b) with identical P(x), input evidence is uninformative. Each batch then adds expected log-odds κ·n·I_y, where I_y is the per-item expected log-likelihood ratio between the two contexts' label models. The posterior crosses the 0.9 switch threshold after about

$$k\approx\frac{\log(9)+\log((1-h)/h)}{\kappa\,n\,I_y}\ \text{batches}.$$

This is [approx]: expectations only, and it assumes the label models are accurate. With h = 0.02, κn = 4: about 6/I_y batches. Permuted labels with accurate heads have large I_y, so detection takes about 1 batch. A small-mass difference has I_y → 0, so the delay is unbounded (L-c).

**What is proved:** the first batch after a switch cannot beat the shared predictor at that input (L-a), whatever the filter does.

### B8.3 Decisions and the snapshot ring

| Condition | Action |
|---|---|
| MAP s changes and posterior > 0.9 | **switch** |
| posterior mass on *new* > 0.5 | create a provisional context |
| neither | stay |

**On a switch:**
1. Record the episode end.
2. Store the snapshot (M4) from the **ring**: L_ring = 2 copies of θ, taken at the start of each of the last 2 batches. Use the copy from just before the first batch where the old context's filtered posterior fell below 0.5. If that batch is older than the ring, use the oldest copy and flag the snapshot as contaminated [corrected after review].
3. Reset Adam's m on the plastic layers **and** its bias-correction counter, so m̂ equals the first new gradient and does not shrink the step by a factor of 1/(1−β_1).
4. Set c_now.

Statistics m_c, W_S and the heads H_c are recomputed at each refresh from buffer items, using the slow view.

### B8.4 Provisional contexts [src: P165 motivation; design]

A provisional context's items enter ℬ, but **not 𝒬**.

**Commit.** At each refresh, commit the context if it has n ≥ 50 items and its filter mass has stayed > 0.5 for ≥ T_min batches.

**Merge.** Merge it into the nearest context if the log Bayes factor from input and label evidence is < 2.

**Probe intake** starts at commit.

### B8.5 Routing and correctness [corrected per critique, point 9]

**In M3 there is no routing for prediction.** The global head predicts. The filter only controls episode bookkeeping, usage per cell, and the Adam reset. So filter errors do not change predictions directly; they change only the training dynamics.

**Routed prediction** (adapters or context heads) is introduced in M5, and is **confidence-gated**:
- route to ĉ only if p(ĉ | ·) ≥ τ_route (0.9);
- otherwise use the shared global head.

**Correctness statement** [proved, decomposition]:

$$\mathrm{acc}=\Pr(\text{routed})\cdot\mathrm{acc}_{routed}+\Pr(\text{shared})\cdot\mathrm{acc}_{shared}.$$

Routing errors affect acc_routed, and with disjoint label sets they can drive it to 0 for misrouted items. Accuracy therefore **does** depend on recognition once routing is on. D3 (B2.2) measures how much.

---

## B9. Snapshots and reactivation (M4)

### B9.1 What a snapshot must be [corrected per critique, points 7–8, and after review]

**What an EMA can and cannot do.** An EMA retains whatever later states still contain. It does **not** guarantee recovery of a historical state that later states have overwritten.

**Store.**
- **Immutable base checkpoints** B_g, full copies of θ.
- **Episode-end snapshots** Ψ_c = B_{g(c)} + L_cR_cᵀ: a per-layer low-rank difference from the base current at storage, rank r_Ψ = 16. It is applied to every layer whose relative Frobenius change exceeds 1%.

Reconstruction uses B_{g(c)}, which never changes, so later drift cannot affect it.

**Compression check** (agreement, not truth) [corrected after review]. On c's 𝒬^sel_c, all items:
- accept the compression if the **measured** top-1 disagreement between the full θ and the reconstruction is ≤ max(1, 0.02·|𝒬^sel_c|) items;
- the fraction certified by the agreement certificate (B7.2) is logged, not used as the threshold, because small-margin items are uncertifiable even when they agree.

Failure leads to rank 2r_Ψ. Failure again leads to a new base B_{g+1} = θ, with Ψ_c = B_{g+1}.

**Base growth is logged.** If more than 1 base is created per 5 snapshots, compression is failing and this is reported.

**Eviction.** A base is evicted only when no live snapshot references it. Snapshots are evicted by B6.3.

### B9.2 Reactivation [design; guards from P183, P168, P128]

**Trigger.** c is MAP after a switch, and either an alarm fires on c or CR_buf(c) < 0.9.

**Candidates** (full models):
- θ' = θ + β(Ψ_c − θ), β ∈ {0.5, 1};
- the plastic-only version with β = 1.

**Select** on 𝒬^sel_c.

**Accept** on 𝒬^acc_c (disjoint from selection) if both hold:
1. the exact one-sided McNemar p < 0.05;
2. on the pooled 𝒬^sel of all other committed contexts, the net loss of correct items ≤ max(2, 0.01·n_pool), **and** no single other context loses more than 2 items [corrected after review: a per-context 0.02 bound was below one item].

**Savings control:** apply Ψ_{c'} for c' ≠ c [src: P128].

### B9.3 Eligibility and power [arithmetic proved; power computed exactly; corrected after review]

**Significance.** With b improvements and 0 regressions, the exact one-sided McNemar p = 2^{−b}, so significance needs b ≥ 5. With c regressions, the threshold is the exact binomial on b + c discordant pairs.

**Power.** Expected improvements nΔ ≥ 5 is **not** a power condition. With only improvements, at rate Δ per item, power = Pr(Bin(n, Δ) ≥ 5). Exact values:

| n | power at Δ = 0.125 | power at Δ = 0.10 |
|---|---|---|
| 40 | 0.57 | 0.37 |
| 53 | 0.81 | 0.62 |
| 60 | 0.88 | 0.73 |
| 80 | 0.98 | 0.91 |

**Rule.** A context is **eligible** if its |𝒬^acc_c| gives power ≥ 0.8 at the target Δ_target = 0.125. With no regressions this means n ≥ 53; with regressions, power is computed per test.

**Default.** With 10 acc items per class, a context needs ≥ 6 classes, or more probes. Contexts below this are not reactivated, and they are reported. Single-class contexts need a raised per-class quota (q_cls = 180) to be eligible. That is a cost, stated, not hidden.

---

## B10. Adapters, two-level readout, controllers, consolidation (M5)

### B10.1 Adapters [src: P195, P176; corrected]

$$W^{(c)}=W+U_c\mathrm{diag}(g_c)V_c^\top,$$

with U_c and V_c orthonormal and kept so by retraction [src: P195], g_c ∈ ℝ^k and k = 4.

**Creation (conflict split).** Every R_conf = 20 steps:
- pick one stored context c ≠ c_now, with probability ∝ u_c;
- compute the cosine of g_new and g_rep,c per plastic layer on 32 items;
- keep an EMA χ.

If χ < −0.1 for ≥ 10 updates and the cap N_ad^max is not reached, create an adapter for c_now. At the cap, block. [src: P153 motivation, P145-inspired]

**Merge** [corrected: function, not cosine]. Merge two adapters only if their *functional* difference is small on buffer items of both contexts:

$$\frac{\mathbb E_x\lVert(U_1G_1V_1^\top-U_2G_2V_2^\top)z_x\rVert^2}{\mathbb E_x\lVert U_1G_1V_1^\top z_x\rVert^2}<0.05.$$

The merged adapter is the rank-k SVD of the average, re-checked on both contexts' 𝒬^sel: measured top-1 disagreement with the unmerged model ≤ max(1, 0.02·|𝒬^sel|) items (agreement, as in B9.1).

**Replay items** use **their own** context's adapter [corrected].

### B10.2 Two-level readout (gated)

Detectors d_c(x) = σ(a_cᵀh̄ + b_c) are trained with the buffer as negatives [src: P196] and re-centred by offsets o_c on 𝒬^sel [src: P120]. Routing is confidence-gated (B8.5).

**Kept only if** it beats the global head (B4.1) on confirmation streams, with oracle-routing D3 reported.

### B10.3 Learning-rate controller [src: P148; raise rule guess]

**Signal:** σ_t = acc(φ) − acc(θ) on incoming batches, before training.

| Condition | Action |
|---|---|
| σ > 3%, no rise for 3 windows, and CR plateaued | η_p ← η_p/2 |
| σ < 0.5% for 3 windows, no alarms | η_p ← min(1.25η_p, η^cliff/3) |

The window is R_lr = 1,000 steps.

### B10.4 Consolidation [src: P174, P185; rollback design]

**Triggers:**
- every T_cons = 5,000 steps;
- or an alarm outside the refractory period;
- or 2,000 items of **cells first seen since the last consolidation** [corrected after review: counting all new items fired every 125 steps].

**Steps:**
1. Snapshot the full state: θ, φ, optimiser, heads, detectors, statistics, adapters, pool.
2. Refresh everything that reads features.
3. Optional offline replay: n_off = 500 steps on ℬ only, with protection.
4. Merge adapters.
5. Validate CR on the pooled 𝒬^sel of all committed contexts. If the number of certified items drops by more than max(2, 0.01·n), restore steps 3–4 from the snapshot and keep step 2. If it still drops, restore everything.
6. Log the newest context separately [src: P122, P185].

**Cost.** Consolidation cost is reported as **measured frequency × measured cost**, since alarms and intake trigger it irregularly [corrected].

---

## B11. Interactions and conditions

**I1. One magnitude brake.** The brake is ω𝓗 inside K. There is no other penalty, mask, or plastic-layer dropout [src: P157]. The decay toward θ_0 (B4.2) is a separate, weak pull toward the pre-trained weights; E2 tests it at λ_wd = 0.

**I2. Descent.** The protected step is a descent direction for m̂: exact for the Woodbury solve, and for every PCG iterate started at 0 [proved, B5.2]. The floor blend preserves this [proved, B5.3]. Warm starts are forbidden. Transfer is **not** guaranteed: aligned components can lose alignment after braking. Gate E4 measures forward transfer.

**I3. Timescales** [corrected].

$$R_{pre}=50\ \ll\ 1/\alpha_\phi=1000\ \le\ \tau_u\in[10^3,10^4].$$

- **Release horizon:** 3τ_u ≤ 3·10⁴ steps to half-weight.
- **Snapshots:** they are durable (B9), so trace lifetime no longer has to exceed the gaps. That condition is gone.
- **Head refresh:** drift-triggered (B4.5), not part of this ordering.

**I4. Independence of filter and monitor.**
- The filter reads the slow view and labels.
- The monitor reads actual logits on 𝒬^mon against fixed references.
- They share θ^st and φ, and the monitor's outputs drive control. So they are **not statistically independent** [corrected], only separated in data.

**I5. Compute** (per incoming item, in forward-backward units):

$$\mathcal C=\frac{1}{1-r}+c_{solve}+\frac{n_HJ_\xi}{R_{pre}\,n_{new}}+c_{\bar h}+\frac{32}{R_{conf}n_{new}}+\frac{\text{measured consolidation cost}}{\text{measured interval}\cdot n_{new}}+\dots$$

Every level is compared with **compute-matched ER**: plain replay given the same 𝒞 [new].

**I6. Head staleness.** The per-input bound is |Δlogit| ≤ 2ς‖Δh‖/‖h‖. The drift trigger controls class-mean drift only, so per-input staleness is monitored by CR, not bounded.

**I7. Release, anchor and repair.** Release never stops replay. On an alarm, both replay and the brake are restored. If certified retention keeps falling, ω_unit is halved (B7.3). This closed loop is gated (E7).

**I8. Self-reference audit.**

| Estimator | Anchor |
|---|---|
| 𝓗 pool | real items at the current θ |
| heads and statistics | real buffer features |
| filter | real inputs and real labels |
| monitor | fixed reference logits on real held-out items |
| reactivation | disjoint real probes |
| learning-rate controller | real incoming labels |

**Self-referential parts:**
- context ids, anchored by label evidence and merge tests;
- φ, used only through anchored checks;
- reference logits, which are the model's own past outputs. They are used to detect *change*, not correctness, and correctness is checked separately on labels.

**I9. Budget.** 𝔅 is enforced by assertion (V4 T6). Adapter, snapshot and probe growth all draw from it.

---

## B12. Edge-case map

The v1.1 cases 1–42 are kept, with their guards updated to v1.2 mechanisms. These rows are new or changed:

| # | Case | Guard (v1.2) | Residual |
|---|---|---|---|
| 2 | Brake makes the new task unlearnable | current-step blend floor, which always exists (B5.3) | the floor ≠ useful learning; Π measured |
| 6 | Stale statistics | pool recomputed every R_pre at current θ | ≤ R_pre steps old [approx] |
| 10 | Forgotten looks novel | filter uses labels as well as inputs | first batch after a switch |
| 11 | Overlapping contexts | label evidence; global head unaffected in M3 | L-a, L-c limits |
| 14 | Negative savings | McNemar on a disjoint acceptance set; eligibility by 80% power at 12.5 points | contexts with < 53 acc items are not reactivated |
| 15 | No trace survives the gap | immutable bases plus snapshots | compression error, measured |
| 43 | K-FAC over-protection | exact layer-output metric (M2a); per-item projected-logit sketch (M2b) | M2b under-protects: rank ≤ n_HJ_ξ |
| 44 | Adam composition uphill | joint SPD solve; exact or PCG from 0 (B5.2) | — |
| 45 | Floor has no root | blend form always has a root | — |
| 46 | Snapshot base drift | immutable bases | — |
| 47 | Pure label shift | per-context label heads; delay ≈ 6/I_y batches (B8.2) | first batch at the shared level (L-a); small I_y means long delay |
| 48 | Routing errors affect accuracy | no routed prediction until M5; confidence gate; D3 reported | misrouted items in M5 |
| 49 | Adaptive reuse of the monitor | data tiers; claims only on confirmation and final streams | development choices may overfit development streams |
| 50 | Total memory growth | byte budget 𝔅 with eviction | unavoidable loss for unrelated contexts (L-b) |
| 51 | Cumulative drift despite small steps | certified retention against fixed references | sample-based |
| 52 | Cross-layer interactions | M2b within Θ_P; stable layers too as gated E4c | memory for G grows with N_P |
| 53 | Same-direction adapters with different magnitudes | functional merge test | — |
| 54 | Warm-started CG uphill | PCG from 0 only | — |
| 55 | Token-shared weights | M2b differentiates through tokens | — |
| 56 | Trace normalisation cancels release | normalise with u ≡ 1 | — |
| 57 | Brake on the class being learned | u_act dose for active cells | domain-IL in M2 has no release |
| 58 | Classes with no probes (class-IL) | per-class quota, fast intake | 6% of each class held out |
| 59 | Low reference accuracy causes permanent alarm | CR only over S^y (correct at reference) | — |
| 60 | Snapshot taken after the switch | ring of 2 states; change-point choice | flagged if older than ring |
| 61 | m reset shrinks the next step | reset the bias-correction counter too | — |
| 62 | m̂ = 0 | zero-step guard | — |

---

## B13. Hyperparameters

v1.1 values are kept except for these:

| Change | Value |
|---|---|
| ς | 10 |
| α_φ | 1e-3 (one slow trace, slow view only) |
| λ_wd | 0.01 toward θ_0, swept {0, 0.01, 0.1} |
| n_H, J_ξ | 64, 1 (M2b); A_u from 256 items (M2a) |
| R_pre | 50 |
| k_CG | 10, from 0 (M2a only) |
| r_D | 1 |
| ω̃ | {0.1, 0.3, 1, 3, 10}; normalised with u ≡ 1 |
| ρ_min | 0.2 [guess] |
| τ_u | [10³, 10⁴], τ_u^0 = 5·10³ |
| u_act, T_act | {0, 0.25, 1} default 0.25; 500 steps |
| E_max | 8 (absolute) |
| q_cls, p_q | 30 (10/10/10), 0.25 |
| alarm counts | n_fail ≥ max(3, 10%), CR_buf < 0.9; recovery 0.95 |
| r_Ψ | 16 |
| compression/merge | disagreement ≤ max(1, 2%) items |
| eligibility | power ≥ 0.8 at Δ = 0.125 |
| h | 1/T̄ |
| n_eff, s_new | 4, 2 |
| L_ring | 2 |
| τ_route | 0.9 |
| n_warm | 1,000 items (≈ 63 steps) |
| 𝔅 | fixed components first, then buffer 0.7 / snapshots 0.2 / probes+stats 0.1 |

That is still about 40 settings. They are set on development streams only, and confirmed once.

---

## B14. Main loop

```
init θ←θ0, φ←θ0; ℬ,𝒮,𝒬←∅; filter p0 ← single context; t_warm ← ⌈n_warm/n_new⌉
rng_main, rng_M2 ← separate generators                                      # V6
for t = 1, 2, ...:
    X ← next batch; push θ to ring (L_ring = 2)                              # B8.3
    ŷ ← global head (M1–M4) | gated routed readout (M5)                      # B4.1, B8.5
    filter: multiply by input evidence; MAP decision → maybe switch           # B8.2–B8.3
      on switch: snapshot Ψ from ring (M4); reset m and its counter; record gap
    Y ← labels revealed; filter: multiply by label evidence                   # B8.2
    log test-then-train metrics
    Xq ← route probe items (per-class quota); X' ← X \ Xq                    # B7.1, V5 asserts
    t_last[cells in X'] ← t; update 𝒜_t                                      # B5.4
    B ← sample_replay(q_κ, n_rep); e_i += 1; swap capped items on arrival     # B6
    grads on X' and B (replay items use their own adapter, M5); Adam moments
    Δ0 ← −η D⊙m̂;  if Δ0 = 0: Δ ← 0
    elif t < t_warm or M1: Δ ← Δ0
    else: Δω ← Woodbury (M2b) | PCG from 0 (M2a) on (D⁻¹ + ω𝓗)Δ = −η m̂     # B5.2
          Δ ← Δω; if ‖Δω‖² < ρmin‖Δ0‖²: blend with λ                         # B5.3
    θ ← θ + Δ; decay toward θ0; φ ← φ + αφ(θ − φ)                             # B4.2
    if t = t_warm: refresh heads/stats; build pool, ω; write reference logits # B4.3
    every R_pre (t > t_warm): resample pool (rng_M2), G or A_u, u, ω           # B5.1–B5.2
    every R_check: drift test → maybe refresh heads/statistics/filter stats   # B4.5
    every R_mon: certificates on 𝒬^mon and buffer; alarms (counts, refractory)# B7.3
    on alarm (M4+, c = c_now): reactivation with sel/acc split                # B9.2
    every R_conf (M5): conflict cosine → maybe adapter                         # B10.1
    every R_lr (M5): LR controller                                             # B10.3
    consolidation triggers (M5): full snapshot, refresh, validate, rollback    # B10.4
    enforce byte budget 𝔅 (assert)                                             # B6.3
```

---

## B15. Levels and gates [corrected per critique, point 16]

**Each level depends only on lower levels** [corrected: v1.1's M2 needed contexts from M4].

| Level | Contents | Depends on |
|---|---|---|
| M0 | harness + verification suite V1–V7 | — |
| M1 | global cosine head, per-layer rates, carried v, refreshed head, balanced replay, budget | M0 |
| M2 | protection (M2a, then M2b), floor, class-level usage with u_act, per-class probes, monitor with certificates, exposure cap (single context treated as committed) | M1 |
| M3 | context filter (no routed prediction), per-cell usage, Adam m reset | M2 |
| M4 | snapshots + reactivation | M3 |
| M5 | adapters, gated two-level readout, controllers, consolidation | M4 |

**Gate rules.**
- **Superiority:** a paired difference over ≥ 5 seeds, 95% confidence interval > 0, on ≥ 2 stream types.
- **Non-inferiority on the others:** the lower 95% bound must be above **−1.0 point** (fixed in advance).
- **Comparisons:** against compute-matched ER **and** the previous level, on the 𝔅 frontier and at equal buffer item count (B6.3).
- **Settings:** on development streams. One confirmation run per decision.

**Factorial interaction tests**, small 2×2 designs [new]:
- protection × replay ratio (E4);
- release × floor (E7);
- snapshots × filter quality: oracle vs learned filter (E8).

**What a negative result means** [corrected]. A negative result rejects the *mechanism in that setting*. It does not falsify H1–H3 in general. Example: if E9 shows no benefit from anchoring, then anchoring isn't useful *here*.

**First experiments to run:** V4's toy cases, before any benchmark. Passing large benchmarks would not fix a defect that a toy case exposes.

---

## B16. Audit

**Proved here, under the stated assumptions:**
- the limits L-a and L-b;
- exactness of M2a's metric;
- unbiasedness of M2b's sketch, including token and cross-layer terms within Θ_P;
- descent of the exact solve and of every PCG iterate from 0;
- existence of the floor-blend root, with descent preserved;
- the release horizon;
- the per-example retention certificate and the Markov population bound;
- the McNemar significance threshold (power is computed, not proved);
- the routing accuracy decomposition;
- the agreement certificate;
- the L-a limit on the first batch after a switch. (The detection delay is an approximation, B8.2.)

**Approximations (stated, not bounded):**
- linearisation in 𝓗;
- layer-local metric in M2a; M2b's restriction to Θ_P and its sampling under-protection;
- pool staleness of up to R_pre steps;
- sample representativeness for population claims;
- the likelihood temperature in the filter.

**Design and guess:** everything labelled as such. The largest bets are:
1. usage-weighted release with τ_u from measured gaps;
2. restore-on-alarm with the ω_unit back-off;
3. the conflict split;
4. reactivation from compressed snapshots.

**Risks:**
- about 40 settings;
- compute: solve, pool, slow view and per-context heads, roughly 1.5–2× ER per item (estimate), to be measured;
- memory: M2b's G is several GB for ViT-B, so at equal 𝔅 ER can store far more data;
- interactions only analysed on paper;
- the small-scale evidence base;
- at scale, pre-training plus a slow learning rate plus random replay may already capture most of the benefit (P124, P205).

---

## Changelog v1.1 → v1.2

**Changed:**
- Problem definition and assumptions added (B0).
- Laws renamed to hypotheses.
- Data tiers added.
- Γ renamed and decomposed.
- Verification suite added (B3).
- K-FAC replaced by an exact layer-output metric (M2a) and a per-item projected-logit gradient sketch (M2b).
- Per-sample linear usage weights.
- Uniform pool sampling.
- Joint SPD solve (Woodbury for M2b, PCG from 0 for M2a), replacing the Adam composition.
- V-new removed.
- Current-step blend floor.
- Removed: eigen-energy, the bisection floor, and the K-trace EMA bank (one φ kept).
- Immutable-base snapshots with measured compression error.
- Sticky Bayesian context filter with label evidence.
- No routed prediction before M5.
- Confidence-gated routing with an accuracy decomposition.
- Probes per class, with sel/acc/mon split at intake; McNemar eligibility from computed power.
- Certified retention from stored reference logits.
- Alarms on certificate counts, labelled as heuristics; refractory period; ω_unit back-off.
- Absolute exposure cap, full weight retained when no fresh item exists.
- Total byte budget with eviction and a frontier.
- Functional adapter merge; replay items use their own adapter.
- Startup policy.
- Consolidation trigger on new-item count; measured consolidation cost.
- Levels made self-contained.
- Gates: savings criterion, non-inferiority margin, factorial tests, compute-matched ER, development and confirmation streams.
- Tensor order fixed.
- Decay claim narrowed to separation.

## Review round on v1.2 (before release to Lucky)

An independent reviewer read the first draft of v1.2 and found 21 defects: 0 critical, 12 major, 9 minor. I checked the main ones numerically before fixing them. All 21 are fixed above.

**Major:**
1. Warm-started, truncated CG can go uphill (74 of 200). Now: Woodbury (exact) for M2b, PCG from 0 for M2a. Every iterate is a descent direction [proved; 0 of 2,000 failures].
2. Unpreconditioned CG at ω = 0 did not reproduce Adam (65% error). Now ω = 0 returns Δ_0 exactly in both solvers.
3. The ξ-sketch null space was misdescribed. It is a superset, i.e. under-protection; now stated, with J_ξ as the knob.
4. Token-shared weights broke the per-token form, and the cost figures were wrong by about 100×. Now M2b differentiates through the network; costs recomputed.
5. Trace normalisation with u cancelled release. Now normalised with u ≡ 1.
6. The brake fell mostly on the class being learned. Now the u_act dose applies to active cells.
7. Probe quotas could not support per-cell alarms or reactivation, and M2 depended on M3. Now per-class quotas, fast intake, count-based alarms per class or context, and the single context counts as committed in M2.
8. CR's set was undefined, and the 0.98 threshold could not be met. Now S^y is explicit, an agreement certificate is added, and compression and merge use measured disagreement counts.
9. Label evidence was zero under a shared class set. Now per-context heads.
10. The one-batch detection claim was unproved. Now an approximate delay formula.
11. The eligibility rule was not a power condition. Now exact power (n ≥ 53 at 12.5 points).
12. Byte shares were infeasible for model-sized state. Now fixed components are allocated first, 𝔅 counts only memory beyond θ and Adam, and results are reported on a frontier plus at equal buffer item count.

**Minor:**
- the consolidation trigger fired every 125 steps;
- warm-up was measured in items in one place and steps in another, and its ordering was wrong;
- ρ̂ edge cases (0/0, and ρ̂ > 1);
- the Adam bias-correction counter after an m reset;
- the snapshot was taken after the switch (now a ring of 2 states);
- the anchor decay was inert and λ_wd was unstated (now decay toward θ_0, with λ_wd given);
- stale references (V-all, ω_0, u_c, scratchpad, p_in(new), recovery, uniform sampling);
- the V6 bit-for-bit conditions (now separate random generators);
- the other-context guard was below one item;
- certificate wording.

