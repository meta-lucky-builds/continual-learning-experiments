# 09 p14v7 "memory of errors": can a robot carry *how fast to learn* (not where to start) from practice to new motor tasks?

**Date run:** 2026-10-19 (designed and piloted the same day, inside T-P14D as amendment 2). **Folder:** design in /mnt/project-files/research/architecture/p14-v7/ (CORE.md, PRINCIPLES-draft.md); test code and results in /mnt/project-files/research/tests-v6/p14d/ (learners/p14v7.py, pilot/PILOT_amend2-4.md, pilot/PILOT_active.md). **Built by:** designed by a fresh P14 architect (v7) who never built the test; implemented from the spec by the independent T-P14D builder. **Seeds:** pilot seeds 101-108 (8 seeds), untuned default configurations; w-body and w-lhome, plus w-active for the full arm. **Verdict:** FAILED at pilot level. First try equal to fresh and never clearly harmful, gain at mid overlap 0.33 (w-body) and 0.43 (w-lhome), but it failed proportionality and "mainly faster learning", and **its own kill test K3 fired**: the simpler twin without the "memory of errors" rule did better (G(o50) 0.47 vs 0.33 in w-body). It was 1.14x (mid) to 1.92x (high overlap) slower than heb, the registered existing-methods bar, in w-body. No main run.

Context for this file: the test itself (world, metrics, other arms, rules) is explained in file 08. This file explains the p14v7 design, its arms, and what happened to it.

---

## 1. Why we ran it

**The problem (P14).** Lucky's P14 is transfer of motor *ability*: after practising tasks that need timing, precision, control under difficulty and handling slips, a robot learns a new task that needs them faster. The speed-up is proportional to the shared abilities, near zero with nothing shared, never negative, larger after varied practice, kept for life, and mainly faster learning, not a better first try (notes/problems.md §14; Lucky's clarification 2026-10-18 quoted in research/tests-v6/p14d/PROTOCOL.md §1).

**What the T-P14D pilot had already shown** (file 08). Every arm that transferred something carried *values*: a starting point, a pooled model, or an inferred latent. They all started better (first try 0.52-0.67 of expert at mid overlap vs 0.03 for fresh) and none showed a learning-speed gain. The previous P14 design (p14v6) carried no values but gained very little (G 0.09).

**The design idea, in plain words** (research/architecture/p14-v7/CORE.md §0, PRINCIPLES-draft.md). The architect's principle P1: when a learner does not choose its data and refits on all of it, only two things can make each episode teach more without changing the first try:

1. **per-direction sensitivity**: which directions of the model to move, and how readily (formally a prior covariance, the "gain" of a Kalman filter);
2. **per-error sensitivity**: how much each error should count (formally the error law, or influence function).

"Motor ability" is defined as these two things learned from practice. The design carries only:

- a **zero-mean metric** V on the model space (a prior covariance with no mean, so no starting point is carried), learned across tasks from the **cross-moment of two independent half-task fits**. This is the "memory of errors". It comes from Herzfeld et al. (2014), who found that human error sensitivity grows when consecutive errors agree in sign and shrinks when they alternate. The architect's statistical reading: the product of two *independent* estimates of the same task's model is an unbiased estimate of the persistent part, because their noise cancels. So "raise sensitivity where independent evidence agrees" becomes "learn the second moment of the corrections tasks really need, net of estimation noise";
- a **learned error law**: a heavy-tailed, context-dependent noise model fitted on held-out residuals, so that slips count less and noisy contexts are down-weighted;
- per test task, a **leave-one-episode-out competition** among sensitivity hypotheses that always includes the fresh learner itself, so that if no hypothesis predicts held-out data better, the output stays close to fresh.

Consequences claimed by construction: the first try is exactly fresh's; a globally sign-reversed task is as likely under the prior as the original; on an unpractised effector the learner *is* fresh, so G = 0 there exactly.

**Why at that moment.** p14v6 had just failed in the pilot, and the T-P14D main run had not happened yet, so a new candidate could still be pre-registered and plugged in before it. The coordinator asked for a fresh architect who had not seen p14v6 (the principles were written before reading p14-v6/CORE.md, and the architect did not read the test's world code, deliberately).

**Lucky's ruling that it counts as ability.** The architect flagged that a zero-mean prior covariance could be read as "a learned prior over where changes lie", which Lucky had excluded from P14 (CORE.md §9, attack A1: "This is definitional and needs Lucky's ruling"). Lucky ruled at 12:50 UTC on 2026-10-19: **the zero-mean learned metric counts as P14 "ability" because it never moves the first try**; the decision rules stay unchanged, and property 6 (mainly faster learning) is the one it should pass (PROTOCOL.md §10c amendment 2; memory note tests-v6-status.md, "Lucky rulings 12:50: p14v7 metric counts as ability"). So p14v7 was judged as a legitimate P14 candidate, not disqualified on definitions.

---

## 2. What we did

### 2.1 The mechanism, more exactly

Source: CORE.md §3. [checked: read]

The task model is a matrix Θ that maps a regressor φ (state, the last 7 commands, a constant) to the next state increment; for effector E1 each of the 4 output rows has d = 19 entries.

- **M1 (practice → metric).** After each practice task (20 episodes), split its episodes by parity into halves A and B. Fit each half with a fixed, history-free robust estimator (Student-t regression with a small ridge), giving a and b. Add the symmetrised cross-product ½(a bᵀ + b aᵀ) to a running sum, per output row. The metric V̂ is the running mean with each eigenvalue shrunk by z standard errors (computed from a reservoir of stored contributions) and clipped at zero: directions whose cross-moment is not clearly above zero across tasks are dropped. A scalar v gives the scale for an isotropic floor.
- **M2 (practice → error law).** Residuals of each half's data under the *other* half's fit go into a reservoir. A Gaussian scale-mixture law (up to K_e components, a shared slip/noise component per transition, a log-linear dependence of the noise scale on |φ|) is refitted by EM.
- **M3 (a new task's estimate).** Hypotheses: F (fresh) and a grid of (α, τ), each giving every row a zero-mean Gaussian prior with covariance α V̂ + τ v I, together with the learned law. Each hypothesis's posterior is fitted from the task's own data (a few variational iterations). Each is scored by leave-one-episode-out prediction under a fixed Student-t judge. Weights ∝ prior weight x exp(η x score), with F's prior weight ω_F = 0.5. Before any episode, the output is fresh's model. After that, it is the weighted mix.
- Memory: about 86,000 floats for E1, under the test's 200,000 budget (CORE.md §6).

### 2.2 The arms (the design's own ablations)

Source: CORE.md §6; PROTOCOL.md §10c.

| Arm | What it changes | Purpose |
|---|---|---|
| `p14v7` | full design | the candidate |
| `p14v7nm` | "naive moment": the metric is built from θ̂θ̂ᵀ of a single full-data fit, no split, no SE shrink (z = 0) | tests the Herzfeld consistency rule; kill test K3 |
| `p14v7g` | the learned law is off (fixed Student-t with per-task scale) | geometry only |
| `p14v7w` | the metric is off (V̂ = 0), law only | law only |
| `p14v7nh` | practice is a no-op | must equal fresh exactly (implementation check K4) |

The simpler twin in Lucky's ledger ("a simpler twin with no memory of errors") is `p14v7nm`: it keeps the zero-mean metric, the law and the competition, but replaces the split-half cross-moment by the naive second moment of each task's own estimate.

### 2.3 Pre-registered kill tests and predictions

Source: CORE.md §0 and §7, registered in PROTOCOL.md §10c before any p14v7 result was seen.

- **K1:** G(o50) ≤ 0.05 in both w-body and w-lhome ("the geometry carries no speed"; the central bet).
- **K2:** any test cell has a lower 95% bound of G below -0.05.
- **K3:** `p14v7nm` ≥ `p14v7` at o50 ("the Herzfeld consistency principle is not needed; the design reduces to zero-mean empirical Bayes").
- **K4:** first try differs from fresh, or G(E2_0) ≠ 0 (implementation error).

Predictions (conjectures): G by cell E2_0/E1_0/o25/o50/o75/o100 = 0.00/0.15/0.20/0.28/0.33/0.38 (±0.15) in w-body; G(o50) in w-lhome 0.22 [0.05, 0.40]; property pass probabilities in w-body P1 0.40, P2 0.97, P3 0.60, P4 0.55, P5 0.65, P6 0.75, P7 0.55; DF 0.70 [0.20, 1.20]; P(shows P14 discovery) 0.15; P(breakthrough) 0.07. Ablations: `p14v7nm` lower than p14v7 by 0.05-0.15 at o50, with P(p14v7 beats p14v7nm at p < 0.05) = 0.60; `p14v7g` about 0.85 x G(p14v7); `p14v7w` about 0.03 [-0.03, 0.10]; `p14v7nh` identical to fresh.

The decision rules of T-P14D (properties P1-P7, DF, the breakthrough rule and the `heb` bar) applied unchanged; see file 08 §2.5. Jargon used below: **G** = 1 - (geometric-mean episodes-to-criterion of the arm / that of fresh), so G = 0.33 means about one third fewer episodes; **EtC** = episodes to reach 0.8 x the expert's closed-loop success; **LSG** = learning-speed gain, which compares the time from first reaching 0.3 x criterion to criterion; **heb** = the hierarchical empirical-Bayes "existing-methods stack" registered as the main bar; **DF** = discovery fraction against the grammar-given reference `gram`.

### 2.4 Implementation choices and deviations

Source: research/tests-v6/p14d/IMPLEMENTATION.md §9. [checked: read]

- The fresh component inside p14v7 used fresh's **default** configuration in pilots (its tuned one only in main runs, which never happened). The builder notes this makes pilot numbers slightly pessimistic for p14v7.
- No `race` hyper-parameter (the competition is its own race); its 15-parameter tuning space replaces the common one, same 16-configuration budget. Tuning never ran, so every pilot number is at the spec's defaults (z = 2, ν_h = 4, K_e = 3, ω_F = 0.5, η = 1, and so on; CORE.md §6 table).
- A numerical failure in model() falls back to fresh's model; zero such fallbacks in the smoke job, "not checked for all pilot jobs".
- Cost: about 99 s per job, the most expensive arm in the test (estimate.md).

---

## 3. What we got

All numbers PILOT: 8 seeds (101-108), untuned defaults. Source for every row: research/tests-v6/p14d/pilot/PILOT_amend2-4.md §3 (candidates) and §4/§5, unless marked. "X" = property fails.

### 3.1 The five arms against the seven properties

| arm | world | G(o50) | P1 ρ (lower) | P2 E2_0 CI | P3 worst lower | P4 V-N diff (p) | P5 S1→S2 | P6 LSG | P7 G 6→24 tasks | DF (CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| p14v7 | w-body | 0.33 | 0.05 (-0.07) X | [0,0] | -0.02 (o75) | 0.03 (0.359) X | 0.32→0.40 | -0.30 X | 0.21→0.33 | 0.92 [0.59,1.30] |
| p14v7 | w-lhome | 0.43 | -0.02 (-0.25) X | [0,0] | -0.03 (o75) | 0.15 (0.008) | 0.31→0.33 | 0.10 X | 0.32→0.43 | 1.08 [0.69,1.52] |
| p14v7nm | w-body | 0.47 | 0.41 (0.28) X | [0,0] | 0.00 (E2_0) | 0.16 (0.023) X | 0.52→0.50 | 0.08 X | 0.36→0.47 | 1.46 [0.87,2.12] |
| p14v7nm | w-lhome | 0.45 | 0.10 (-0.24) X | [0,0] | -0.03 (o75) | 0.18 (0.016) X | 0.36→0.43 | -0.07 X | 0.42→0.45 X | 1.13 [0.79,1.51] |
| p14v7g | w-body | 0.46 | 0.63 (0.54) X | [0,0] | 0.00 (E2_0) | 0.07 (0.102) X | 0.57→0.64 | 0.22 X | 0.39→0.46 X | 1.37 [0.97,1.81] |
| p14v7g | w-lhome | 0.50 | 0.51 (0.24) X | [0,0] | 0.00 (E2_0) | 0.11 (0.008) | 0.53→0.60 | 0.30 X | 0.40→0.50 | 1.27 [0.93,1.71] |
| p14v7w | w-body | 0.03 | -0.15 (-0.30) X | [0,0] | -0.01 (o100) | -0.01 (1.0) X | 0.03→0.02 X | -0.24 X | 0.03→0.03 X | 0.08 [0.00,0.15] |
| p14v7w | w-lhome | 0.07 | -0.23 (-0.36) X | [0,0] | -0.01 (conf) | -0.02 (1.0) X | 0.03→0.03 X | -0.05 X | 0.05→0.07 X | 0.13 [0.03,0.26] |
| p14v7nh | both | 0.00 | 0.00 X | [0,0] | 0.00 | 0.00 (1.0) X | 0.00→0.00 X | 0.00 X | 0.00→0.00 X | 0.00 |

Properties passed by the full `p14v7`: in w-body P2, P3, P5, P7 (4 of 7); in w-lhome P2, P3, P4, P5, P7 (5 of 7). It failed P1 (proportional) in both worlds and P6 (mainly faster learning) in both. "Shows P14 discovery": no, in both worlds, for all five arms.

### 3.2 Speed against the competitors

"Ratio" = paired geometric-mean EtC+ ratio, p14v7-family arm over comparator; above 1 means the p14v7 arm needs more episodes. Mid = cell o50; high = cells o75 and o100. The per-competitor ratio pools o50, o75 and o100 and has a 90% t interval [checked: analysis.py `compare`, called with mid + high cells].

| arm | world | vs best competitor: mid / high | vs heb: mid / high | breakthrough |
|---|---|---|---|---|
| p14v7 | w-body | 1.14 / 1.92 (heb) | 1.14 / 1.92 | False |
| p14v7 | w-lhome | 1.33 / 1.73 (sigma_shrink) | 1.00 / 1.66 | False |
| p14v7nm | w-body | 0.89 / 1.09 (heb) | 0.89 / 1.09 | False |
| p14v7nm | w-lhome | 1.29 / 1.56 (sigma_shrink) | 0.97 / 1.50 | False |
| p14v7g | w-body | 0.93 / 1.02 (heb) | 0.93 / 1.02 | False |
| p14v7g | w-lhome | 1.19 / 1.04 (sigma_shrink) | 0.89 / 1.00 | False |

Pooled mid+high ratios in w-body (90% intervals):

| comparator | p14v7 | p14v7nm | p14v7g |
|---|---|---|---|
| heb | 1.62 [1.39, 1.88] | 1.02 [0.81, 1.28] | 0.99 [0.81, 1.20] |
| pbml | 1.45 [1.22, 1.72] | 0.91 [0.79, 1.06] | 0.89 [0.74, 1.06] |
| mbase | 1.39 [1.17, 1.66] | 0.88 [0.80, 0.97] | 0.85 [0.70, 1.04] |
| sigma_shrink | 1.30 [1.09, 1.56] | 0.82 [0.71, 0.96] | 0.80 [0.66, 0.96] |
| adapter | 1.35 [1.00, 1.81] | 0.85 [0.61, 1.19] | 0.83 [0.58, 1.18] |

### 3.3 The kill tests

| Kill | Result | Source |
|---|---|---|
| K1 G(o50) ≤ 0.05 in both worlds | **Not fired**: 0.33 and 0.43 | PILOT_amend2-4.md §3 |
| K2 any cell lower bound < -0.05 | **Not fired**: worst lower bounds -0.02 (w-body) and -0.03 (w-lhome) | same |
| K3 p14v7nm ≥ p14v7 at o50 | **Fired**: 0.47 vs 0.33 (w-body); 0.45 vs 0.43 (w-lhome) | same; PROTOCOL.md §10f |
| K4 first try ≠ fresh, or G(E2_0) ≠ 0 | **Not fired**: first try 0.03 / 0.01 (o50 / E1_0, w-body) as fresh; G(E2_0) interval [0,0]; p14v7nh equals fresh in every cell | PILOT_amend2-4.md §3-§4; PROTOCOL.md §10f |

Two remarks on K3. First, the registered wording is a comparison of point values at o50; the paired test "p14v7 beats p14v7nm at p < 0.05" (which the architect gave probability 0.60) was not printed as such, but p14v7nm is *better*, not merely equal, so the direction is clear in this pilot. Second, the w-lhome difference (0.45 vs 0.43) is small; the w-body difference (0.47 vs 0.33) is large. The pooled w-body ratios against heb (1.02 for nm vs 1.62 for the full arm) say the same.

The law ablation went the same way: switching the learned error law **off** (`p14v7g`) improved G from 0.33 to 0.46 (w-body) and from 0.43 to 0.50 (w-lhome), and raised the proportionality score ρ from 0.05 to 0.63 (w-body). The architect had predicted p14v7g at about 0.85 x p14v7. So **both of the design's distinctive parts (the split-half "memory of errors" and the learned error law) made it worse at default settings.** The metric itself carries the gain: `p14v7w` (metric off, law only) had G 0.03 / 0.07, as predicted.

### 3.4 Predictions vs pilot (full arm, w-body unless noted)

| Prediction | Pilot |
|---|---|
| G(o50) 0.28 ± 0.15 | 0.33 (hit) |
| w-lhome G(o50) 0.22 [0.05, 0.40] | 0.43 (above the interval) |
| P1 passes with P 0.40 (mean ρ 0.75) | fails, ρ 0.05 |
| P4 passes with P 0.55 | fails in w-body (p = 0.36), passes in w-lhome (p = 0.008) |
| P5 retention 0.90 [0.75, 1.05] | G rose after S2: 0.32 → 0.40 (passes) |
| P6 passes with P 0.75, LSG/G about 0.85 | fails: LSG -0.30 (w-body), 0.10 (w-lhome) |
| P7 G(6 tasks) about 0.55 x G(24) | 0.21 vs 0.33, ratio 0.64 (passes) |
| DF 0.70 [0.20, 1.20] | 0.92 [0.59, 1.30] (within) |
| P(shows P14 discovery) 0.15; P(breakthrough) 0.07 | neither |
| Robustness cost none (P 0.80) | push-cost and corrupt-cost 0.000 ± 0.000 (no cost) |
| p14v7nm lower by 0.05-0.15 | higher by 0.14 (miss; K3) |
| p14v7g about 0.85 x G | 0.46, i.e. higher than the full arm (miss) |
| p14v7w about 0.03 | 0.03 (hit) |
| p14v7nh identical to fresh | identical (hit) |

Other readouts (PILOT_amend2-4.md §5): separate-abilities life, G on combinations 0.31 (w-body) / 0.10 (w-lhome) and on out-of-range values 0.00 / 0.00. p14v7nm's G on combinations was higher (0.62 / 0.52).

### 3.5 In the w-active world (no weight in any decision)

Source: research/tests-v6/p14d/pilot/PILOT_active.md §3, §6. The w-active world failed its validity check (file 08 §3.7), so these rows carry no weight. p14v7 had G(o50) 0.37; vs heb mid ratio 1.02 and high 2.15 (pooled 1.68 [1.29, 2.19]); vs the best competitor there (adapter) mid 1.80, high 2.18. Its episodes rose by a factor 1.23 when it was given active data under the default rule, compared with the fixed data.

---

## 4. What we inferred

**What the pilot says.**
1. **The "carry sensitivity, not values" idea did give a real, safe gain from a fresh start.** With the first try identical to fresh's, p14v7 still cut episodes by about a third at mid overlap (G 0.33 / 0.43), never had a cell clearly below fresh (worst lower bound -0.03), and retained its gain after later practice. That is better than p14v6 (G 0.09) and the L-reduced arm (G about 0).
2. **But the P14 pattern was not there.** The gain was not proportional to shared ability (ρ 0.05 / -0.02), and by the test's measure it was not "mainly faster learning" (P6 failed in both worlds).
3. **The specific mechanism the design was named for was refuted at pilot level.** K3 fired: the Herzfeld-style split-half cross-moment was worse than the naive moment. The learned error law also hurt. What remained useful was the plain zero-mean empirical-Bayes metric, i.e. what the architect himself called "zero-mean empirical Bayes" in the K3 wording.
4. **Against existing methods it lost.** p14v7 was 1.14x (mid) and 1.92x (high) slower than heb in w-body, and 1.33x / 1.73x slower than the best competitor (sigma_shrink) in w-lhome. Its mid-overlap point ratio against heb was 1.00 in w-lhome (and 1.02 in w-active); for these mid-only comparisons the analysis prints no interval, so whether they meet the protocol's formal tie rule is not recorded. The memory note of 2026-10-23 says p14v7 "tied heb at mid overlap in w-active only"; the files show the same point ratio (1.00) in w-lhome too.

**What it does NOT say.**
- It is not tuned. With 15 hyper-parameters at their defaults, a tuned p14v7 could do better, and so could its ablations and competitors. The K3 result in particular could depend on z (the SE multiplier, default 2) and the ridge of the half-fits.
- It does not show that "sensitivity transfer" is wrong in general; its zero-mean simpler twins (p14v7nm, p14v7g) were roughly level with heb on pooled mid+high in w-body (ratios 1.02 and 0.99) and somewhat faster than sigma_shrink and mbase there, with first tries equal to fresh. But those were diagnostic ablations, they also failed P1 and P6, they were not registered as candidates, and in w-lhome they were slower than the best competitor.
- It says nothing about real robots or RoboJEPA.

---

## 5. Why we think it happened

- **The zero-mean metric works because task variation in this world is low-dimensional.** The builder measured that the across-task variation of the true model Θ has 99%-variance rank 8 of 76 entries in w-body and about 12-13 in w-lhome [checked: PROTOCOL.md §10d]. A prior covariance concentrated in that subspace lets a new task resolve its model from few episodes, which is what the architect's §4.1 argument predicts [inferred].
- **Why the split-half cross-moment lost to the naive moment.** Not tested; possible reasons [inferred unless marked]:
  - Each half has only 10 of the task's 20 episodes, so each half-fit is noisier, and with a ridge the cross-moment shrinks poorly excited directions toward zero (CORE.md §4.3b calls this "the conservative side"). Its own sanity check showed the split-half in-subspace trace at 1.93 against a truth of 2.56 [checked: CORE.md §4.3b, sanity only]. A metric that is too small in true directions gives too little plasticity exactly where it is needed.
  - The z = 2 standard-error shrink over at most 24 contributions can remove real directions.
  - The naive moment's known flaw (it inflates poorly excited directions with estimation noise) may matter little when each practice task has 20 episodes and the per-task fits are already fairly accurate; and extra prior variance in poorly excited directions may even help, because that is where fresh is slow [guess].
- **Why the learned error law hurt.** Not tested. The architect noted that the law is fitted on residuals that include the other half's estimation error, so it is "slightly too wide" (CORE.md §3 M2 step 3). The world's noise grows with the command along one body direction, while the law's context model is log-linear in the absolute regressor values, a different family [inferred from IMPLEMENTATION.md §1 and CORE.md O2]. A misfitting law can down-weight informative transitions. Which of these mattered is unknown [guess].
- **Why proportionality failed.** The architect named it as the main risk in advance (CORE.md §4.2 "declared mismatch"): the design transfers by *geometric* overlap, the test grades *declared* overlap. In w-body, body dynamics (A5) are shared by every E1 task, so even tasks with low declared overlap lie largely in the practised geometry, flattening G across cells [inferred].
- **Why P6 failed.** The arch-v8 architect's reading [inferred, research/architecture/arch-v8/NOTES.md line 46]: LSG counts only the duration after an arm first reaches 0.3 x criterion. An arm whose first try equals fresh's but which reaches 0.3 x criterion sooner gets G > 0 with LSG near zero or negative: its gain is in the early episodes, which LSG books neither as first try nor as speed. If so, part of p14v7's P6 failure may come from how P6 was operationalised rather than from the mechanism. This was noticed after the pilot; the rule was not changed.

---

## 6. Caveats and what could be wrong

- **Pilot only, untuned defaults, 8 seeds, 2 worlds.** All K3 and ablation conclusions are pilot conclusions. The protocol says so: "Untuned defaults; the verdict is the main run's, not the pilot's" (PROTOCOL.md §10f). The main run never happened.
- **The internal fresh used fresh's default configuration** in pilots, which the builder says is slightly pessimistic for p14v7.
- **Numerical fallbacks** to fresh in model() were not counted across pilot jobs; if they were frequent, p14v7 would have been partly fresh.
- **The P6 metric may be mis-specified** for fresh-start arms, as above.
- **The "ability" label rests on Lucky's ruling.** If one read P14 strictly as excluding any learned prior over where changes lie, p14v7 (and heb, pbml) would not count at all. Lucky ruled that the zero-mean metric counts.
- **The mechanistic explanations in §5 are inferences or guesses**; none was tested by a targeted run.
- **Multiple ablations, small n.** With five arms and two worlds on 8 seeds, some differences (for example nm vs full in w-lhome, 0.45 vs 0.43) are within noise.

---

## 7. What it changed in the project afterwards

- **Marked FAILED** in notes/all-tests-ledger-2026-10-28.md §9 ("early honest no"): first try equals fresh and never hurts; saves about 1/3 of episodes at mid overlap; fails proportionality and varied > narrow; its own kill test fired; 1.1-1.9x slower than the best existing methods. (The ledger's "fails varied > narrow" is true for w-body; in w-lhome P4 passed.)
- **arch-v8** (2026-10-20) took the lesson into its design notes: "Covariance learners (heb, p14v7) do not produce learning-speed gain or proportionality. p14v7 ρ .05, heb ρ .53, LSG fails for all" and "p14-v7's split-half covariance lost to the naive moment (K3)" (research/architecture/arch-v8/ARCH.md §I2 and M5). It adopted heb's hierarchical empirical-Bayes prior as "memory of how to adapt" rather than p14v7, and proposed a different invented P14 mechanism (M10, form evidence), which was never tested.
- The memory note of 2026-10-23 recorded that the only "wins" so far were pilot-level, including "p14v7 beat weak baselines and tied heb at mid overlap in w-active only" (memory tests-v6-status.md). Note that w-active carries no weight by its own registered rule, because it failed validity.
- No further p14v7 work was done. The T-P14D main run, which would have tuned and tested p14v7 on 48 seeds and all worlds, was never run.
