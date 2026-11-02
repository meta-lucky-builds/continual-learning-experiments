# 10 T-DORMANT: should a memory forget a stored skill only when new evidence contradicts it, never just because time passed?

**Date run:** 2026-10-23 (brief at 14:18 UTC after Lucky chose "Yes, start"; result reported about 20:50 UTC). **Folder:** /mnt/project-files/research/tests-v7/dormant/ **Built by:** a separate builder agent (a Sonnet model, per memory note tests-v6-status.md) who did not design the hypothesis; the hypothesis was the coordinator's. **Seeds:** main 1001-1030 for the linear arms (30 seeds); 1001-1012 for `cxl` (12 seeds, and only n ∈ {2, 8}); tuning 9001-9003 (cxl 9001-9002); pilot 9101-9103. **Verdict:** KILLED by the pre-registered bar (12 wins, 11 losses, 1 tie). Clock-driven forgetting, when switched on, was very harmful, which supports the "never forget by the clock" half. But plain no-forgetting RLS beat the evidence-driven champion (directional-forgetting RLS) by about 4 nats at 20 frames and about 50 nats at 200 frames, and the memory-unmatched existing stack `cxl` was best overall. The world had no memory pressure, so "never forget" could not lose there.

---

## 1. Why we ran it

**Project problems.** P2 (protection of old skills: keep earlier skills close to their best after long sequences of other experience) and P5 (relearning/savings: a skill that returns after a gap should come back fast) in notes/problems.md. A "dormant" skill is one that is not used for a long time and then returns. The worry is that a memory loses it during the gap even though nothing contradicted it.

**The idea under test** (research/tests-v7/dormant/BRIEF.md; coordinator's hypothesis, the builder "owes it nothing"): *a memory should lose a stored skill only when new evidence contradicts it, never just because time passed.*

- **Clock-driven stores** forget with elapsed steps whatever the data. Examples: decay gates in modern linear-attention memories, and recursive least squares (RLS) with an exponential forgetting factor.
- **Evidence-driven stores** lose precision only along directions that new data actually excite. Examples: RLS with directional forgetting (SIFt-RLS), precision relaxation toward a prior (Palimpsa), and plain RLS with no forgetting at all (the corner that "never forgets, never tracks").

**Where the idea came from.** Lucky uploaded a literature map of memory research (research/literature-v6/MEMORY-FIELD-MAP-2026-10-23.md). Its main finding, as summarised in the project's memory note, is that clever memory methods tend to tie trivial ones at matched bits and tuning. Its open problem **OP1, "Clock-driven vs evidence-driven forgetting under idle steps"**, predicts that if a key is written n times and then left idle (no writes), the idle time until its read gain halves (Δ½):

- rises with ln n for "precision-accumulating" layers (RLS, MesaNet, with forgetting factor γ): Δ½ ≈ [ln n + ln(h₁/λ)] / ln(1/γ), i.e. a slope of 1/ln(1/γ) per unit of ln n;
- is flat in n for gated layers (Gated DeltaNet, Mamba-2, GLA): Δ½ ≈ ln 2 / ln(1/α).

The map's text notes "No filler-token test exists in any of these layers" (MEMORY-FIELD-MAP §7, OP1). The coordinator linked OP1 to P2/P5 (marked as an inference in the memory note) and proposed "evidence-driven, not clock-driven, forgetting" as a research direction, with odds "about 50/50 that it holds beyond algebra; 15-25% solid paper" (memory tests-v6-status.md, 2026-10-23 14:15). Lucky chose "Yes, start" at 14:18.

**What decision it was meant to settle.** Whether "forget only on contradicting evidence" is a rule that actually beats existing memory rules at equal memory, and so deserves to be a design principle for protecting dormant skills, or whether it is just algebra that existing methods already satisfy.

---

## 2. What we did

### 2.1 Jargon

- **nats**: natural-log units of log-likelihood. **Excess nats over the oracle** = the sum, over a block of frames, of (oracle log-likelihood minus arm log-likelihood). 0 means as good as knowing the true mean; ignoring the skill entirely costs about 24 nats per frame here.
- **E20, E200**: excess nats summed over the first 20 and the first 200 frames after the skill returns.
- **RLS (recursive least squares)**: exact online least squares; it keeps an information (precision) matrix R and updates the weights by R⁻¹. An **exponential forgetting factor** λ < 1 multiplies R by λ every step, so old information fades with time whatever the data (clock-driven).
- **SIFt-RLS / directional forgetting** (Kulhavý and Zarrop 1993; arXiv 2404.10844, Alg. 1): RLS that forgets only along the direction of the current input, so information about directions the data do not excite is kept (evidence-driven). Arm name `drls`.
- **Gated delta rule / Gated DeltaNet** (arXiv 2412.06464, Eq. 10): a linear-attention memory matrix S updated by a delta-rule write (correct the prediction error along the current key) and multiplied by a scalar decay gate a each step. With a < 1 it forgets by the clock. Arm `gdn`.
- **KDA (Kimi Delta Attention)** (arXiv 2510.26692, Eq. 1): the same with a per-channel decay vector instead of a scalar. In the source paper the gate is learned and data-dependent; here a fixed vector was used, because there is nothing to pre-train (a disclosed deviation). Arm `kda`.
- **MesaNet-style decayed ridge** (`mesa_ef`): decay both sufficient statistics G and H by γ but keep a fixed, undecayed regulariser. Added by the builder because OP1's prediction is about exactly this variant.
- **Palimpsa** (arXiv 2602.09075, Eqs. 30-31): diagonal precision that relaxes toward a prior; evidence-weighted but still with a clock term (1 - a). Classified evidence-driven because the brief named it so.
- **cxl**: the existing-methods, label-free stack built for T-CAUSE (research/tests-v6/cause/arm_cxl.py), which keeps a library of training episodes, clusters them into contexts and stores small low-rank weight edits and snapshots [inferred from T-CAUSE IMPLEMENTATION.md, third amendment]. It was run unchanged. It is **not memory-matched**.
- **LoRA + replay**: a low-rank adapter trained with Adam on the current frame plus frames from a uniform reservoir buffer.

### 2.2 World

Source: research/tests-v7/dormant/PROTOCOL.md §1-§3, IMPLEMENTATION.md §1. [checked: read]

- A standalone world built on T-CAUSE's frozen-base machinery (Student-t ν = 3 likelihood, constant scale 0.8, 64 output tokens, 40-dimensional input). The builder explained why a T-CAUSE world could not be reused: there, every frame excites every direction of the base's hidden layer, so a gap that "does not excite A's directions" cannot be built; and T-CAUSE contexts return rather than persist or change.
- The base is almost linear (h = tanh(0.5 E v) with E orthonormal), so the problem is close to linear-Gaussian. 12 of the 40 input coordinates are active (axis-aligned, chosen per seed).
- **Skill A** is a rank-2 correction of the base's output along two key coordinates: c1 "shared" and c2 "unshared". Its Frobenius norm is 8.
- **Phases:** training = n visits x 125 frames (n ∈ {1, 2, 4, 8}); a gap of Δ frames (Δ ∈ {30, 90, 300, 900, 3000, 9000, 30000}; 30000 = 30 x the longest training length); then A returns for 200 frames. Every Δ is a prefix of one gap stream, and the return block is identical across Δ (paired).
- **Four gap types:**
  1. *idle*: gap inputs have no energy on A's four directions; A stays true but is never visible;
  2. *overlap*: gap inputs excite A's shared direction and the complement, not the unshared one; truth stays A, so the gap re-confirms half of A;
  3. *contradict* (the fairness control): at the first gap frame A changes to an independent A' of the same norm; any store that refuses to update must lose;
  4. *null*: there is never a skill; tests false writes.
- **Oracle validity** (PROTOCOL.md §3; "oracle validity" means checking with the true model that the world measures what it claims): V1 base excess at 20 frames ≥ 100 nats in idle/overlap/contradict, 0 in null (observed 539.6 / 539.6 / 538.6 / 0.0); V2 perfect retention of the original A costs 0 in idle/overlap/null and ≥ 300 nats in contradict (observed 0 / 0 / 913.3 / 0); V3 oracle excess 0; V4 the unshared coordinate is silent in idle/overlap gaps (0.001), the shared one silent in idle (0.001) and active in overlap (0.795); V5 plain RLS can learn the truth (52 nats at 200 frames vs base about 5400). Information check: gap frames carry 0.000 nats/frame about A in idle, 13.05 in overlap, and the old A loses 16.2 nats/frame in contradict. **All passed.**

### 2.3 Arms

Source: PROTOCOL.md §4; IMPLEMENTATION.md §2-§3. Equal data, equal knowledge, a budget of 4200 stored floats.

| group | arm | stored floats |
|---|---|---|
| clock | `gdn` (Gated DeltaNet) | 2560 |
| clock | `kda` (Kimi Delta Attention, fixed gate) | 2600 |
| clock | `rls_ef` (RLS, exponential forgetting) | 4160 |
| clock | `mesa_ef` (decayed ridge, fixed regulariser) | 4160 |
| evidence | `drls` (SIFt-RLS directional forgetting) | 4160 |
| evidence | `rls_plain` (RLS, no forgetting) | 4160 |
| evidence | `palimpsa` (diagonal precision relaxing to a prior) | 2600 |
| bar | `res_replay` (uniform reservoir + normalised LMS) | 4120 |
| bar | `fifo` (recency buffer, same learner) | 4120 |
| bar | `lora_replay` (rank 1-2 LoRA + Adam + reservoir) | ≤ 4200 |
| bar | `fresh` (plain RLS started at the return) | 4160 |
| bar | `base` (frozen, no correction) | 0 |
| bar | `cxl` (T-CAUSE stack, unchanged, not matched) | see §3.1 |

Arms below the budget were **not padded** (disclosed: this favours the large-state RLS and replay arms over gdn, kda and palimpsa).

### 2.4 Tuning

Each tuned arm had exactly 8 configurations (cxl its own 8-row T-CAUSE table) on held-out tuning seeds. Objective J = mean of (E20 + E200) over tuning jobs and all Δ, **averaged over all four gap types**: one configuration per arm for every world, because "the learner does not know which regime it is in" (PROTOCOL.md §5). The per-gap-type best configuration was stored for a secondary "regime-aware" run. Evidence champion = lowest-J arm among {drls, rls_plain, palimpsa}. The drls forgetting grid was shifted to {1, .999, .99, .9} because, with 12 excited directions, its per-direction forgetting is about (1 - λ)/12, so this spans the same effective half-lives as the RLS-EF grid [inferred by the builder]. cxl was tuned on a reduced grid (n = 2, 2 seeds, Δ ≤ 3000) for cost.

### 2.5 Pre-registered bar

Source: PROTOCOL.md §7 (sharpened from the brief, with the changes marked there).

The idea **HOLDS** only if all of:

- (a) in the idle and overlap worlds, at both 20 and 200 frames, over the decisive gaps Δ ∈ {3000, 9000, 30000}, the evidence champion beats **each** of gdn, kda, rls_ef, mesa_ef (every clock arm, not just the best), cxl and res_replay: one-sided paired t-test and Wilcoxon signed-rank test both p < 0.01, mean difference > 0, and the difference at least 5% of the comparator's mean (a practical floor; a tie counts against);
- (b) contradict control: the champion's mean E is within 1.25 x the best arm's + 2 nats;
- (c) null control: the same inequality in the null world (added by the builder).

**KILLED** if any comparator beats the champion in any (a) cell at p < 0.01 with the floor; **CEILING** otherwise. A secondary "mechanism verdict" uses the same bar without cxl; a tertiary reading compares drls only with rls_ef and mesa_ef (same RLS machinery, only the forgetting rule differs).

Power: compute was not binding, so N = 30 (linear) and 12 (cxl) were fixed in advance; the analysis prints the minimum detectable difference (MDE) at power 0.8 per contrast (PROTOCOL.md §6).

Predictions with probabilities were written before the pilot and tuning (PROTOCOL.md §8): P(HOLDS) 0.10, KILLED 0.65 ("mostly through cxl"), CEILING 0.25; P(tuning picks γ = 1 for at least one clock arm) 0.50. After tuning, before main data (§11): P(HOLDS) 0.05, KILLED 0.45, CEILING 0.50.

### 2.6 Amendments and the builder's peek disclosure

Source: PROTOCOL.md §10. [checked: read]

- **A1** (after the 3-seed pilot had been looked at): control (c) as written was unsatisfiable, because `base` is exact in the null world (E = 0) and every learner pays an estimation cost; replaced by "best arm excluding base". The pilot also showed LMS-type arms are 10-20x less efficient than RLS-type ones.
- **A2:** controls (b) and (c) use all seven Δ and all n, because adaptation to a contradiction is informative only at Δ ≤ 900.
- **A3:** the full tuning table is printed.
- **A4** (after linear tuning, before main): because every selected clock arm turned out not to forget, fixed-forgetting **diagnostic arms** (γ = .999 and .9999) were added as stage `main_diag`, outside the bar; and a regime-aware secondary run `main_ra` was added.
- **A5:** cxl run with 200-frame gap episodes (T-CAUSE's own visit length) and training chunked into ≤ 500-frame episodes; earlier timing jobs used other lengths and are not used.
- **A6:** the selected ridge for rls_plain and fresh (30) sits at the edge of their 8-value grid (J still falling); not re-gridded because the 8-configuration budget was fixed. rls_plain and rls_ef with γ = 1 are the same algorithm up to the ridge.
- **A7 (the peek and rerun disclosure).** During the main run, after 4 seeds had been seen, the builder added a second half-life measure computed from the stored state itself (`errA`: the error ‖M̂ - M_true‖_F on A's two key coordinates), because OP1 defines the half-life on read gain and E20 is a nonlinear function of it, and because the first version (error on all 40 coordinates) "turned out polluted by noise-fitting in never-excited coordinates (rls_ef gamma=1 err grew 3.5 -> 12 at Delta=30000 with no effect on E20)". The four partially-run seeds of the main, main_diag and main_ra stages were **deleted and re-run** with the new fields. The builder wrote: "no decision was taken on them (I did look at the 4-seed means of E20 before the restart: drls was 20% worse than rls_ef in idle, better in contradict; no protocol element was changed because of them)." So the builder saw the direction of the main result on 4 seeds before the full run. Nothing in the bar changed afterwards, and the added measure is supplementary, but the reader should know the peek happened.

Other deviations listed in RESULTS.md: cxl on n ∈ {2, 8} and 12 seeds; mesa_ef added; every clock arm in the bar; for about 2 minutes a fourth worker process ran (limit 3). Compute: 17.4 core-hours, under the 30-hour budget.

---

## 3. What we got

### 3.1 Key table

Source: research/tests-v7/dormant/analysis_main.md and RESULTS.md (same numbers). Mean excess nats, n ∈ {1, 2, 4, 8} and Δ ∈ {3000, 9000, 30000} (cxl: n ∈ {2, 8}, 12 seeds; others 30 seeds). Lower is better.

| arm (tuned config) | idle E20 | idle E200 | overlap E20 | overlap E200 | contradict E20 | null E20 | stored floats |
|---|---|---|---|---|---|---|---|
| base | 480.0 | 4730 | 480.0 | 4730 | 479.1 | 0.0 | 0 |
| fresh | 397.0 | 1651 | 397.0 | 1651 | 397.2 | 45.5 | 4160 |
| gdn (gate a = 1) | 113.9 | 806 | 74.5 | 591 | 35.1 | 48.7 | 2560 |
| kda (gate g = 1) | 113.9 | 806 | 74.5 | 591 | 35.1 | 48.7 | 2600 |
| rls_ef (λ = 1) | 19.0 | 146 | 10.4 | 82.5 | 12.8 | 19.0 | 4160 |
| mesa_ef (γ = 1) | 19.0 | 146 | 10.4 | 82.5 | 12.8 | 19.0 | 4160 |
| **drls** (λ = .99; champion) | 23.2 | 197 | 14.3 | 131 | 6.1 | 23.2 | 4160 |
| rls_plain (ρ = 30) | 23.1 | 169 | 12.8 | 94.3 | 13.2 | 14.6 | 4160 |
| palimpsa (a = 1) | 24.8 | 193 | 15.6 | 126 | 15.7 | 18.8 | 2600 |
| res_replay | 417.4 | 3955 | 317.1 | 3014 | 184.0 | 330.4 | 4120 |
| fifo | 120.5 | 872 | 79.7 | 644 | 38.4 | 54.1 | 4120 |
| lora_replay | 2197.5 | 8059 | 1362.2 | 4783 | 83.7 | 2050.3 | 4160 |
| cxl (not matched) | 9.8 | 98.0 | 11.6 | 102.5 | 11.7 | 1.6 | 3,158,632 |

On cxl's memory: IMPLEMENTATION.md §2 and PROTOCOL.md §8.1 give "about 1.4e5 (one measured job: 135952)", while the main analysis reports 3,158,632. The files do not explain the difference. A likely reason is that cxl stores the gap frames it is fed as episodes, and 30,000 frames x 104 floats ≈ 3.1 million [guess]. Either way it holds tens to hundreds of times the 4200-float budget.

### 3.2 Tuning outcome

Source: analysis_main.md (tuning J table); PROTOCOL.md §11; configs_selected.json.

**Every clock arm chose no forgetting**: gdn a = 1, kda g = 1, rls_ef λ = 1, mesa_ef γ = 1 (palimpsa also a = 1). With one configuration for all four worlds, forgetting by the clock cost more in idle, overlap and null at long gaps than it gained in the contradict world. Tuning J: drls 718 < gdn/kda 794 < rls_plain 810 < rls_ef = mesa_ef 825 < fifo 857 < palimpsa 860 < lora_replay 2829 < res_replay 2891 < base 4015; cxl 388. The forgetting settings were expensive in J: rls_ef at its most forgetful settings 5551 and 5448; gdn's worst 2637.

drls became the evidence champion because it had the lowest J of all linear arms; what separated it from no-forgetting RLS was faster adaptation in the contradict world.

The pre-registered prediction "P(tuning picks γ = 1 for at least one clock arm) = 0.50" came true more strongly than predicted: all four did. As a result the tuned "clock" arms cannot forget by time at all, and gdn = kda and rls_ef = mesa_ef numerically.

### 3.3 The bar

Source: analysis_main.md, "Decision bar". d = comparator minus champion (positive = champion better), idle and overlap, decisive gaps.

| world | horizon | vs gdn / kda | vs rls_ef / mesa_ef | vs cxl | vs res_replay |
|---|---|---|---|---|---|
| idle | 20 | WIN (+90.7) | LOSS (-4.2) | LOSS (-8.4) | WIN (+394.2) |
| idle | 200 | WIN (+608.9) | LOSS (-51.3) | LOSS (-72.4) | WIN (+3757.4) |
| overlap | 20 | WIN (+60.2) | LOSS (-3.9) | tie/unresolved (-0.2) | WIN (+302.7) |
| overlap | 200 | WIN (+460.5) | LOSS (-48.2) | LOSS (-16.0) | WIN (+2883.7) |

Losses against rls_ef/mesa_ef had one-sided p = 1 for "champion better" and MDEs of 1.0 (E20) and 4.4-4.5 (E200), so they are resolved. Controls (all n, all seven Δ):

| control | champion E | best arm | best E | threshold | pass |
|---|---|---|---|---|---|
| contradict E20 | 218.5 | cxl | 151.4 | 191.2 | NO (also without cxl) |
| contradict E200 | 1634.8 | cxl | 750.8 | 940.5 | NO (also without cxl) |
| null E20 | 31.9 | cxl | 1.6 | 4.0 | NO (also without cxl) |
| null E200 | 259.6 | cxl | 2.0 | 4.5 | NO (also without cxl) |

**Primary verdict: KILLED** (12 wins, 11 losses, 1 tie; a_pass = False, any_loss = True, controls_ok = False). **Mechanism verdict without cxl: KILLED.** Tertiary (drls vs rls_ef and mesa_ef, same RLS machinery): drls loses in all 8 cells.

### 3.4 Retention over the gap

Source: analysis_main.md, "E20 by Delta".

Idle world, E20 by gap length:

| arm | Δ=30 | 300 | 3000 | 30000 |
|---|---|---|---|---|
| rls_ef (λ=1) | 48 | 34 | 21 | 18 |
| drls | 49 | 35 | 23 | 24 |
| rls_plain | 46 | 37 | 25 | 22 |
| gdn (a=1) | 106 | 117 | 113 | 115 |
| res_replay | 225 | 227 | 346 | 478 |
| lora_replay | 90 | 141 | 564 | 5060 |
| cxl | 10 | 10 | 10 | 10 |

No tuned arm lost the skill by the clock: their curves are flat or still improving out to Δ = 30000. Half-lives were "inf" (not reached) for all γ = 1 arms. Only the replay arms forgot, through buffer overwriting or weight drift: lora_replay idle E20 90 → 5060.

Contradict world, E20 by gap length (how fast each arm re-learns after the change):

| arm | Δ=30 | 90 | 300 | 900 | 3000 |
|---|---|---|---|---|---|
| drls | 660 | 517 | 265 | 70 | 7 |
| rls_ef (λ=1) | 675 | 549 | 327 | 138 | 32 |
| gdn (a=1) | 536 | 379 | 114 | 37 | 34 |
| cxl | 491 | 491 | 23 | 19 | 13 |

### 3.5 Diagnostic arms with the clock switched on (outside the bar)

Source: analysis_diag.md; RESULTS.md.

| arm | idle E20 at Δ = 900 | 9000 | 30000 |
|---|---|---|---|
| rls_ef, λ = .999 | 28 | 809 | 3283 |
| mesa_ef, γ = .999 | 28 | 388 | 399 |
| gdn, a = .999 | 309 | 476 | 476 |
| kda, g = .999 | 298 | 467 | 475 |
| palimpsa, a = .999 | 35 | 489 | 498 |

Clock-on forgetting is ruinous here: compare the tuned no-forgetting arms at 18-24 nats.

**OP1 slope check** (half-life measured on E20, idle, slope of Δ½ against ln n): gdn@.999 129 (OP1 prediction 999), kda@.999 170 (999), rls_ef@.999 361 (999), mesa_ef@.999 152 (999), palimpsa@.999 319 (999); gdn@.9999 2088 and kda@.9999 2017 (prediction 9999). All slopes positive and about 0.13-0.36 of the prediction. The error-based half-life (A7, errA) gives slopes from 4 (gdn@.999) to 293 (gdn@.9999). Note that the *gated* arms also showed half-lives that rise with n, whereas OP1 predicts a flat slope for gates. The builder's conclusion: "the simple 'half-life grows by 1/ln(1/gamma) per ln n' prediction is not borne out in this world".

### 3.6 Regime-aware secondary run

Source: analysis_main_ra.md. When each arm uses its best configuration *for that gap type* (no cxl in this run), the verdict is **CEILING** (12 wins, 0 losses, 8 ties; controls still fail in the null world). The ties arise because, in the idle and overlap worlds, drls's regime-specific choice becomes λ = 1, which makes it the same algorithm as rls_ef (identical E20 19.0 / 10.4 in that run). Under the pre-registered one-configuration rule, KILLED stands.

---

## 4. What we inferred

**What the result says.**
1. **"Do not forget just because time passed" was confirmed in this world, but almost trivially.** Tuning, given one configuration for all regimes, switched the clock off for every clock arm; the clock-on diagnostics were very costly (up to 3283 nats at 20 frames). That supports the negative half of the hypothesis.
2. **"Evidence-driven forgetting is the better rule" was not confirmed.** The evidence champion (drls) lost to no-forgetting RLS in every idle and overlap cell (by about 4 nats at E20 and about 50 nats at E200). Its only advantage was faster re-learning after a real contradiction (contradict E20 6.1 vs 12.8 over the decisive gaps; 70 vs 138 at Δ = 900), and that advantage was not enough to pass the contradict control, which cxl dominated.
3. **The existing stack cxl was best overall**, but with roughly 750 times the memory budget (3.16 million floats vs 4200) and a richer structure, so that comparison does not isolate the forgetting rule.
4. The regime-aware run shows the verdict depends on whether the arm may pick its forgetting setting per regime; with that freedom drls simply turns its forgetting off where it hurts, and the comparisons become ties.

**What it does NOT say.**
- It does not test forgetting under **memory pressure**. 12 active keys x 64 outputs fit easily into a 4200-float RLS state, so nothing is gained by forgetting anything except when the truth changes. In such a world "never forget" cannot lose in idle and overlap gaps.
- It says nothing about trained deep sequence models: KDA's gate was fixed, not learned; OP1 itself was posed for trained layers with clamped writes on language-model tasks.
- It does not show evidence-driven forgetting is useless in general; it shows that in this near-linear, uncrowded world it offered no protection advantage over not forgetting at all.

---

## 5. Why we think it happened

- **Why tuning switched every clock off.** Three of the four worlds (idle, overlap, null) keep the truth fixed through the gap, and only one (contradict) changes it. With J averaged over all four, keeping everything is the better single setting [checked: J table; inferred reason, also stated by the builder in PROTOCOL.md §11].
- **Why drls loses to no-forgetting RLS in idle and overlap.** The builder's statement: "drls's directional forgetting still shrinks the information in excited directions slightly" [checked: RESULTS.md]. In more detail [inferred]: during an idle gap the inputs excite the *complement* coordinates, so drls keeps forgetting there and holds less precise estimates of those (truly zero) coefficients than no-forgetting RLS does; and during the 200 return frames, whose inputs are isotropic, drls with λ = .99 forgets along A's directions too, which is why the gap grows from about 4 nats at E20 to about 50 nats at E200.
- **In the idle world the linear arms' return error is estimation noise, not forgetting.** For rls_ef and drls, idle E20 equals null E20 exactly (19.0 = 19.0 and 23.2 = 23.2; E200 145.9 vs 145.8 for rls_ef) [checked: analysis_main.md]. In the null world there is no skill to forget. A linear least-squares learner's error does not depend on the true coefficients, so equal numbers mean the idle-world excess of these arms is entirely the cost of estimating the model from noisy data, and none of it is lost skill [inferred]. The idle E20 *falls* with gap length (48 → 18 for rls_ef) because the gap inputs excite the complement coordinates and let the learner pin those coefficients at zero more precisely [inferred].
- **Why the LMS-type arms (gdn, kda, fifo, res_replay) are far worse even with no decay.** They take normalised gradient steps rather than exact least-squares solves; the pilot showed them 10-20x less statistically efficient (PROTOCOL.md §10 A1) [checked]. That is an efficiency confound, which is why the protocol added the tertiary drls-vs-RLS comparison.
- **Why cxl is best.** The builder's advance expectation: cxl carries a low-rank, site-structured hypothesis class and a library that never decays, while the full-matrix linear arms must estimate 12 x 64 numbers from noisy data (PROTOCOL.md §8.1) [inferred by the builder]. Its contradict curve (491 until Δ = 90, then 23 at Δ = 300) suggests it switches to a new context once enough contradicting evidence has accumulated [guess].
- **Why the OP1 slopes were lower than predicted.** Not tested. The builder notes the E20-midpoint definition and censoring could be responsible, and that A7 offered one alternative definition (RESULTS.md, "What could still be wrong") [inferred]. OP1's formula is for a single key written n times with idle steps that write nothing; here gap inputs do write along other coordinates, and the read-out is a 20-frame excess, not a read gain, so the setting differs [inferred].

---

## 6. Caveats and what could be wrong

- **No memory pressure.** This is the biggest caveat and the builder names it: "the world has no capacity pressure on the 4200-float state ... so decay has no benefit. A tighter memory budget or a world where unrelated skills compete for state would be a fairer test of forgetting; I did not run one."
- **The verdict rests on a mild effect plus cxl's structural advantage.** drls loses 4 nats of E20 to rls_ef. The losses are statistically resolved (MDE 1.0), but they are small in absolute terms against a base excess of 480.
- **Sensitivity to the tuning rule.** One configuration across regimes gives KILLED; per-regime configurations give CEILING.
- **The world favours some arms.** Axis-aligned key coordinates favour diagonal arms (kda, palimpsa) and make the problem near-linear-Gaussian, which favours RLS-family arms. Arms under budget were not padded.
- **Implementation choices.** KDA's gate fixed rather than learned; Palimpsa's diagonal approximation; SIFt-RLS's epsilon fixed at 1e-6 and untuned; rls_plain and fresh at the edge of their ridge grid; T-CAUSE's own LoRA, fresh and base replaced by equal-bytes re-implementations.
- **cxl comparison is unbalanced**: 12 seeds, n ∈ {2, 8} only, reduced tuning grid, far more memory. Its stored-float count is reported inconsistently across the folder's files (§3.1).
- **The peek (A7).** The builder saw 4-seed means before restarting the main run. No protocol element changed, and the 4-seed direction matched the final result; I see no sign the peek affected the verdict, but it is a departure from a clean blind run.
- **The test answers the coordinator's framing of OP1, not OP1 itself.** OP1 asks about trained layers on a recall task with clamped writes; this test uses hand-set linear memories on a regression world.

---

## 7. What it changed in the project afterwards

- **Recorded as KILLED** in notes/all-tests-ledger-2026-10-28.md §10, with the caveat "the world had no memory pressure".
- **Kept as a principle, not a method.** The ledger lists "don't forget on a clock" among "the ideas that held up as principles", while noting that none became a better method.
- **A capacity-pressure re-test was offered to Lucky** (memory tests-v6-status.md: "Offered Lucky a capacity-pressure re-test; awaiting reply"). It was never run; it is listed in file 11 ("never tested").
- No architecture change followed from it in the files I read.
