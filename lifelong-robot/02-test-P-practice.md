# 02 Test P: does practice cover the edit directions of never-practised task families? (first P14 test, condition (L))

**Date run:** 2026-10-17/18 (protocol written 2026-10-17 before any main run) **Folder:** /mnt/project-files/research/toy-tests/P-practice/ **Built by:** one Opus test agent working from research/toy-tests/BRIEF.md (memory note first-principles-architecture-draft-v31.md); not the designer of the core. REPORT.md was written by that agent and saved by the coordinating session because the harness blocked the agent from writing report files (REPORT.md, first line). **Seeds:** 20 (seeds 0–19), all finished **Verdict:** **UNCLEAR** on the core's conjecture C-P14 by the pre-registered rule; **condition (L) clearly FAILED** (met in 0 of 20 seeds). Practice did speed up learning of the recombination family about 2× over an isotropic prior, but the coverage behind it was mostly generic, not specific.

Primary sources: `research/toy-tests/P-practice/PROTOCOL.md`, `REPORT.md`, `results/tables.md`, `results/posthoc.txt`. Numbers below come from these unless another file is named.

---

## 1. Why we ran it

**The project problem.** P14, "transferable, task-agnostic motor memory": the robot should learn *new* motor tasks faster because it has practised *other* motor tasks, and more so after varied practice than narrow practice (`notes/problems.md`, problem 14).

**How the v3.6 core tried to solve it.** CORE.md Principle III ("learning to learn", sections B3.1–B3.5) proposed that practice builds a **learned prior over edits**, called Π_E. Some background:
- The robot adapts by adding **rank-one edits** to a frozen network: at one weight matrix (a "site"), an edit is a·bᵀ, an output direction a times an input direction b.
- Each new edit needs a prior: a guess, before seeing data, of which directions are likely. The default is an **isotropic** prior (all directions equally likely). Π_E is a mixture of that isotropic component and a **factored** component, a ~ N(0, Σ_out), b ~ N(0, Σ_in), where Σ_out and Σ_in are learned from the edits made during practice by **empirical Bayes** (fitting the prior's parameters to past data).
- If new families' edit directions fall inside the leading ranges of Σ_out and Σ_in, the robot needs fewer records to find and shape them, so it learns faster.

**The conjecture C-P14** (CORE.md B3.5). For a never-practised family, each of its measured edit directions k has an **uncovered fraction** u_k (how much of a_k and b_k lies outside the leading ranges of the learned Σ's). The **covered fraction** is ρ = 1 − mean(u_k). The conjecture: ρ rises with the number of practised families m, is higher after varied than after narrow practice, and on the recombination family after varied practice it meets **condition (L)**.

**Condition (L)** is a bound from an owed proof ("P-link") in CORE.md B3.5. It turns coverage into a predicted speed-up: a quantity D_k(n), roughly the "effective number of dimensions" that must be paid for to learn direction k with n records under Π_E, must average at most half the isotropic count, ½·d_iso = 63.5 at this toy's site size. If (L) holds, the core says episodes to criterion fall to at most about half of the isotropic prior's.

CORE.md listed C-P14 as item 1 of "What is least sure" (§10): "Real families may not recombine practised input and output directions; ρ(F_new) may stay small."

**Why at that moment.** Same as Test M: on 2026-10-17 Lucky ordered toy tests of the two riskiest bets before writing the full spec, so that results would be fixed in the core (`research/toy-tests/BRIEF.md`). The decision this test was meant to settle: does the core's P14 mechanism (a learned edit prior) actually cover new families, enough for the theory's guarantee to apply?

**The trap the brief named.** If families are built by hand as recombinations of shared directions, ρ is high by construction. So families had to be defined **physically** and their directions **measured** from fits, never designed (`research/toy-tests/BRIEF.md`).

## 2. What we did

### World (`PROTOCOL.md` §1)
- A simulated 2-link planar arm in a vertical plane with a gripper, 100 Hz control, 5 Hz frames, 2-second episodes (10 frames).
- A camera renders 5 keypoints as blobs in a 16×16 image.
- Pre-life task: reach random targets by random deadlines.
- One controller for every family (inverse kinematics, minimum-jerk plan, computed-torque tracking using the **true** physics of each instance). So each family's motor commands come from physics, not from hand-written patterns.
- A frozen predictor (a small residual MLP, inputs 137-dim, output a 9-dim "motor code") trained on 640,000 pre-life records. Its four 64×64 block matrices are the **sites** where edits can go; d_in = d_out = 64, d_iso = 127 per site.

### Families (defined physically; `PROTOCOL.md` §2)
- **Seven practised families:** payload, curl force field, weak elbow, shoulder friction, tilted gravity, spring, stiff grip.
- **R (recombination):** payload **and** curl field together. A recombination of practised *physics*; whether it is a recombination of practised *directions* is what ρ measures.
- **O (orthogonal):** the camera rotates 20–45°, no change in dynamics or task. Chosen in advance as the family least likely to share directions with practice.
- **Four per-ability probes:** timing (moving target), precision (slot insertion), control under difficulty (time-varying load), recovery from slips (regrasp a displaced object).
- No edit direction is written anywhere in the code.

### Learner and arms (`PROTOCOL.md` §3)
Each seed runs practice "lives":
- **varied_m{1,2,3,5,7}**: 6 contexts from each of m practised families.
- **narrow_m{1,3,7}**: the same number of contexts, all from one family.
- **twin_m7**: a learner whose Π_E is frozen at isotropic on the varied_m7 stream (checks whether the prior's own targets are biased by the prior, B3.2).
- **boot_v7**: a bootstrap refit, to measure sampling error of the estimate.

Then, on new families (3 instances each, 30 episodes = 300 records per instance), these arms learn:

| Arm | What it is |
|---|---|
| iso | isotropic prior, empty dictionary (no practice) |
| PiE_v7 | Π_E after varied practice on all seven, empty dictionary (the prior alone) |
| PiE_v7_dict | Π_E plus the dictionary of committed directions (the full learner after practice) |
| PiE_n7 | Π_E after narrow practice |
| SC_reptile (SC-b) | the core's "strongest competitor": a PCA prior over saved full-matrix adapters from the same practice stream, starting from their mean (a **Reptile**-style initialisation; Reptile is a meta-learning method that learns a starting point by repeatedly moving it toward weights adapted to each task), then **test-time training** (TTT: gradient fitting of an adapter on the new stream) |
| SC_knn (SC-a) | the same, starting from the saved adapter whose context key is nearest |
| SC2_reptile, SC2_knn | added post hoc (see deviations): the same with a tempered likelihood chosen on held-back records |

SC's score per instance is the best of the SC arms, chosen after the fact, which favours SC.

### Measurements (`PROTOCOL.md` §4)
- **Directions of a family:** the isotropic learner run on 2000 oracle records per instance; each committed rank-one edit is a measured direction.
- **ρ**, plus two chance levels: ρ_chance (random directions, given the selected ranks) and ρ_chance,id (random directions that the instance's own data can express; added before main runs, after a sanity check showed only about 9 output directions are identifiable at the last site).
- A rank-free diagnostic: coverage by the top 8 eigenvectors (chance 0.125).
- **D_k(n_c) and (L)**, with n_c the iso arm's records to criterion.
- **Episodes to criterion:** the first checkpoint where held-out negative log-likelihood closes 80% of the gap between the frozen base and an oracle isotropic fit. Not reached by 30 episodes: censored and scored 60.

### Pre-registered decision rules (`PROTOCOL.md` §5)
1. ρ near 0: CI upper bound < 0.10.
2. ρ rises with m: CI of the slope above 0.
3. Varied above narrow at equal m: CI > 0.
4. Narrow beats varied on the narrow family itself.
5. (L) holds if the CI upper bound of E_k[D_k] ≤ 63.5; fails if the lower bound > 63.5; else undecided.
6. Π_E's episodes ≤ ½ of SC's on R.
7. Twin agreement.
8. Orthogonal clause: Π_E no worse than isotropic on O.
9. **C-P14 verdict:** fails if ρ(R) near 0 (or chance-level and (L) fails); holds if rules 2, 3 and 5 all pass on R; otherwise unclear.

All intervals are 95% t intervals over seeds.

### Deviations (`PROTOCOL.md` §10; `REPORT.md` §3)
1. A fix to the rank search, which could cycle (before main runs).
2. The data-expressible chance level added (before main runs; can only make a pass harder).
3. The planted-edit sanity check S-E redefined.
4. Refits every 14 contexts instead of 6, for timing.
5. **Post hoc:** the SC2 arms were added **after 6 seeds had finished and interim rule-6 numbers had been seen (and passed)**. The pre-registered SC arms got *worse* with more data (on R, the fraction of the oracle gap closed fell from 0.44 at 4 episodes to 0.24 at 30), meaning the untempered TTT overfit. Rule 6 was changed to compare against the best of all four SC arms. Both versions are reported; the change can only make a pass harder.
6. A guard against numerical divergence, after seeds 8, 12 and 19 crashed; those seeds were rerun from scratch, some several times as the guard was strengthened.

Also: on rule 5 the agent reports (L) as **failing** on the basis of 0/20 seeds and a capped version of D_k, which goes beyond the letter of the pre-registered rule (which gave "undecided"). The agent says so explicitly (`REPORT.md` §3).

## 3. What we got

### Sanity checks (`REPORT.md` §1)
- Variational onset evidence was 1.15–2.42 nats below Monte Carlo (conservative, valid). The collapsed-Laplace evidence that the core specified gave false onsets on pure noise: max +41.7 nats, 5% above the level (`results/sanity_prior.json`, as reported in REPORT.md).
- Empirical Bayes recovered a planted Σ when targets were enough: ranks exact in 4/4 cases, subspace overlap 0.92–0.98.
- The core's in-sample rank rule (S3) picked ranks 25–42 even on targets with no structure (`results/sanity_eb_power.json`, as reported), so a cross-validated rule was used instead.
- Planted edits through the real network were recovered at the output-side sites (functional recovery 0.84–0.99) but **never at their own site** for the input-side sites (W_i, before the nonlinearity): later sites absorbed them.
- Null contexts: 7 onsets in 240 null contexts in the main runs (seed 3: 4, seed 9: 2, seed 11: 1) (`results/tables.md`).

### Coverage ρ after varied practice on seven families (`results/tables.md`, "Rule 1")

| Family | ρ [95%] | ρ − chance | ρ − chance,id | Rule 1 |
|---|---|---|---|---|
| R (recombination) | 0.429 [0.319, 0.538] | 0.171 [0.097, 0.246] | 0.224 [0.153, 0.295] | not near 0 |
| O (orthogonal) | 0.372 [0.272, 0.471] | 0.128 [0.061, 0.194] | 0.175 [0.111, 0.238] | not near 0 |
| T (timing) | 0.311 [0.213, 0.409] | 0.075 [0.029, 0.122] | 0.120 [0.077, 0.162] | not near 0 |
| Pr (precision) | 0.381 [0.278, 0.485] | 0.114 [0.045, 0.183] | 0.175 [0.123, 0.227] | not near 0 |
| C (control) | 0.428 [0.327, 0.528] | 0.169 [0.102, 0.235] | 0.222 [0.158, 0.287] | not near 0 |
| Rc (recovery) | 0.369 [0.268, 0.470] | 0.129 [0.062, 0.197] | 0.178 [0.113, 0.242] | not near 0 |

Where the measured directions of R sit: 55% at site 1 and 36% at site 3, the two output-side sites (`results/tables.md`, "Oracle directions").

### ρ against m and varied against narrow
- **Rule 2:** slope of ρ(R) per added family 0.0234 [−0.0004, 0.0471] → **not shown**. Slope of ρ − chance on R 0.0176 [0.0026, 0.0326] → above. Pooled slope 0.0210 [0.0001, 0.0419] → above (`results/tables.md`).
- **Rule 3:** varied − narrow at m = 7 on R: −0.022 [−0.167, 0.122] → **not shown**; pooled −0.034 [−0.168, 0.100] (`results/tables.md`).
- Post hoc (`results/posthoc.txt`): top-8 coverage rises clearly with m (R slope 0.0429 [0.0303, 0.0555]), and varied beats narrow on top-8 coverage at m = 7 on R (+0.247 [0.146, 0.347]). But over the data-expressible chance level, varied − narrow at m = 7 is −0.023 [−0.113, 0.066].
- Mean ranks at m = 7, site 3 (q_out/q_in): varied 9.8/32.6, narrow 34.2/34.5 (`results/tables.md`). Narrow practice picked larger ranks.

### Condition (L) (`results/tables.md`, "Rule 5" and post-hoc table)

| Family | Seeds with E_k D_k ≤ 63.5 | Median over seeds of E_k D_k | Capped E_k D_k [95%] |
|---|---|---|---|
| R | 0/20 | 187.2 | 112.8 [106.9, 118.7] |
| O | 0/20 | 181.3 | 111.9 [105.0, 118.7] |
| T | 0/20 | 144.2 | 102.0 [96.2, 107.8] |
| Pr | 0/20 | 159.2 | 107.4 [102.8, 111.9] |
| C | 0/20 | 179.1 | 113.3 [107.7, 118.9] |
| Rc | 0/20 | 196.8 | 110.5 [103.3, 117.7] |

The pre-registered mean-CI rule gave "undecided" for every family because a few directions at sites with a tiny fitted remainder α (10⁻⁶ to 10⁻¹²) produced D_k values around 10⁹. Terms of D_k for R (mean over seeds): rank term 31.9, **remainder term 144.1**, weight term 1.2; median α 0.53, median δ 6.74, π_fac 0.310. The remainder term alone is more than twice the 63.5 budget.

### Episodes to criterion (`results/tables.md`; geometric means over instances and seeds)

| Family | iso | Π_E v7 | Π_E v7 + dict | Π_E n7 | best SC (all four) |
|---|---|---|---|---|---|
| R | 17.2 | 7.6 | 6.8 | 13.5 | 28.0 |
| O | 17.4 | 15.1 | 13.3 | 15.8 | 23.7 |
| T | 53.3 | 54.5 | 54.5 | 54.5 | 52.6 |
| Pr | 25.3 | 23.8 | 23.7 | 23.2 | 29.3 |
| C | 21.1 | 14.0 | 13.4 | 16.7 | 27.6 |
| Rc | 60 (all censored) | 60 | 60 | 60 | 58.9 |
| narrow (held-out instances) | 21.0 | 11.5 | 5.4 | 3.5 | 22.3 |

Censored instances on R: iso 5/60, Π_E + dict 3/60, best SC 35/60. T: 52–54 of 60 censored in every arm; Rc: 59–60 of 60.

### Key ratios (`results/tables.md`, "Rule 6"; geometric-mean ratios, 95% CI)

| Family | Π_E+dict / best SC | Π_E / iso | best SC / iso | Π_E n7 / Π_E v7 |
|---|---|---|---|---|
| R | **0.24 [0.16, 0.37]** | **0.44 [0.33, 0.59]** | 1.63 [1.13, 2.33] | 1.77 [1.22, 2.56] |
| O | 0.56 [0.37, 0.85] | 0.87 [0.73, 1.03] | 1.36 [1.01, 1.84] | 1.04 [0.88, 1.24] |
| Pr | 0.81 [0.51, 1.27] | 0.94 [0.83, 1.07] | 1.16 [0.79, 1.71] | 0.98 [0.84, 1.13] |
| C | 0.49 [0.31, 0.76] | 0.66 [0.54, 0.81] | 1.30 [0.92, 1.84] | 1.20 [0.78, 1.84] |
| narrow | 0.24 [0.13, 0.46] | 0.55 [0.37, 0.82] | 1.06 [0.76, 1.49] | 0.30 [0.18, 0.51] |

- Rule 6 on R holds against the best of all four SC arms (0.24) and against the pre-registered pair (0.21 [0.13, 0.33]). Censor sensitivity: 0.35 at censor 30, 0.24 at 60, 0.17 at 120.
- Rule 4: narrow beats varied on the narrow family itself, 0.30 [0.18, 0.51].

### Twin and orthogonal clause (`results/tables.md`)
- **Rule 7, twin:** ρ_twin − ρ_varied on R = −0.119 [−0.221, −0.017]. Subspace overlap twin vs main: out 0.140, in 0.370; bootstrap vs main: out 0.221, in 0.438. Verdict **undecided**.
- **Rule 8, orthogonal:** 0 violations of the mixture cap in 2400 checks; Π_E / iso on O = 0.87 [0.73, 1.03] → no harm shown. Mean L_iso − L_mix on O = −246.6 nats against log(1/π_iso) = 0.717 nats.

### Rule 9 verdict
"Rule 9 verdict on C-P14: **unclear** (rule 1 on R: not near 0; rule 2 on R: not shown; rule 3 on R at m=7: not shown; rule 5 on R: undecided)" (`results/tables.md`).

## 4. What we inferred

**What the result says.**
1. **Practice did make the recombination family faster to learn.** With the same learner and the same records, the prior learned from varied practice needed 0.44× [0.33, 0.59] the episodes of the isotropic prior on R (7.6 against 17.2), and 0.40× with its dictionary (6.8). On C it was 0.66×. This is a real, measured speed-up from practice. [checked]
2. **But the coverage that should explain it is mostly generic.** The orthogonal family O, a camera rotation with no dynamics change, was covered almost as much as R (0.37 against 0.43). The "orthogonal" family was not orthogonal in the network's terms. So high ρ is not, by itself, evidence that practice learned something specific to recombinations. [checked numbers; interpretation is the agent's and mine, inferred]
3. **The core's predicted signatures of real learning-to-learn did not appear on ρ.** ρ did not clearly rise with m on R, and varied was not above narrow on ρ. [checked] Two caveats pull the other way: the rank-free top-8 coverage did rise with m and did favour varied (post hoc), and in *episodes* varied practice beat narrow on R by 1.77×. So the effect on learning speed matched the core's prediction (varied better on new families, narrow better on its own family) even though ρ, the core's chosen measure, did not show it. [checked]
4. **The theory's guarantee did not apply.** Condition (L) failed in every seed of every family. So the speed-up we saw is **not** explained by the core's own theory (P-link). The bound is far too loose to predict anything at this size. [checked]
5. **The margin over the "strong competitor" is mostly not from practice.** Π_E beat SC by about 4× on R (0.24), passing the core's ≤ ½ bar. But SC, full-matrix test-time training, was *slower than the isotropic rank-one learner with no practice at all* (1.63×). So most of the gap to SC comes from the rank-one edit form, not from practice. Only the Π_E/iso ratio (0.44) isolates the effect of practice. [checked]
6. **The empirical-Bayes estimate is poorly determined with about 200 targets.** Even a bootstrap of the main estimate overlaps itself only 0.22 (out-side) and 0.44 (in-side). [checked]

**What it does NOT say.**
- It does not say practice fails. It helped 2× on R.
- It does not test P14 as Lucky later clarified it (transfer of motor *ability*, graded by shared demand, never negative). This test models P14 as a *parameter prior*. The ledger records: "We later learned this modelled P14 wrongly: a parameter prior is not transfer of ability" (`notes/all-tests-ledger-2026-10-28.md`, item 2).
- It does not compare against a **low-rank** competitor such as LoRA test-time training. The SC arms were full 64×64 matrices. So "4× faster than a strong TTT baseline" (ledger wording) overstates the strength of that baseline.
- The timing (T) and slip-recovery (Rc) probes tell us nothing: almost every instance in every arm was censored at 30 episodes.
- A small wording issue in the ledger: it says "Most of the speed-up came from the simple rank-one form, not from practice." Strictly, the files support "most of the **margin over SC** came from the rank-one form". The speed-up over the isotropic prior (0.44×) is attributable to practice. [checked against `REPORT.md` Summary]

## 5. Why we think it happened

**Why coverage is generic.**
- The frozen network has only 9 outputs. At the last site only about 9 output directions are identifiable from data (sanity check S-E: raw cosines 0.25–0.4 while the cosine in the data's metric was about 1.0). So every fitted direction, practised or new, lands in the same small output subspace that the network fixes. Covering it is easy for any family. [checked: `PROTOCOL.md` §10 item 2 and `REPORT.md` §1; the conclusion is inferred]
- 80–90% of measured directions sit at the two output-side sites (R: 55% and 36%), and input-side edits are absorbed by later sites. So most directions live where the generic subspace dominates. [checked]
- On O, the mixture's evidence beat the isotropic evidence by about 246 nats on average. The factored component, learned from dynamics families, fits camera-rotation edits very well, which only makes sense if both live in the same generic output-side subspace. [checked number; interpretation inferred]

**Why ρ did not separate varied from narrow.** Narrow practice selected much larger ranks at the output site (34.2/34.5 against 9.8/32.6). Larger ranks cover more of any direction, which offsets narrow's less varied content. A rank-thresholded fraction is therefore sensitive to the rank rule, not just to what was learned. [checked ranks; explanation is the agent's, inferred]

**Why (L) failed even though practice helped.**
- D_k charges, for every dimension outside the learned leading ranges, a cost of ½·log(1 + nδα). With 64 − q dimensions per side, α ≈ 0.53 and δ ≈ 6.7, this remainder term alone was about 144, more than twice the budget. [checked]
- That charge assumes every ambient dimension has to be paid for. But the data identify only a few of them (the 9-output point above), so the true cost of unidentifiable dimensions is near zero. The bound is in the wrong metric. [inferred; this is also the CORE2 architect's diagnosis, `CORE2-RESPONSE.md` row P4, which itself says the cause is "[inferred]: the toy logged no per-instance Fisher"]
- (L) is a *sufficient* condition, not a necessary one. Failing it does not forbid a speed-up; it just means the theory cannot guarantee one. [inferred]
- An extra analysis for CORE2 found that the first onset's evidence gain did not predict which instances practice sped up (Spearman +0.01 over 240 instances; `CORE2-RESPONSE.md` row P18, citing CORE2 §8.28(e)). That suggests the gain comes from shaping the edit after detection, not from detecting it sooner. [checked as reported in CORE2-RESPONSE.md; I did not rerun it]

**Why SC was slow.** SC's adapters were full 64×64 matrices per site, fitted by test-time training. A full matrix has far more free parameters than a rank-one edit, so it overfits a short stream. The pre-registered untempered SC got *worse* as data grew (0.44 → 0.24 of the oracle gap on R between 4 and 30 episodes), and even the tempered SC2 was slower than the isotropic rank-one learner. [checked; the overfitting explanation is the agent's, inferred]

**Why the twin disagreed on ρ.** The twin's ρ was 0.12 lower. Two explanations fit: a prior that learns may collect different targets than a frozen one (feedback, which B3.2 was designed to prevent), or the estimate is simply unstable at about 200 targets. The overlap comparison could not separate them because the bootstrap's self-overlap was also low. [checked numbers; both explanations inferred]

## 6. Caveats and what could be wrong

- **A post-hoc change to the competitor.** The SC2 arms and the "best of four" rule were added after interim rule-6 results had been seen and had passed. The change can only make passing harder, and both versions pass, but it is still a post-hoc change.
- **The (L) "fail" reading goes beyond the pre-registered rule**, which said "undecided" because of a heavy tail. The 0/20 and capped values are post hoc. I find them convincing, but they were not pre-registered.
- **SC picked per instance after the fact.** This favours SC. Even so, SC was weak, because of its full-matrix form.
- **No low-rank competitor.** The single most important missing arm.
- **Censoring.** Unreached criterion was scored 60 episodes; ratios move with that choice (R: 0.35 at censor 30, 0.24 at 60, 0.17 at 120). T and Rc are almost entirely censored.
- **The mean learning curves do not obviously match the episode metric.** On R, the mean fraction of the oracle gap closed at 4 episodes was 0.32 for iso but 0.02 for Π_E, and at 11 episodes 0.61 against 0.59 (`results/tables.md`), yet Π_E reached the 80% criterion earlier on the geometric mean. Possible reasons: means over instances are dragged by a few instances with large negative values, and "first checkpoint that crosses the criterion" counts a crossing even if the curve later rises again. The files do not record whether crossings persisted. [inferred; worth checking before relying on the 0.44 figure]
- **Toy scale.** 64×64 sites, 9 outputs, 7 practised families, about 200 targets per life, a small MLP on 16×16 images. Not RoboJEPA.
- **Null onsets.** 7 of 240 null contexts produced an onset after tempering, so the onset test is not perfectly calibrated here.
- **Numerical crashes.** Three seeds crashed on a diverged network and were rerun with a guard. The guard fired 2, 8 and 2 times in seeds 8, 12 and 19 (`REPORT.md` §3). Seeds were not dropped for their results.
- **A wrong model of P14.** The test faithfully checks the core's mechanism, but that mechanism (a parameter prior) was later judged not to be what P14 asks for.

## 7. What it changed in the project afterwards

**CORE2 (2026-10-18).** The same Opus architect rewrote the core on the toy evidence (`research/architecture/spec-v3.6/BRIEF-core2.md`). The memory note first-principles-architecture-draft-v31.md summarises the result as: "CV ranks; condition (L) deleted; P14 judged by Γ vs same learner without practice + scale-only control + low-rank SC-i ... Least sure: practice may teach only edit size." In detail (`CHANGELOG.md`, `CORE2-RESPONSE.md` rows P1–P34, `CORE2-LESSONS.md`):
- **Condition (L), D_k and the coverage route P-link were deleted.** ρ was kept as a **diagnostic only**, with both chance levels and top-k coverage beside it.
- **The claim was restated on a practice ratio Γ**: episodes with practice divided by episodes for the same learner without that history, with a pre-set bar of Γ(R) ≤ ½ and five clauses, including specificity.
- **A scale-only control** was added: Π_E with each learned Σ replaced by a multiple of the identity with the same trace. This checks whether practice teaches *directions* or only the *typical size* of an edit (`CORE2-RESPONSE.md` P34). Test P had no such arm.
- **A low-rank competitor SC-i** (rank-r LoRA test-time training with Reptile, a PCA prior and nearest-neighbour recall) was added; later rounds added SC-j (long-context in-context learner) and SC-k (learned-prior TTT).
- **Cross-validated ranks** replaced the in-sample rank rule; **proper one-step predictive densities** replaced the collapsed-Laplace onset evidence; **pre-life tempering was removed**.
- **The edit regret was restated in the records' Fisher metric**, where unidentifiable directions cost nothing, with a new owed proof ("P-shape") meant to predict episodes from the risk after detection.
- Onset members became (component, site) pairs, so an edit commits at whichever site its evidence is largest, addressing site ambiguity.
- Bookkeeping fixes: δ given one definition (‖Δ‖_F); the family count corrected to eleven.

Status counts for the 34 Test P rows of `CORE2-RESPONSE.md`: fixed 15, test only 13, holds already with evidence 6. **These are the architect's claims; Test P was never re-run on CORE2.** The CORE2 audit and fix loop (memory note first-principles-architecture-draft-v31.md) concentrated on the write rule and was paused after three rounds.

**P14 itself was redefined the same day.** On 2026-10-18 Lucky clarified that what transfers is motor *ability*, graded by shared demand and never negative, not a parameter starting point (`notes/problems.md`, problem 14; `notes/retrospective-2026-10-19.md` §1 and §4 item 4). The retrospective concludes that "every earlier design and the running P14 test had modelled the wrong thing".

**Later P14 tests.**
- T-P14 on the merged core-v4 (2026-10-18/19): a learned practice prior Γ made learning **slower** (1.30× the frames with Γ on versus off), 1.41× Reptile's frames, narrow beat varied, and the learner slowed with more practice (`notes/all-tests-ledger-2026-10-28.md`, item 5).
- T-P14D (pilot, independent builder): no arm of any kind, ours or existing, showed full P14, and our designs lost to generic adapters (`notes/all-tests-ledger-2026-10-28.md`, item 8).

**Lasting lessons from Test P.** A coverage number needs a chance baseline before it means anything; a "strong competitor" must be the strongest reasonable one (here, a low-rank adapter), or the margin is mostly about parameterisation; and a theory that cannot predict the effect it is meant to explain (here, (L) failing while a 2× speed-up appeared) is not yet a theory of that effect. I did not read the primary files of T-P14 or T-P14D for this document.
