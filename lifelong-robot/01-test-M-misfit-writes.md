# 01 Test M: do misfit commits ("bad writes") pile up when the frozen base model is wrong?

**Date run:** 2026-10-17 (protocol pre-registered that day; main runs about 14:30 to 21:35 UTC) **Folder:** /mnt/project-files/research/toy-tests/M-misfit/ **Built by:** one Opus test agent working from research/toy-tests/BRIEF.md (memory note first-principles-architecture-draft-v31.md); not the designer of the core **Seeds:** 8 per well-specified sanity arm, 3 for the positive control, 4 to 6 for the other arms, 6 for each misspecified arm, 12 for MIX, 4 for each driver arm (after compute cuts C1 and C3) **Verdict:** **FAILED** at toy width under pre-registered rule D7. The safety bound held, but misfit commits kept growing roughly linearly with time and did not level off.

A note on sources before anything else. The folder has `PROTOCOL.md`, `code/`, and `results/` (`tables.md`, `extra.md`, `summary.json`, logs). **There is no `REPORT.md`.** The brief asked for one, but the harness blocked the agent from writing report files. The agent's own summary survives only as a copy pasted into the next design brief, `research/architecture/spec-v3.6/BRIEF-core2.md`. In this document every number comes from `research/toy-tests/M-misfit/results/tables.md` or `results/extra.md` unless another file is named. Where I use the agent's interpretation I cite `BRIEF-core2.md`.

---

## 1. Why we ran it

**The design idea under test.** The v3.6 core (`research/architecture/spec-v3.6/CORE.md`, section B2) has a principle that can be summed up as "write rarely, and only on statistical evidence". The robot carries a frozen base model. It never changes the base. Instead it keeps a small *committed state* on top of it: a dictionary of edit directions, their coefficients, and "stamps" (saved snapshots of the committed state that can be recalled later). The committed state changes only through a **commit** (a "write").

A commit is decided by a sequential test. In plain words:
- At regular times the robot opens a **start**: a test that begins reading the stream from that moment.
- Each start keeps, for every candidate edit (a **member**: a jump in one coefficient, a new direction called an **onset**, a **refit** of a direction's shape, or a **recall** of a stamp), a running ratio: how much better the candidate has predicted the frames since the start than the current committed model has. This running ratio is an **e-value**. An e-value is a quantity whose expected value is at most 1 when the null hypothesis is true, so a large value is evidence against the null.
- If the weighted mixture of these e-values reaches a level 1/η (here log(1/η) = 16.71 **nats**; a nat is the natural-log unit of evidence or information, so 16.71 nats means the candidate predicted about e^16.7 ≈ 18 million times better in likelihood), the best candidate is committed.

The core proves a bound (CORE.md B2.3, §8.1): if the committed model is the *true* law of the data, the expected number of **false commits** is at most λ_𝔅 per commanded second per slot, with λ_𝔅 = 1 per 1000 commanded hours. That is Ville's inequality applied to e-values.

**The worry.** The proof only covers the case where the committed model is correct. A frozen base model is wrong almost everywhere. When it is wrong, the same rule makes **misfit commits**: edits that genuinely predict better than a wrong reference. CORE.md itself listed this as item 2 of "What is least sure" (§10): "misfit commits may be frequent and grow K and the stamp count until the remover binds. The bound is clean; the drift bar (T3′ misspecified arm) may fail."

**Project problems targeted.** Mainly P3 (stability–plasticity: "It does not drift when nothing changes"), with consequences for P9 (memory that stays bounded in size) and P2 (old skills not silently damaged). The brief labels the test "core §10 item 2; T3′ misspecified arm; P3" (`research/toy-tests/BRIEF.md`).

**Why at that moment.** On 2026-10-17, after many rounds of paper spec writing and red-teaming that did not converge, Lucky decided to run toy tests *before* writing the full spec: "first do the toy test both of them and then we wil check about the spec so if any results we know we will fix them in the core only" (`research/toy-tests/BRIEF.md`). Test M was one of the two riskiest bets. The decision it was meant to settle: can the write rule be kept as it is, or does it need a structural change before anything else is built on it?

## 2. What we did

### The world
(From `PROTOCOL.md` §1.)
- One factor, one site, a **linear-Gaussian head**. The input is 5 smooth AR(1) signals plus a constant (6 numbers); the output is 4 numbers; noise standard deviation 1. Frames arrive at 5 Hz, so one frame = 0.2 commanded seconds.
- The true law: y = W·x + κ·Q(x) + u_t·a*(b*ᵀx) + noise. Q is a missing quadratic term (products of neighbouring inputs). u_t is a slow hidden variable (relaxation 3000 frames = 10 minutes). The noise can be Gaussian, Student-t (heavy tails), or autocorrelated.
- The **frozen base** is a least-squares fit on 200,000 pre-life frames. In the NL arms the pre-life inputs were shifted by +0.7, so the base has a fixable linear error as well as the unfixable quadratic one. In the well-specified arms the base is exactly the truth.
- The world is **stationary**. Nothing changes during a life (except in the WS_change power check). So in the misspecified arms every commit is by definition a misfit commit.

### What was implemented
The commit rule was implemented literally as CORE.md states it (`PROTOCOL.md` §2): four slots (jump, onset, refit, recall), starts opened on the dyadic "covering" (r = 2, φ = ½: every frame opens a start, and a start at rung ℓ lives 4·2^ℓ frames), mixture weights fixed when a start opens, and **commit-and-continue**: when one start commits, the winning start closes, and every other start keeps its accumulated statistic while the reference switches to the new committed state. At most one commit per frame. The protocol states: "Not simplified, because it is what the test is about: the commit rule" (§7).

Simplifications declared before the runs (`PROTOCOL.md` §7) include: one site (S1); no sensor side, so no cross-side separation (S2; this can only remove commits, so counts here are upper bounds for that part of the rule); a uniform move forecaster F̂ (S3; the protocol argues this likely *under*-counts commits); law members only on constant directions (S6); and **no remover** (S9), so the test measures what a remover would have to absorb, not the remover itself.

### Arms
| Arm group | What it is | Frames per run | Seeds (after cuts) |
|---|---|---|---|
| WS_count λ×1, ×100, ×10⁴ | Well-specified; commits counted but not applied, so the reference stays the truth. Checks the bound. | 2^15 | 8 |
| WS_count_cheat λ×10⁴ | Positive control: the onset statistic replaced by a non-predictable plug-in ratio, which must break the bound. | 2^15 | 3 |
| WS_apply λ×1 | Well-specified, commits applied. | 2^17 | 4 |
| WS_change | Well-specified, then an abrupt rank-one change (δ = 0.3) at frame 2^15. Checks power. | 2^16 | 6 |
| NL1–3 | Missing nonlinearity, κ = 0.05, 0.15, 0.5 | 2^16 | 6 |
| HT1–3 | Student-t noise, ν = 20, 6, 3 | 2^16 | 6 |
| AC1–3 | AR(1) noise, ρ = 0.1, 0.3, 0.7 | 2^16 | 6 |
| LAT1–3 | Slow hidden variable, s_u = 0.05, 0.15, 0.5 | 2^16 | 6 |
| MIX | κ = 0.15, ν = 6, ρ = 0.3, s_u = 0.15 together | 2^17 (7.28 commanded h) | 12 |
| MIX drivers | λ×0.1, λ×10, φ = ¼, r = 4 | 2^16 | 4 |
| MIX_close | Diagnostic, **not the core's rule**: every start closes at a commit | 2^16 | 4 |

There were no competitor methods in this test. It is a test of one rule against its own claim, not a race.

### Metrics
Per run (`PROTOCOL.md` §4): every commit with its kind; K (number of directions), active directions, stamps; **held-out excess loss** (nats per frame above the true conditional law, on a forked 10,000-frame future of the same world); a **task proxy** (the percentage of held-out frame-channels whose predicted mean is within 0.1 noise-sd of the true conditional mean; there is no controller in the toy); the share of **spurious** active directions (switching them off does not hurt held-out loss).

### Pre-registered decision rules
(`PROTOCOL.md` §5.)
- **D1, bound:** crossings X against B = λ·T·4·seeds must satisfy P(Poisson(B) ≥ X) ≥ 0.01; the positive control must fail.
- **D2, power:** ≥ 90% of seeds commit within 2000 frames of the planted change, and held-out loss recovers to within 0.01 nats/frame.
- **D3, growth:** commits counted in doubling windows of time [2^j, 2^(j+1)). A Poisson regression on the last four windows gives an exponent α (cumulative commits grow like t^α). **Saturates** if the upper 95% bound of α < 0; **keeps growing** if the lower bound ≥ 0, and **linear-like** if the lower bound is also > 0.5.
- **D5, drift:** Δtask = task(end) − task(day 1), with "day 1" = frame N/30 (the run is treated as a compressed month). Bar met if the lower bound > −1 point.
- **D7, the bet:** **Holds** if every misspecified arm saturates or at worst keeps growing with α's upper bound < 0.5, *and* the drift bar is met. **Fails** if any arm keeps growing linear-like or any arm fails the drift bar.

### Deviations and amendments (all logged in `PROTOCOL.md` §10)
- **C1 (before any main run, timing only):** runs were 2–2.5 ms per frame, so the plan would have taken over 25 CPU-hours on 2 cores. Run lengths and seeds were cut; NL2 and LAT2 driver arms dropped.
- **C2:** D2's "9 of 10 seeds" read as "≥ 90% of seeds" because only 6 seeds ran. No threshold changed.
- **C3 (during main runs, compute only, after seeing seed 0 of every arm and all 12 MIX seeds):** the 2^16 arms went from 8 to 6 seeds, drivers from 6 to 4, and the long 2^19-frame MIX run was **dropped**. Decision rules unchanged.
- **C4 (reading notes after results):** α is unreliable for sparse arms (AC2: 18 commits, NL1: 22); all 6 HT3 runs and one NL3 and one AC3 run hit the K = 256 or 2048-stamp cap and stopped early (between frames 45,056 and 61,440), which biases HT3's α *downward*.

The aim of 20 seeds per arm (BRIEF) was met only for MIX's original plan and then cut to 12.

## 3. What we got

### Sanity checks: the bound and power

**D1, false commits with the reference equal to the truth** (`results/tables.md`, "D1"):

| Arm | Seeds | Crossings X | Bound B | P(Poisson(B) ≥ X) | Verdict |
|---|---|---|---|---|---|
| WS_count_lam1 | 8 | 0 | 0.06 | 1 | pass |
| WS_count_lam100 | 8 | 0 | 5.83 | 1 | pass |
| WS_count_lam10000 | 8 | 22 | 582.54 | 1 | pass |
| WS_count_cheat_lam10000 (positive control) | 3 | 2866 | 218.45 | 0 | FAIL (as required) |

Mean e-value at a start's expiry, per slot, was below 1 in every valid arm: for example at λ×1, jump 0.898, onset 0.338, refit 0.334, recall 0.858. In the cheat control the onset slot's mean was 34.8 and P(E ≥ 10) was 0.49 (`results/tables.md`).

**Well-specified with commits applied:** 4 seeds × 2^17 frames, commits per run [0, 0, 0, 0] (`results/tables.md`).

**D2, power after a planted change** (δ = 0.3 at frame 32768): first commit after the change at [983, 1404, 987, 824, 928, 1137] frames, 6/6 within 2000. Held-out excess loss: 0.0114 before, 0.0522 just after, 0.0117 at the end (recovery gap 0.0003 [−0.0010, 0.0017]). **Pass.**

### D3: growth of misfit commits (`results/tables.md`, "D3")

| Arm | Commits per run | Pooled counts per doubling window (j = 10 … last) | α [95%] | D3 verdict | Late rate per commanded hour |
|---|---|---|---|---|---|
| NL1 | 3.7 | 0, 0, 0, 5, 7, 10 | 0.94 [0.70, 1.39] | keeps growing (linear-like); sparse, flagged C4 | 0.9 |
| NL2 | 16.7 | 5, 17, 11, 13, 24, 30 | 0.52 [0.24, 0.85] | keeps growing | 2.7 |
| NL3 | 2361.0 | 296 … 6939 | 0.96 [0.88, 1.05] | keeps growing (linear-like) | 635.3 |
| HT1 | 1.5 | 0, 0, 4, 5, 0, 0 | −1.33 [−14.41, −0.61] | saturates | 0.0 |
| HT2 | 762.0 | 79 … 2175 | 0.99 [0.88, 1.11] | keeps growing (linear-like) | 199.1 |
| HT3 | 4118.5 | 411 … 7767 (last window truncated by the cap) | 0.62 [0.52, 0.73] | keeps growing (linear-like) | 711.1 |
| AC1 | 0.0 | all 0 | n/a | saturates (no commits) | 0.0 |
| AC2 | 3.0 | 0, 0, 0, 0, 8, 10 | 1.60 [0.76, 30.59] | keeps growing (linear-like); sparse, flagged C4 | 0.9 |
| AC3 | 1053.7 | 302 … 1895 | 0.44 [−0.08, 0.84] | unclear | 173.5 |
| LAT1 | 0.0 | all 0 | n/a | saturates (no commits) | 0.0 |
| LAT2 | 18.0 | 1, 0, 7, 22, 38, 40 | 0.66 [0.29, 0.96] | keeps growing | 3.7 |
| LAT3 | 117.5 | 18 … 345 | 0.95 [0.89, 1.03] | keeps growing (linear-like) | 31.6 |
| MIX | 1452.4 | 228, 262, 560, 930, 2135, 4237, 8938 | **1.06 [1.00, 1.13]** | keeps growing (linear-like) | **204.6** [183.4, 222.7] |

Even leaving out the two sparse arms that C4 flags (NL1, AC2), five arms are linear-like with tight intervals: NL3, HT2, HT3, LAT3 and MIX.

### What the commits were (`results/tables.md`, "K, stamps, cascades, spurious directions")

| Arm | K at end | Active dirs at end | Stamps at end | No-op share | Cascade share (n_cross > 1) | Spurious share of active dirs | Runs hitting cap |
|---|---|---|---|---|---|---|---|
| NL2 | 7.8 | 6.7 | 14.0 | 0.05 | 0.09 | 0.00 | 0 |
| NL3 | 156.0 | 5.7 | 1775.3 | 0.11 | 0.33 | 0.28 | 1 |
| HT2 | 26.5 | 0.0 | 136.7 | 0.75 | 0.90 | n/a | 0 |
| HT3 | 247.2 | 1.3 | 1952.7 | 0.34 | 0.81 | 0.90 | 6 |
| AC3 | 171.2 | 20.7 | 894.7 | 0.11 | 0.44 | 0.20 | 1 |
| LAT3 | 14.5 | 2.0 | 72.0 | 0.06 | 0.11 | 0.37 | 0 |
| MIX | 89.2 | 9.8 | 961.7 | 0.19 | 0.63 | 0.24 | 0 |

Commit kinds pooled over MIX's 12 seeds: recall_exact 11,047, recall_plus 3,662, onset 1,070, law 1,010, refit 432, scale 139, jump 44, switch_off 15, return_const 10 (`results/tables.md`). So about 14.7k of about 17.4k MIX commits were recalls.

### D5: drift of the task proxy and held-out loss (`results/tables.md`, "D5")

| Arm | Δtask, points [95%] | Bar (≥ −1) | Task: base → day 1 → end | Excess loss nats/frame: base / day 1 / end |
|---|---|---|---|---|
| NL1 | +6.32 [5.61, 7.18] | met | 86.0 → 85.2 → 91.5 | 0.0120 / 0.0113 / 0.0073 |
| NL2 | +18.18 [12.74, 27.73] | met | 54.3 → 48.1 → 66.3 | 0.1004 / 0.8463 / 0.0465 |
| NL3 | +5.49 [3.14, 7.91] | met | 25.8 → 15.1 → 20.6 | 0.9292 / 1.3253 / 0.7307 |
| HT2 | 0.00 [0.00, 0.00] | met | 100.0 → 100.0 → 100.0 | 0.1254 / 0.1230 / 0.1254 |
| **HT3** | **−54.30 [−77.00, −25.10]** | **FAILED** | 100.0 → 95.2 → 40.9 | 0.7811 / 1.2504 / 5.0947 |
| AC3 | +17.23 [8.50, 24.14] | met | 11.3 → 11.9 → 29.1 | 1.3399 / 1.4884 / 0.6872 |
| LAT1 | −0.74 [−1.89, 0.18] | unclear | 98.9 → 99.6 → 98.9 | 0.0018 / 0.0006 / 0.0018 |
| LAT2 | +3.10 [−4.19, 8.50] | unclear | 87.4 → 87.5 → 90.6 | 0.0135 / 0.0112 / 0.0096 |
| LAT3 | +5.01 [−3.99, 13.86] | unclear | 59.8 → 57.7 → 62.7 | 0.1359 / 0.1449 / 0.1252 |
| MIX | +3.41 [2.11, 4.63] | met | 23.6 → 23.7 → 27.1 | 0.4145 / 0.4052 / 0.3645 |

(HT1, AC1, AC2 had Δtask within ±0.3 points; full table in `results/tables.md`.)

HT3's collapse happened late. In `results/extra.md` the HT3 trajectory shows task 97.5 at frame 32,768 and 35.0 at frame 49,152, with K going from 175.0 to 241.7 and excess loss from 1.03 to 4.72 nats/frame over the same stretch.

### D6: what drives the rate (`results/tables.md`, "D6", all at 2^16 frames)

| Variant of MIX | Seeds | Commits per run [95%] | Mean alive starts per slot | α [95%] |
|---|---|---|---|---|
| λ×1 (first half of MIX runs) | 12 | 707.6 [629.7, 784.3] | 27.0 | 1.02 [0.88, 1.16] |
| λ×0.1 | 4 | 517.2 [461.5, 575.8] | 25.6 | 0.86 [0.74, 1.06] |
| λ×10 | 4 | 1100.0 [991.5, 1243.0] | 24.8 | 1.00 [0.82, 1.16] |
| φ = ¼ | 4 | 1235.2 [1161.2, 1337.5] | 47.1 | 1.01 [0.93, 1.07] |
| r = 4 | 4 | 968.0 [854.0, 1085.5] | 36.7 | 0.92 [0.74, 1.05] |
| close-on-commit (not the core's rule) | 4 | 255.0 [221.5, 283.0] | 15.1 | 1.13 [0.95, 1.38] |

d log(commits) / d log λ = 0.16 [0.13, 0.20]. Ratios I computed from this table: φ = ¼ gives 1.745× the commits with 1.744× the alive starts; r = 4 gives 1.368× the commits with 1.359× the alive starts.

### The D7 verdict
`results/tables.md` does not print a single D7 line; applying the pre-registered rule to the tables above gives **Fails**, because several arms keep growing linear-like and HT3 fails the drift bar. The agent's own summary gives the same verdict: "Failed, verdict 'fails at toy width' by pre-registered rule D7" (`research/architecture/spec-v3.6/BRIEF-core2.md`). The project ledger records "Verdict: FAILED (pre-registered rule D7)" (`notes/all-tests-ledger-2026-10-28.md`).

## 4. What we inferred

**What the result says.**
1. **The safety bound is real and the implementation is sound.** Valid arms stayed far under the bound (22 against 583 at λ×10⁴), and the deliberately invalid statistic broke it by a factor of about 13 (2866 against 218). This means the test had power to see a broken bound, and the bound as proved holds when its assumption holds. [checked]
2. **The rule can detect real change.** A planted change was committed in 824–1404 frames (about 3 to 5 commanded minutes) in every seed, and the model recovered. [checked]
3. **The bound says nothing about misfit.** When the base is wrong, the same rule writes at a rate that keeps growing with time. In MIX (all four kinds of misfit at moderate size), commits grew with exponent 1.06 [1.00, 1.13]: indistinguishable from linear growth, about 205 commits per commanded hour late in the run, with no sign of levelling off over 7.3 commanded hours. [checked]
4. **Much of the writing was waste made by the rule itself, not by the world.** In MIX, 19% of commits changed nothing (no-ops), 63% were cascades (several starts above the level at the same frame), and about 14.7k of 17.4k were recalls. HT2 is the clearest case: 762 commits per run, 0 active directions at the end, task unchanged. [checked]
5. **Heavy tails are dangerous to a Gaussian predictive.** HT3 (ν = 3) lost 54 task points and its excess loss rose about sixfold over the base; every HT3 run hit the K cap. [checked]
6. **Elsewhere misfit commits were genuine learning.** MIX +3.4 points, NL2 +18.2, AC3 +17.2, with held-out loss falling. The test did not show that misfit commits are harmful in general; it showed that they are endless and mostly wasteful, and harmful in the heavy-tail case. [checked]
7. **The false-write level λ is not a control knob for misfit writes.** A hundredfold change in λ (0.1 to 10) changed commits only from 517 to 1100 per run; the elasticity is 0.16. [checked]
8. **The rate scales with the number of alive starts**, almost exactly in proportion. [checked]

**What it does NOT say.**
- It does not say the core's bound is wrong. It held.
- It does not test the remover, the sensor side, a trained forecaster F̂, or any real network. There was no remover, so "K grows without limit" means "the remover would have to absorb this", not "memory overflows in the real design".
- It does not say what happens over a real month. The longest run was 7.28 commanded hours; the planned 29-hour run was dropped (C3). Saturation at much later times is untested, not refuted. [checked: `PROTOCOL.md` §10 C3]
- It does not compare against any other learner (LoRA, replay, etc.). Whether a different method would drift less on the same streams is not addressed here.
- The task metric is a proxy (mean precision), not task success.
- The HT, AC arms with a 100% base task score show that the task proxy is blind to some misfit: for HT the true conditional mean is still exactly linear, so the base already scores 100 and only the noise shape is wrong. [inferred from `PROTOCOL.md` §1 and the base task column]

## 5. Why we think it happened

**Why the bound held but writes did not stop.**
- The bound comes from the e-value having expectation ≤ 1 when the reference is the true law. Under misfit, the reference is not the true law, so the e-value of a better-predicting member grows on average, and nothing in the rule stops a new start from finding the next small improvement. [checked: CORE.md B2.3 states the claim only for the well-specified null]
- Under misfit, the evidence a start accumulates has a heavy upper tail. The agent's later analysis (copied into `CORE2-RESPONSE.md` row M19) estimates this tail at a scale of about 6 nats rather than 1, which would make the commit rate scale as λ^(1/6) ≈ λ^0.17, close to the measured 0.16. [inferred; the 6-nat figure comes from CORE2-RESPONSE.md, not from a Test M result file]

**Why commits scale with alive starts.** Each alive start is a separate chance to cross the level. If each has a roughly constant crossing hazard under stationary misfit, the total rate is the sum over starts. The measured 1.745× versus 1.744× (φ = ¼) and 1.368× versus 1.359× (r = 4) fit that almost perfectly. [checked numbers; the hazard explanation is inferred]

**Why so many recalls, no-ops and cascades.** Under the literal "commit-and-continue" rule, after one start commits, every other alive start keeps the evidence it had built up against the *old* reference. That evidence mostly says "the old reference was wrong", which the first commit already used. So several starts can cross on successive frames, each committing something (often a recall of a stamp equal to the state just left or just entered). A recall to the current state is a no-op but still counts as a write and closes a stretch, creating a new stamp. [inferred from the rule as implemented (`PROTOCOL.md` §2) and the pilot notes in `PROTOCOL.md` §9, which describe "literal commit-and-continue cascades"; supported by HT2's 0.90 cascade share and 0.75 no-op share — checked]

The MIX_close diagnostic supports this partly: closing every start at a commit cut commits 2.8× (708 to 255 per run). But note that its growth exponent was still 1.13 [0.95, 1.38]. Removing cascades lowered the level of the curve; it did not show saturation. [checked] So cascades are not the whole story. The remaining growth is plausibly the persistent misfit that the model family cannot represent (see next point). [inferred]

**Why misfit cannot be "used up" in some arms.**
- In NL3 and LAT3, part of the misfit (the quadratic term; the hidden slow variable) cannot be represented by the rank-one linear edits the core allows. The best linear correction depends on which region of input space the robot is currently in, so as the state wanders, a different correction becomes locally best, and the rule keeps switching. [inferred; this is also the agent's later explanation in `CORE2-RESPONSE.md` row M7, "prequentially correct switching, not a defect of the rule"]
- In NL1, misfit is mostly the fixable linear error from the pre-life input shift; only 22 commits occurred, task rose 6.3 points and held-out loss fell. This looks like learning that might saturate given more time, but the counts are too sparse to say. [inferred; C4 flags it]

**Why heavy tails did damage.** The reference is Gaussian. Under Student-t noise with ν = 3, single frames far out in the tail are much more likely than the Gaussian allows. A member that happens to explain one such frame gains a lot of log-likelihood at once. The CORE2 analysis computes that one 6-standard-deviation frame gives about 18 nats, more than the 16.71-nat level (`CORE2-RESPONSE.md` row M13). The pilot notes record "single heavy-tailed frames triggering onsets with large c" (`PROTOCOL.md` §9). A large wrong coefficient on a direction would corrupt the predicted mean, which is what the task proxy measures. HT3's spurious share of active directions was 0.90. [checked: pilot notes, spurious share; the 18-nat figure is from CORE2-RESPONSE.md; the causal chain to the late task collapse is inferred — the result files do not record which commit caused the drop from 97.5 to 35 points]

**Why HT2 churned with nothing to show.** In HT2 (ν = 6) the base mean is exactly right, so there is nothing for the mean to learn. Recalls dominated (3,792 recall_exact of about 4,572 pooled commits), with 0.75 no-ops and 0.90 cascades, and 0 active directions at the end. Outliers triggered commits that later recalls undid. [checked numbers; mechanism inferred]

**Why held-out loss spiked early in several arms.** NL2's mean excess loss went 0.10 (base) to 0.85 (day 1) and back to 0.065 by frame 4096; MIX went from 0.42 to 1.09 at frame 256 and back to 0.42 by frame 512 (`results/extra.md`). These are seed means over 6 or 12 runs, so one bad run can drive them. The CORE2 analysis attributes the spikes to commits installing a point estimate without its uncertainty, so the reference jumps (`CORE2-RESPONSE.md` row M27). [checked numbers; explanation inferred, not verified in Test M's files]

## 6. Caveats and what could be wrong

- **No REPORT.md.** The agent's own written interpretation was not saved in the folder. I rely on the tables, the protocol and the summary copied into `BRIEF-core2.md`.
- **Few seeds.** 6 seeds per kind × level arm and 4 per driver, far below the brief's aim of 20. Intervals on sparse arms (NL1, AC2, HT1) are very wide or degenerate.
- **Short runs.** The design treats 2^17 frames (7.28 commanded hours) as a "compressed month". That is an assumption of the test design, not a fact about real time. The long run was dropped, so late saturation is untested.
- **Capped runs.** All 6 HT3 runs stopped early at the K or stamp cap. Their last window is truncated, so HT3's α (0.62) is biased downward; its "end" metrics are at the stopping frame (`PROTOCOL.md` §10 C4).
- **Toy scale.** One site, 6 inputs, 4 outputs. The protocol itself says the net direction of the effect at real scale is unknown (S1).
- **Missing parts.** No remover, no sensor side, uniform F̂. S3 argues a trained F̂ would *increase* misfit commits; S2 argues separation would *decrease* them. Neither was measured.
- **Proxy metric.** Task = precision of predicted mean, with no controller. For HT arms the base already scores 100.
- **The D5 bar is one-sided.** It is "met" whenever the task does not fall by more than 1 point; large improvements count as "met" even though |Δtask| > 1 was supposed to be flagged.
- **A small ledger inaccuracy.** The project ledger says "13 worlds ... 6 seeds each" (`notes/all-tests-ledger-2026-10-28.md`). In fact there were 12 kind × level arms with 6 seeds and MIX with 12 seeds.
- **Rate per hour.** "205 per hour" is the late rate in commanded hours in MIX's last window (204.6 [183.4, 222.7]). If that linear rate continued for a real month (720 hours), it would be on the order of 150,000 commits; this is my extrapolation, not a measured number. [inferred]

## 7. What it changed in the project afterwards

**Immediately (2026-10-18): CORE2.** Lucky asked the same Opus architect to write `research/architecture/spec-v3.6/CORE2.md`, solving every toy-test finding "at its root" without editing CORE.md (`research/architecture/spec-v3.6/BRIEF-core2.md`; memory note first-principles-architecture-draft-v31.md). For Test M, the changes recorded in `CHANGELOG.md`, `CORE2-RESPONSE.md` and `CORE2-LESSONS.md` were:
- **Re-measuring after every commit.** In the first CORE2, every alive start was "re-anchored" at each commit so that it measures candidates against the newly adopted explanation, not the abandoned one. The architect claimed this makes no-op, repeat and cascade commits impossible [proved in CORE2 §8.24–8.25, per the architect]. Over three fix rounds this was replaced, first by "leader compounds" and an "adopted family", then by an "adopted set" profile plus a "block member" that treats several simultaneous changes as one member (`CHANGELOG.md`, rounds 1–3).
- **Identity edits removed from members**, so a recall to the current state can no longer be a commit (`CORE2-RESPONSE.md` M10).
- **The residual law became committed state**: per channel group, a log-scale, a Student-t tail and an OU (autocorrelation) relaxation, so heavy tails and autocorrelated noise are modelled rather than read as change. A new declared number τ₁ was added. A bounded-influence lemma states that one frame can move a statistic only a bounded amount (`CHANGELOG.md`; `CORE2-RESPONSE.md` M13, M16).
- **Installing the member's posterior** at a commit, not a point estimate, to stop the reference jumping (M27).
- **A rate law and named sources of misfit** (B2.6). Misfit the family cannot hold was declared as a cost (D23), "not fixed with reason" (M7). The insensitivity to λ was also "not fixed with reason": λ bounds false writes only (M19).
- **A re-specified toy test T3′-M** with new rules and predictions, for example 0 no-ops and repeats, HT and AC arms saturating, and MIX's rate below one third of 205 per hour (`CORE2-LESSONS.md`).

Status counts for the 30 Test M rows of `CORE2-RESPONSE.md`: fixed 4, ceiling with test 1, test only 17, holds already 6, not fixed with reason 2. **These are the architect's claims; the toy was never re-run on CORE2.** The independent audit of CORE2 (`AUDIT-CORE2.md`, summarised in memory note first-principles-architecture-draft-v31.md) found the design not ready, with new critical problems near the write rule; for example, "misfit rate grows with log age" was listed among the majors. Three fix rounds each produced about four new major problems at the seams of the write rule, and the loop was paused (memory note first-principles-architecture-draft-v31.md; `notes/retrospective-2026-10-19.md`).

**Later (2026-10-18/19): core-v4 and T-CHURN.** A fresh merged core (core-v4) carried the lesson forward as "re-check after each write". Its test, T-CHURN, found that writes now **did level off**, at about 4 per million frames against the old ~205 per hour, but LoRA with replay was slightly more accurate in every misfit world, so that bet failed on accuracy; its false-alarm control world turned out to be invalid (`notes/all-tests-ledger-2026-10-28.md`, item 4; `notes/retrospective-2026-10-19.md` §2). I did not read T-CHURN's primary files for this document.

**Lasting lesson.** The retrospective lists "writing only on held-out evidence stops runaway bad writes" as the project's one real mechanism win, and the ledger lists "re-check after each write" among the principles that held up, while noting that none became a better method than existing ones (`notes/retrospective-2026-10-19.md` §3; `notes/all-tests-ledger-2026-10-28.md`, "The one-line pattern"). Test M is where that lesson started: it showed that a bound on false writes is not a bound on writes.
