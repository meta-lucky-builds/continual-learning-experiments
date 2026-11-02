# 04 T-CHURN: under a misspecified base, do structure writes now level off, and is accuracy as good as LoRA with replay?

**Date run:** 2026-10-18 to 2026-10-19 (UTC). The first pass of `merge_ref`, `lora_replay` and oracle finished overnight. The last `merge_noref` files are stamped 2026-10-19 05:29–06:01. Final tables were generated 2026-10-19 06:18. **Folder:** /mnt/project-files/research/toy-tests-v4/compose-churn/ **Built by:** "Agent 1", the same test agent as T-COMPOSE, under `research/toy-tests-v4/BRIEF.md`. The agent's session ended at the 01:45 UTC container restart, and the orchestrating session relaunched the remaining runs. No `REPORT.md` was written. Independent audit: `research/toy-tests-v4/VERIFY.md`. **Seeds:** 20 main seeds (0–19) per world for `merge_ref`, `lora_replay` and oracle. `merge_noref` has 2–4 seeds per world (incomplete). Pilot seeds were 900–901 (tuning), 920–927 (sanity S3) and 930 (timing). **Verdict:** FAILS under the pre-registered rule, because LoRA+replay was more accurate in every misfit world. Writes no longer grow linearly, which is a real improvement on Test M. But the "≤ 1 write per decade" criterion was missed in worlds a and b5. The control world (d) "failure" was an **invalid control**, per VERIFY's re-check. It is not evidence against the Ville bound.

---

## 1. Why we ran it

**The problem.** This test targets the project's oldest failure class: structure writes that keep piling up when the base model is wrong ("misfit"). TESTS.md names ledger rows M6/M7/M16 and R1-N3. The phase-1 Test M had shown, for the v3.6 core, that writes grew roughly linearly under misfit, about 205 per hour in the mixed world with growth exponent 1.06 (`notes/all-tests-ledger-2026-10-28.md` §1). [inferred] In terms of the problem list, unchecked writes are a stability–plasticity failure (P3: "learning systems can degrade even when nothing in the world has changed"). They also threaten P2 (protection of old skills) and the compute budget.

**The design idea under test.**
- The merged core admits a structure move m only if a *held-out*, *channel-averaged* e-process E_m exceeds exp(L(m)), where L(m) is a description-length price (RECOMMENDATION Q3).
- The core also adds **copy-on-write refinement**: an admitted direction can be refined into a new copy, and the parent is kept (RECOMMENDATION Q4).
- TESTS.md warns that refinement "re-opens an age route": a new way for writes to keep happening as the system ages.

**Why decisive.** TESTS.md calls T-CHURN "a close fourth" that "should run alongside, because it is cheap and guards the oldest failure class." Its primary measure is held-out accuracy against an oracle. Write counts alone are explicitly not enough (ledger R2-N2).

**Jargon.**
- An *e-process* is a running product of likelihood ratios ("e-values"). Under the null it is a nonnegative supermartingale with mean ≤ 1.
- *Ville's inequality* says such a process ever exceeds 1/α with probability at most α. The core's version is A's Lemma 2. It holds **if the current model is the true conditional law** (VERIFY §1.1).
- *Nats* are natural-log units.
- *LoRA+replay* is one continually trained low-rank adapter, with each new minibatch mixed with frames drawn from a fixed-size reservoir of past frames.
- A *decade of t* is a tenfold increase in frames, here 10^5 to 10^6.

## 2. What we did

**Worlds** (each run 10^6 frames, no shortening; PROTOCOL §2.2). The same toy base as T-COMPOSE was used, with a planted rank-1 change at W1 (Frobenius 12).
- **(a)** iid Student-t noise with ν = 2.5. The model's ν grid {1, 2, 3, 5, 10, 30, ∞} does not contain 2.5, so the noise shape is misspecified.
- **(b5), (b9)** AR(1) t(ν = 4) noise with ρ = 0.5 and 0.9. Frames are not independent, but the model assumes they are.
- **(c)** correlated channels. Secondary, not run.
- **(d)** the control: iid t(ν = 4), with the true direction already in the dictionary at the start. Meant to be "a true model in the span".

**Arms.**

| Arm | What it is | Tuning |
|---|---|---|
| `merge_ref` | Merged core with copy-on-write refinement | 9 configs (fit_steps × fit_lr) on worlds a and e0, pilot seeds 900–901, 10^5 frames. Best (100, 0.1) |
| `merge_noref` | Same, without refinement moves | same config |
| LoRA+replay | Rank-4 LoRA at all four sites. Each step uses 32 new frames plus 32 from a reservoir **capped at 20 000 frames**, the same cap as the merge's fit-record reservoir. ν chosen from its own residuals on the same grid | 16 configs on worlds a, **b9** and e0, pilot seed 900, 10^5 frames. Best Adam 1e-3 with decay |
| Oracle | True change and true noise law, scored with the best-in-class (ν, κ). A stronger ceiling than the model's own grid | — |

A-only, B-only and the declared fallback `merge_ref_block` were second priority or conditional.

**Pre-registered rule** (TESTS.md, copied into PROTOCOL §1, with the operationalisation in PROTOCOL §2.2):
- **Holds** in (a)–(c) if fresh structure writes in (10^5, 10^6] average ≤ 1 per decade *and* the mean relative gap to the oracle is ≤ 2 %.
- **Fails** if writes grow linearly, or if accuracy is significantly worse than LoRA+replay (one-sided paired sign-flip test, p < 0.05). "Linear" means mean writes ≥ 3 and a late/early write-rate ratio ≥ 0.5; a log law gives about 0.32.
- **World (d):** writes after the first decade must be within the Ville bound. The tester read the bound as "≤ 1 expected write per seed" and set a one-sided exact Poisson test of total ~ Poisson(20) over 20 seeds.
- **World (b):** if the plain merge does not hold, the pre-declared block-e-value fallback must be run.

**Deviations and gaps:**
- LoRA+replay was tuned at a 10^5-frame horizon, not 10^6 (PROTOCOL §7.4).
- The oracle ceiling choice (best-in-class rather than grid ν) was fixed before the main runs.
- `merge_noref` was required but has only 2–4 seeds per world.
- `merge_ref_block` was required by the protocol's own (b) clause and was never run (VERIFY §3).
- World (c), A-only and B-only were not run.

## 3. What we got

**Accuracy and writes at 10^6 frames** (`research/toy-tests-v4/compose-churn/results/tables.md`, T-CHURN table; relative gap to oracle, mean and 95 % CI):

| World | Arm | n | Rel. gap at 10^6 | Fresh writes per decade | Refine writes per decade | Late/early rate | Directions K at 10^6 |
|---|---|---|---|---|---|---|---|
| a | LoRA+replay | 20 | 0.0018 [0.0017, 0.0019] | – | – | – | – |
| a | merge_ref | 20 | 0.0022 [0.0020, 0.0023] | 1.30 [0.65, 1.95] | 1.70 [1.09, 2.31] | 0.41 | 4.2 |
| a | merge_noref | 3 | 0.0023 | 3.00 | 0 | 0.63 | 6.0 |
| b5 | LoRA+replay | 20 | 0.0002 [0.0002, 0.0003] | – | – | – | – |
| b5 | merge_ref | 20 | 0.0016 [0.0014, 0.0018] | 1.55 [1.01, 2.09] | 1.75 [1.27, 2.23] | 0.49 | 4.3 |
| b5 | merge_noref | 3 | 0.0019 | 4.67 | 0 | 0.79 | 7.0 |
| b9 | LoRA+replay | 20 | 0.0007 [0.0007, 0.0007] | – | – | – | – |
| b9 | merge_ref | 20 | 0.0072 [0.0062, 0.0081] | 0.85 [0.30, 1.40] | 0.25 [0.04, 0.46] | 0.15 | 4.0 |
| b9 | merge_noref | 2 | 0.0096 | 1.00 | 0 | 0.32 | 5.0 |
| d | LoRA+replay | 20 | 0.0020 [0.0020, 0.0021] | – | – | – | – |
| d | merge_ref | 20 | 0.0011 [0.0011, 0.0011] | 1.25 [1.04, 1.46] | 0.50 [0.18, 0.82] | 0.24 | 2.5 |

**Per-world verdicts** (`compose-churn/results/summary.json`, `churn_verdicts`):

| World | Slope ≤ 1/decade? | Gap ≤ 2 %? | Merge − LoRA gap (pp), 95 % CI, p(worse) | Linear? | Verdict |
|---|---|---|---|---|---|
| a | no (1.30; upper 1.83) | yes (0.22 %) | +0.036 (0.018, 0.054), p = 0.0002 | no | **FAILS** |
| b5 | no (1.55; upper 1.99) | yes (0.16 %) | +0.139 (0.118, 0.160), p < 1e-4 | no | **FAILS** |
| b9 | yes (0.85; upper 1.31) | yes (0.72 %) | +0.647 (0.549, 0.744), p < 1e-4 | no | **FAILS** |
| d | – | – | – | – | **FAILS** by the protocol: 40 writes after frame 10 over 20 seeds against 20 allowed, Poisson p = 5.3e-5. 35 of them are after 10^5 |

For `merge_noref` (n = 2–4), the analysis flags "linear growth" in a and b5 (late/early 0.63 and 0.79). With so few seeds, no decision attaches to it. Note: the folder holds 4 `merge_noref` files for a and b5 (seed 3 finished at 06:00–06:01), but the 06:18 table counts n = 3. I did not find why.

**Cumulative fresh writes, mean** (`tables.md`): world a, merge_ref: 0 at 10^3, 1.65 at 10^4, 2.85 at 10^5, 3.20 at 3×10^5, 4.15 at 10^6. b5 and b9 look similar (4.35 and 3.95 at 10^6).

**Where the writes went.** I counted write events in the 20 `merge_ref` files per world under `compose-churn/results/churn/` (my count). Sites: 0 = lens, 1 = W1 (the true change site), 2 = W2, 3 = scl (the log-scale head).
- World a: 83 fresh writes in all (W1 30, lens 25, scl 19, W2 9) plus 35 refinements. After 10^5: 26 fresh and 34 refine. Selected ν was 3 in every seed.
- World d: 40 writes. 25 fresh at scl and 10 refine at scl, all after 10^5. Plus 5 early fresh lens writes. Selected ν was 5 in every seed.

**Accuracy trajectory** (`tables.md`): in world a the merge was *ahead* of LoRA+replay early on (gap 0.0062 vs 0.0195 at 10^4). LoRA+replay overtook it by 10^5 (0.0020 vs 0.0034). The same pattern holds in b5 and b9.

**VERIFY's re-check of world (d)** (VERIFY §1):
- World d uses ν = 4, but the model's ν grid (the core's own grid N25: 1, 2, 3, 5, 10, 30, ∞) has no 4. The model picked ν = 5 in all channels. So **the true law was not in the model class**, and (d) was not the "true model in the span" control TESTS.md asked for.
- On the 18 seeds available then, 30 of 35 writes were at the scale head, with log E ≈ 2 × L(m): 462–632 nats of evidence against prices of 240–361. VERIFY calls this "a real predictive gain, not chance". A scale-head direction partly compensates for the t5-versus-t4 shape mismatch.
- **Re-run:** VERIFY re-ran seed 0 for 350 000 frames with 4 added to the grid (world "d4"). ν = 4 was selected from frame 4 000 on, with **0 writes**, against a site-3 write at 300 k in world d. The gap to the oracle was 2.3e-6.
- The pre-main sanity check S3 (8 seeds × 200 k frames in d4) also had 0 writes (`results/sanity/S3_falsewrite_d4_a.json`, `_b.json`). The closest approach of log E to L(m) was 32 nats below the threshold.
- The 5 early lens writes came from evidence gathered while the coordinates and ν were still converging. Under that condition Lemma 2's premise ("current law = true law") does not hold either.
- VERIFY's conclusion: (d) "does not test the Ville bound". It shows the merge writes scale-head structure to absorb noise-shape misfit. d4 would probably hold, but that rests on 9 short runs, not on 20 seeds × 10^6 frames.

**Two other findings from VERIFY about (d):**
1. The analysis counted refinement writes, but RECOMMENDATION R1-N2 says copy-on-write refinements must **not** count as false writes. On the final 20 seeds, dropping the 10 refine writes leaves 30 against 20 expected. My Poisson tail is p ≈ 0.022, still a FAIL by the protocol.
2. The protocol's "≤ 1 expected write per seed" is far *more lenient* than the core's own per-candidate bound e^{−L(m)}, with L of 80–360 nats. Under the strict reading any single null write would be a violation.

**LoRA+replay fairness** (VERIFY §3):
- It used the same stream, no labels and no oracle information.
- Its held-out set was never trained on.
- Its buffer was bounded at the same 20 000 frames as the merge's reservoir.
- It had rank 4 at all four sites, more expressive than the merge's rank-1 directions. TESTS.md fixes rank only for LoRA-TTT, so this was allowed.
- It saw world b9 in tuning, while the merge did not. VERIFY calls this a mild asymmetry.

## 4. What we inferred

**What it says:**
- **Accuracy:** both merge arms stay within 0.16–0.72 % of a strong oracle in a, b5 and b9, so the "within 2 %" half passes. But a plain, bounded LoRA+replay is significantly more accurate in all three misfit worlds. That triggers the pre-registered fail clause. [checked: `summary.json`]
  - In world a the margin is tiny (0.036 percentage points), but the verdict is mechanical and correct by the rule (VERIFY §3).
  - In b9, the strongly autocorrelated world, the merge is about 10× further from the oracle than LoRA+replay (0.72 % vs 0.07 %).
- **Writes:**
  - Linear growth was not declared in any world for `merge_ref`. Late/early rate ratios were 0.41, 0.49 and 0.15, all below the 0.5 cut-off and closer to a log law (0.32) than a linear one (1.0). That is a real improvement on Test M. [checked]
  - But the pre-registered slope criterion (≤ 1 fresh write per decade) was **met only in b9**. In a and b5 the means were 1.30 and 1.55, and refinement adds another 1.70–1.75 per decade. So "writes level off" is accurate only in the weak sense of "sub-linear, consistent with roughly logarithmic growth". The project summaries ("writes did level off, at about 4 per million frames"; "≈1 per tenfold") round this favourably. [checked: `tables.md`; the judgement about the summaries is mine]
  - Even without the LoRA comparison, a and b5 would have been INCONCLUSIVE, not HOLDS. [inferred from the rule]
- **Refinement:** with refinement (`merge_ref`) there were fewer fresh writes and fewer directions at 10^6 than without (`merge_noref`): K 4.2 vs 6.0 in a. The no-refinement arm looked more linear. With 3 seeds this is a hint, not a result. [checked numbers; inference weak]
- **The control:** world (d) does not show that the Ville bound is broken. It was misspecified. [checked by VERIFY's re-run]

**What it does NOT say:**
- The core's false-write safety bound was **not tested**. The proper control (d4, 20 seeds × 10^6 frames) was never run as a main run, as far as I could find.
- It does not show the merge to be inaccurate in absolute terms. The gaps are fractions of a percent.
- Merge versus LoRA in world d is not a fair accuracy comparison. The merge started with the true direction in its dictionary, which explains why it was more accurate there (0.0011 vs 0.0020). [inferred]
- The write numbers are not directly comparable with Test M's "205 per hour": the units differ (per decade of frames versus per hour) and the worlds differ. I did not find a conversion in the files I read.

## 5. Why we think it happened

- **Why the merge loses accuracy to LoRA+replay** [inferred]:
  - LoRA+replay keeps adjusting a rank-16 adapter (rank 4 × 4 sites) by gradient on fresh and replayed frames. It can soak up misfit anywhere, a little at a time.
  - The merge can only move coordinates on a few fixed rank-1 directions. It adds new structure only when held-out evidence clears a large description-length price (L(m) values of roughly 80–360 nats in world d per VERIFY §1.1, and 180–870 nats in the S3 sanity files). So residual misfit that is real but spread thin is never captured.
  - This is the price of "write rarely, only on evidence". The early-life trajectory fits this picture: the merge is ahead at 10^4 frames, where few parameters help, and behind from 10^5 on, where flexible capacity pays.
- **Why b9 is the worst world** [inferred]: with ρ = 0.9 the effective sample size is about 1/10 of the frame count. The merge's per-frame evidence and coordinate updates treat frames as independent, so it is overconfident. LoRA+replay's minibatch SGD is less sensitive to this. The block-e-value fallback, which would address the evidence side, was not run. VERIFY judged it unlikely to rescue accuracy, because it changes only admission. [inferred by VERIFY]
- **Why writes are sub-linear now** [inferred]: every admission must win against a held-out baseline, and the price grows with the precision needed. Once the main misfit is absorbed by a few directions (including scale-head ones), later candidates have little evidence to gain. In Test M the old core had nothing like this re-check, so cascades of writes could feed on one another (memory v31 notes "recall churn + cascades (63 %)" for Test M).
- **Why world d wrote** [checked by VERIFY]: ν = 4 is off the grid, so a scale-head direction gives a real predictive gain. VERIFY's d4 re-run removed this cause and the writes vanished.
- **The early-life writes** [checked by VERIFY, mechanism partly inferred]: evidence piled up while the coordinates and ν were still converging (frames 1 000–10 000). It was kept across refits, and later cashed in when a candidate's L(m) shrank. VERIFY: Lemma 2 "gives no protection against writes driven by inference lag, because their null ... never holds while learning." This is a genuine limitation of the bound's scope, not a coding bug.
- **The settle gate** (VERIFY §1.2): shadow evidence was collected only on chunks where the top hypothesis had prior weight ≥ 0.9. That gate is not in the core and is not listed in `IMPLEMENTATION.md`. It keeps validity but loses evidence. Its effect on worlds a and b was not checked (VERIFY §8).

## 6. Caveats and what could be wrong

- Required items are missing: `merge_noref` (2–4 seeds instead of 20) and `merge_ref_block` (not run). Neither can turn the accuracy failure into a pass unless the block fallback moved accuracy by more than 0.14–0.65 pp, which VERIFY judged unlikely. [inferred]
- LoRA+replay was tuned at 10^5 frames, and it saw an AR world in tuning while the merge did not. The asymmetries cut against the merge, but only mildly.
- The oracle is deliberately strong: the true noise law with best-in-class ν. This makes all relative gaps larger than with the model's own grid, by 0.2–1.6 % in AR worlds per PROTOCOL §7.5. It affects all arms equally.
- The world (d) count includes refinement writes, against R1-N2. It still fails the protocol test without them, but (d) is invalid anyway.
- Small toy and CPU only, with one planted change. The merge is a partial implementation of the core: no ψ, settle gate, plug-in baseline, best single candidate (VERIFY §1.2).
- Thread-count changes after the restart altered floating-point summation order. Bit-for-bit equality was not checked (VERIFY §6).
- No `REPORT.md` exists. The numbers come from `analysis.py` outputs, which VERIFY re-ran and reproduced.

## 7. What it changed in the project afterwards

- The retrospective (`notes/retrospective-2026-10-19.md` §2–§3) records two things. First, writes now level off "≈1 per tenfold of frames ... That is a real fix", while LoRA with replay is slightly more accurate. Second, "the false-alarm control world was invalid, so the core's safety bound is untested rather than broken." It lists "writing only on held-out evidence stops runaway bad writes" as the project's **one real mechanism win**. Note that the slope bar was strictly met only in b9.
- The ledger's closing list of principles that held up includes "re-check after each write" (`notes/all-tests-ledger-2026-10-28.md`, "The one-line pattern"). It also notes that no principle became a better *method*.
- The test added LoRA+replay to the "honest map of the competition": COIN, replay and plain LoRA are strong, and any breakthrough has to beat them (retrospective §3).
- VERIFY proposed a proper d4 control (20 seeds × 10^6 frames, about 2 worker-hours). Memory notes this as "awaiting Lucky" (memory v31, line 76). I found no record in the notes or memory I read that it was run.
