# 05 T-P14 (practice prior Γ_s): does a direction prior learned from practice make new change families faster to learn than no prior, Reptile, or plain LoRA-TTT?

**Date run:** 2026-10-18 to 2026-10-19 (UTC). Sanity checks and tuning ran on 10-18 between 16:20 and 22:46. The main runs went from 22:46 on 10-18 to 05:22 on 10-19, interrupted by the 01:45 container restart. The headline analysis was written at 05:22:56. **Folder:** /mnt/project-files/research/toy-tests-v4/p14/ **Built by:** "Agent 2", a test agent working under `research/toy-tests-v4/BRIEF.md` (project memory says it was a Sonnet 5.5 agent). After the restart, its resumable `driver.py` was relaunched by hand. No `REPORT.md` exists in the folder. Independent audit: `research/toy-tests-v4/VERIFY.md` §4. Follow-up design: `research/toy-tests-v4/p14-ability/DESIGN.md`. **Seeds:** 16 main seeds (100–115). Tuning used pilot seeds 901–903, sanity used 904–907. The sensitivity worlds were started and then stopped at Lucky's request at 05:24 on 10-19, with no result files written. **Verdict:** FAILS under the pre-registered rule. With Γ_s on, the merge needed **1.30×** the frames it needed with Γ_s off (worse, not better), and 1.41× Reptile's. Narrow practice beat varied practice. The merge got slower the more it practised. Even a planted, true Γ_s gave no gain. VERIFY found the result robust.

---

## 1. Why we ran it

**The problem.** P14 is transferable, task-agnostic motor memory: practice on anything should make new motor tasks faster to learn (`notes/problems.md` §14). In the core-v3.6 red team, the absence of any principle for P14 was a FATAL finding (A-01; memory v31, line 44). By core-v4, the only mechanism for P14 in the merged core was the life prior's direction prior **Γ_s** (RECOMMENDATION §2.2, O8: "It is the P14 carrier").

**What Γ_s is.** In B's core, each new direction has two factors (a, b) at a site s. Γ_s = (Γ^a_s, Γ^b_s) is the prior covariance of those factors: a low-rank part plus an isotropic part. It is fitted from the directions admitted so far, in admission order, as a mixture with a "birth" (uninformed) component whose weight is at least ε_open (`research/architecture/core-v4/B/CORE.md`, lines 389–398). Γ_s acts in two places:
1. it is the prior of the within-episode scratch correction ψ (B/CORE.md, line 304);
2. it sets the code length that prices new directions (line 487).

The hope: after practising many families of change, Γ_s concentrates on the subspaces where changes tend to lie. New changes would then be found from fewer frames, and admitted more cheaply.

**Why decisive.** TESTS.md Bet 4: "A-01 was a FATAL ledger row; Γ_s is the only mechanism for it; toy P showed no theory-predicted gain and a weak competitor." RECOMMENDATION §8 names it as one of the three bets that would make the merge a breakthrough.

**History.** The phase-1 Test P had found the practice prior 2.5× faster than an isotropic prior, but the gain was generic: unrelated families were covered almost as much, and varied practice was no better than narrow (`notes/all-tests-ledger-2026-10-28.md` §2). This test was meant to compare against a stronger competitor (Reptile) under fair tuning.

**An important change of understanding happened during this test.** On 2026-10-18 Lucky clarified P14: what transfers is motor *ability*, graded by overlap and never negative, "not a parameter starting point" (`notes/problems.md` §14; retrospective §1). The retrospective says this means "every earlier design and the running P14 test had modelled the wrong thing". So this test measures the bet as TESTS.md wrote it (a parameter/direction prior). It is not a test of P14 as now defined.

**Jargon.**
- *Reptile* is a first-order meta-learning method. It learns an initialisation from which a few gradient steps fit any practised task well. Here, Reptile-init meta-learns the starting LoRA for test-time training.
- *LoRA-TTT* trains a fresh low-rank adapter online on the new task.
- *Frames to 90 %* is the number of frames until an arm reaches, and keeps, 90 % of the oracle's held-out gain.

## 2. What we did

**World** (`p14/IMPLEMENTATION.md` §1):
- A frozen random two-layer base with 64 tokens and latent 32, and t(ν = 4) noise.
- Two change sites: F, a late output projection acting on hidden features, and G, a lens-like map acting on the input.
- 20 family generators. Each has a home site and a 4-dim output and input subspace. A member is a rank-2 change inside its family's subspaces.
- **Choice C1, "the most consequential choice in the test":** TESTS.md gives no structure shared *across* families, so the tester put all families' subspaces inside a shared per-site "ambient" subspace, plus 8 % leak. Without any cross-family sharing, no learner can gain from practice. The headline world `mid` has ambient dimensions 12/12/8. Sensitivity worlds (tight, loose, none, and other energies) were planned but stopped before producing results.

**Life and test.**
- Stage 1 practises 10 families. **Varied** practice is 5 members × 300 frames per family. **Narrow** practice is 1 member × 1 500 frames per family. Both get equal practice time.
- Snapshots are taken after m = 2, 5 and 10 families (nested).
- The test runs on the 10 unpractised families, 2 members each (20 members), each from the end-of-life snapshot, for up to 1 024 frames.
- Held-out gain is measured on 48 frames per member.
- Frames-to-90 % must be sustained over two checkpoints and is log-interpolated. Members that never get there are censored at 2 048.
- The per-seed statistic is the median over the 20 members (PROTOCOL §2, O2–O5).

**Arms** (IMPLEMENTATION §2):

| Arm | What it is |
|---|---|
| Merge, Γ_s on | Merged core, with Γ_s fitted from admitted directions and used as the ψ prior and in code lengths |
| Merge, Γ_s off | Identical code, with Γ_s fixed at the birth component for the whole life |
| diag: Γ-on life, birth prior at test | The same life as "on", but Γ_s switched off at test only |
| diag: planted Γ_s | Γ_s fitted from the **true** rank-1 components of the practised members, used at test on the Γ-off life. An upper bound on what a learned Γ_s could give |
| Reptile-init | LoRA-TTT with a Reptile-meta-learned (A0, B0) over the same stage-1 data |
| LoRA-TTT (scratch) | Rank 2 or 4 at both sites, tuned optimiser, no stage 1 |
| Oracle | Told the true change. Fraction 1 by definition |
| Oracle-subspace (reference) | Knows the true family subspaces and home site, and fits only the 4×4 core |

**Tuning.** Every tuned arm got the same budget: 20 trials (the default, 9 random draws, 10 local perturbations) × 3 pilot seeds × 10 test members. A trial's score was the geometric mean of frames-to-90 %, and crashed or censored members counted as 2 048. The selection rule was fixed at 17:06 on 10-18, after the first merge trials but before any LoRA or Reptile trial (PROTOCOL §4, §8). Selected tuning scores: merge_on 144.7, merge_off 136.4, lora_ttt 135.3, reptile 143.7.

**Pre-registered rule** (TESTS.md, copied verbatim; statistics in PROTOCOL §3; paired t on log of the per-seed statistic, df = 15):
- **A (the fail side):** Γ_s-on faster than Γ_s-off at one-sided α = 0.05. If A is not met, the verdict is **Fails** ("P14 is unmet and the core must say so").
- **B1:** varied Γ_s-on ≤ 0.6× Γ_s-off at α = 0.01.
- **B2:** varied Γ_s-on ≤ 0.8× Reptile at α = 0.01.
- **C:** "narrow < varied in gain": the early-curve gain is lower for narrow than for varied, at α = 0.05.
- **Holds** only if A, B1, B2 and C all hold. If Reptile matches Γ_s-on, the claim drops to "parity".

**Deviations and incidents:**
1. Two bugs were fixed before tuning: a PPCA fitting bug found by sanity S4, and an early-frame bug (PROTOCOL §8).
2. **Singular-matrix crash.** Main runs crashed with `LinAlgError` in 8 Γ-off seeds (101–104, 106–108, 111) and 1 Γ-on seed (109). The orchestrator added a least-squares fallback that runs only on failure, and those runs were re-run (PROTOCOL §8, 01:57 on 10-19).
3. Crashed tuning trials were scored 2 048. That hit both merge arms; the PROTOCOL log mentions only one of them (VERIFY §4).
4. On-vs-off mixes Γ_s with configuration: the two arms were tuned separately and chose different τ0, c0 and seed_scale (VERIFY §4).

## 3. What we got

**Headline table, world `mid`** (`research/toy-tests-v4/p14/results/tables.md`; geometric mean over 16 seeds of the per-seed median frames-to-90 %, with 95 % t interval; lower is better):

| Arm | Varied m=2 | Varied m=5 | Varied m=10 | Narrow m=2 | Narrow m=5 | Narrow m=10 |
|---|---|---|---|---|---|---|
| Merge Γ_s on | 117 [95, 145] | 156 [108, 226] | **182 [147, 225]** | 98 | 107 | 123 |
| Merge Γ_s off | 95 [90, 101] | 113 [104, 123] | **140 [127, 153]** | 87 | 95 | 106 |
| diag: Γ-on life, birth prior at test | 112 | 133 | 162 [152, 172] | 99 | 110 | 122 |
| diag: Γ-off life, planted Γ_s at test | 100 | 117 | 144 [134, 155] | 95 | 102 | 113 |
| Reptile-init | 186 | 149 | **129 [121, 138]** | 175 | 171 | 138 |
| LoRA-TTT (scratch) | 116 | 116 | **116 [107, 126]** | 116 | 116 | 116 |

**Decision statistics** (`p14/results/analysis_mid.json`):

| Test | Ratio (geometric mean) | 95 % CI | p | Met? |
|---|---|---|---|---|
| A: on vs off | 1.302 | [1.081, 1.569] | p(on faster) = 0.996 | **No → FAILS** |
| B1: on ≤ 0.6× off | 1.302 | upper 99 % 1.635 | ≈ 1.0 | No |
| B2: on ≤ 0.8× Reptile | 1.407 | [1.170, 1.693] | 0.9993 | No |
| C: narrow < varied in gain | narrow − varied = **+0.065** | [0.010, 0.120] | 0.988 | No (narrow was *better*) |

Verdict: "FAILS". `reptile_parity: true`.

**Slowing with practice** (`tables.md`, per-seed OLS slope of frames-to-90 % against the number of practised families m; frames per added family):

| Arm | Varied | Narrow |
|---|---|---|
| Merge Γ_s on | +7.1 [6.2, 8.1] | +3.2 [2.2, 4.2] |
| Merge Γ_s off | +5.7 [4.4, 6.9] | +2.5 [1.8, 3.2] |
| Reptile-init | **−6.9** [−8.2, −5.5] | −4.9 [−7.2, −2.7] |

Only Reptile got faster with more practice. Both merge arms got slower, with or without Γ_s.

**Early-curve gain** (mean fraction of the oracle's gain over checkpoints 16–256 frames, m = 10; `tables.md`):

| Arm | Varied | Narrow |
|---|---|---|
| Merge Γ_s on | 0.754 | 0.818 |
| Merge Γ_s off | 0.798 | 0.816 |
| Reptile-init | 0.653 | 0.630 |
| LoRA-TTT | 0.658 | 0.658 |

So on the early curve the merge arms lead the LoRA arms. On the pre-registered frames-to-90 % measure they trail.

**Cleaner contrasts and robustness checks.** I recomputed these from the 16 per-seed files in `p14/results/main/mid/`, using `analyze.py`'s rule (NaN = censored at 2 048, median over members, paired t on log). VERIFY §4 did the same with 15 seeds.

| Contrast (varied, m = 10) | Ratio | 95 % CI | Source |
|---|---|---|---|
| on / off (headline) | 1.302 | [1.081, 1.569] | `analysis_mid.json` |
| on / off, without seed 109 | 1.202 | [1.107, 1.306] | my computation (VERIFY, n = 14: 1.18 [1.09, 1.27]) |
| Γ_s on at test vs off at test, **same life, same config** | 1.124 | [0.894, 1.414] | my computation from `to90` vs `to90_testoff` (VERIFY n = 15: 1.12 [0.87, 1.43]) |
| the same, without seed 109 | 1.011 | [0.974, 1.049] | my computation |
| **planted true Γ_s vs Γ_s off** (same Γ-off life) | 1.034 | [0.959, 1.114] | my computation from `to90_planted` (VERIFY: 1.03 [0.95, 1.12]) |
| Γ_s off / Reptile | 1.081 | [1.017, 1.149] | my computation |
| LoRA-TTT / Γ_s off | 0.831 | [0.741, 0.933] | my computation |
| LoRA-TTT / Reptile | 0.899 | [0.815, 0.991] | my computation |
| Oracle-subspace, geometric mean | **5.45 frames** | – | my computation from `oracle_sub__s*__varied__m0.json`; VERIFY "about 5.5" |

**Seed 109.** The Γ-on arm took 767 frames: 9 of 20 members were censored. Its dictionary held 36 directions, against 74–98 in the other seeds. The same life with Γ_s off at test took 138 frames (`merge_on__s109__varied__m10.json`). Seed 109 was the one Γ-on seed that crashed before the least-squares fix and was re-run with it. VERIFY could not tell whether the fallback caused the odd life (VERIFY §4).

**Dictionary growth** (mean number of directions K, my computation from the result files): Γ-on varied 17.6 / 43.2 / 79.0 at m = 2 / 5 / 10; narrow 6.6 / 17.4 / 34.4. Γ-off is almost identical (16.7 / 41.5 / 78.8 and 6.9 / 16.8 / 33.2).

**Sanity checks** (`p14/results/sanity_s*.json`):
- **S1:** the oracle-subspace reaches about 1.0 of the true gain, in 2.7–7.2 frames to 90 %.
- **S2:** the merge recovers planted changes (final fraction 0.96–0.997, 89–180 frames). LoRA reaches 0.93–0.98 in 97–271 frames.
- **S3:** null control, 60 streams: 0 admissions, mean E at frame 320 = 1.03, max log E − L = −75.7.
- **S4 (planted Γ_s check):** fitting Γ_s from *planted true* directions recovers the planted subspace well above chance. For one family, 0.90–0.98 of the subspace is in span against chance of 0.016–0.125. For the ambient subspace from 30–70 directions, 0.67–0.84 is in span against chance of 0.03–0.375.

  S4 checks the fitting procedure on true directions. It does not check that Γ_s fitted from the merge's *own admitted* directions recovers the subspace. I found no file that measures that directly.

## 4. What we inferred

**What it says:**
- **The bet fails cleanly.** Γ_s-on was slower than Γ_s-off (1.30×, significant in the wrong direction). It was slower than Reptile (1.41×). Narrow practice gave a higher early gain than varied (+0.065), the opposite of the hoped-for direction. [checked]
- **Γ_s itself has about zero effect at test in this implementation.** Same life and config, Γ_s on versus off at test: 1.12× with seed 109, 1.01× without. A *planted, true* Γ_s gives 1.03×. So it is not just that Γ_s was learned badly: even a perfect one does not help the merge's test-time pipeline here. [checked numbers; the reading is VERIFY's and mine]
- **The on/off gap therefore comes mostly from the life and configuration.** Without seed 109, on/off is 1.20×, while "Γ at test" is 1.01×. The rest (the "Γ-on life, birth prior at test" row: 162 vs 140) reflects a life grown under Γ-on code lengths, together with a different tuned configuration. The data cannot separate those two. [inferred from the table]
- **Loss of plasticity.** Both merge arms get slower as practice grows (Γ-off 95 → 113 → 140 frames). Only Reptile improves (186 → 149 → 129). This is a P3 (stability–plasticity) symptom: "the robot learns as fast late in its life as early on" is violated. [checked]
- **The plain competitor wins.** LoRA-TTT from scratch, with no practice at all, was the fastest learner on the primary measure (116 frames). It was faster than Γ-off (0.83×) and slightly faster than Reptile (0.90×). In this toy, practice as implemented by any arm bought little or nothing over a well-tuned fresh learner. Reptile's practice benefit only brings it near LoRA-TTT at m = 10. [checked; my computation of the paired ratios]
- **A large headroom exists.** The oracle-subspace reaches 90 % in about 5.5 frames, against 116–182 frames for every learner. Knowing *where* a new family's change lies is enormously valuable in this world. No learner got that knowledge from practice. [checked]

**What it does NOT say:**
- It does not test P14 as Lucky defined it on 10-18 (ability transfer graded by overlap, never negative). The families here are subspaces of parameter change, and the test has no notion of ability overlap or a zero-overlap control (`p14-ability/DESIGN.md` §1).
- It does not show that no direction prior could help. The planted Γ_s here encodes the practised families' directions, i.e. the shared ambient subspace, not the test family's own 4-dim subspace. And the merge's test-time use of a prior may be the bottleneck (VERIFY §4). [inferred]
- The sensitivity worlds (tight / loose / none sharing, other energies) were never run. The verdict is conditional on the `mid` sharing level, which IMPLEMENTATION C1 says has a "large and positive" effect on the merge's chances.

## 5. Why we think it happened

- **The prior is not used where it matters** [inferred, with VERIFY]: the merge's test-time path runs through hypothesis refresh, ψ (scratch) fitting, candidate generation on a schedule (t = 10, 20, 40, ... × sched_f) and evidence-based admission. A sharper ψ prior changes the scratch estimate, but frames-to-90 % may be dominated by how quickly coordinates and admitted directions settle, not by the ψ prior. The planted-Γ_s null (1.03×) supports "the bottleneck is elsewhere". The exact step that limits speed was not isolated. [guess as to which step]
- **Why the merge slows with practice** [guess, consistent with the data]: the dictionary grows from about 17 to about 79 directions, and the number of stored contexts grows with every visit (IMPLEMENTATION S12: every visit is stored). With more stored contexts and directions:
  - the refresh has more wrong candidates to rule out;
  - the shared prior mass for stored contexts (0.4) is spread over more contexts;
  - coordinate inference has more dimensions.

  The narrow life has fewer visits (10 against 50) and about half the directions (34 vs 79), and it is faster at every m. That fits "store size slows test-time learning", but no ablation separates the store size from other differences.
- **Why narrow beat varied** [inferred]: by the same store-size mechanism, narrow practice leaves a smaller, less cluttered memory. Varied practice did not create a more useful prior, so the clutter cost dominates.
- **Why Γ_s-on lives are worse** [guess]: Γ_s changes code lengths, and so which candidates get admitted during life. It may admit more directions specialised to practised families, which then become distractors for new families. The K counts are similar on and off (79.0 vs 78.8), so if this happens it is about *which* directions, not how many. Not checked.
- **Seed 109** [not established]: an odd life with half the usual directions and 9 of 20 members censored. Whether the least-squares fallback contributed is unknown (VERIFY §8).
- **Why LoRA-TTT is fast here** [inferred]: the world is linear in the change at both sites, and a rank-2/4 LoRA with Adam fits a rank-2 change efficiently within about 100 frames. Reptile's initialisation only helps if the meta-learned starting point is closer than zero to new families' changes, and with new families it mostly is not.

## 6. Caveats and what could be wrong

- **The test modelled the wrong P14**, as the project later concluded. A parameter or direction prior is not transfer of motor ability. The negative result is about Γ_s as a P14 mechanism, not about P14 itself.
- Only the `mid` world was run. A world with tighter sharing might favour Γ_s more (C1), though even a planted Γ_s gave no gain in `mid`.
- On versus off differ in tuned configuration as well as Γ_s. The cleaner same-life contrasts (test-off, planted) address this and agree.
- The singular-matrix fallback was added mid-run. VERIFY found no sign that it favours either arm (seeds re-run with it give on/off 1.19×, the others 1.40×; n small).
- Crashed tuning trials were scored 2 048. VERIFY re-ran one crashed Γ-on configuration on one pilot seed and found it worse (280 vs 153). The other crashed configurations were not re-checked.
- The merge implementation simplifies the core:
  - no sel or aud records;
  - one pool at a time;
  - no pre-life dictionary;
  - Γ_s fitted from unrefined directions;
  - a full-covariance ψ prior instead of the core's diagonal Laplace step (C7), which *favours* Γ_s.

  The only simplification flagged as hurting Γ_s directly is S9 (unrefined directions), and the planted-Γ_s diagnostic bounds it.
- The oracle-subspace and the internal correctness of the Reptile and LoRA-TTT code were not audited (VERIFY §8).
- No `REPORT.md` exists. The numbers come from `analyze.py` outputs, which VERIFY reproduced.

## 7. What it changed in the project afterwards

- The retrospective records: "practice made learning slower (1.30×). Even a planted true life prior gave no gain. The merge also gets slower the more it practises, which is a loss of plasticity (P3)." It also lists "the wrong P14 for the whole project" as a root cause of the stall: "Every core modelled P14 as a parameter prior. The test confirmed that mechanism can't work" (`notes/retrospective-2026-10-19.md` §2, §4). The test showed this mechanism failing in this toy. "Can't work" in general is stronger than one toy shows.
- Under RECOMMENDATION §8, failing Bet 4 means the merge is not a breakthrough.
- **It led to a redesigned test, `research/toy-tests-v4/p14-ability/DESIGN.md`** (2026-10-19, design only, nothing built or run in that folder). Its main points:
  - It measures *behaviour*, not a mechanism: episodes to criterion on new motor tasks after controlled practice histories. Any learner can be put through it, and Reptile is a competitor, not ruled out by definition.
  - The world is a small simulated body with **two independent effectors** (a two-joint arm and a separate pointer), so a true zero-overlap condition exists.
  - Five task-independent **abilities**: timing, precision, control under difficulty, slip handling and body knowledge. Tasks are demand vectors over these abilities.
  - Overlap is declared by a formula, and *checked* by an oracle-factored learner whose gain must rise with overlap and be zero at zero overlap.
  - The conditions cover overlap levels {0, 0.25, 0.5, 0.75, ~1}, conflict tasks (to test "never negative"), varied versus narrow practice at equal episodes, an S1-then-S2 lifelong condition (not overwritten), variations of known tasks, and amount of practice.
  - The pre-registered rule has seven properties: proportional, near zero at o = 0 (δ = 0.05), never negative, varied > narrow (α = 0.01), not overwritten (≥ 0.8×), mainly faster learning, and growing with practice. A breakthrough additionally requires ≤ 0.8× the best competitor's episodes at mid and high overlap.
  - The competitors are a fresh learner, Reptile/MAML, multi-task pretraining with heads, an adapter library, a model-based learner, and the oracle-factored learner.
  - The design states that it expects the merge's Γ_s to fail if run through it, "an inference, not a measurement".
- The retrospective proposed focusing first on the cluster "learning to learn motor skills" (P14, P5, P13, P3) and building this ability test (retrospective §5). The ledger shows it was later realised as **T-P14D** in `research/tests-v6/p14d/`. There, at pilot level, no arm of any kind showed full P14, and the project's own designs lost to generic adapters (`notes/all-tests-ledger-2026-10-28.md` §8–§9). I did not read the T-P14D files themselves for this document.
