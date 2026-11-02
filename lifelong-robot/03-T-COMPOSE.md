# 03 T-COMPOSE: can the merged core predict a never-seen mix of known causes by inference alone, with no write?

**Date run:** 2026-10-18 to 2026-10-19 (UTC). Main compose runs finished by 2026-10-18 18:28; the `merge_exact` diagnostic finished 2026-10-19 05:18; final tables were regenerated 2026-10-19 06:18. **Folder:** /mnt/project-files/research/toy-tests-v4/compose-churn/ **Built by:** "Agent 1", a test agent working under `research/toy-tests-v4/BRIEF.md` (project memory says it was a Sonnet 5.5 agent, at Lucky's request). The agent's session ended after a container restart at 01:45 UTC on 10-19. The orchestrating session relaunched the unfinished runs. No `REPORT.md` was ever written in this folder. The independent audit is `research/toy-tests-v4/VERIFY.md`. **Seeds:** 16 main seeds (0–15). The `merge_exact` diagnostic has 8 seeds (0–7). Pilot seeds were 900–903. **Verdict:** FAILS under the pre-registered rule. The audit says the verdict stands. The interacting-arm half of the failure is probably an artifact of the implementation.

---

## 1. Why we ran it

**The idea being tested.** The merged core-v4 design (B's backbone, with parts taken from A) keeps a shared dictionary of small, fixed "directions": rank-1 weight corrections at named sites of a frozen base predictor. A situation is described by *coordinates* on those directions. B's **Claim C1**, which B calls "the breakthrough claim", is this: when two causes the robot has met separately occur together for the first time, the filter identifies the combination within the first episode, **with no structure write** (`research/architecture/core-v4/B/CORE.md`, line 712). In plain words: if you know what a heavy payload does and what dim lighting does, you should predict "heavy payload in dim lighting" at once, just by setting two coordinates. You should not need to learn anything new.

**Why it was decisive.** `research/architecture/core-v4/AUDIT/TESTS.md` (Bet 2) says: "This is the merge's (B's) claim C1 and the main difference from Snapshot-COIN. If composition needs a write, the merge collapses to a snapshot library with extra machinery." The audit's `RECOMMENDATION.md` §8 lists Bet 2 as one of the three bets that would make the merge a breakthrough. It adds: "If Bet 2 or Bet 4 fails, the merge is an excellent integration (COIN-over-LoRA with honest writes), not a breakthrough." TESTS.md ranks T-COMPOSE second of the "top three" tests, after T-SPAN, which needs a GPU.

**Problems targeted.** TESTS.md does not name P-numbers for this bet. [inferred] The claim bears mainly on P1 (adaptation: working out quickly what changed, without labels) and P13 (sample efficiency). It also touches P4 (transfer), because known causes are supposed to transfer to their combinations for free.

**Why at that moment.** On 2026-10-18 the core-v4 audit had just recommended the merge. Lucky chose "CPU tests first, GPU later" (project memory, 10-18 15:50 UTC). This test, T-CHURN and T-P14 were the three CPU-runnable bets.

**Jargon used below.**
- *COIN* (contextual inference) is a Bayesian model of motor memory. It infers which "context" is active and keeps a separate memory for each. Here, *Snapshot-COIN* means a COIN-style filter over a library of separately fitted LoRA adapters, one per context. It has no shared dictionary.
- *LoRA* is a low-rank adapter added to a frozen weight matrix. *LoRA-TTT* (test-time training) trains a fresh LoRA online on each test episode.
- *Nats* are natural-log units of log-likelihood. "Excess nats" is how much worse an arm predicts than the oracle, summed over the frames scored. Lower is better.
- An *e-process* is a running evidence statistic whose false-alarm rate is controlled by Ville's inequality. The core writes new structure only when this held-out evidence exceeds a description-length price L(m).

## 2. What we did

**World.** A toy frozen base: a random two-layer predictor of width 256, with a 32-d latent, an 8-d action and 64 tokens in 4 channels. It has four sites: lens, W1 ("fwd"), W2 and scl (the log-scale head). Noise is Student-t with ν = 4 (`compose-churn/IMPLEMENTATION.md` §1). The world is a 3×3 factorial:
- factor X ("payload-like") is a rank-1 change at W1, Frobenius norm 8 per level;
- factor Y ("lighting-like") is a rank-1 change at the lens, Frobenius norm 3 per level;
- levels are 0, 1 and 2.

Training covered five cells, (1,0), (2,0), (0,1), (0,2) and (1,1). That was 60 shuffled episodes of 500 frames, i.e. 12 visits per cell, with a "sleep" after each episode. The test cells were the three unseen mixed cells (2,1), (1,2) and (2,2). (0,0) was run as a control only (`compose-churn/PROTOCOL.md` §2.1).

**Interacting arm.** The same cells, but with an extra rank-1 change at W2 (Frobenius 10 per unit, coefficient max(xy−1, 0)) that is present in no training cell. Here the joint effect is *not* the sum of the parts. The core should notice this and *write* a new direction. Zero writes there counts as a failure ("misfit absorbed").

**Arms.**

| Arm | What it is | Tuning |
|---|---|---|
| Merge | The merged core as implemented: directions, context filter with whole-store refresh, birth learner, channel-averaged held-out e-process, description-length price | 9 configs of (fit_steps, fit_lr), tuned on the **churn** pilot worlds a and e0, not on compose. Every compose-relevant setting left at its default (PROTOCOL §6; VERIFY §2) |
| Snapshot-COIN | One rank-4 LoRA snapshot per training cell, fitted offline **on that cell's labelled frames**. A COIN-style filter over {5 snapshots, one novel context}, with the novel LoRA learned online | 12 configs on the compose objective itself, pilot seeds 900–903. Best: Adam 3e-3 |
| `coin_sum` (extra) | COIN, but after 3 frames the novel LoRA starts from the sum of the two most responsible snapshots (task arithmetic) | as COIN |
| LoRA-TTT | A fresh rank-4 LoRA at all four sites per episode, trained online every frame | 24 configs in two rounds. Best: Adam 1e-3, 3 steps per frame |
| base | The frozen base, no adaptation | — |
| `merge_exact` (diagnostic) | The merge **given the two true directions at the start**. It learns only contexts and coordinates. It is not exact inference: memory notes that it was briefly misnamed that way (memory v31, line 75) | seeds 0–7 only |
| Oracle | The true cell perturbation with the true t(ν = 4) law | — |

A-only and B-only variants were second priority and were not run (PROTOCOL §3).

**Metric.** The *excess* is the oracle's log-score minus the arm's prequential log-score (nats) on the first 20 frames of the first test episode of each unseen mixed cell, averaged over the three cells. The same is computed for 200 frames, which is reported but not used in the decision. Writes were counted over the 12 test episodes per cell.

**Pre-registered rule** (copied word for word from TESTS.md into PROTOCOL §1):
- **Holds** if the merge's 20-frame excess is ≤ 25 % of Snapshot-COIN's *and* ≤ 50 % of LoRA-TTT's, with zero writes in ≥ 90 % of seeds.
- **Fails** if its excess is not below Snapshot-COIN's at α = 0.05 (one-sided exact paired sign-flip test, 2^16 patterns).
- The interacting arm must *produce* a write within the "T8 latency". The agent fixed that latency at 12 episodes, because T8 had no number in the text it read. The arm passes at ≥ 90 % of seed-cells with a write and fails below 50 %.
- Otherwise the verdict is inconclusive.

**Deviations, all logged in PROTOCOL §7:**
1. Three test cells decide, not four. (0,0) is the base cell, not a composition.
2. T8 latency was fixed at 12 episodes by the tester.
3. `merge_exact` was added as a diagnostic.
4. The merge skips (0,0) at test, to save time.
5. Seed 1's merge produced NaN (a near-singular covariance made a log-determinant undefined). The code got a fallback that only triggers on failure, and seed 1 was re-run. The originals are kept in `results/compose_nan_original/`.
6. After the restart, the relaunched processes ran about 10× slower until the BLAS thread count was set back to 1. Only speed changed.

## 3. What we got

**Main table, normal world** (`research/toy-tests-v4/compose-churn/results/tables.md`; means with 95 % t intervals; nats):

| Arm | n | Excess, 20 frames | Excess, 200 frames | Seeds with 0 writes |
|---|---|---|---|---|
| base | 16 | 462.5 [423.2, 501.7] | 4806.6 [4474.4, 5138.8] | – |
| LoRA-TTT | 16 | 449.3 [411.1, 487.5] | 3403.8 [3185.5, 3622.2] | – |
| Snapshot-COIN | 16 | 111.6 [91.6, 131.5] | 789.8 [695.5, 884.0] | – |
| coin_sum | 16 | 91.1 [79.1, 103.0] | 638.8 [587.2, 690.3] | – |
| **Merge** | 16 | **144.8 [104.4, 185.2]** | 1641.7 [1237.1, 2046.2] | 14/16 |
| merge_exact | 8 | 52.3 [39.1, 65.4] | 727.2 [496.1, 958.3] | 8/8 |

**Decision statistics** (`compose-churn/results/summary.json`, `compose.verdict`):

| Quantity | Value | Bar | Met? |
|---|---|---|---|
| Merge/COIN ratio, 20 frames | 1.298 [1.043, 1.555] | ≤ 0.25 | No |
| p (merge below COIN), one-sided | 0.978 | fail if > 0.05 | **Fail condition met** |
| Merge/TTT ratio, 20 frames | 0.322 [0.251, 0.393], p = 1.5e-5 | ≤ 0.50 | Yes |
| Merge/COIN ratio, 200 frames | 2.079 [1.749, 2.408] | (not used) | – |
| Zero-write seed fraction | 0.875 (Clopper-Pearson 0.617–0.984) | ≥ 0.90 | No |
| Interacting arm: seed-cells with a write within 12 episodes | 0.0625 (3 of 48; CP 0.013–0.172) | ≥ 0.90 to pass; < 0.50 fails | **Fails** |

In the normal world, writes happened in seed 3 (2 writes, the first in test episode 5) and seed 12 (1 write, episode 3) (`summary.json`, `normal/merge`).

**Interacting world** (`tables.md`): merge 20-frame excess 1091.7, COIN 1041.8, coin_sum 1018.2, TTT 1263.7, base 1310.6. At 200 frames: merge 10 589.8, COIN 5554.5, TTT 7843.2. So at 200 frames the merge was the worst adaptive arm in this world. Writes happened in seeds 1 (episode 2), 9 (episode 8) and 12 (episode 6) (`summary.json`, `interact/merge`).

**The `merge_exact` (true-structure) diagnostic.** I recomputed this from the eight per-seed files `compose-churn/results/compose/normal__merge_exact__s{0..7}.json` and the matching COIN, coin_sum and merge files. All values are 20-frame excess averaged over the three mixed cells.

| Seed | merge_exact | COIN | coin_sum | Merge (learned structure) |
|---|---|---|---|---|
| 0 | 41.4 | 94.6 | 79.3 | 133.2 |
| 1 | 46.4 | 201.4 | 141.5 | 203.9 |
| 2 | 47.4 | 92.9 | 87.5 | 153.1 |
| 3 | 44.9 | 82.4 | 78.5 | 76.7 |
| 4 | 45.5 | 85.9 | 76.2 | 140.2 |
| 5 | 45.9 | 90.6 | 74.0 | 61.1 |
| 6 | 61.9 | 113.9 | 93.6 | 176.0 |
| 7 | 84.6 | 96.5 | 86.6 | 143.4 |
| **mean** | **52.3** | **107.3** | **89.6** | **135.9** |

From these files (my computation, not stored in any project file):
- merge_exact/COIN on the same 8 seeds is **0.487**. My percentile bootstrap gives about [0.36, 0.64].
- merge_exact is below COIN in 8 of 8 seeds (exact one-sided sign-flip p = 1/256 ≈ 0.004).
- merge_exact/coin_sum is 0.58; merge_exact is below coin_sum in 8 of 8 seeds.
- merge_exact/TTT is 0.12.
- merge_exact made no write in any of the 24 seed-cells.
- At 200 frames, merge_exact averaged 727 against COIN's 752 and coin_sum's 619 on these seeds. Here it is roughly level with COIN and worse than coin_sum.

**A discrepancy in the summaries.** The project summaries give the true-structure ratio as "0.40–0.43× COIN" (`notes/all-tests-ledger-2026-10-28.md` §3) or "0.43×" (`notes/retrospective-2026-10-19.md` §2; memory v31, line 78). VERIFY §2 gives 0.40 at n = 5 (45.1 vs 111). From the files:
- the first 5 seeds give 0.406, matching VERIFY;
- the first 7 seeds give about 0.44;
- **all 8 seeds give 0.49**, because seed 7 is weak (84.6).

I could not reproduce "0.43 at 8/8". The final 8-seed value is about 0.49. That is further from the 0.25 bar than the summaries say. Either way, the conclusion "even with true structure, the bar is missed" stands.

**Structure learned in training versus result** (`compose-churn/results/compose_structure_by_seed.json`; the file has no header). My reading of its columns: seed, the sites of the directions held after training (0 = lens, 1 = W1, 2 = W2, 3 = scl), the training write log, merge 20-frame excess, COIN 20-frame excess, and a sixth column whose meaning is undocumented. I checked columns 4 and 5 against the per-seed result files and they match. The true payload site is W1 (site 1). [checked from file; the grouping is my post hoc analysis, not pre-registered]

| Seeds | Learned a W1 direction? | Merge mean | COIN mean | Seeds where merge < COIN |
|---|---|---|---|---|
| 0, 1, 3, 4, 5, 9, 10, 13, 15 (9 seeds) | yes | 100.7 | 108.7 | 6 of 9 |
| 2, 6, 7, 8, 11, 12, 14 (7 seeds) | no (the payload went to W2 or extra lens directions) | 201.4 | 115.1 | 0 of 7 |

Overall, by my count, the merge was below COIN in 6 of 16 seeds (3, 5, 9, 10, 13, 15). VERIFY §2 says 5 of 16 and omits seed 10 (merge 128 vs COIN 161 in the structure file).

**Sanity checks before the main runs** (PROTOCOL §5):
- S1: the oracle score equals the analytic expected log-density.
- S2b: COIN-novel, TTT and the merge each recover a planted cell in a 3000-frame episode (`results/sanity/S2b_compose_fitters.json`).
- S4: finite-difference gradient and Fisher checks passed.

## 4. What we inferred

**What the result says:**
- **The bet fails as pre-registered.** The merge's 20-frame excess is about 1.3× Snapshot-COIN's, not ≤ 0.25×, and it is not even significantly below COIN (p = 0.978). [checked: `summary.json`]
- The zero-write condition was also missed (14/16 = 0.875 against 0.90). The interacting arm also failed (3/48 seed-cells wrote).
- The merge clearly beats LoRA-TTT (0.32×). That half of the rule passed. But LoRA-TTT barely beat the frozen base (449 vs 462 at 20 frames), so this is a win over a weak reference. [checked: `tables.md`]
- **The true-structure diagnostic separates two questions.** Given the right directions, inference-only composition works in a real sense: no writes, and below COIN in every seed. It still does not reach the pre-registered 0.25× (it is about 0.49× on 8 seeds), and it does not beat task-arithmetic COIN by that margin (0.58×). By 200 frames it is no better than COIN. [checked from files; my computation]
- So the failure has two layers. [inferred]
  1. **Structure learning** often puts the payload factor at the wrong site. When it does, composition degrades badly: 0 of 7 seeds beat COIN. This is the larger part of the gap.
  2. **Even with perfect structure**, coordinate inference over 20 frames is not 4× better than a labelled snapshot library. The "breakthrough" margin is absent even in the best case.

**What the result does NOT say:**
- It does not show that composition by inference is a bad principle. It shows that this implementation, in this toy, does not deliver the pre-registered margin.
- It does not test the full core. The toy merge has no scratch state ψ, no sel/aud records, and an extra evidence gate (VERIFY §1.2, §2.4).
- The interacting-arm failure ("misfit absorbed") should **not** be cited as a property of the core. VERIFY §2.4 found evidence was collected on only 35 of 672 chunks (5 %) in a probe, because of an undocumented "settle" gate, and candidates came from a reservoir dominated by training frames. VERIFY calls this likely an implementation artifact.

## 5. Why we think it happened

- **W1/W2 non-identifiability.** In this toy, a rank-1 change at W1 can be closely imitated by a rank-1 change at W2, because both shift the output along a fixed direction by a feature of the input. The tester saw this in pilots: 3 of 4 pilot seeds wrote the payload at W2 (`IMPLEMENTATION.md` §4a; PROTOCOL §4). When the payload is learned at the wrong site, "level 2 of X" is no longer exactly twice the learned direction, so composition is only approximate. [checked that it happens: structure file and pilots. The causal link to the excess is inferred from the 0-of-7 pattern]
- **Redundant approximate directions are never removed.** The toy only compresses *unreferenced* directions. It does not run the core's "compress whenever ΔL < 0" (`IMPLEMENTATION.md` §4c). The tester deliberately did not add it, because its threshold would have had to be chosen after seeing the failure. [checked that it is absent] [inferred that this hurts composition]
- **Linearised coordinate updates.** A level-2 cell needs a coordinate twice the training value. Gauss-Newton updates converge more slowly when the coordinate is far from the prior (`IMPLEMENTATION.md` §4d). [inferred] This is a plausible reason why even `merge_exact` needs about 52 nats over 20 frames, and why some of its cells have large 200-frame excess (seed 0, cell (2,1): 808 nats, against 80 for cell (1,2), from `normal__merge_exact__s0.json`). [guess as to cause]
- **The comparison leans toward COIN in places** (VERIFY §2):
  - COIN's snapshots were fitted with **cell labels**, whereas TESTS.md defines Snapshot-COIN as inferring contexts.
  - COIN was tuned on the compose objective, while the merge reused churn tuning.
  - COIN's best learning rate sat at the edge of its grid, which would favour COIN further if extended.

  [checked by VERIFY] Even so, `merge_exact`, which has *more* information than COIN, misses the bar. So the fairness issues can move the verdict at most from FAIL to INCONCLUSIVE, and only if structure learning were perfect. [inferred, agreed by VERIFY]
- **Interacting arm.** The interaction cost the merge about 9 000 extra nats per 200-frame episode, yet the W2 candidate's evidence moved only from log E 459.7 to 469.4 against L ≈ 540. The settle gate shut off evidence on 95 % of chunks. [checked by VERIFY on one seed-cell; generalisation inferred]

## 6. Caveats and what could be wrong

- `merge_exact` has only 8 seeds. Its ratio estimate moved from 0.40 (5 seeds) to 0.49 (8 seeds), and its interval is wide.
- The "structure versus result" split above is post hoc and was not pre-registered. With 9 and 7 seeds it is suggestive, not established.
- The toy merge deviates from the core in ways not all listed in `IMPLEMENTATION.md`: no ψ, the settle gate, a plug-in baseline rather than the mixture predictive, best single candidate rather than joint subsets (VERIFY §1.2). These bear most on writes and the interacting arm.
- LoRA-TTT was weak in this toy (close to the frozen base after 24 tuned configs). The "≤ 0.5× TTT" half is therefore easy.
- The cell-level and site-level choices (payload at W1, Frobenius norms 8 and 3, 12 training visits per cell) were fixed by the tester. Other choices might make structure learning easier or harder.
- A small toy (64 tokens, width 256). No scale argument is made.
- No `REPORT.md` was written. The verdict comes from `analysis.py` output (`tables.md`, `summary.json`), which VERIFY re-ran and reproduced.

## 7. What it changed in the project afterwards

- The retrospective records all three core-v4 CPU bets as failing "as built". It summarises this one as: the merge's error is 1.30× COIN's, and even with the true structure it reaches only about 0.43×, against a 0.25× bar (`notes/retrospective-2026-10-19.md` §2). By the files, the 8-seed value is about 0.49×.
- Under RECOMMENDATION §8's own terms, failing Bet 2 means the merge is "an excellent integration ... not a breakthrough". [checked: RECOMMENDATION.md]
- The lesson recorded in memory (v31, line 75): "composition by inference works when structure is right; merge's structure learning picks wrong site." The project then moved away from the unified core. It focused first on one cluster (learning to learn motor skills), with an "evidence first" rule: build the strongest existing stack and invent only at measured break points (retrospective §5).
- The idea "store causes, not situations" came back later as cluster C's self-test and then as the independent T-CAUSE pilot (`notes/all-tests-ledger-2026-10-28.md` §6–§7). There an existing label-free stack (cxl) was best or near-best in every world.
- VERIFY proposed a fair repeat: unlabelled COIN, compose-specific tuning for the merge, and an interacting arm with ψ and without the gate. I found no record that it was run.
