# 06 Cluster designers' self-tests: did each cluster architect's own CPU toy show an edge over strong methods?

**Date run:** 2026-10-19 (architects about 06:25–07:05 UTC; audits about 07:05–08:35 UTC). **Folder:** /mnt/project-files/research/architecture/clusters-v5/ **Built by:** four Opus architects (one per cluster, L, C, M, S), each building and running its *own* toy; then two independent Opus auditors (Auditor 1: L and C; Auditor 2: M and S; both wrote a "seams" file). **Seeds:** L 2 worlds × 12 tasks per cell; C 2 seeds (4 after the audit); M 5 seeds × 300 episodes (P6) and 3 lives (P8); S 10 seeds. **Verdict:** every toy flattered its own design. After audit: L had one FATAL finding; C, M and S had no fatal finding but each headline edge shrank to a tie, a planted effect, or a known method. No pre-registered bar was involved; these were feasibility checks, not tests.

Tags used below: **[checked]** = stated in a result or audit file I read (path given); **[inferred]** = my reasoning from those files; **[guess]** = plausible but not supported by a file. All paths are relative to /mnt/project-files.

---

## 1. Why we ran it

**Background.** By 2026-10-19 the project had spent five days designing one architecture for all 14 problems at once, and the three CPU tests of the merged core-v4 had all failed: the merge had 1.30× the error of a COIN context library on unseen combinations, LoRA with replay was more accurate in every misfit world, and a learned practice prior made learning 1.30× slower (notes/retrospective-2026-10-19.md §2; notes/all-tests-ledger-2026-10-28.md tests 3–5) [checked]. The retrospective's diagnosis was that depth had been split 14 ways and that fixes broke each other at the seams (retrospective §4) [checked].

**The decision.** Lucky agreed to "contracts, then four cluster architects" (memory note cluster-round-v5.md) [checked]. Each architect owned a few related problems and had to produce "at least one genuine breakthrough idea", defined as "something the strongest existing method cannot do, on your owned problems, shown by a falsifiable test" (research/architecture/clusters-v5/BRIEF.md) [checked]. The four clusters (CONTRACTS.md §3) [checked]:

| Cluster | Owns | Plain question |
|---|---|---|
| L, learning to learn | P14 transferable motor ability, P13 sample efficiency, P3 stability–plasticity | Does the robot get better at *learning* motor skills with practice? |
| C, change and memory of change | P1 adaptation, P2 protection of old skills, P4 interference/transfer, P5 relearning | What changed, and how is old knowledge kept and regained? |
| M, memory economy | P6 habits, P7 episodic memory, P8 consolidation, P9 multi-stage memory | What is stored where, and how does memory stay bounded? |
| S, self-knowledge and correction | P10 metacognition, P11 error correction, P12 recovery | Does the robot know its own competence and recover from errors? |

**Why the toys existed.** CONTRACTS.md §6 asked each architect, for each owned problem, to name the strongest existing method, state a falsifiable claim with "a CPU toy experiment that could show it false", and give the oracle's expected result as a validity check. The brief allowed "small CPU checks ... to sanity-check a mathematical claim", single-threaded, under ~10 minutes each, tagged [shown] (BRIEF.md "Conduct") [checked]. So the toys were meant as quick feasibility checks inside a design document. They were *not* pre-registered, and the architect who designed the mechanism also designed the world, the competitors and the measure. That last fact is the main subject of this file.

**Why the audits.** CONTRACTS.md §8 planned an independent auditor per pair of cores. Lucky approved the audit at 07:05 UTC (cluster-round-v5.md) [checked]. The auditors re-ran every script on copies, added competitor arms, and checked the claims against the contracts and the brain checklist (AUDIT/AUDIT-{C,L,M,S}.md) [checked].

---

## 2. What we did

All toys were small numpy programs written by the architect, run on the shared 4-CPU container.

### L (learning to learn)
- **Idea.** Motor ability is a model of the robot's own sensorimotor *channel*: when a command takes effect (delay), how noise grows, what a slip looks like. Practice writes immutable channel "forms" (shift, scale, event forms) only on multi-task held-out evidence; each new task fits only the channel's values and its own content, in a windowed "race" against a fresh learner (L/CORE.md sharp idea) [checked].
- **Toy** (L/checks/lens_toy.py). 6 action channels, an 8-lag window, fresh task content per task. Two worlds (seeds 7 and 11) × 3 practice histories (varied, timing-only, narrow) × 5 test conditions (C1–C5) × 12 tasks. Episodes of 20 samples. Criterion: relative coefficient error ≤ 0.25; "EtC" = episodes to criterion, capped at 61; G = gain relative to fresh (L/CORE.md §D.6) [checked].
- **Arms.** `fresh` (an unstructured 48-coefficient window regression), `pooledR` (pooled model-based values plus a learned per-lag prior, raced against fresh; the "Γ-style" competitor), `lensR` (L's candidate), `oracle` (knows the true lag, noise and slip law) [checked].
- **Not run by L:** an amortised RMA / meta-RL identifier (named by L as the real strongest competitor), any S1→S2 lifelong run, any P3 probe panel (L/CORE.md §D.10) [checked].

### C (change and memory of change)
- **Idea.** Store *causes*, not situations. Each cause is the shortest edit (a few rank-1 terms at the "sites" of the network where the change lives) that held-out data supports. A situation is a sparse sum of causes, recognised by an exact Bayes mixture over all such sums, with no write; only what no sum explains is written as a new immutable cause, gated by a Ville-bounded test (a statistical bound on false writes) (C/CORE.md Phase F) [checked].
- **Toy.** The T-COMPOSE world from toy-tests-v4: a 3×3 factorial of two causes X and Y, train on five cells, test on the unseen mixed cells (2,1), (1,2), (2,2). Measure: excess log-score over the oracle, in nats (natural-log units of prediction error; lower is better), summed over the first 20 and 200 frames (C/CORE.md §D.6 P4) [checked].
- **Arms.** C's full unlabelled loop (`ledger_pipeline.py`), against Snapshot-COIN (a library of one snapshot adapter per labelled context, recalled by Bayesian context inference; Heald et al. 2021) *with* labels, `coin_sum` (labelled snapshots summed, i.e. task arithmetic) and LoRA-TTT (a low-rank adapter updated at test time). Helper scripts: `sitesel.py` (labelled minimal edits), `snapacc.py` (summed rank-4 snapshots), `rebuild_bound.py` (relearning bound), plus an "interacting" world with an extra term when x·y > 1 [checked].
- **Seeds.** 2 (seeds 0 and 1); interacting world 1 seed [checked].
- **Simplifications** declared by C: birth rule a fixed margin against the best single hypothesis rather than the mixture with alpha-spending; no sleep-time redundancy removal or re-description (C/CORE.md Appendix) [checked].

### M (memory economy)
- **Idea.** "Hand a job to a cheaper stage on probation": planner → habit, window → record, record → exact statistic. A habit carries its own prediction of its actions' consequences, and a CUSUM test (a cumulative-sum change detector) on real consequence residuals hands control back to the planner, like a JIT compiler de-optimising (M/CORE.md §F) [checked].
- **Toys** (M/checks/). Toy 1 (`toy_p6_guard.py`): a broad familiar manifold, secret rotations of the action map (30°–150°), sign flip, gain change, cosmetic change; 5 seeds × 300 episodes. Toy 2 (`toy_p6_guard2.py`): a detour task and a narrow corridor between hazards. Toy 3 (`toy_p8_deletion.py`): consolidation followed by forced deletion of all records; 3 lives × 45,000 samples (M/CORE.md §D.6) [checked].
- **Arms.** The guard (GRD), ensemble-disagreement arbitration ("When to Plan"), a progress monitor (PROG, built by M itself as "the obvious consequence-based alternative"), oracle; for P8, EXACT (exact ridge sufficient statistics over random features) vs ER-MLP (experience replay into a small MLP) [checked].
- **Not run by M:** any P7 or P9 toy; the distil-then-delete competitor (P&C) for P8 (M/CORE.md §D.10) [checked].

### S (self-knowledge and correction)
- **Idea.** "Commit, receipt, audit". Confidence comes from forecasts committed before each attempt, from "receipts" of every persistent write (the paired change in held-out "vault" score the write causes, called a *plasticity efference copy*), and from one-sided e-detectors (sequential tests with a guaranteed false-alarm bound) whose null is "my committed forecast is calibrated" rather than "my world model is true" (S/CORE.md sharp idea) [checked].
- **Toys.** T-S1 (`t_s1.py`): 10 seeds, 12 skills, 8,000 attempts per life, forgetting caused by write interference, world change for 3 skills at mid-life. Measure: Brier score (mean squared error of a probability forecast; lower is better) overall, on a "forgotten" slice, on changed skills; ECE (expected calibration error). T-S3 (`t_s3.py`): recovery within an episode, 4 hidden situation types, 5 routes, 1,500 episodes, 10 seeds (S/CORE.md §D.6) [checked].
- **Arms.** S; per-task online Platt scaling (the knowledge base's recommended method); a pooled SGD logistic with the same features; oracle; for P12, MPC, MPC + tabu list, S recovery, oracle [checked].
- **Not run by S:** T-S2 (P11), T-S1R (the real-learner version), the step-level detector (S/CORE.md §D.10) [checked].

### The audits
Each auditor re-ran the scripts unchanged and then added arms or variants that the architect had not built. Grades: FATAL (breaks the central bet), LOCAL (a specific claim fails), SPECULATIVE (AUDIT-*.md headers) [checked]. Nothing in the architects' folders was changed.

No pre-registered pass/kill bar applied to the toys themselves. Each core stated a bar for a *later* decisive test (for example C: "at most 0.4× COIN ... on 16 seeds"; L: "≤ 0.8× the amortised arm's episodes"), but no toy was judged against a bar fixed before it ran [checked: CORE.md files; inferred that no bar was fixed in advance, since none is recorded].

---

## 3. What we got

### L
Architect's table (L/CORE.md §D.6), median EtC [checked]:

| Varied history | fresh | pooledR | lensR (L) | oracle |
|---|---|---|---|---|
| all three factors, world 11 | 34 | 18 | **4** | 3 |
| all three factors, world 7 | 53 | 30 | **20** | 6 |
| delay out of practised range, world 11 | 42 | 46 | **32** | 3 |
| delay out of practised range, world 7 | 38 | 40 | **28** | 4 |

Audit (AUDIT/AUDIT-L.md §3.2), selected rows, median EtC; `gramTR` = a learner given the same structural "grammar" as L (every single-lag profile plus a per-task robust slip scale) but **no practice at all**; `lensAllR` = L's practised noise and slip forms with all lags allowed [checked]:

| world | history | cell | fresh | pooledR | lensR (L) | gramTR (no practice) | lensAllR | oracle |
|---|---|---|---|---|---|---|---|---|
| 11 | varied | C1 all three | 34 | 18 | 4 | 5 | 4 | 3 |
| 11 | varied | C4 delay out of range | 42 | 46 | 32 | 12 | 3 | 3 |
| 11 | timing only | C4 | 51 | 51 | 51 | 5 | 27 | 4 |
| 11 | narrow | C1 | 38 | 28 | 16 | 6 | 3 | 3 |
| 7 | varied | C1 | 53 | 30 | 20 | 6 | 5 | 6 |
| 7 | varied | C4 | 38 | 40 | 28 | 6 | 4 | 4 |
| 7 | timing only | C4 | 61 | 61 | 61 | 8 | 36 | 4 |

- Both original runs reproduced byte for byte (`diff` empty) [checked: AUDIT-L §2].
- Over all 30 cells, lensR vs gramTR: lensR slower in 20, equal in 7, faster in 3 [checked: AUDIT-L §3.2 "Reading"].
- lensAllR was faster than or equal to gramTR in every cell where noise and slip forms were learned, by 0–9 episodes [checked].
- Finding L-F1 graded **FATAL**: "The [shown] headline does not survive a fair history-free competitor" [checked].

### C
Architect's result (C/CORE.md §D.6, Appendix) [checked]:
- Mean 20-frame excess over the 3 unseen cells: C (unlabelled) 29.8 / 55.8 nats (seeds 0 / 1) vs labelled COIN 94.6 / 202.4, coin_sum 79.3 / 143.2, LoRA-TTT 362.9 / 559.9. Ratio C/COIN 0.31 / 0.28.
- Labelled minimal single-site edits summed: 3.5–13.4 nats; summed rank-4 snapshots: 81–310 nats.
- Interacting world (1 seed): C 747.8 vs COIN 797.0 at 20 frames, but C 5538 vs COIN 5342 at 200 frames (C worse by 196 nats). C reported this itself as a failed half-claim.

Audit (AUDIT/AUDIT-C.md §2), per seed-cell 20-frame excess, C / COIN / coin_sum [checked]:

| seed | (2,1) | (1,2) | (2,2) |
|---|---|---|---|
| 0 | 29.5 / 47.4 / 47.4 | 10.3 / 109.0 / 68.5 | 49.5 / 127.4 / 122.1 |
| 1 | 30.7 / 147.8 / 138.9 | 16.5 / 109.7 / 65.6 | 120.2 / 349.7 / 225.0 |
| 2 (new) | 11.4 / 86.8 / 86.8 | 10.5 / 72.4 / 53.1 | 11.9 / 119.7 / 119.7 |
| 3 (new) | 12.5 / 85.4 / 71.7 | 17.9 / 96.2 / 96.2 | 43.3 / 90.8 / 90.8 |

- All original runs reproduced identically [checked].
- C below both COIN arms in 12/12 seed-cells; ratio C/COIN per seed 0.31, 0.28, 0.12, 0.27; ratio of means over 4 seeds 0.25 (30.4 vs 120.2 nats) [checked].
- COIN was not under-tuned for this measure: changing its learning rate or training snapshots 5× longer changed its 20-frame excess by ≤ 6% [checked: C-F11].
- With n = 4 the smallest possible one-sided sign-flip p-value is 1/16 = 0.0625, so even 4/4 is not significant at 0.05 (C-F3) [checked].
- The (2,0) cause was born with a spurious extra lens term in 4 of 4 seeds; duplicate causes in 2 of 4 seeds; in seed 1, no sum was recognised in 2 of 3 test cells (C-F2) [checked].
- Relearning check: savings ratio 0.004 for an exact return, 0.39 at 10% deviation, 0.99 at 30% (C-F8) [checked].

### M
Architect's results (M/CORE.md §D.6), failure rates [checked]:

| Condition (toy 1 unless stated) | Ensemble | Guard | Progress monitor | Oracle |
|---|---|---|---|---|
| rotation 100° | 84% | 33% | ties guard | 33% |
| rotation 60° | 21% | 4.3% | ties guard | 4.3% |
| sign flip | 78% | 39% | ties guard | 39% |
| cosmetic change: needless hand-overs | 98% | 0% | — | — |
| detour, familiar: false hand-overs (toy 2) | 17.7% | 0 | 1.9% | — |
| corridor rotation 25° (toy 2) | 0.1% | 73% | — | — |

P8 forced deletion (toy 3): old-situation MSE after deleting all records and learning 5 new situations, EXACT 0.0139 vs ER-MLP 0.0415 (0.0390 with 4× replay memory); ER-MLP better on the new situations (0.0102 vs 0.0121) [checked].

Audit (AUDIT/AUDIT-M.md) [checked]:
- All three toys reproduced bit-for-bit.
- Noise probe (`aud_guard_noise.py`, 5 seeds × 300 episodes), failure rates GRD / PROG / ORC: σ = 0.05, rot60: 0.147 / 0.079 / 0.082; σ = 0.10, rot60: 0.623 / 0.351 / 0.345; σ = 0.10, rot100: 0.813 / 0.587 / 0.585 (M-F3). Neither arbiter was re-tuned for the new noise level, and at σ = 0.10, rot30 the oracle validity check failed.
- In the gain-0.4 condition the habit alone failed 0.000 of episodes, yet the guard handed over in 93% of episodes at 3.57 planner calls per episode (M-F4).
- The P8 edge is the known edge of exact analytic continual-learning readouts (RanPAC, ACIL/GACL); the world was one stationary function, and ER's persistent knowledge after deletion was an ~836-float MLP against EXACT's 66,560-float statistic (M-F10).
- Totals: FATAL 0, LOCAL 12, SPECULATIVE 3.

### S
Architect's T-S1 table (S/CORE.md §D.6) [checked]:

| | S | Per-task Platt | Pooled SGD (same features) | Oracle |
|---|---|---|---|---|
| Brier, all | 0.110 | 0.116 | 0.115 | 0.104 |
| Brier, forgotten slice | 0.278 | 0.467 | 0.280 | 0.213 |
| Brier, changed skills | 0.265 | 0.270 | 0.286 | 0.241 |

- Paired S minus per-task Platt on the forgotten slice: −0.19 (10/10 seeds). S vs pooled SGD on the forgotten slice: −0.003 (6/10), a tie by S's own rule.
- Episode audit: 30/30 changed skills detected, median delay 16 attempts, 0 alarms on unchanged skills.
- T-S3: completion S 0.941 vs MPC + tabu 0.915 vs oracle 0.983; S − tabu +0.026 ± 0.010 (9/10).
- S stated openly that "the receipt–forgetting link was built into the toy world".

Audit (AUDIT/AUDIT-S.md) [checked]:
- Both toys reproduced bit-identically.
- In `t_s1.py`, after every attempt of skill k the code resets `I[k] = 0.0; wlog[k] = 0.0; rcpt[k] = 0.0` (line 57), so forgetting lasts exactly one attempt and any method that learns from outcomes cannot learn it (S-F1).
- With persistent forgetting (`I[k] *= 0.7` per attempt), S − per-task Platt (best learning rate) on the forgotten slice became −0.106. Against a per-task Platt given the same features it became −0.023 (persistent world) or −0.089 (one-shot world) (S-F1).
- A pooled SGD with a per-task fast offset (S's own slow/fast split, fitted by SGD) tied S overall (−0.0002) and beat S on changed skills (+0.0039, with S winning in 0 of 10 seeds) (S-F2).
- T-S3: a "tabu + route-success counts" competitor completed 0.935 vs S 0.941, S − bandit_tabu +0.006 ± 0.009 (6/10), a tie (S-F7).
- AUROC (a discrimination measure): S 0.736 vs pooled SGD + offset 0.737 overall (S-F12).
- Totals: FATAL 0, LOCAL 12, SPECULATIVE 3.

Note: the project ledger summarises the S result as "−0.19 fell to −0.02 when made fair" (notes/all-tests-ledger-2026-10-28.md test 6). The audit's numbers are −0.106, −0.089 or −0.023 depending on the variant; −0.02 is the most favourable-to-competitor variant, not the only one [checked].

---

## 4. What we inferred

- **L.** The toy did not show that practice helps. It showed that *structure* helps: once a learner without any practice was given the same grammar, practice through L's timing ("shift") forms was net harmful (20 of 30 cells slower). A narrower effect survived: carrying the body's noise direction and slip scale (lensAllR) brought every cell where they were learned to near-oracle level. But the fair test of that residual, a no-practice per-task estimate of the same noise direction, was not run by anyone, so even the residual is not shown to need practice (AUDIT-L §3.2, §9) [checked].
- **C.** The core result reproduces and holds on new seeds, which is more than the other three can say. But the edge over COIN comes from C's stored-item shape (rank-1, single-site edits) matching the world's planted truth (rank-1, single-site changes). A *labelled* library of such edits did better still (3.5–13.4 nats). So C's own contribution is doing it without labels, at a cost of 5–45 nats over the labelled version (C-F1). Four seeds cannot reach significance. The result says nothing about worlds where changes are spread across sites, vary in strength, or interact [checked; the last sentence is inferred from the toy world's construction].
- **M.** The consequence guard has one real measured property: it ignores cosmetic change and catches consequence-only changes that input-uncertainty arbitration misses. It does not beat a progress monitor (tie at the toy's noise level, worse at higher noise with the declared defaults, untuned). P7 and P9 had no runs. The P8 edge belongs to existing analytic methods. The "one act everywhere" unification is three different rules (AUDIT-M §9, M-F11) [checked].
- **S.** The receipt idea (plasticity efference copy) was untested: its toy link was planted. The other edges became ties once fair competitors were built. What survived is the *principle* that an alarm should test the calibration of a forecast the robot committed to, a null that can hold even when every model is wrong; in the toy this gave 30/30 detections and no false alarms on unchanged skills (AUDIT-S §9) [checked].
- **Across all four.** Nothing here is evidence of a breakthrough. The orchestrator's own summary said so at the time: "breakthrough candidates with toy evidence, not breakthroughs" (SUMMARY.md) [checked]. The audits then showed that even "candidate" was too strong for L and for most of M and S.
- **What it does NOT say.** It does not say any of the ideas is wrong at robot scale; nothing ran on RoboJEPA. It does not say COIN, the progress monitor or pooled calibration are the best possible competitors; they are the ones that were built. A tie in a toy is not a loss in general [inferred].

---

## 5. Why we think it happened

The pattern is the same in all four: the designer's assumptions ended up inside the test, so the test could not falsify them. The memory note designer-never-builds-test.md states this as the lesson: "a designer's toy encodes the designer's assumptions, so it can't falsify them" [checked]. The specific routes:

1. **Knowledge given to one arm only (L).** L's design says the loop grammar (single transport lag, rank-1 structure, scaling noise, sparse slips) is "fixed by physics". L gave that grammar to its own learner, while `fresh` fitted all 48 window coefficients and `pooledR` had a weak per-lag prior (`lens_toy.py:84–113` vs `:53–65`) [checked: AUDIT-L L-F1]. The large gap (4 vs 34) was mainly the difference between fitting 6 parameters and 48 from 20 samples, a dimension effect, not a practice effect (L-F4) [checked]. Worse, the practised lag set H restricts new tasks to lags that practice happened to cover; when a test delay falls outside H, L is slower than a no-practice learner (C4 rows, world 7's missing lags 2 and 3) [checked: L-F2]. That restriction is, in effect, a learned prior over where solutions lie, the very thing P14's wording excludes [checked: AUDIT-L L-F2; the link to P14's "is not" is the auditor's].

2. **World built to match the stored shape (C).** The T-COMPOSE truth is "exactly one rank-1 change at one declared site per factor" (`sumset_check.py:24–44`), and C stores exactly that; COIN stores rank-4 all-site snapshots that do not add [checked: C-F1]. In such a world, summing minimal edits is right by construction. C said so itself ("In the toy it is true by construction", C/CORE.md §D.10) [checked]. C's unlabelled loop then added spurious terms because each cause was fitted from a single 375-frame segment (C-F2) [checked].

3. **High signal-to-noise and an in-class model (M).** The noise was σ = 0.01 against actions of size about 0.1–1, so one wrong action left a residual 10–100× the noise, and the consequence model was exactly linear like the world (`toy_p6_guard.py:28, :66`) [checked: M-F2]. In that regime any consequence check works at once; the guard's handover step was 2.0–2.2 in every changed condition. Raising the noise exposed that the guard's fixed threshold (k_g = 4) trades detection power for robustness (M-F3, M-F13) [checked]. In the P8 toy the world was one stationary function, so a global exact statistic has zero interference by construction (M-F10) [checked].

4. **A planted link (S).** The forgetting that receipts were supposed to predict was generated *from* the same write quantities that the receipts measured, and it was reset after one attempt, so outcome-learning competitors were structurally unable to catch it (S-F1) [checked]. Competitors were also left untuned (fixed learning rates 0.05 and 0.03) and denied S's features (S-F2) [checked].

5. **The strongest competitor left out.** Each architect named a strong competitor and did not build it: L the amortised RMA identifier; C a labelled minimal-edit library and compositional lifelong learners; M distil-then-delete (P&C) and innovation-test arbiters; S pooled calibration with a fast offset (CORE.md §D.10 files; AUDIT C-F11, L-F10, M-F1, S-F2) [checked]. When auditors built the simplest missing one, it tied or won in L, M and S [checked].

6. **Few seeds, no bar fixed in advance.** C had 2 seeds and L had 2 worlds, against 16 seeds asked for by their own decisive-test plans (C-F3, L-F6) [checked]. With no pass/kill rule written before the run, a favourable number could be reported as the headline and a tie further down. AUDIT-M notes the guard's tie was listed after the favourable comparison in M's sharp-idea paragraph (AUDIT-M §4) [checked]. I add that this was not concealment: every architect reported its own ties and losses openly, and the audits credit them for it [checked: "admitted shortfalls" lists in all four audits]. The bias is in what was built, not in what was reported [inferred].

7. **Time pressure** [guess]. Each architect wrote a long core (8,000–13,000 words, by `wc`) and built toys in roughly half an hour to an hour (cluster-round-v5.md timings). That leaves little room to build the strongest competitor. This is a plausible contributing cause; no file states it.

### Seams (SEAMS-1.md, SEAMS-2.md), briefly
The seam audits found no fatal conflict but several services that did not match across clusters [checked]:
- C publishes a belief over *sums of causes*, while M needs a discrete situation label (S-4, S2-1).
- L hands task adapters to "C's protected store", but C has no intake for them, so protection of L-learned skills has no owner (S-3).
- Four clusters each hold out the same scarce real episodes, with no arbiter (S2-4).
- Declared memory shares sum to more than 100% of the budget (">150%" read literally, SEAMS-1 S-6; "≥ 12.7–15.1 GB against 10 GB", SEAMS-2 S2-7).
- S's receipts target write-caused forgetting, which C's and L's immutable writes largely design out (S2-5).

These matter here because they show that even the toy wins, had they held, would not have composed into one system without more work [inferred].

---

## 6. Caveats and what could be wrong

- **The audits are themselves toy-level.** The auditors used the architects' worlds and added arms in the same small setting. For example, the M noise probe did not re-tune either arbiter (M-F3), and S's competitor learning rates were picked on the test seeds, "which favours them slightly" (S-F2) [checked]. An audit tie is also small-sample evidence.
- **The L FATAL finding depends on what counts as fair knowledge.** Giving a no-practice learner the full grammar asks whether practice adds anything *beyond* known structure. If the robot really must discover that structure, a grammar-equipped reference is a ceiling, not a competitor. The later T-P14D brief took exactly this view: the grammar-given learner became a "reference arm" and discovery was measured against it (research/tests-v6/BRIEF-tests.md, Test B) [checked]. So "L lost" is right for the claim L made, but it does not settle whether structure can be discovered by practice [inferred].
- **C's result is the most robust but the narrowest.** 12/12 seed-cells is consistent, but n = 4, and the world plants C's assumption. Whether it survives a world that breaks the assumption was not known from these files; that is what T-CAUSE (file 07) was built to test [checked: AUDIT-C C-F18; BRIEF-tests.md].
- **Numbers were not re-derived by me.** I read the result tables in the CORE and AUDIT files and did not re-run any script. The audits report byte-identical re-runs, which I take as given [checked that the audits say so; not re-run].
- **The ledger compresses.** The project ledger's one-line summaries (for example "−0.19 fell to −0.02", "lost to a simple progress monitor at noise 0.05–0.1") are fair in direction but drop qualifiers: several variants for S, and no re-tuning in the M probe [checked against AUDIT-S, AUDIT-M].
- **Not recorded:** exact run times of each architect's toy, and whether any architect tried and discarded other toy settings before reporting. If they did, the reported results would carry selection bias beyond what is described here [not recorded].

---

## 7. What it changed in the project afterwards

- **A standing rule: designers never build their own tests.** The memory note designer-never-builds-test.md records it: decisive tests must be built by an agent that did not design the mechanism, every competitor gets the same structural knowledge, and every test includes an oracle validity check and a "same knowledge, no mechanism" arm [checked]. The ledger's verdict for this phase says the same: "This is where the rule 'designers never build their own tests' came from" (all-tests-ledger test 6) [checked].
- **Two independent tests were commissioned at 08:37 UTC the same day** (cluster-round-v5.md) [checked], with the audit findings written into their briefs (research/tests-v6/BRIEF-tests.md) [checked]:
  - **T-CAUSE** (file 07) took over C's idea, with the audit's demanded additions: a labelled minimal-edit library (`lmel`, C-F18 (i)), worlds where changes are not single-site, vary in strength, or interact (C-F18 (ii)), and C's own pre-declared continuous-strength variant.
  - **T-P14D** took over L's question, with a hidden body structure, the auditor's grammar-given learner as a reference arm, an amortised identifier arm, and "the L-reduced candidate" (practised noise law and slip signature only, AUDIT-L's reduced bet).
- **The P14 architect for v6 was told to deliver a learner spec with "no own evidence toy"** (cluster-round-v5.md) [checked].
- **Surviving principles carried forward.** The later adopted-stack architecture (arch-v8) included "one SR e-detector auditor on committed forecasts", which is S's surviving audit principle, and "verified plan-cache habits" (memory note tests-v6-status.md) [checked that arch-v8 lists these; the link to S's and M's cores is inferred]. The ledger lists "audit your own committed forecasts" as one of three ideas that held up as principles, while noting none became a better method [checked].
- **What was not pursued.** No T-S1R, T-S2, P7 or P9 test was built afterwards, as far as the ledger records (all-tests-ledger "Never tested" lists other items and does not mention these) [checked that they do not appear in the ledger; whether they were consciously dropped is not recorded].
