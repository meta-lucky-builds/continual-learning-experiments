# 07 T-CAUSE: does "store causes, not situations" beat existing methods when the world does not hand it changes of exactly the kind it stores?

**Date run:** 2026-10-19 (build, sanity checks and pilot only; protocol written 09:30 UTC, amendments 2 and 3 the same day). **Folder:** /mnt/project-files/research/tests-v6/cause/ **Built by:** an independent Sonnet test builder that did not design the mechanism (memory note cluster-round-v5.md; brief research/tests-v6/BRIEF-tests.md); the `cxl` competitor was specified by a separate researcher ("Researcher CH", research/adopt-v6/change/STACK.md, ARM-SPEC.md). **Seeds:** PILOT ONLY: 2–3 seeds per cell (seeds 9101–9103; 1 seed for the oracle in w-site), every arm on its default, untuned configuration. The pre-registered main run (24 seeds, tuned arms) **never happened**: it needs the rented 192-vCPU machine. **Verdict:** pilot only, not evidence. In the pilot, C (in every variant) would fail both pre-registered bars; the label-free existing-methods stack `cxl` was best or near-best in every world.

Tags: **[checked]** = stated in a file I read (path given); **[inferred]** = my reasoning or my arithmetic on those files; **[guess]** = plausible, no file supports it. Paths are relative to /mnt/project-files.

---

## 1. Why we ran it

**The idea under test.** Cluster C's core (research/architecture/clusters-v5/C/CORE.md, Phase F) bets that a robot should store *causes*, not situations [checked]:
- each cause is kept as the shortest edit that held-out data supports: a few rank-1 terms at the network "sites" where the change lives (in this toy, four sites: the input lens, layer W1, layer W2, the scale head);
- a situation is a sparse sum of stored causes;
- the robot recognises "what changed" with an exact Bayes mixture over all sparse sums plus "something new", so an unseen combination of known causes, or a return after any gap, is recognised from the first frame with no write;
- only what no sum explains is written, as a new immutable cause, after a held-out test with a Ville bound (a statistical guarantee on the rate of false writes).

**Problems targeted.** P1 adaptation, P2 protection of old skills, P4 interference and transfer, P5 relearning (CONTRACTS.md §3) [checked]. The decisive one for C is P4: a new combination of known causes should be handled at once, and a conflicting prior should never make things worse than starting fresh (C/CORE.md §D.6 P4) [checked].

**Why at that moment.** C's own toy had shown unlabelled C at 0.31 / 0.28 of labelled COIN's error (2 seeds), and the independent audit reproduced this on 4 seeds (12 of 12 seed-cells, ratio of means 0.25). But the audit found that the world was built to match C's assumption: the true changes were exactly rank-1 at one site, which is exactly what C stores, and a *labelled* library of such minimal edits did even better (AUDIT/AUDIT-C.md C-F1) [checked; see file 06]. The audit asked for two things before any claim: a labelled minimal-edit library arm, and "a world whose true change is not at any single declared site" (C-F18) [checked]. The cluster round had also produced the rule that designers never build their own tests (memory note designer-never-builds-test.md) [checked].

So T-CAUSE was meant to settle one question: **does C's mechanism still win when the world does not hand it changes of exactly the kind it stores?** The brief put it as: "A test 'tells us something big' only if a positive result could not have been produced by the test's construction" (BRIEF-tests.md) [checked]. Lucky launched it at 08:37 UTC with "start all three ... may give us a breakthrough" (cluster-round-v5.md; BRIEF-tests.md) [checked].

---

## 2. What we did

### 2.1 Worlds
All worlds use the frozen toy base from the v4 tests (`toyworld.py`, byte-identical to C's copy), four edit sites, and Student-t noise with true ν = 3 that every learner also uses, so the true noise law is inside every learner's class (fixing the v4 error where ν = 4 was off the grid) (PROTOCOL.md §2) [checked]. Two causes X and Y. Training: the T-COMPOSE schedule, five cells (1,0), (2,0), (0,1), (0,2), (1,1), 12 visits each, 500 frames per episode, 60 episodes. Test: the unseen combinations (2,1), (1,2), (2,2), 200 frames, first visit primary [checked].

| World | How a cause shows up | Role |
|---|---|---|
| w-site | X = rank-1 at W1, Y = rank-1 at lens (same as T-COMPOSE) | C's home world; control |
| w-spread | each cause spread over several sites and ranks; no single-site description | kill world (decisive) |
| w-magnitude | as w-site, but strength varies continuously per visit; the label is only a bin | kill world (decisive) |
| w-interact | as w-site plus an interaction term at W2 that is zero in every training cell | reported ceiling only; nothing can learn it |
| w-interact2 | as w-interact, but two extra training cells (3,1), (1,3) make the interaction visible | kill world (decisive), added by amendment 2 |
| w-alias | Y is an almost exactly equivalent edit at a different site (10% perturbation) | ceiling |
| w-null | no cause; truth is the frozen base | false-write control |

Effect sizes were matched so that one unit of a cause costs the frozen base 5–7 nats per frame in every world (sanity/calibrate_effects.txt, cited in PROTOCOL §2) [checked].

**Oracle validity checks** ("oracle validity" = a check that an arm knowing the true law shows the structure the world claims; if not, the world is wrong, not the learner). Each world had a pass criterion on large-data fits; for example, in w-spread the best single-site rank-3 fit must leave a gap > 1.0 nat/frame while the true structure leaves < 0.35. The sanity README states "all six PASS", and w-interact2 also passed (sanity/README.md; PROTOCOL §2.1) [checked]. I did not open each validity JSON beyond what PROTOCOL quotes for w-interact2 (additive gap 4.0 / 4.7 / 29.8 nats/frame, learned-interaction gap 0.02 / 0.03 / 0.24) [checked].

### 2.2 Arms
Units: "excess" = summed log-likelihood shortfall against the oracle, in nats (natural-log units), over the first 20 (ex20) or 200 (ex200) test frames, averaged over the three unseen combinations. Lower is better (PROTOCOL §3.5) [checked]. Descriptions from IMPLEMENTATION.md and PROTOCOL §4 [checked]:

**Candidates (C family, unlabelled)**
- `C`: C's full loop, tuned over 8 configs in the main run. In the pilot every arm ran its "default (first-in-grid)" config (PROTOCOL §9), which for C is config c0, the "literal core" (configs.py line 19) [checked]. So pilot `C` is the literal-core version [inferred from those two facts].
- `C_pub`: a re-implementation of the audited `ledger_pipeline.py` logic (fixed margin birth rule, refits the MAP cause in place, which breaks immutability) [checked].
- `C_gain`: C with a continuous strength per cause (C's own pre-declared alternative), with two beams for compute [checked].
- `C_core` exists in the main job list but has no pilot results [checked: results/pilot file list].

**Labelled references (given the cell label; bin label in w-magnitude)**
- `coin`: Snapshot-COIN, a library of one rank-4 adapter per label, recalled by Bayesian context inference with a fixed hazard (Heald et al. 2021). The primary bar.
- `coin_sum`: COIN plus summing the two most responsible snapshots (task arithmetic).
- `lmel`: "labelled library of minimal edits": the same edit-growing rule as C, but fitted per label on pooled data (about 18× more data per edit than C gets per segment). The audit's requested arm; it separates C's parameterisation from C's unlabelled inference.
- `lt_sft_sum`: sparse fine-tuning differences (LT-SFT style), summed.

**Unlabelled existing methods**
- `cxl`: "context + pseudo-labelled library", built only from existing parts: a continuous context basis (CAVIA-style), k-means+BIC pseudo-labels, an lmel-style minimal edit and a COIN-style snapshot per cluster, pairwise sums, and a fixed-share Bayes mixture over all of these plus fresh and rebuild learners; the library grows at episode ends (PROTOCOL §4.2; STACK.md §0). The **main existing-methods bar** (amendment 3).
- `ctx`: a fully continuous context learner (edits linear in a per-episode context vector z, CAVIA / auto-decoder style). Not an RMA amortised encoder.
- `ctx_ident`: ctx given the true number of mechanisms (d = 2), a declared structural hint.
- `ctx_nn`: ctx with an MLP encoder of the last 20 frames (RMA-style).
- `moca_lite`: MOCA-style changepoint detection (BOCPD) plus Bayesian linear regression in residual space.
- `ssl_lmel`: contrastive self-supervised encoder → k-means pseudo-labels → lmel library.
- `lora_replay`: one persistent LoRA (a low-rank adapter) trained with a 20,000-frame replay reservoir; `lora_replay_targeted` chooses replay by a forecast of which frames an update would hurt.
- `lora_ttt`: one persistent LoRA updated online on every frame (test-time training), no replay.
- `fresh`: a zero-initialised LoRA per episode. `base`: frozen. `oracle`: true law.

Amendment 2 arms (six) were taken from a literature summary; the builder "did NOT read the original papers; the arms are numpy approximations of the ideas" (IMPLEMENTATION.md, Second amendment) [checked].

### 2.3 Life protocol
Same training episodes in the same order to every arm; 30 further "gap" episodes of the distractor cell (1,1) to arms that keep learning; then the unseen-combination test (2 visits, prequential, meaning each frame is scored before any learner sees it); then returns to each of the 5 trained cells. A return is "recognised" if its ex20 ≤ 0.1 × the frozen base's ex20 + 2 nats (PROTOCOL §3) [checked].

### 2.4 Tuning (main run only)
Every tuned arm gets exactly 8 configs, selected on held-out tuning seeds 9001–9002 in four worlds, then used everywhere. C's grid includes the C_pub configuration as c7, so tuned C is at least as good as C_pub on the tuning cells (PROTOCOL §4, §10) [checked]. **None of this was run**: results/tune and results/main are empty, and configs_selected.json has an empty `selected` field [checked: directory listing, file contents].

### 2.5 Pre-registered decision rules (PROTOCOL §6) [checked]
- Statistic: per seed, D = log max(m_A,1) − log max(m_B,1); one-sided sign-flip randomisation test, exact for ≤ 20 seeds; effect = ratio of means with a paired bootstrap interval.
- **Bar 1:** C beats labelled COIN at α = 0.01 in all six decisive cells (w-spread, w-magnitude, w-interact2 × 20 and 200 frames).
- **Bar 2 (amendment 3):** C also beats `cxl` in the same six cells.
- A loss or tie in w-interact2 or w-alias is reported as a ceiling. Any unlabelled arm beating C is reported first.
- w-null: C's births must not be Poisson-improbable against the Ville bound (0.05 expected births per seed over 5 decades of frames).
- Sensitivity sweep over α ∈ {0.05, 0.01, 0.001} and floor ∈ {0.5, 1, 5} nats.
- **Seeds:** N = max(24, power rule); the power rule (power.py) asks ≥ 0.8 power at α = 0.01 to detect a C/COIN ratio of 0.7, with σ floored at 0.4; it gave 16, so N = 24 (n_main_seeds.json: {"n": 24}) [checked].

### 2.6 Predictions written before the pilots in other worlds (PROTOCOL §8) [checked]
- P(C passes bar 1) = 0.10; for C_gain 0.15. P(C passes both bars) = 0.03 (§8.2).
- w-site: C beats COIN (ratio 0.3–0.8); lmel beats C; ctx competitive with C.
- w-spread: C does not beat COIN (P(C wins both n) ≤ 0.25).
- w-magnitude: C ties or loses at 200 frames; C_gain < C; ctx ≤ C.
- w-alias: C ties COIN.
- The builder had already seen one w-site pilot seed when writing §8 (C 61.6, COIN 75.1, lmel 4.6, ctx 32.9) and said so in the protocol header [checked].

### 2.7 Amendments (all before any main or tuning run)
1. 08:57 UTC: `ctx` and `C_gain` added at Lucky's request.
2. 12:17 UTC (Lucky approved): six more arms; w-interact2 added and the old w-interact demoted to a ceiling, because nothing could learn an interaction absent from every training cell.
3. Same day: `cxl` and nine optional ablations added, with the second bar.
Other logged changes: interaction size reduced to Frobenius 4 (the legacy size made every arm fail identically); a +2 nats term added to the "recognised" rule; C_gain beamed after an unbeamed job ran > 35 minutes; pilot scope cut for time (PROTOCOL §10) [checked]. The builder classed these as pre-registration amendments, not deviations, since no main run existed [checked]. I note that amendments 2 and 3 came after some pilot numbers were known, so the arm set was not fixed blind; the decision rule for C was not loosened by them, only made harder (a second bar) [inferred].

---

## 3. What we got

**Everything below is PILOT.** Default untuned configs, 2–3 seeds, a shared container. The pilot summary header says "NOT evidence, used for debugging, timing and sigma only" (pilot/summary.md) [checked]. That header says "2-4 seeds per cell", but the tables show 1–3; I report the tables [checked].

### 3.1 Per-world tables
Source: research/tests-v6/cause/sanity/PILOT_report.md §1 (same numbers as pilot/summary.md, plus visit-2 and returns). Mean over seeds; sorted by ex200 [checked; the tables below were extracted from that file by script, not retyped].

#### w-site

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| oracle | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 1.00 |
| lmel | 3 | 3.6 | 12.6 | 3.3 | 3.0 | 1.00 |
| cxl | 3 | 5.9 | 17.2 | 5.6 | 3.4 | 1.00 |
| C_pub | 2 | 12.3 | 95.7 | 11.9 | 6.3 | 1.00 |
| ctx | 3 | 41.5 | 164.4 | 41.6 | 43.6 | 0.27 |
| ctx_ident | 3 | 40.8 | 168.5 | 41.4 | 39.3 | 0.20 |
| C_gain | 3 | 39.7 | 250.6 | 46.6 | 27.1 | 0.40 |
| ssl_lmel | 3 | 46.9 | 349.9 | 47.1 | 31.1 | 0.33 |
| C | 3 | 62.2 | 472.4 | 75.7 | 22.3 | 0.73 |
| coin_sum | 3 | 64.6 | 474.7 | 42.9 | 6.5 | 1.00 |
| lora_ttt | 3 | 100.1 | 557.2 | 36.7 | 140.7 | 0.20 |
| coin | 3 | 86.9 | 621.4 | 50.4 | 6.4 | 1.00 |
| lt_sft_sum | 3 | 96.9 | 854.5 | 108.2 | 44.2 | 0.27 |
| lora_replay | 3 | 122.8 | 1150.1 | 122.1 | 125.7 | 0.20 |
| lora_replay_targeted | 3 | 149.1 | 1514.7 | 167.4 | 133.0 | 0.20 |
| ctx_nn | 3 | 164.6 | 1591.4 | 148.0 | 96.1 | 0.07 |
| moca_lite | 3 | 275.3 | 2680.2 | 276.3 | 194.3 | 0.00 |
| fresh | 3 | 418.7 | 3057.9 | 434.1 | 230.0 | 0.00 |
| base | 3 | 429.0 | 4202.0 | 439.8 | 235.9 | 0.00 |

#### w-spread

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| cxl | 3 | 45.9 | 435.6 | 48.2 | 12.4 | 0.80 |
| ctx | 3 | 83.7 | 523.6 | 85.1 | 63.0 | 0.00 |
| coin_sum | 3 | 139.1 | 1057.8 | 98.2 | 8.2 | 0.93 |
| lora_ttt | 3 | 183.7 | 1093.1 | 86.8 | 144.2 | 0.00 |
| coin | 3 | 159.5 | 1137.0 | 117.5 | 8.1 | 0.93 |
| lmel | 3 | 156.9 | 1337.4 | 132.6 | 50.3 | 0.20 |
| ssl_lmel | 3 | 167.2 | 1377.9 | 147.1 | 67.7 | 0.07 |
| ctx_ident | 3 | 173.2 | 1437.3 | 157.4 | 80.8 | 0.00 |
| C_pub | 2 | 187.9 | 1615.8 | 183.2 | 66.2 | 0.00 |
| lt_sft_sum | 3 | 201.4 | 1629.0 | 201.0 | 75.9 | 0.00 |
| ctx_nn | 3 | 203.8 | 1822.7 | 264.1 | 73.9 | 0.07 |
| C | 3 | 220.9 | 1968.9 | 230.9 | 65.1 | 0.00 |
| lora_replay | 3 | 195.2 | 2046.2 | 231.8 | 131.1 | 0.20 |
| C_gain | 2 | 228.1 | 2075.1 | 262.2 | 74.8 | 0.00 |
| lora_replay_targeted | 3 | 204.5 | 2252.4 | 247.8 | 133.9 | 0.20 |
| moca_lite | 3 | 320.4 | 3353.7 | 359.2 | 203.2 | 0.00 |
| fresh | 3 | 519.7 | 4042.4 | 557.3 | 236.5 | 0.00 |
| base | 3 | 534.9 | 5565.9 | 568.7 | 240.9 | 0.00 |

#### w-magnitude

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| cxl | 3 | 11.9 | 65.6 | 13.1 | 8.3 | 0.80 |
| lmel | 3 | 19.2 | 138.1 | 15.9 | 13.7 | 0.73 |
| C_pub | 3 | 20.2 | 152.3 | 22.1 | 9.6 | 0.80 |
| ctx | 3 | 40.7 | 153.6 | 47.8 | 45.8 | 0.07 |
| ctx_ident | 3 | 42.5 | 179.9 | 48.5 | 40.3 | 0.00 |
| C_gain | 2 | 50.5 | 302.2 | 63.2 | 35.2 | 0.40 |
| ssl_lmel | 3 | 61.1 | 430.2 | 62.3 | 40.1 | 0.33 |
| coin_sum | 3 | 74.1 | 512.4 | 61.0 | 19.7 | 0.60 |
| lora_ttt | 3 | 100.3 | 533.0 | 42.5 | 170.4 | 0.00 |
| C | 3 | 73.7 | 547.4 | 85.4 | 17.7 | 0.60 |
| coin | 3 | 96.3 | 653.4 | 64.1 | 14.4 | 0.60 |
| lt_sft_sum | 3 | 111.2 | 955.0 | 123.3 | 46.5 | 0.20 |
| lora_replay | 3 | 131.2 | 1328.3 | 137.6 | 132.4 | 0.00 |
| ctx_nn | 3 | 188.1 | 1711.9 | 150.5 | 101.6 | 0.07 |
| lora_replay_targeted | 3 | 154.7 | 1713.1 | 182.2 | 137.2 | 0.00 |
| moca_lite | 3 | 282.6 | 2779.7 | 280.3 | 183.5 | 0.00 |
| fresh | 3 | 426.7 | 3170.8 | 439.0 | 207.0 | 0.00 |
| base | 3 | 435.5 | 4362.3 | 444.9 | 212.2 | 0.00 |

#### w-interact2

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| oracle | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 1.00 |
| cxl | 3 | 37.2 | 335.7 | 45.6 | 6.4 | 1.00 |
| ctx | 3 | 58.3 | 452.7 | 96.3 | 72.1 | 0.10 |
| ctx_ident | 3 | 74.0 | 702.8 | 84.5 | 76.2 | 0.10 |
| lmel | 3 | 73.2 | 761.6 | 80.0 | 6.5 | 1.00 |
| ssl_lmel | 3 | 84.3 | 773.5 | 87.1 | 49.7 | 0.29 |
| coin | 3 | 103.4 | 827.6 | 75.3 | 8.4 | 1.00 |
| C_pub | 3 | 84.6 | 913.4 | 99.4 | 31.0 | 0.81 |
| coin_sum | 3 | 118.6 | 1003.2 | 91.0 | 8.4 | 1.00 |
| C | 3 | 114.2 | 1118.0 | 130.0 | 48.9 | 0.48 |
| C_gain | 3 | 142.9 | 1215.3 | 149.7 | 78.8 | 0.24 |
| ctx_nn | 3 | 186.1 | 1452.8 | 157.0 | 143.0 | 0.00 |
| lora_ttt | 3 | 287.7 | 1649.2 | 108.7 | 204.9 | 0.10 |
| lt_sft_sum | 3 | 204.3 | 1797.8 | 194.3 | 101.5 | 0.14 |
| lora_replay | 3 | 256.8 | 2192.6 | 183.3 | 184.0 | 0.14 |
| lora_replay_targeted | 3 | 254.2 | 2454.6 | 214.4 | 191.3 | 0.10 |
| moca_lite | 3 | 349.9 | 3463.5 | 328.5 | 267.2 | 0.00 |
| fresh | 3 | 573.2 | 4013.6 | 537.7 | 349.1 | 0.00 |
| base | 3 | 589.0 | 5774.3 | 546.8 | 357.7 | 0.00 |

#### w-interact (old; reported ceiling only, not in any decision)

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| lmel | 3 | 217.5 | 1564.9 | 151.6 | 3.0 | 1.00 |
| cxl | 3 | 220.0 | 1607.4 | 96.2 | 3.4 | 1.00 |
| lora_ttt | 3 | 285.4 | 1650.4 | 112.2 | 140.7 | 0.20 |
| C_pub | 2 | 213.0 | 1734.5 | 121.5 | 6.3 | 1.00 |
| C_gain | 2 | 234.2 | 1769.1 | 164.1 | 25.2 | 0.40 |
| coin | 3 | 280.4 | 1779.0 | 164.9 | 6.4 | 1.00 |
| coin_sum | 3 | 270.3 | 1818.3 | 171.5 | 6.5 | 1.00 |
| C | 3 | 270.6 | 2032.5 | 204.0 | 22.3 | 0.73 |
| ctx | 3 | 254.2 | 2270.2 | 185.7 | 43.6 | 0.27 |
| lora_replay | 3 | 307.6 | 2873.6 | 239.8 | 125.7 | 0.20 |
| fresh | 3 | 573.2 | 4013.6 | 537.7 | 230.0 | 0.00 |
| base | 3 | 589.0 | 5774.3 | 546.8 | 235.9 | 0.00 |

#### w-alias

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| lmel | 2 | 3.2 | 15.0 | 2.0 | 3.1 | 1.00 |
| cxl | 3 | 7.0 | 31.1 | 6.1 | 4.6 | 1.00 |
| C_pub | 2 | 7.6 | 91.8 | 11.7 | 5.2 | 1.00 |
| ctx | 2 | 26.8 | 118.4 | 25.4 | 29.2 | 0.70 |
| lora_ttt | 2 | 81.5 | 420.8 | 24.7 | 39.1 | 0.60 |
| coin | 2 | 77.4 | 493.6 | 37.5 | 4.4 | 1.00 |
| C | 2 | 95.9 | 643.6 | 96.6 | 7.2 | 1.00 |
| lora_replay | 2 | 96.9 | 1057.6 | 103.2 | 22.7 | 0.60 |
| fresh | 2 | 426.4 | 3376.0 | 428.8 | 233.3 | 0.00 |
| base | 2 | 432.1 | 4600.0 | 434.5 | 239.1 | 0.00 |

#### w-null (control; excess near 0 for everything is expected)

| arm | seeds | ex20 | ex200 | ex20 visit 2 | return ex20 | returns recognised |
|---|---|---|---|---|---|---|
| base | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 1.00 |
| C | 2 | 0.0 | 0.1 | 0.0 | 0.1 | 1.00 |
| C_pub | 2 | 0.0 | 0.1 | 0.0 | 0.1 | 1.00 |
| lmel | 2 | 0.0 | 0.1 | 0.0 | 0.1 | 1.00 |
| coin | 2 | 0.4 | 3.9 | 0.6 | 0.6 | 1.00 |
| cxl | 3 | 0.5 | 6.3 | 1.2 | -0.3 | 1.00 |
| lora_replay | 2 | 0.5 | 8.1 | 1.6 | 0.7 | 1.00 |
| ctx | 2 | -0.4 | 17.2 | 1.7 | -0.0 | 1.00 |
| fresh | 2 | 0.8 | 97.5 | 1.6 | 2.2 | 0.50 |
| lora_ttt | 2 | 11.9 | 106.3 | 10.9 | 16.5 | 0.00 |

### 3.2 Decision statistics (PILOT_report.md §2, §5, §7) [checked]

Bar 1, C vs labelled COIN:

| cell | seeds | mean C | mean COIN | ratio (95% CI) | verdict |
|---|---|---|---|---|---|
| w-spread/20 | 3 | 220.9 | 159.5 | 1.38 (1.36, 1.43) | tie |
| w-spread/200 | 3 | 1968.9 | 1137.0 | 1.73 (1.67, 1.78) | tie |
| w-magnitude/20 | 3 | 73.7 | 96.3 | 0.76 (0.72, 0.79) | tie |
| w-magnitude/200 | 3 | 547.4 | 653.4 | 0.84 (0.66, 1.11) | tie |
| w-interact2/20 | 3 | 114.2 | 103.4 | 1.10 (0.93, 1.39) | tie |
| w-interact2/200 | 3 | 1118.0 | 827.6 | 1.35 (1.24, 1.51) | tie |

Bar 2, C vs cxl: ratio C/cxl 4.82 and 4.52 (w-spread), 6.19 and 8.34 (w-magnitude), 3.07 and 3.33 (w-interact2), at 20 and 200 frames. C_gain vs COIN in w-interact2: 1.38 and 1.47; C_gain had only 2 seeds in w-spread and w-magnitude ("insufficient"). Neither bar was met for C or C_gain, and the sensitivity sweep gave "False" in all 9 (α, floor) settings.

**Every verdict is "tie", including ratios of 8×.** With 3 seeds, the exact one-sided sign-flip test cannot give a p-value below 1/8 = 0.125, so it can never reach α = 0.01; the report's p-values are 0.125 at best [checked: p-values in PILOT_report §2–4; the 1/8 floor is my arithmetic]. The "tie" labels therefore mean "too few seeds to decide", not "equal". The direction is what the pilot can show.

**Writes (births of new causes)**, mean per seed at end of training + gap (PILOT_report §6) [checked]:

| world | C | C_pub | C_gain |
|---|---|---|---|
| w-site | 29.3 | 6.5 | 3.3 |
| w-spread | 31.0 | 8.5 | 37.5 |
| w-magnitude | 12.0 | 23.3 | 3.5 |
| w-interact2 | 15.7 | 13.0 | 21.3 |
| w-alias | 2.0 | 6.5 | — |
| w-null | 0.0 | 0.0 | — |

For reference, the world has two true causes. w-null: C and C_pub made 0 births in 2 seeds each; the bound "holds" (p_exceed 1.0) [checked].

### 3.3 Derived comparisons (my arithmetic on the pilot means) [inferred]
Because the main run would tune C over a grid that includes C_pub's configuration, the fairest pilot view of "C at its best" is the best of C, C_pub and C_gain in each world:

| world | best C variant, ex200 | cxl ex200 | ratio best-C / cxl | labelled COIN ex200 | ratio best-C / COIN |
|---|---|---|---|---|---|
| w-site | C_pub 95.7 | 17.2 | 5.6 | 621.4 | 0.15 |
| w-spread | C_pub 1615.8 | 435.6 | 3.7 | 1137.0 | 1.42 |
| w-magnitude | C_pub 152.3 | 65.6 | 2.3 | 653.4 | 0.23 |
| w-interact2 | C_pub 913.4 | 335.7 | 2.7 | 827.6 | 1.10 |
| w-alias | C_pub 91.8 | 31.1 | 3.0 | 493.6 | 0.19 |

C_pub has only 2 seeds in w-site, w-spread and w-alias.

---

## 4. What we inferred

1. **The existing-methods stack beat every C variant in every world, by 2.3× to 27× at 200 frames** (section 3.2 and the table above) [checked numbers; inferred conclusion]. This is the clearest pilot signal: no seed count would plausibly close a gap of 2–8× in the decisive worlds, though the pilot cannot prove it [inferred]. The builder also chose not to compute power for bar 2: "the question is not detectability of a C advantage but whether any exists" (PROTOCOL §7) [checked].
2. **C beats labelled COIN only where its assumption holds or nearly holds.** Its best form (C_pub) was well below COIN in w-site (0.15×), w-magnitude (0.23×) and w-alias (0.19×), and above COIN in w-spread (1.42×) and w-interact2 at 200 frames (1.10×) [inferred from the pilot table]. These are the only "wins" for C in the project's record, and they are against a classic baseline, not the strongest one (all-tests-ledger-2026-10-28.md test 7) [checked].
3. **The edge in w-site comes from the parameterisation, not from C's unlabelled inference.** The labelled minimal-edit library `lmel` reached 12.6 nats at 200 frames, against 95.7 for C_pub and 472.4 for C [checked]. `cxl`, which uses lmel's edit recipe on pseudo-labels, reached 17.2 without labels [checked]. So "store minimal site-local edits and sum them" works in the home world, but it works better inside an existing label-free pipeline (clustering + edits + snapshots + context learners + mixture) than inside C's own birth-by-birth loop [inferred].
4. **Discrete causes did not earn their place against a plain continuous context learner.** `ctx` had lower mean excess than C in all four decisive and control worlds (for example w-spread 523.6 vs 1968.9; w-magnitude 153.6 vs 547.4) [checked]. The protocol had said in advance that if ctx matches C in w-spread and w-magnitude, "discrete causes are not earning their place" is the reported conclusion for those worlds (PROTOCOL §8) [checked]. ctx did better than match. Caveat declared in advance: the additive worlds are inside ctx's class by construction (PROTOCOL §8) [checked].
5. **What it does not say.** It does not say C's mechanism has no value at RoboJEPA scale; nothing here touched RoboJEPA. It does not say cxl is optimal, or that the ordering would hold after tuning. It does not test P2 dormancy at long gaps (the gap is 15,000 frames) or P5 savings on changed returns beyond the returns rule (IMPLEMENTATION.md §7) [checked].

**Predictions vs pilot** (PROTOCOL §8 vs the pilot tables) [inferred]:
- Held: C beat COIN in w-site by mean (0.72× and 0.76×, inside the predicted 0.3–0.8); lmel beat C in w-site; C did not beat COIN in w-spread; C_gain < C in w-magnitude (302.2 vs 547.4); moca_lite weakest structured arm; ctx_nn worse than ctx at 200 frames; cxl within [0.7, 1.5] of lmel in w-site (1.37×).
- Missed: ctx was much better than "competitive" with C in w-site (ctx/C 0.35 at 200); C was worse than COIN in w-alias (643.6 vs 493.6), not a tie; ctx_ident was *worse* than ctx in w-spread (1437.3 vs 523.6), against the prediction "ctx_ident ≤ ctx"; lora_replay_targeted was 1.32× lora_replay in w-site, outside the predicted [0.85, 1.15]; cxl vs C in w-interact2 was 3.3×, far outside the predicted [0.7, 1.4].

---

## 5. Why we think it happened

- **w-spread breaks C's description length rule.** A cause there needs 3–4 terms over several sites (PROTOCOL §2 table) [checked]. C grows at most `max_terms` rank-1 terms per cause (3 in the literal config; 6 in some grid configs) from one segment of about 333 fit frames, and each term must pass an inner held-out gain of log 20 (IMPLEMENTATION.md §3, rows 2 and 4) [checked]. "3 rank-1 terms cannot describe a cause of 4+ terms (w-spread)" (same row) [checked]. So each birth is an incomplete description, and a sum of incomplete descriptions is wrong at the unseen combination [inferred]. COIN's rank-4 all-site snapshots, cxl's snapshots and ctx's every-site basis do not have this restriction [inferred from IMPLEMENTATION §5–6].
- **The literal config over-births.** In the literal core's `birth='top'` mode, a discrete strength 2X becomes a sum of increments X + X′, so the unseen (2,2) needs four causes, more than the s_max = 2 the mixture allows; in an early pilot seed this mode produced 20 births and kept writing in the gap phase (IMPLEMENTATION.md §3 row 5) [checked]. The pilot C averaged 29.3 births per seed in w-site against two true causes [checked]. This explains much of the gap between pilot C (472.4) and C_pub (95.7) in w-site [inferred]. The main run's tuning would probably have selected a 'base'-mode config [guess].
- **w-magnitude: fixed-size causes vs continuous strength.** Each new strength is a deviation from any stored cause; C either births another cause or leans on its rebuild learner. C_gain addresses this but learns strength only inside the span of stored causes, with a beam that can drop the right hypothesis (IMPLEMENTATION.md §4) [checked]. ctx and cxl's context learners represent strength as a coordinate (STACK.md C2) [checked]. That matches the observed order cxl < lmel ≈ C_pub ≈ ctx < C_gain < C at 200 frames [inferred].
- **w-interact2: unit-strength sums cannot represent an interaction of size 1 or 3.** The builder noted in advance that C "contains sums of causes with unit strength, so I at size 1 or 3 is not directly representable; C_gain and ctx can scale it" (IMPLEMENTATION.md, w-interact2) [checked]. COIN starts its novel learner from the mean snapshot, which C's own core had already identified as a better start than C's zero or MAP-sum start in an interacting world (C/CORE.md §E.1) [checked].
- **Why cxl wins broadly.** cxl's fixed-share mixture contains, as members, the kinds of predictor that win in each world (minimal edits for w-site, snapshots for spread changes, context learners for strength), and a fixed-share mixture's log-loss is within about log(1/prior mass) of its best member (STACK.md §0, cited there as known, "not re-checked") [checked that STACK says so; the bound itself not verified by me]. So cxl "tracks the best existing arm per world without labels", as predicted in PROTOCOL §8.2 [checked]. In w-interact2 it was also best, likely because its context learners can scale the visible interaction [guess].
- **Why ctx_ident did worse than ctx in w-spread.** With d = 2 and rank ≤ 2 per module, it may be too small to describe causes spread over 3–4 site-terms each [guess; the builder only predicted the opposite].

---

## 6. Caveats and what could be wrong

- **Pilot only, and untuned.** Every arm ran its first config, not its tuned one. Tuning could reorder arms, especially those whose first config is a poor default (the literal-core C is a clear case). The C_pub row is the best guide to tuned C, and it still lost to cxl everywhere [checked numbers; inferred interpretation].
- **2–3 seeds; the test cannot reach significance.** The minimum attainable p is 0.125. The intervals in the report are bootstrap intervals over 2–3 seeds and are far too narrow to trust [inferred].
- **The main run never happened.** It is built, pre-registered and packaged: 7 worlds × 20 arms × 24 seeds = 3,360 jobs, about 118 core-hours, plus 25 core-hours of tuning, 143 core-hours in total; about 0.8 hours of wall time on 190 workers if perfectly packed; the optional cxl ablation block adds about 170 core-hours (estimate.md) [checked]. The project ledger lists it under "never tested ... waiting on the machine", about $10 for both tests (all-tests-ledger "Never tested") [checked]. The memory note says the question of how to get machine access (Remote Control vs Lucky running the README command) went unanswered (tests-v6-status.md) [checked].
- **Faithfulness of C's implementation.** The builder re-ran C's original pipeline and got identical numbers; the C_pub port matched in behaviour, not in digits, on one seed (C_pub 10.1 / 12.0 / 43.8 vs the pipeline's 29.5 / 10.3 / 49.5 at 20 frames) (IMPLEMENTATION.md §1) [checked]. Sixteen open choices are listed with their likely effect; some favour C (free segmentation, no beam), some hurt it (noisier causes from 333-frame fits; sleep re-description and volatility not implemented) (IMPLEMENTATION.md §3) [checked]. The missing M4(iii) re-description is the step C's core expected to reach ≤ 0.25× COIN (C/CORE.md §D.6) [checked].
- **cxl was not independent of the project.** It was specified by a project researcher, and a single-seed sanity version had already been seen to do well in w-site and w-spread before amendment 3 added it (STACK.md §0) [checked]. It is assembled from existing methods, but the assembly is the project's own; "existing stack" means "existing parts". Its pilot gap episodes used only the first 100 frames of each, for compute (IMPLEMENTATION.md, Third amendment) [checked].
- **Approximate competitors.** The six amendment-2 arms are numpy approximations from a literature summary, not checked against the papers [checked]. ctx is not an RMA encoder; ctx_nn is a non-recurrent 20-frame encoder [checked]. A weak approximation would make that arm look worse than the real method [inferred].
- **Toy scale.** A small frozen toy base, iid noise, abrupt changes at known episode boundaries (given free to every arm), short 200-frame tests, gaps of 15,000 frames rather than 10⁵–10⁶ (IMPLEMENTATION.md §3 row 1, §7) [checked].
- **A record discrepancy.** The memory note tests-v6-status.md says "ssl_lmel/ctx_nn etc. didn't beat C". In the pilot tables, ssl_lmel had a lower mean ex200 than C in w-site (349.9 vs 472.4), w-spread (1377.9 vs 1968.9), w-magnitude (430.2 vs 547.4) and w-interact2 (773.5 vs 1118.0), and ctx_nn was lower in w-spread (1822.7 vs 1968.9) [checked: pilot/summary.md]. None of these is significant at 3 seeds, so "didn't beat" is defensible only in the significance sense. The primary file should be preferred [inferred].
- **Power rule mismatch.** power.py simulates a t-test, while analysis.py uses a sign-flip randomisation test (power.py docstring and code; PROTOCOL §6) [checked]. At 24 seeds the two should behave similarly, but the declared power is for the t-test [inferred].

---

## 7. What it changed in the project afterwards

- **C's idea was not carried forward as a method.** The ledger's pilot verdict: "C would fail its bar. The existing stack wins" (all-tests-ledger test 7) [checked]. Memory note tests-v6-status.md records: "existing label-free stack `cxl` best in every world incl. valid w-interact2; C 3-10x worse, fails both bars (COIN and cxl)" [checked].
- **The project moved toward adopting existing stacks.** The next architecture (arch-v8, 2026-10-20) was built "on adopted stacks": frozen RoboJEPA plus additive, never-overwritten records "under one key (cxl edits, heb stats, episode vault, verified plan-cache habits, offline gated LoRA w/ replay)", with 9 of 10 mechanisms adopted and 1 invented (tests-v6-status.md) [checked]. STACK.md had already argued that "the C candidate must be compared against `cxl`" (STACK.md §0) [checked].
- **The remaining gaps were restated.** STACK.md named three break points for existing methods: label-free composition when causes alias or interact; savings for slightly changed returns and a "memory of errors"; protection of dormant knowledge at scale while a shared component keeps learning (STACK.md §0) [checked]. The dormant-knowledge gap later led to T-DORMANT (all-tests-ledger test 10; tests-v6-status.md) [checked that T-DORMANT followed; the causal link is inferred].
- **A pattern confirmed.** The ledger's one-line pattern, "our principle was sound in a clean toy, sometimes beat a classic baseline, and then lost to a simpler or existing method once that method was tuned and given the same knowledge", fits T-CAUSE exactly, with the caveat that here the competitors were not yet tuned either (all-tests-ledger "The one-line pattern") [checked quote; caveat inferred].
- **Still open.** The pre-registered main run, tuning and cxl ablations remain unrun. If the machine is ever rented, running them would turn this pilot into evidence: README-RUN.md gives the exact steps, and every job is resumable (README-RUN.md) [checked].
