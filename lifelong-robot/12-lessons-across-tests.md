# 12 What the tests say together, and why we think it keeps happening

This file reads files 01–10 as one body of evidence. Claims are tagged:
- **[checked]** when a result file shows it;
- **[inferred]** when it is our reading across tests;
- **[guess]** when nothing tests it.

## 1. The scoreboard

| # | Test | Our idea | Strongest competitor in the race | Outcome |
|---|---|---|---|---|
| 01 | Test M | evidence-gated writes (e-values) | none (absolute bar) | FAILED: bad writes grew about linearly under misfit |
| 02 | Test P | practice prior over edit directions | isotropic rank-one, full test-time training | UNCLEAR; condition (L) met in 0/20; margin over test-time training mostly from the rank-one form |
| 03 | T-COMPOSE | compose known causes by inference | COIN snapshot library | FAILED: 1.30× COIN's error (bar ≤0.25×) |
| 04 | T-CHURN | re-check after every write | LoRA + replay | FAILED on accuracy; writes now grow sub-linearly, but the pre-registered slope bar was met in only 1 of 3 misfit worlds |
| 05 | T-P14 | learned practice prior Γ | Reptile, LoRA from scratch | FAILED: Γ on needed 1.30× the frames of Γ off |
| 06 | cluster toys | L, C, M, S cores | per-cluster simple baselines | All four flattered by their own toys; L fatal |
| 07 | T-CAUSE (pilot) | store causes (C, C_pub) | existing label-free stack cxl | cxl beat every C variant in the decisive worlds (C_pub 2.3–3.7× worse at 200 frames; 5.6× in the w-site control); C_pub beat classic COIN in 3 worlds; 1–3 seeds, so no result is significant |
| 08 | T-P14D (pilot) | p14v6 | adapters, PBML, heb, RMA | No arm showed full P14; ours below generic adapters |
| 09 | p14v7 (pilot) | memory of errors | simpler twin, heb, adapters | Own kill test fired |
| 10 | T-DORMANT | evidence-driven forgetting | no-forgetting RLS, cxl | KILLED (12 wins / 11 losses / 1 tie) |

**Count [checked across files].**
- Pre-registered bars passed by one of our ideas against the strongest competitor: **0**.
- Times one of our ideas beat a *classic* baseline:
  - C_pub vs labelled COIN in T-CAUSE (pilot);
  - Test P's prior vs full test-time training;
  - core-v4 merge vs LoRA-TTT;
  - p14v7 vs weak baselines.

## 2. Five patterns that repeat

### Pattern A: the principle holds, the method loses [inferred, strong across tests]
Several ideas were right *as statements about what a good learner must do*:
- **Test M / T-CHURN.** Writing under a wrong base model must be re-checked, and once that was added, writes stopped growing linearly for merge_ref (sub-linear growth; the strict slope bar was still missed in 2 of 3 worlds; merge_noref stayed flagged linear at n = 2–4) [checked].
- **T-DORMANT.** Forgetting on a clock is ruinous: error went from 28 to 3283 as the idle gap grew in the worst arm; other clock arms reached 399–498 [checked; diagnostics outside the pre-registered bar].
- **The S audit.** Self-knowledge should audit your own committed forecasts [checked].

But the method we built to carry each principle lost to a simpler method that respects the same principle for free:
- LoRA + replay re-checks implicitly, through replay.
- No-forgetting RLS never forgets by the clock.
- cxl keeps snapshots and a fresh learner.

**Why [inferred].** A clean principle tends to have a simple implementation that someone already built. Our versions added machinery (e-values, stamps, directional forgetting, cause algebra). Every extra part had to be estimated from data, and the estimation error cost more than the principle gained.

### Pattern B: structure learning is where our ideas break [inferred, strong]
- **T-COMPOSE.** Composition by inference reached about 0.49× COIN (8 seeds; 0.41× on the first 5) when given the *true* structure, and 1.30× COIN when it had to learn the structure itself. Its structure learning picked the wrong site, which the toy made non-identifiable [checked].
- **Cluster L.** It worked only because it was handed the body's structure. A no-practice learner handed the same structure did better (fatal in the audit) [checked].
- **T-P14D.** When the robot had to *discover* its body structure, no arm showed P14 [checked, pilot].

**Why [inferred].** Our designs assume a factored world (causes that add, body channels, edit directions) and then must find the factors from a few frames of data. Finding factors is the hard part; using them is easy. Existing stacks avoid committing to a factorisation, using snapshots, minimal edits and mixtures, so they don't pay for a wrong one.

### Pattern C: "learning faster" was never achieved by anyone [checked, pilot level]
Our P14 attempts so far:
- Test P: a parameter prior;
- T-P14: a practice prior Γ;
- cluster L: motor forms;
- p14v6;
- p14v7: memory of errors.

None made learning faster in the sense Lucky defined: proportional to shared abilities, never negative, larger after varied practice. No published arm in T-P14D did either. In T-P14 more practice made our learner *slower*, which is loss of plasticity [checked]. In T-P14D generic adapters beat our designs [checked, pilot].

**Why [inferred].** Every attempt stored something *about parameters* (where changes lie, how big they are). Lucky's P14 is about transferring an *ability*. The literature has no mechanism for that either (research/literature-v6/BOTTLENECKS-LIT.md). This is a genuinely open problem, not only our failure.

### Pattern D: designer-built toys flatter the designer [checked]
- **Phase 3.** All four cluster toys were flattered: L was handed the grammar, S's receipt advantage was planted, and C's world matched C's assumption [checked: clusters-v5/AUDIT].
- **Earlier self-coverage.** Self-counts in the spec rounds ran high: the auditor counted 10 answered rows where the architect claimed 16–19.

**Why [inferred].** A designer's toy encodes the designer's model of the world, so the toy can't contradict it. Since Phase 4, tests are built by someone else, and competitors get the same knowledge.

### Pattern E: toys favour simple linear-Gaussian methods, and that cuts both ways [inferred]
- T-DORMANT's world was near-linear with no capacity pressure, which favours RLS-family arms. That is a stated deviation in its RESULTS.md [checked].
- T-CAUSE worlds are low-dimensional with a frozen base.

So "simple wins" partly reflects the toys. But our ideas were *designed* to win in such clean worlds, and they still lost there. A failure in a clean toy is strong evidence. A pass would have been weak evidence (research/architecture/tests-explained.md).

## 3. What we believe now, and how sure

| Belief | Confidence |
|---|---|
| The strongest existing combinations (cxl-type change stack, heb body identifier, replay, a fresh learner in a mixture) are the baseline to beat, and we have not beaten them | high [checked, pilot level] |
| Don't forget on a clock; keep snapshots; re-check writes; audit your own committed forecasts. These are sound principles | moderate to high [checked in toys] |
| Learning *faster* (real P14) is unsolved by anyone, including us | high for our arms; moderate for the field |
| Our toy wins would carry to RoboJEPA scale | unknown: no GPU test exists (file 11) |
| More design rounds without new data will help | low [inferred from the v3.x history: specs grew from 23k to 90k words without converging] |

## 4. What would change these beliefs
1. **Tuned main runs of T-CAUSE and T-P14D.** Pilots can flip: untuned arms, 2–8 seeds.
2. **One run on RoboJEPA with a GPU** (T-SPAN, or a public benchmark such as LIBERO-Lifelong).
3. **A capacity-pressure re-run of T-DORMANT.** Directional forgetting has its only real chance there.
4. **A single focused shot at learning faster** with a kill rule, tested on discovered rather than given structure.
