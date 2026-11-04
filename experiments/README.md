# Standalone experiments (TOE project): world models and memory

This folder records every standalone experiment run in Lucky's TOE project (continual learning, memory and world models), up to 2026-11-02.

For each test, the folder records:
- **why** we ran it;
- **what** we built and ran, including every sub-run, crash and fix;
- **what we got**;
- **what we inferred**;
- **why we think it happened**.

The research notes the tests came from (geometry of learning, continual learning first principles, the CL paper study, the brain and robotics notes) are in `meta-lucky-builds/toe1`, on branch `meta-lucky-builds-files`.

## The three tests

| # | Test | Date | Question | Verdict | Folder |
|---|---|---|---|---|---|
| 1 | World-model geometry: Lucky's table vs the earlier combination | 2026-10-23 | Which latent geometry design gives the better JEPA world model? | On a toy damped point mass, the structured phase-space design (B) beat the table design (A) on every metric. The test is confounded and tilted toward B, so it is not decisive. | [`experiments/01-world-model-geometry`](01-world-model-geometry/EXPLANATION.md) |
| 2 | Fast/slow product-key memory with sleep consolidation | 2026-10-23 | Does consolidating fast memory into slow memory over shared addresses beat a single memory? | **Mostly killed.** Copy-all equals one memory exactly. All the gain comes from selection, and a plain counter ties the best hierarchy. One niche survives: noisy repeats without item identity. | [`experiments/02-sleep-consolidation`](02-sleep-consolidation/EXPLANATION.md) |
| 3 | Idea B: is the haystack wall a delayed write-credit problem? | 2026-10-23/24 | Does a local, write-time objective break the recurrent-memory wall without a curriculum? | **Mechanism killed (pre-registered).** The local probe got 10/10, but the end-only probe got 9/10 (p = 0.5). Exploratory lead: a linear state reader breaks the wall, which suggests read–write bootstrapping. | [`experiments/03-idea-b-write-credit`](03-idea-b-write-credit/EXPLANATION.md) |

Each folder has:
- `EXPLANATION.md`, the full story;
- `RESULTS.md`, the original write-up from the day;
- the code;
- the raw per-seed results.

## Timeline

1. **2026-10-23, repository review.** I read all the toe1 files and gave an honest take on each, plus the cross-connections. The geometry synthesis (`geomoflearn.md`) is a strong literature map. Its master template can't be falsified, but its sub-claims can.
2. **Same day, geometry discussion → Test 1.** I proposed a geometry per job of the world model. Lucky proposed a simpler table, and the smallest head-to-head test was run (3 seeds, about 7 minutes each on CPU).
3. **Same day, memory-hierarchy analysis → Test 2.** The analysis and a literature search found no breakthrough in the files. "Consolidation within a shared address space" was proposed as the most promising gap, and the toy kill test mostly killed it.
4. **Same day, memory field document review → idea B → Test 3.** A field document produced in another chat was reviewed. Three candidate ideas were extracted, Lucky chose idea B, and it was pre-registered and run on the source paper's exact code (43 runs, about 7 CPU-hours over two stages).

## What the three tests say together

These are inferences across tests, not results of any single test.

1. **In every test, the headline mechanism was not what mattered. Something simpler was.**
   - Test 1: the planning metric only helped when the dynamics under it were good, and B's advantage most likely comes from its structure matching the world.
   - Test 2: the memory hierarchy did nothing. *Selection* (which items get written and kept) did everything.
   - Test 3: *immediate* credit did nothing extra. Having *a reader at all* did everything.
2. **"Who decides what gets written, and with what information" keeps coming up as the real variable.**
   - Test 2: the counter wins because it knows item identity.
   - Test 3: writes improve only once something can read them.
   - This matches the "null cluster" in the memory field document: clever memory methods tie trivial rules once state bits and baselines are matched.
3. **Two of the three headline ideas were killed by their own pre-stated kill condition.** That is the system working as intended. Each kill also left a sharper hypothesis behind:
   - Test 2 → consolidation without item identity on cross-session streams;
   - Test 3 → read–write bootstrapping.
4. **Method lessons that held up:**
   - derive the trivial cases before coding (Test 2's identity result);
   - check parity against the source code (Test 3);
   - write the pre-registration before any treatment run (Test 3);
   - make checkpoints that resume bit-identically, because cloud containers restart;
   - record measurement bugs as bugs, not results (Test 1's singular-matrix crash);
   - list confounds explicitly (Test 1 changed three things at once).

## Reproduction check (2026-11-04)

`02-sleep-consolidation/main.py` was rerun from a clean copy on another machine (Windows, numpy 2.5.4). Every metric matched `results.json` to within 3.3e-16, which is floating-point noise. `noisy.py` and the torch experiments (01 and 03) were not rerun. Experiment 03's verdict was recomputed from its stored per-seed results, and the scripts in 02 and 03 now resolve paths from their own folder.

## Proposed but never run

| Proposed test | From | What it would settle |
|---|---|---|
| 2×2 ablation {free vs structured dynamics} × {diagonal vs low-rank belief}, on a harder world with a one-way gate and a bounce | Test 1 | Which component of design B actually helps, and whether it survives non-smooth dynamics |
| Zipfian keys + noisy repeats, with TF-IDF slot scoring and sparse allocation | Test 2 | Whether better write selection rescues same-address consolidation |
| Consolidation without item identity on cross-session streams (savings, spacing) | Test 2 | The sharper surviving version of the consolidation idea |
| Decoupled reader built from the model's own key projection, at L = 256 and with collision keys | Test 3 | Whether the "reader breaks the wall" lead is a real training method |
| "Machine Sperling": linear probe on the frozen walled models' memory state | Test 3 | Whether the wall is a write failure or an access failure |
| Muon vs Adam plasticity under non-stationarity; Fisher-trust-region SFT vs SFT vs RL | Repo review (`geomoflearn.md` falsifiers) | The two cheapest falsifiers of the geometry synthesis |
| Idea A (memory as reverse water-filling) | Memory thread | Whether bit allocation by demand × uncertainty beats eviction and uniform quantisation under skewed demand |

## Not tests

These were analysis or writing, with no experiment run:
- the spectral-gauge-flow critique (math checks only);
- the repository review;
- the memory literature searches;
- the field-document review;
- the step-1 research prompt (`memory-field-doc/step1_prompt.md` in the project files).
