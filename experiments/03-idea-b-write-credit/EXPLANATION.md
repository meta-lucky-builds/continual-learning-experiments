# Test 3: Idea B, is the haystack "interference wall" a delayed write-credit problem?

**When:** 2026-10-23 to 2026-10-24, thread "Memory hierarchy across the papers".
**Files in this folder:**

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | Hypothesis, arms, validity checks and decision rules, written **before** any treatment run |
| `RESULTS.md` | The original write-up |
| `hay.py` | Model, task, training and evaluation. A copy of the arXiv:2609.16183 harness plus the probe |
| `parity_test.py` | Proves `hay.py` matches the original code |
| `resume_test.py` | Proves checkpoint-resume is bit-identical |
| `analyze.py` | Tallies lock-ins and applies the pre-registered decision rules |
| `run_stage1.sh`, `run_stage2.sh` | Job queues |
| `results/*.json` | 43 per-seed results |
| `logs/` | Run logs |

**Licence note.** `hay.py` copies code from github.com/JIBSIL/dualgoose (MIT License, © 2026 Julian Boesch and Andrew Wee), as stated in its header. The scripts now resolve paths from their own folder; the original runs used `/mnt/project-files/idea-b-write-credit/`, which still appears in `logs/`.

## 1. Why we ran it

1. **The memory field document.** Earlier in the thread, a document produced in another chat found a "null cluster": most clever memory methods tie trivial rules once state bits, tuning and baselines are matched. It also cited arXiv:2609.16183, where fixed-state recurrent memories (Gated DeltaNet and relatives) sit at chance on haystack retrieval until a **training curriculum** fixes them (recall 0.021 → 1.000 with no architecture change). That paper diagnoses "interference under sparse supervision": the write gate gets almost no gradient.
2. **Idea B.** When Lucky asked for elegant, breakthrough-tier ideas across all files, three candidates came up. Idea B said:
   - the *deeper* cause of the wall is **delayed credit for the write**;
   - a write, or a damaging distractor write, is judged only by a loss that arrives at the far-away query;
   - the curriculum works because it shortens that delay.
3. **The prediction.** A **local write-time objective**, which credits each write immediately, should break the wall *without a curriculum*. If true, it would be a cheap, practical training method for long-context recurrent memory.

Lucky picked idea B and asked for a test that is *"as representative and accurate to the idea as possible… no mistakes in the test… extremely honest"*. That is why the test reuses the paper's exact code, checks parity, and was pre-registered.

## 2. What we built

**Testbed: the paper's own model and task, copied from its public code.**
- **Cell:** causal "armed" Gated DeltaNet (gated delta rule + width-4 short conv), 2 layers, d = 32, 49,508 parameters.
- **Task:** haystack retrieval with lexical distractors. L = 64, 4 key–value pairs, vocabulary 64; chance ≈ 1/54 ≈ 0.019.
- **Training:** 3,000 steps, batch 256, AdamW lr 1e-3 with cosine schedule to 1e-4, gradient clip 1.0.
- **Evaluation:** the paper's protocol. Maximum gap, 2,048 sequences, the model's own masked-token prediction.
- **Rebinding control:** swap the table values. If accuracy collapses to chance, the model is really reading the binding.

**The treatment (the only new code, off unless enabled):**
- Each layer's recurrent state S_t gets a probe per table key: r = S_t·pᵢ.
- A linear head decodes valueᵢ from r. The loss is cross-entropy, with λ = 1 fixed in advance.
- **The probe is not used at evaluation.**

**Arms:**

| Arm | Supervision | Curriculum | State probe | Role |
|---|---|---|---|---|
| A0 naive | 1 query | no | none | Checks that the wall reproduces |
| A1 dense | 4 queries | no | none | Control |
| A2 dense + curriculum | 4 queries | yes | none | Positive control (the paper's fix) |
| **A3 dense + local probe** | 4 queries | no | **every step** after the pair is written | **Treatment** (immediate credit) |
| A4 dense + end-only probe | 4 queries | no | **last position only** | Locality control: same probes, heads and targets, but credit must travel the whole gap |

**Pre-registered rules** (see `PREREGISTRATION.md` for the full text):
- **Lock-in:** a seed counts as locked in when eval accuracy > 0.8 *and* rebinding accuracy < 0.1. Counts are compared with a one-sided Fisher exact test.
- **Validity checks:**
  - V1: A0 is at chance on every seed.
  - V2: A2 locks in on ≥ 3/10.
  - V3: A1 locks in on ≤ 2/10.
- **Verdicts:**
  - SUPPORTED if A3 ≥ 6/10, A3 > A1 and A3 > A4.
  - MECHANISM KILLED if A3 > A1 but not A3 > A4.
  - KILLED if A3 is not above A1.
- **Staging:** stage 2 (A4) runs only if A3 > A1.

## 3. Every run we did, in order

| # | Run | Why | What happened |
|---|---|---|---|
| 1 | **Parity test** (`parity_test.py`) | Any difference from the paper's code would make the result meaningless | Passed. Init, forward pass, loss, data and a full 20-step train + eval run are bit-identical to the original package. |
| 2 | **Timing run** | Size the compute budget | About 1 s per step on one CPU thread, so about 50 min per 3,000-step run. 4 runs in parallel. |
| 3 | **Pre-registration written** | Lock the rules before seeing any treatment result | `PREREGISTRATION.md`. |
| 4 | **Stage 1, first launch** | Main runs | **Killed by a machine restart** about 40 minutes in, before any run finished. The cloud container pauses when the session is idle, and the job had been started detached. No results were written. |
| 5 | **Checkpointing plus resume test** (`resume_test.py`) | Make restarts harmless | Checkpoint every 250 steps. A resumed run is **bit-identical** to an uninterrupted one, tested on a plain arm and on the local-probe arm. |
| 6 | **Stage 1, relaunched** as a session-tracked task | Keep the machine awake | Ran through idle periods. Each tracked task is capped at 2 hours, so it was relaunched at each cap from checkpoints. One relaunch waited for Lucky's approval. Interim peeks were taken from the logs only. |
| 7 | **Stage 1 complete**: A0 × 3, A1 / A2 / A3 × 10 seeds | – | V1–V3 pass. A3 > A1 (p = 6e-5), so stage 2 is triggered. |
| 8 | **Stage 2**: A4 × 10 seeds | Pre-registered locality control | A4 = 9/10, so A3 vs A4 gives p = 0.50. |
| 9 | **Analysis** (`analyze.py`) | Apply the rules mechanically | Verdict: **MECHANISM KILLED**. |

## 4. What we got

| Arm | Lock-in | Per-seed eval accuracy |
|---|---|---|
| A0 naive | 0/3 | 0.018, 0.022, 0.013 |
| A1 dense | 1/10 | all ≈ 0.02 except seed 3 = 1.000 |
| A2 dense + curriculum | 6/10 | 0.957, 0.916, 0.999, 1.000, 0.100, 0.996, 0.795, 0.286, 0.553, 0.975 |
| **A3 local probe** | **10/10** | all 1.000 |
| **A4 end-only probe** | **9/10** | all 1.000 except seed 8 = 0.205 |

- **Rebinding:** every locked-in seed falls to about 0.02 under rebinding, so the binding is genuine.
- **Controls match the paper:** dense 1/10 (paper: 1/10), curriculum 6/10 (paper: 5/10 causal, 7/10 bidirectional).

| Fisher one-sided | Counts | p |
|---|---|---|
| A3 > A1 | 10/10 vs 1/10 | 6.0e-5 |
| **A3 > A4** | 10/10 vs 9/10 | **0.50** |
| A4 > A1 | 9/10 vs 1/10 | 5.5e-4 |
| A2 > A1 | 6/10 vs 1/10 | 0.029 |

**Probe accuracy at the end of the sequence** (layer 1, layer 2):
- A3: both layers at 1.0 on every seed.
- A4: layer 1 at chance on 9/10 seeds (only seed 4 is above it), layer 2 at about 1.0 except seed 8 (0.164). The first write-up said 8/10; corrected against `results/A4_*.json`.

## 5. What we inferred

1. **Pre-registered verdict: idea B's mechanism is killed.** Immediate credit to each write is *not* what breaks the wall. The same probe applied only at the end, where credit has to cross the whole haystack exactly like the task loss, works almost as well (9/10 vs 10/10).
2. **Exploratory: a decoupled linear reader of the memory state breaks the wall,** with no curriculum and better than the curriculum (A3 vs A2: p = 0.043; A4 vs A2: p ≈ 0.15). This was found after the fact. The probe adds **no new information**: its targets are the same 4 values the dense queries already supervise. What it adds is a *path*, a fixed linear reader of each layer's state.
3. **Exploratory: locality buys speed, not lock-in.** From task-loss curves logged every 250 steps, A3 first drops below loss 1 at step 500–750 on every seed. A4 gets there at 750–2,250, and one seed never does.
4. **Exploratory: local supervision forces storage into every layer it touches.** In A4 the binding lives only in layer 2. In A3 both layers hold it.

## 6. Why we think it happened (hypothesis, needs its own pre-registered test)

**The wall looks like a read–write co-adaptation (bootstrapping) problem, not a delay or capacity problem.**
- The write path (what gets put into S) can only be shaped by gradient that flows back through *a reader* of S.
- The model's own reader is a two-layer circuit that must itself be learned. Until writes are clean, it has nothing to learn from, so it stays poor. Until it exists, writes get no useful gradient. That makes a chicken-and-egg loop, and most seeds never escape it (dense alone: 1/10).
- The probe supplies a reader that is useful from step 0: a single linear map from S. Gradient reaches the write path immediately, writes become clean, and then the model's own reader can learn on top of clean writes.
- **Why distance didn't matter.** The gated delta-rule state is (gated-)linear in its writes. So once a linear reader exists, gradient from the last position flows back through S to an early write with little distortion. The obstacle was the *absence* of a reader, not the *length* of the path. That is why the end-only probe works nearly as well as the every-step probe.
- **Why the curriculum also helps.** Short gaps let the model's own reader learn on easy cases first, which is the same bootstrapping solved a different way.

## 7. Caveats

- **One toy setting:** L = 64, d = 32, lexical distractors, the causal cell. The paper's harder settings (L = 256/512, collision keys) were not run.
- **The probe is key-indexed.** There is one probe per key, which is possible because keys are always tokens 5–8 here. In a language model keys are not fixed symbols, so this probe does not transfer as-is.
- **λ = 1 was fixed in advance and not tuned.** The probe heads use their own optimiser, created after the base model, so base-model initialisation is identical across arms.
- **Runs were interrupted and resumed,** by container restarts and the 2-hour task cap. `resume_test.py` shows resume is bit-identical.
- **One run-script launch failed** with "Permission denied" in `logs/runner.out` (the .sh was not executable). The job was then started another way.
- **The verdict is not a breakthrough.** It is a clean negative on the mechanism plus a modest lead. An auxiliary probe loss on internal state is a known family of tricks (deep supervision).

## 8. What we would run next (proposed, never run)

1. **Decoupled reader without key identity.** Build the probe query from the model's own key projection of the query token. Test at **L = 256**, where the paper's curriculum fails on every seed, and on collision keys. Kill condition: no gain over dense supervision at L = 256. Pre-register before running.
2. **"Machine Sperling" probe** (proposed 2026-10-25). Freeze the walled dense-only models and fit a linear probe on their memory state after training.
   - If the values are recoverable, the wall is an *access* failure: written but not readable.
   - If not, the writes themselves failed.
   - This directly tests the bootstrapping reading.

## 9. How to reproduce

```bash
pip install torch numpy
# run from this folder
git clone https://github.com/JIBSIL/dualgoose /tmp/dualgoose && python3 parity_test.py /tmp/dualgoose
python3 resume_test.py
bash run_stage1.sh && bash run_stage2.sh
python3 analyze.py
```
