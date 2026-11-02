# Idea B: pre-registration (written before any treatment arm was run)

Written 2026-10-23, before running any treatment arm. Only the parity test and a timing run had been done at that point.

## The idea under test

Fixed-state recurrent memories hit a "haystack wall" on retrieval: 4 key–value pairs, then distractors, then a query. They sit at chance, and a gap curriculum fixes this (arXiv:2609.16183). That paper diagnoses "interference under sparse supervision": the write gate gets almost no gradient.

**Idea B** says the deeper cause is *delayed credit for the write*. A write, or a damaging distractor write, is judged only by a loss that arrives at the far-away query. The curriculum works because it shortens that delay.

**The prediction:** a *local write-time objective*, which credits each write immediately, should break the wall *without* a curriculum.

## Testbed

This is the paper's own model and task, copied verbatim from its public code (`hay.py`). `parity_test.py` checks it is numerically identical to the original.

| Setting | Value |
|---|---|
| Cell | Causal armed Gated DeltaNet (`armed_gdn_fwd`): gated delta rule + width-4 short conv |
| Size | 2 layers, d = 32 |
| Task | Haystack retrieval with lexical distractors, L = 64, 4 pairs, vocabulary 64 (chance 1/54 ≈ 0.019) |
| Training | 3000 steps, batch 256, AdamW lr 1e-3 with cosine schedule to 1e-4, gradient clip 1.0 |
| Evaluation | The paper's protocol: maximum gap, 2048 sequences, the model's own masked-token prediction, plus the rebinding control |

## Arms

**Treatment arms** add a probe on each layer's recurrent state, r = S_t·p_i, with one probe per table key. A linear head decodes value_i from it. The loss is cross-entropy, with weight λ = 1 fixed in advance. The probe and its head are **not used at evaluation**.

| Arm | Supervision | Curriculum | Auxiliary state probe |
|---|---|---|---|
| A0 naive | 1 query | no | none |
| A1 dense (control) | 4 queries | no | none |
| A2 dense + curriculum (positive control) | 4 queries | yes (uniform gap) | none |
| **A3 dense + local write objective (treatment)** | 4 queries | no | at **every step** after the pair is written |
| A4 dense + probe at the end only (locality control) | 4 queries | no | **only at the last position** (same probes, heads and targets) |

## Compute and staging (fixed before running)

**Cost.** One run takes about 1 s per step on one CPU thread, so roughly 50 minutes for 3000 steps. Four runs go in parallel.

**Stage 1.**
- A0: 3 seeds (0–2). This only checks that the wall reproduces, so it uses 3 seeds rather than the 5 written below.
- A1, A2, A3: seeds 0–9, interleaved.

**Stage 2.** A4 (seeds 0–9) runs only if A3 > A1 at p < 0.05. Otherwise the locality control cannot change the verdict.

**Parity.** `parity_test.py` passed before any run. Initialisation, forward pass, loss, data and a full 20-step training-plus-evaluation run are bit-identical to the original package.

## Outcome and statistics

- **Lock-in:** a seed counts as locked in when eval accuracy > 0.8, following the paper's Table 5.
- **Seeds:** 10 per arm for A1–A4. A0 gets 5 seeds, since it only checks that the wall reproduces.
- **Test:** Fisher exact test, one-sided, on lock-in counts.
- **Genuine binding:** locked-in seeds must show rebinding accuracy < 0.1. Otherwise they don't count as genuine binding.

## Validity checks (if any fails, the test is VOID rather than evidence)

- **V1:** A0 is at chance on every seed (accuracy < 0.05).
- **V2:** A2 locks in on at least 3 of 10 seeds. The paper reports 5 of 10 for this causal cell.
- **V3:** A1 locks in on at most 2 of 10 seeds, so a wall exists to be broken. The paper reports 1 of 10 for its bidirectional cell; it does not report the causal cell without curriculum.

## Decision rules (fixed now)

- **SUPPORTED:** A3 locks in on at least 6 of 10 seeds, A3 > A1 (p < 0.05), and A3 > A4 (p < 0.05).
- **MECHANISM KILLED:** A3 > A1, but A3 is not above A4 (p ≥ 0.05). Then probe supervision helps, but locality is not the reason.
- **KILLED:** A3 is not above A1 (p ≥ 0.05), with V1–V3 holding.

**Diagnostic:** the probe's end-of-sequence accuracy shows whether the bindings are *held in the state*, whether or not the model's own read uses them.

**Pre-declared exploratory follow-up (labelled post hoc if run):** if A3 is killed, I run λ ∈ {0.1, 10} at 5 seeds each. This checks that a negative isn't caused by one bad weight. These runs cannot turn a KILLED into a SUPPORTED. They would only motivate a new, separately pre-registered run.
