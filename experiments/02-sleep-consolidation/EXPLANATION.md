# Test 2: Fast/slow product-key memory with sleep consolidation

**When:** 2026-10-23, thread "Memory hierarchy across the papers".
**Files in this folder:**
- `pkmsleep.py` (model and stream)
- `main.py` (exact repeats)
- `noisy.py` (noisy repeats)
- `results.json`, `results_noisy.json` (raw numbers)
- `RESULTS.md` (the original write-up from the day)

The scripts contain absolute paths under `/mnt/project-files/memory-consolidation-test/`. Edit those, or create that folder, to rerun them.

## 1. Why we ran it

1. **The question.** Lucky asked what the toe1 files collectively say about memory: hierarchy, stages, and whether there is a "breakthrough" idea.
2. **The analysis.** It said that every memory write is one step of the same "fit new evidence, stay close to the old state" problem at a different timescale. That gives a ladder: context → fast weights → adapters → base weights. The analysis also said **no breakthrough idea was present** in the files.
3. **The literature search.** A search of 2026 memory architectures found that FwPKM (fast-weight product-key memory) already has a fast PKM and a slow PKM playing episodic and semantic roles, **but nothing moves between them**.
4. **The idea proposed** as the most promising open direction: **consolidation within one shared address space.** Give the fast and slow memories the same product-key addresses. During an offline "sleep", copy heavily used fast slots into the slow store, then clear the fast store. The hope was that a local, sparse transfer would avoid the interference that full distillation causes between consolidations.
5. **The kill condition, stated in advance:** if retention half-life is the same with and without sleep transfer, the idea is dead.

Lucky asked for a toy that is *"as accurate and representative of the real idea's math or core insight as possible"*.

## 2. What we built

**Memory (follows the PKM / FwPKM / delta-rule math):**
- **Addressing.** The query (d = 32) is split into halves. Each half is scored against 64 random unit sub-keys. The top 8 per half are combined into candidates, and the top k = 8 slots are kept out of 4,096, with softmax weights (temperature 0.05, effective k ≈ 5).
- **Read.** y = Σ wᵢ (Sᵢ + Fᵢ): slow plus fast, with shared addresses and an additive read.
- **Fast write.** NLMS/Kaczmarz one-step rewrite on the residual of the combined read: Fᵢ += wᵢ (v − y)/‖w‖². After the write, y equals v exactly.
- **Sleep.** At every task boundary, a consolidation rule may move content F → S, then F is cleared.

**Stream:**
- 100 tasks. Each task has 10 targets shown 3 times each, plus *n* distractors shown once, in random order.
- Only targets are ever queried. After every sleep, all targets seen so far are queried.
- Accuracy is nearest-neighbour decoding over the 1,000 target values (chance ≈ 0.001).

**Conditions:**

| Condition | What it is | Why it's there |
|---|---|---|
| `single` | One memory, writes on every sighting, never cleared | Baseline |
| `single2N` | Same with about 2× the slots | Parameter-matched to fast + slow |
| `oracle` | Single memory that writes only targets | Upper bound on filtering distractors |
| `count_gate` | Single memory that skips the first sighting in a session (needs item identity) | Tests whether the fast store is only acting as a counter |
| `none` | Fast + slow, nothing moves at sleep | Floor |
| `copy_all` | S += F for every slot | Sanity check: should equal `single` |
| `copy_sel` | S += F on slots whose address mass this session is ≥ τ (hard threshold) | The idea, hard version |
| `copy_soft` | S += min(1, mass/τ)·F | The idea, soft version |
| `replay` / `replay_all` | Write items seen ≥ 2× (last sighting / all sightings) into S, CLS-style | Standard alternative (needs item identity) |

## 3. Every run we did, in order

| # | Run | Why | What happened |
|---|---|---|---|
| 1 | **Derivation before any code** | Spot trivial cases first | With shared addresses and an additive read, `copy_all` is *mathematically identical* to a single memory. A fast/slow split cannot help by itself. |
| 2 | **Sanity checks in code** | Make sure the code matches the math | `copy_all` matched `single` with **0.0 difference**. Each fast write stored its item exactly. |
| 3 | **Threshold tuning** on seed 100, 40 distractors only | Avoid tuning on test seeds | τ fixed at 0.4 (hard) and 3.0 (soft), then frozen. |
| 4 | **Main run** (`main.py`): 10 conditions × 3 distractor levels (0 / 40 / 100) × seeds 0–4 | The core comparison | See §4. |
| 5 | **Noisy repeats** (`noisy.py`): each sighting uses a noisy key and queries use the clean key; noise 0.3 and 0.6, 40 distractors, seeds 0–4 | Real repeats are never exact, so a real system has no item identity | See §4. |

## 4. What we got

**Exact repeats** (final retention over all 1,000 targets, mean of 5 seeds, sd ≤ 0.017):

| Condition | 0 distractors | 40 | 100 |
|---|---|---|---|
| single | 0.932 | 0.546 | 0.271 |
| single2N (parameter-matched) | 0.972 | 0.823 | 0.522 |
| oracle | 0.932 | 0.935 | 0.927 |
| **count gate** | 0.932 | 0.935 | 0.927 |
| fast+slow, nothing moves | 0.001 | 0.001 | 0.001 |
| fast+slow, copy all | 0.932 | 0.546 | 0.271 |
| same-address copy, hard | 0.907 | 0.812 | 0.649 |
| **same-address copy, soft** | 0.932 | 0.927 | 0.877 |
| item replay | 0.932 | 0.934 | 0.926 |

**Noisy repeats** (40 distractors):

| Condition | Noise 0.3 | Noise 0.6 |
|---|---|---|
| single | 0.465 | 0.258 |
| single2N | 0.746 | 0.444 |
| oracle (identity given) | 0.812 | 0.521 |
| count gate (identity given) | 0.824 | 0.490 |
| **soft slot copy (no identity)** | **0.827** | **0.495** |
| replay of last sighting (identity given) | 0.774 | 0.397 |
| replay of all sightings (identity given) | 0.817 | 0.525 |

## 5. What we inferred

1. **The kill prediction was confirmed in its strong form.** A fast/slow hierarchy with sleep beats a single memory **only through selection**. On exact repeats, a plain counter on one memory ties the best hierarchy and the oracle. The fast store was acting as a frequency counter, nothing more.
2. **Same-address slot copy is not a better transfer than item replay.** It is equal at moderate load and worse at heavy load (0.877 vs 0.926).
3. **The one real niche is noisy repeats without item identity.** There, soft slot copy uses only per-slot address mass, yet matches methods that are *given* identity, and beats the parameter-matched single memory by 0.05–0.08. Its value is cheap bookkeeping, not better consolidation.
4. **The original hope, that local transfer avoids interference between consolidations, was not tested.** Item replay already reached the oracle, so there was no interference left to remove.

## 6. Why we think it happened

- **The identity result is linear algebra.** With an additive read over shared addresses, the read only ever sees S + F. Moving content from F into S without changing the sum changes nothing. So the only lever a sleep rule has is *which* content survives the clear: selection.
- **The counter ties the oracle because of how the stream is built.** Targets repeat 3× per session and distractors appear once. "Write only from the second sighting on" is then a perfect target filter, *given that you know which sightings are the same item*. The hierarchy was rediscovering this rule the hard way.
- **Slot copy loses under heavy load because slots are shared.** Each slot holds a mix of about 10 items, so per-slot usage statistics cannot fully separate targets from distractors. The more distractors there are, the more of them hitch a ride on heavily used slots.
- **Slot copy wins on noisy repeats** because near-duplicate keys land on overlapping slots (59% overlap at noise 0.3, 32% at 0.6). Per-slot mass therefore adds up the sightings of one item *without anyone knowing they are the same item*. Methods that need identity only get it for free in the exact-repeat toy, which is the "smuggled information" pattern from `cross.md` D7.

**Follow-up from reading the project files the same day.** The files already anticipated this:
- the brain knowledge base treats **write control** as its own quantity, separate from timescales;
- the n2r notes say untargeted plasticity hurts;
- the CL doc's sparse-memory-finetuning section ranks slots by **TF-IDF** (usage × rarity), and its ablation shows TF-only ranking (which is what `copy_sel` used) forgets more.

The CL doc's principle 8, "use multiple timescales", was the overclaim: this toy shows timescales without a selection rule do nothing.

## 7. Caveats

- Sub-keys are fixed and random, not learned as in FwPKM. The memory is linear, with no backbone and no backprop through the memory. This measures storage and recall, not generalisation.
- Importance is signalled only by repetition within a session. Items never return across sessions, so savings and spaced repetition were not tested.
- Thresholds were tuned at noise 0 and reused under noise. Five seeds, one stream generator.
- Distractors are uniform random, so IDF-style "background" statistics would not help here. Real usage is Zipfian.

## 8. What we would run next (proposed, never run)

- A rerun with **Zipfian keys and noisy repeats**, to test TF-IDF slot scoring and sparser, pattern-separated allocation (about 15 minutes).
- **Consolidation without item identity on cross-session streams**, measuring savings and spacing.

## 9. How to reproduce

```bash
pip install numpy
mkdir -p /mnt/project-files/memory-consolidation-test && cp *.py /mnt/project-files/memory-consolidation-test/
python3 main.py     # writes results.json
python3 noisy.py    # writes results_noisy.json
```
