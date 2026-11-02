# Fast/slow product-key memory with sleep consolidation: results

Run 2026-10-23. Code: `pkmsleep.py` (model), `main.py` (exact repeats), `noisy.py` (noisy repeats). Raw numbers: `results.json`, `results_noisy.json`.

## What was tested
The idea under test: a fast and a slow sparse memory share one product-key address space. At "sleep" (each task boundary), slots that were used heavily are copied from fast to slow, and the fast store is cleared.

- Addressing follows PKM: split the query in two halves, use 64 sub-keys per half, giving 4,096 slots. Keep the top 8 slots, with softmax weights (effective k ≈ 5).
- Reads are additive over the shared addresses: y = Σ wᵢ (Sᵢ + Fᵢ).
- Fast writes are FwPKM-style one-step rewrites (NLMS/Kaczmarz) on the residual of the combined read: Fᵢ += wᵢ (v − y)/‖w‖². Each write stores its item exactly, which is checked in code.
- The stream has 100 tasks. Each task has 10 targets shown 3 times each, plus distractors shown once. Only targets are queried. After every sleep, all earlier targets are tested by nearest-neighbour decode over the 1,000 target values (chance ≈ 0.001).
- The thresholds for copy_sel and copy_soft were tuned on seed 100 at 40 distractors, then frozen. Results are for seeds 0–4, reported as mean ± sd.

## A result from the math before any run
With shared addresses and an additive read, copying every fast slot at sleep (S += F, then F = 0) is **exactly** one memory with no sleep at all. In code the difference is 0.0. A fast/slow split alone adds nothing. Any gain has to come from *what is selected* at sleep.

## Exact repeats (final retention over all 1,000 targets)
| condition | 0 distractors | 40 | 100 |
|---|---|---|---|
| single memory | 0.932 | 0.546 | 0.271 |
| single, 2× slots (parameter-matched) | 0.972 | 0.823 | 0.522 |
| oracle (writes only targets) | 0.932 | 0.935 | 0.927 |
| count gate (single memory, skips first sighting) | 0.932 | 0.935 | 0.927 |
| fast+slow, no transfer | 0.001 | 0.001 | 0.001 |
| fast+slow, copy all | 0.932 | 0.546 | 0.271 |
| **same-address copy, hard threshold** | 0.907 | 0.812 | 0.649 |
| **same-address copy, soft (min(1, mass/3))** | 0.932 | 0.927 | 0.877 |
| item replay (session key buffer, count ≥ 2) | 0.932 | 0.934 | 0.926 |

Standard deviations are ≤ 0.017 everywhere. Half-life: the single memory falls to half its acquisition after 54 tasks (40 distractors) and 24 tasks (100 distractors). Soft copy, replay, oracle and count gate stay above half over all 100 tasks.

## Noisy repeats (each sighting uses a noisy key, queries use the clean key; 40 distractors)
| condition | noise 0.3 (cos 0.96, 59% slot overlap) | noise 0.6 (cos 0.86, 32% slot overlap) |
|---|---|---|
| single | 0.465 | 0.258 |
| single 2× slots | 0.746 | 0.444 |
| oracle (identity given) | 0.812 | 0.521 |
| count gate (identity given) | 0.824 | 0.490 |
| same-address copy, soft (no identity) | 0.827 | 0.495 |
| replay of last sighting (identity given) | 0.774 | 0.397 |
| replay of all sightings (identity given) | 0.817 | 0.525 |

## Verdict
1. **The prediction that would kill the idea is confirmed in its strong form.** A fast/slow hierarchy with sleep beats one memory only through *selection*. On exact repeats, a plain counter on a single memory (count gate) does exactly as well as the best hierarchy. Here the fast store is acting as a frequency counter, nothing more.
2. **Same-address slot copy is not better than item replay.** It is equal at moderate load (0.927 vs 0.934) and worse at high load (0.877 vs 0.926). A slot mixes several items, so statistics kept per slot can't fully separate targets from distractors. The hard threshold is clearly worse. Soft copy lowers acquisition slightly (0.99 vs 1.00).
3. **Where it does earn something: approximate repeats without identity.** When repeats are noisy, no exact counter or item buffer exists unless the system already knows which sightings belong to the same item. Soft slot copy uses only per-slot address mass, and matches the methods that are *given* identity (0.827 vs 0.824 and 0.817 at noise 0.3; 0.495 vs 0.490 and 0.525 at noise 0.6). It beats the parameter-matched single memory by 0.08 and 0.05. So its real value is cheap bookkeeping, which comes from aggregating nearby keys for free. It is not a better transfer.
4. On the original hope that local transfer avoids interference between consolidations: in this toy, item replay already reaches the oracle, so there was no interference left to remove. That claim is untested here, not supported.

## Limits (how far this toy represents the real idea)
- Sub-keys are fixed and random, not learned (FwPKM learns them with an addressing loss). The memory is linear in its values, with no backbone and no backprop through the memory. It measures storage and recall, not generalisation.
- Importance is signalled only by repetition within a session. Items never come back across sessions, so savings and spaced repetition are not tested.
- Thresholds were tuned at noise 0 and reused under noise without retuning. Five seeds, one stream generator.
- "Stored but not accessible" was not probed separately.
