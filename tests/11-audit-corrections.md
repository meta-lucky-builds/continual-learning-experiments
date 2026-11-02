# 11. Corrections after Lucky's evidence audit: tests A1–A3

**Source:** ewt.md §14. Audit: `EVIDENCE_AUDIT_AND_SYNTHESIS.md`. Pre-declared in `audit_plan.md`.

## Why we did it
Lucky's external audit checked our claims against the code and found that the T-S5 version curve was unfair. Readouts were fitted on different amounts of data. Three tests were pre-declared to settle it.

## Code facts the audit exposed
- T-S5 version readouts were not matched (task-18 version: ~21 items per old class; task-0: 480).
- Every head was `nn.Linear`, not the spec's cosine head.
- The significance rule as coded was 2·SE plus 5/5 signs; ewt had misdescribed it.
- The toy "start-insert" ER-ACE was non-causal.
- T-S5 output files had vanished from disk. They were regenerated and reproduced earlier numbers exactly.

## A1: matched-data probes (CIFAR-100)
Every encoder's LDA is fitted on the same 480 items per class.

| Encoder frozen at task | 0 | 4 | 8 | 12 | 16 | 18 | final | RC |
|---|---|---|---|---|---|---|---|---|
| Online readout (T-S5) | 26.0 | 28.5 | 29.5 | 28.7 | 26.3 | 24.9 | – | 28.2 |
| Matched, 480 per class | 26.0 | 29.3 | 31.9 | 33.4 | 34.5 | 34.9 | 35.3 | 28.2 |
| Matched, 20 per class | 17.7 | 20.6 | 22.8 | 23.8 | 24.6 | 24.7 | 25.1 | 18.2 |

v18 − v8 = +3.0 \*; v18 − RC = +6.7 \*.

## A2: the 4-version bank vs its own memory-matched buffer
BANK_4spread 31.2 vs MIX_half(ER M = 3,439, RC) 30.0: **+1.3 \***. At its own budget, a small spread bank beats a bigger buffer.

## A3: exact-transfer Sleep control (CIFAR-10)
An exact teacher copy plus reset: 52.2 vs ER 52.4 (n.s.). Distillation vs exact copy: −8.2 \*. All of Sleep's loss is the approximate transfer.

## What we inferred
- **The online features never got worse.** They improved for all 20 tasks, even on the first task's classes.
- **What decays is the class statistics stranded in old coordinates.** A new encoder is better but has almost no history in its own coordinates.
- On CIFAR-100, 20 trained-on buffer items gave about the same LDA as 20 fresh ones (24.9 vs 24.7), so the gap is sample count, not replay bias.

This reframing is the origin of the whole "Atlas" line: carry statistics forward into new coordinates.

## Why we think it happened
The earlier design compared encoder+readout pairs where the readout data differed by 20×. Comparisons must be matched by construction. Our self-tests checked the code against the design, not the design against the question, so only an outside reader caught it.

## Wording corrections accepted
- "Established" became "supported in the tested settings".
- Non-significant is not a tie.
- Bias-correction gain ÷ raw accuracy is not a "share of forgetting".
- The M2c "logit scale forced up" story is not proved.
- Sleep noise failing rejects those dreams, not all synthetic data.
