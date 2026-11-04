# 02. Spec v1.3 math checks and new structure (N1–N6)

**Date:** 2026-10-28. **Type:** numerical checks plus new theory, not a learning experiment.
**Source files:** `source/notes/v13-checks/{verify13.py, frontier.py, frontier2.py, heldout_frontier.py, stale_slack_counterexample.py}`; write-up in `source/notes/formal-spec.md`.

## Why we ran it
Lucky's second external audit (of v1.3) raised 15 patch areas, with concrete counterexamples. Each had to be reproduced, and the new structure I added in v1.3 (N1–N6) had to be checked the same way.

## What we did
Reproduced every counterexample in the audit numerically, then tested my own new claims N1–N6 with random cases and constructed cases. A second independent reviewer then read v1.3.

## What we got
**Audit counterexamples, all reproduced exactly:** 3.04 for group-rate post-scaling; −977 vs 27.6 nats for conditional novelty; 0.843 for the never-switching HMM; powers 0.808, 0.363 and 0.204; the (6, 5) certificate false alarm; the (1, 0, 2) new-class steal; the H = 11ᵀ null-space case; the 0.25 → 0.953 usage jump.

**New structure:**
- **N1 [proved].** Every protection step is one B⁻¹-proximal map, which gives surrogate descent and non-expansion.
- **N2.** The centred certificate pairs with the centred metric: 0 violations in 5,748 random cases.
- **N3 [corrected].** My first "frontier" said margin constraints *dominate* the quadratic brake. **That was an in-sample tautology**: damage was measured on the constrained items themselves. On 200 held-out items the quadratic brake did better (damage 64 at progress 171, vs 77 at 160). The follow-up slogan "constraints memorise, penalties generalise" was later withdrawn too (test 04 contradicts it).
- **N4.** Tracked slack gives a cumulative guarantee on the constrained pairs. With stale slack, a linear counterexample falls to −5.5.
- **N5.** An exact detectability threshold for the sticky filter: log L_min = 0.166. Later shown **not necessary** under noisy evidence (test 15, T7).
- **N6.** Inferential reactivation needs n ≈ 170 probes for robust power, so it runs in control mode.
- The second reviewer found 20 more defects in v1.3, including errors in N3 and N4. All were fixed.

## What we inferred
The audit was right on every point. My one positive claim that looked like a strong result (N3, "constraints dominate") was an artefact of measuring on the training items.

## Why we think it occurred
N3: a constraint that forbids damage on a set of items trivially has zero damage *on those items*. The real question is damage on other items, and there the soft penalty generalised better. This was the first instance of the pattern that recurs through the whole project: **anything fitted on a set looks better on that set** (see RELIABILITY.md).

## How reliable this test is
- **High** for the reproduced counterexamples (deterministic).
- **Medium** for N2–N6: random-case checks and constructed cases, not proofs (except N1). N3 shows a check can be numerically correct and still measure the wrong thing.
