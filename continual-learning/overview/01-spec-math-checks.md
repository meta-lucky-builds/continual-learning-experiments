# 01. Spec math checks (v1.2 and v1.3 audits)

**When:** before any learning experiment. **Source:** ewt.md §1. **Kind:** algebra and toy-scale numerical checks, not learning experiments.

## Why we did it
The project began with a written spec for a continual learner. It had a preconditioned "protection" step, a sketch estimator, a context filter and certificates. Two external critiques (v1.2: 16 points; v1.3: 15 patch areas) claimed parts of the math were wrong. Before spending compute on learning runs, we wanted to know whether each claim was true. A learner built on broken math would make every later result uninterpretable.

## What we did
We wrote small scripts that reproduce each critique's counterexample numerically, and checked each proposed fix the same way:
- v1.2: `checks.py`, `fix2.py`, `review_checks.py`.
- v1.3: `verify13.py`, the `frontier*.py` scripts, `heldout_frontier.py`, `stale_slack_counterexample.py`.

## What we got
**v1.2 (all 16 points substantially correct):**
- An Adam-composed preconditioner can go uphill: g·P⁻¹U = −3.04. Solving (D⁻¹ + wH)d = g gives descent (+0.665).
- K-FAC protects directions the outputs cannot see: exact 0.0 vs K-FAC 0.5.
- Woodbury exact solve matches a dense solve to ≤ 4.7e-6 relative, up to w = 1e4. It equals Adam exactly at w = 0.
- PCG had 0 non-descent iterates in 2,000. Unpreconditioned CG at ω = 0 did not reproduce Adam (65% error); fixed.
- The token-coupled sketch is unbiased: exact 106.6 vs estimate 107.1.
- Power: 80% power at a 12.5-point effect needs n ≈ 53.
- An independent reviewer found 21 defects in the v1.2 draft (12 major, 9 minor). All were fixed.

**v1.3 (all 15 areas valid).** Counterexamples reproduced exactly: 3.04 (group-rate post-scaling), −977 vs 27.6 nats (conditional novelty), 0.843 (never-switching HMM), powers 0.808/0.363/0.204, the (6,5) certificate false alarm, the (1,0,2) new-class steal, the H = 11ᵀ null-space case, and the 0.25 → 0.953 usage jump. One error was ours: v1.2's delay formula was off by a factor of κn = 4, and its premise was wrong.

**New structure found and checked:**
- N1 (proved): every protection step is one B⁻¹-proximal map.
- N2: the centred certificate pairs with the centred metric (0 violations in 5,748 cases).
- N3, corrected: "margin constraints dominate the quadratic brake" was an in-sample tautology. On 200 held-out items the brake did better (damage 64 at progress 171 vs 77 at 160).
- N4: tracked slack gives a cumulative guarantee. With stale slack, a counterexample falls to −5.5.
- N5: a detectability threshold log L_min = 0.166. It was later shown not to be necessary (marathon T7).
- N6: inferential reactivation needs n ≈ 170 probes.

## What we inferred
The spec's math was repaired. **Nothing here says whether the learner works.** These checks only remove reasons it would fail for trivial mathematical reasons.

## Why we think it happened
The spec had been written by composing pieces (Adam, K-FAC, CG) whose guarantees do not compose: a product of positive-definite matrices is not positive-definite in general, so descent breaks. N3 was wrong because the "damage" was measured on the very items that were constrained. Any constraint looks perfect on the items it constrains. This was the first appearance of the pattern behind I2 in [20-cross-test-insights.md](20-cross-test-insights.md): fitted-on items flatter you.

## Caveats
A second reviewer of v1.3 found 20 more defects, including errors in N3 and N4. All were fixed. The scripts are in [`source/notes/v12-checks/`](../source/notes/v12-checks/) and [`source/notes/v13-checks/`](../source/notes/v13-checks/).
