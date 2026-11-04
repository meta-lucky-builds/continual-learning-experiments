# 01. Spec v1.2 math checks

**Date:** 2026-10-27/28. **Type:** numerical checks of algebra, not a learning experiment.
**Source files:** `source/notes/v12-checks/{checks.py, fix2.py, review_checks.py}`; write-up in `source/notes/formal-spec.md` Part A; old spec in `source/notes/archive/formal-spec-v1.2.md`.

## Why we ran it
Lucky's first external critique of the formal spec (v1.2) made 16 claims that parts of the spec's math were wrong: its update geometry (how a protection step combines with Adam), its curvature approximation, and its statistics. Before building anything, each claim had to be checked numerically, because a learner built on wrong algebra would make every later experiment meaningless.

## What we did
Small scripts constructed the exact counterexamples the critique described, plus checks of my proposed fixes:
- an Adam-composed preconditioner, tested for descent;
- K-FAC against the exact curvature on a constructed update Δ;
- a Woodbury exact solve against a dense solve, over a range of brake weights w;
- preconditioned conjugate gradient (PCG) over 2,000 iterates;
- the token-coupled sketch estimator against the exact value;
- a power calculation for the planned evaluation.

## What we got
- **The Adam-composed preconditioner can go uphill:** g·P⁻¹U = −3.04. The fix, solving (D⁻¹ + wH)d = g, gives descent (+0.665).
- **K-FAC protects directions the outputs cannot see:** exact 0.0 vs K-FAC 0.5 on the constructed Δ.
- **Woodbury exact solve** matches a dense solve to ≤ 4.7e-6 relative error, up to w = 1e4, and equals Adam exactly at w = 0.
- **PCG:** 0 non-descent iterates in 2,000. Unpreconditioned CG at ω = 0 did *not* reproduce Adam (65% error). This was a reviewer finding, then fixed.
- **The token-coupled sketch is unbiased:** exact 106.6 vs estimate 107.1.
- **Power:** 80% power at a 12.5-point effect needs n ≈ 53.
- **Mine:** v1.2's delay formula was off by a factor of κn = 4, and its premise was wrong.
- An independent reviewer found 21 defects in the v1.2 draft (0 critical, 12 major, 9 minor). All were fixed.

## What we inferred
All 16 critique points were substantially correct. The spec's update geometry was rebuilt around one exact solve.

## Why we think it occurred
The v1.2 spec combined pieces (Adam's diagonal scaling, a curvature brake, sketches) that are each sound alone but whose composition was never checked. Composing two positive-definite scalings in the wrong order does not guarantee a descent direction, so a "protection" step could increase the loss.

## How reliable this test is
- **High for what it checks.** These are deterministic algebra checks with exact or near-exact answers, re-run on 2026-10-29 with the same outputs.
- **It says nothing about whether the learner works.** It only fixes the math.
