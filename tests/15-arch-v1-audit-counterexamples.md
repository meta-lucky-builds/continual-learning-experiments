# 15. ARCH-v1 audit: deterministic counterexamples

**When:** 2026-10-30, after the first Atlas architecture (ARCH-v1). **Files:** `design/audit-v1/check_architecture.py` and `architecture_checks.json` (Lucky's audit, supplied as files).

## Why we did it
ARCH-v1 made several claims about transporting class statistics between encoder versions, fusing evidence and combining experts. An external audit claimed some were false. A claim about math can be settled by a small counterexample, so the audit shipped a script with one counterexample per claim. We re-ran and checked each before accepting or rejecting it.

## What we did and what we got
| Claim in ARCH-v1 | Counterexample | Result |
|---|---|---|
| Old statistics (mean, second moment) identify where a class lands in a new encoder | Two distributions P, Q with identical mean 0 and second moment 1; new feature x⁴ | New means 2 vs 4: **transport from summaries is not identifiable** |
| Re-estimating covariance absorbs a linear drift | 2-class example under rotation | Stale means score 50% (tie); transported means 100% |
| A global covariance stands in for within-class covariance | 4 classes at corners, global cov = I | Within-class variance is exactly 0 |
| Fusing estimates (Kalman-style) shrinks uncertainty | Same observation fused 10 times as if independent | Claimed variance 1/11 ≈ 0.091; real variance 1.0 |
| A product of experts (PoE) is safe | Stable expert 0.9 on the truth; failed expert 1e-12 | Product log loss 12.7; half mixture 0.7985, matching its bound 0.7985 exactly |
| Class means' subspace is the discriminant subspace | Mean difference (2,0), correlated covariance | LDA direction (10.53, −9.47): not aligned with the means |
| Re-anchoring resets error | Estimated mean 1, true 0, identity re-anchor | Error 1.0 before and 1.0 after |

## What we inferred
Seven ARCH-v1 claims were plain errors (listed in `AUDIT-RESPONSE-v1.md`):
1. re-estimating Σ absorbs linear drift;
2. recycling is an exact coordinate change;
3. a random chart gives free calibrated p-values;
4. soft-EM and prediction-credited activation create self-reinforcing loops;
5. Kalman fusion assumed independence;
6. class-balanced replay = frequency replay;
7. the 16:1 ratio (it is 4, per [05](05-marathon-revisions.md)).

This produced ARCH-v2: an atomic publication split, a regression (GREG) estimator, ε-mixture safety, and evidence provenance.

## Why we think it happened
ARCH-v1 treated summaries as if they carried the whole distribution, and repeated estimates as if they were new evidence. Both are classic errors. A finite set of moments does not determine a distribution under a nonlinear map. Reused data is not independent data. A product of experts lets one confidently wrong expert veto everything; an arithmetic mixture with weight ε on a safe expert bounds the damage at −log ε.

## Caveats
These are constructed worst cases. They prove claims false in general; they do not measure how often each failure happens in practice.
