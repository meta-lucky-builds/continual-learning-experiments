# 17. Atlas test notebook: correctness checks with known answers

**When:** 2026-10-30. **Files:** `notebooks/atlas_v2_tests.ipynb` and its sources in `src/v1/`. The same checks, extended, are in `src/v2b/`.

## Why we did it
Lucky asked that the tests be run "accurately so the results are reliable, no mistake in the code of the test". There was no torch in our environment, the CIFAR hosts were blocked, and the original experiment code was never uploaded. So the analysis was written in plain numpy and **tested against synthetic data whose answers are known**, before any real data. The notebook re-runs every check in Colab and stops if any fails.

## What we checked (all pass)
**Library (`test_lib.py`):**
- Ledoit-Wolf covariance: large-n agrees with the sample covariance; p > n stays well conditioned.
- LDA equals brute-force Gaussian log-likelihood argmax.
- Ridge regression equals the closed form, for n > p and n < p.
- GREG recovers exact means under an exact linear relation (error 2.6e-6).
- GREG's variance ratio is ~1 at R² = 0 and ~0.29 at R² = 0.8 (theory ≈ 0.2 plus slope noise).
- Classes absent from the bridge are flagged as fallbacks.
- Chained exact linear hops recover the truth.
- Within-class R² is ~1 for an exact linear map and ~0 for independent data.
- The significance rule fires and does not fire correctly.
- Block ridge drops a wide useless block (κ = 0, R² = 0.999).

**End-to-end analysis (`test_analysis.py`):** a synthetic drifting stream with known truth.
- With exact linear drift, every transport arm recovers the oracle (errors ~1e-9).
- With noise, per-class GREG beats buffer-only, and adding r never makes R² worse.

**scikit-learn cross-check (`sk_check.py`):** our Ledoit-Wolf, ridge and LDA match sklearn exactly (LDA agreement 1.0000).

**Audit design tests (`audit_tests.py`), synthetic:**
- U1: an exact hop carries old error forward as δW (re-anchoring never resets error).
- U2: a rotation is not fixed by a mean shift (error 15.3); a full map fixes it (1e-8).
- U3: fusing the same evidence twice claims half the variance (real/claimed = 1.99).
- U4: a dormant class inside crowded support is retrieved in the top 5 (recall 0.97).
- U5: a label swap is caught by prediction error (0 → 1.00) and not by input novelty (shift 0.02 sd).
- U6: the ε-mixture bound holds against a confidently wrong expert (0 violations).

## Decisions these checks changed (made before any CIFAR result)
1. **Cross-fitting off.** Fitting the GREG slope on half the bridge lost badly at 5 items per class (error 2.43 vs 0.78). The slope is fitted on all bridge items.
2. **Wider ridge grid.** The first grid (down to 1e-6) left measurable shrinkage bias on exact data. The grid now goes to 1e-9.
3. **Separate penalty for r's block.** With one shared penalty, adding 40 noisy r columns lowered out-of-fold R² from 0.85 to 0.74. r now gets its own penalty from {drop, 0.1, 0.3, 1}.
4. **A synthetic test was wrong, not the code.** The first "exact" test projected 12 latent dimensions into 10, so the hop was not linear. Fixed by making it invertible.

## Why this matters
Every one of these would have silently changed a reported number. Self-tests check the code against the design. They do not check the design against the question; that is what the CIFAR results and outside review are for.

## Caveat
The PyTorch cells (training, data loading, feature extraction) were never executed in our environment. They were checked in Colab by a smoke run at the start of each notebook.
