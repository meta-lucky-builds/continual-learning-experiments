# 19. Atlas tests v2b: designed, not yet run

**When:** designed 2026-10-30. **Files:** [`atlas/notebooks/atlas_v2b_tests.ipynb`](../atlas/notebooks/atlas_v2b_tests.ipynb), [`atlas/src/v2b/`](../atlas/src/v2b/). **Status: no results yet.** Nothing below is a finding.

## Why we designed it
Each change answers a specific problem found in [18](18-atlas-v1-cifar100.md):

| Problem in v1 | v2b change |
|---|---|
| Learner too weak (head 8.1%, learned < random) | **Learner calibration:** 6 recipes (v1's; momentum; 3–5 steps per incoming batch; flip+crop augmentation; lr 0.01–0.1) on a separate seed (100), picked by the learned-feature ceiling on held-out control items, never the test set. Gate printed: does the learned ceiling beat the random chart? |
| Reserve vs buffer confounded count with provenance | **Matched-size test:** at every hop, each class gets the same k = min(10, #buffer, #reserve) items from each source |
| Random-chart expert never combined with the learned one | **PoE** of RC_LDA with each migrated readout, weights and temperatures fitted on control items; plus the **ε-mixture** (ε = 0.05), whose log-loss bound is checked on every test item |
| "Equal bytes" config spent half the memory on a reserve | **Config E:** 1800 buffer + 2 control per class = 2000 images |
| Mean error disagreed with accuracy | Decisions use accuracy (overall, old classes, recent classes) and log loss |

## Pre-registered comparisons and predictions
| | Comparison | Prediction |
|---|---|---|
| G1 | A: O_full > RC_LDA | yes, 60% |
| P1 | A: GR_prev_buf > CH_cv_buf | yes, 80% |
| P2 | A: GR_multi_buf > GR_prev_buf | yes, 40% |
| P3 | A: matched reserve > matched buffer (training bias real) | yes, 35% |
| P3b | A: matched raw means, reserve > buffer | yes, 40% |
| P4 | A: PoE[RC+GR_prev_buf] > RC_LDA | yes, 80% |
| P4b | A: PoE > GR_prev_buf | yes, 85% |
| P5 | E: PoE > CH_cv_buf | yes, 85% |
| P5b | E: PoE > RC_LDA | yes, 70% |
| P6 | A: CH_cv_buf > B_only_buf | yes, 40% |
| P7 | A: PoE[RC+O_full] > PoE[RC+GR_prev_buf] (migration loss remains) | yes, 70% |

Hard check: the ε-mixture bound must hold on 100% of test items. It is an identity, so any violation is a code bug.

## Edge cases built in
- A diverging learner candidate cannot be picked; a diverging test run is recorded and excluded.
- No tuning touches the test set.
- Bridge fallbacks are counted.
- Colab disconnects resume from saved runs.
- Results from other versions are never mixed in.
- Augmentation applies to training only.
- With 11 comparisons, expect at most one false positive.

## New checks (all pass)
- Temperature fitting recovers a planted τ = 2 (got 1.88).
- PoE puts weight ~1 on an informative expert vs noise, and ~0.5 on two equal experts (and beats both).
- The ε-mixture equals direct computation, and its bound holds.
- **The matched test detects a planted buffer bias** (reserve wins by 16–53 points) and shows **no gap without bias** (−0.9 to +0.9).
- The no-bridge config runs and skips bridge arms.
