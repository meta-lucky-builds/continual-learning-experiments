# 19. Marathon revision R1: is the bias size a closed-form function of the prior?

**Date:** 2026-10-29. **Source files:** `source/notes/marathon/results.md` §0b, `readout_checks.py`, runs `results/A_*`, checks `mar2.py` N14–N16.

## Why we ran it
Lucky's v1.3 deep audit questioned test 06's claim that logit adjustment by the stream prior, with τ = 1, sets the bias size with zero parameters. The audit also offered an alternative explanation (a global logit scale).

## What we did
Trained under the spec's actual loss (separate equal-weight means, which in my harness is w = 4), measured the prior ratio that loss induces, applied logit adjustment by it, swept τ, and measured the 2-way calibration temperature among the newest classes.

## What we got
- The spec's loss induces a new/old prior ratio of 4 (measured 3.9–4.0), not 16.
- Adjustment by it at τ = 1 **under-corrects badly**: toy A 53.2 vs biascorr 59.1 (−5.9 \*); MNIST 58.9 vs 71.4 (−12.5 \*).
- Oracle-best τ: 2.7–3.4 (toy A), ≥ 4 (MNIST, grid cap). Even under the plain loss, best τ is 1.4–1.7 (toy) and 2.1–2.6 (MNIST).
- Newest-class temperature T_new ≈ 0.9–1.3, so the global logit-scale explanation is **not supported**.
- Replay weighting w = 4 raises raw accuracy (+4.3 \* toy, +1.9 \* MNIST) but not bias-corrected accuracy (+1.4, +0.3, not significant).

## What we inferred
The bias's direction and order of magnitude follow from the prior; its size does not. A data-driven correction is required. The "zero-parameter mechanism" claim is withdrawn.

## Why we think it occurred
With a tiny buffer, up-weighting replay mostly fits the stored items; it does not raise old-class logits on old-class test inputs. So the real offset on test data is larger than the training prior predicts.

## How reliable this test is
- **Medium-high.** Direct test of a specific claim, 5 seeds, checks N14–N16. The cause of τ > 1 is still not identified.
