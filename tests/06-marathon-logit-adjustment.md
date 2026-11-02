# 06. Marathon: closed-form logit adjustment by the stream prior

**Date:** 2026-10-28. **Source files:** as test 05; revision in `source/notes/marathon/results.md` §0b (R1).

## Why we ran it
If the bias is the training prior (test 05), then subtracting τ·log π, with π the class frequencies in the last task's training batches and τ = 1, should remove it with **zero fitted parameters**. That would make the readout fix free. The test also has a sharp prediction: under ER-ACE, only replay items create cross-class competition, and they are class-balanced, so there should be no bias at all.

## What we did
Applied logit adjustment with π from the last task's batches, τ = 1, to ER and to ER-ACE. Control: the *wrong* π (all labels) applied to ER-ACE.

## What we got
- Recovered the bias-correction gain: toy A 83–120%, toy B 130–160% (beats biascorr, +1.6 \*), MNIST 5/class 66–80%, MNIST 50/class 89.4 vs ER 83.2.
- ER-ACE: replay-only π gives +0.1 ± 0.1, and ER-ACE's biascorr gain is ≈ 0, as predicted. The wrong π hurts ER-ACE by −4.9 \*, so the test discriminates.

## What we inferred (first), then what survived
- First: "the bias size follows in closed form from the prior". **Withdrawn after Lucky's v1.3 deep audit (test 19, R1).** Under the spec's actual loss, the induced prior ratio is 4, not 16, and adjustment by it under-corrects badly (−5.9 \* toy, −12.5 \* MNIST). The best τ is 2.7–3.4 (toy) and ≥ 4 (MNIST).
- **What survives:** the bias's *direction* and order of magnitude follow from the stream prior, and ER-ACE's no-bias prediction held. Its *size* must be fitted on held-out data.

## Why we think it occurred
With a tiny buffer, up-weighting replay mostly fits the 40 stored items; it does not raise old-class logits on old-class *test* inputs. So the real bias is larger than any loss-induced prior predicts. A global logit-scale explanation (from the audit) was checked and not supported (newest-class temperature ≈ 0.9–1.3).

## How reliable this test is
- **Medium.** The ER-ACE prediction is a strong discriminating test and held. The size claim was wrong and is withdrawn; the τ grid capped at 4 on MNIST.
