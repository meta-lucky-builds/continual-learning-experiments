# 23. CIFAR-10 E1: does a stable statistics path plus the plastic network beat either alone?

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{plan.md, results.md, cifar.py, analyze_cifar.py, fu_mix.py, fu_bc.py}`, `results/e1/`.

## Why we ran it
The marathon revisions (tests 19–21) pointed to a two-path design: a stable path (fixed encoder with streaming class statistics, accumulated at arrival, never stale) and a plastic path (ER with bias correction), combined by a mixture. This was the first test of that design on real images. Also tested: the exact Bayes mixture on the stream's log loss, which the audit proposed for weighting.

## What we did
Split CIFAR-10, 5 tasks of 2, 5 seeds, M = 500, online single pass, lr 0.03 (chosen on seed 100). Arms: ER, ER+bc, ACE, RC-LDA (random conv statistics), RP-LDA, FZ-LDA (frozen after half of task 1), MIX_stream (Bayes on stream loss), MIX_resv (50 reserve items), MIX_half (0.5/0.5).

## What we got

| Arm | Final |
|---|---|
| ER | 42.7 |
| ER+bc_buf | 52.6 |
| ACE | 50.8 |
| RC-LDA | 54.0 |
| FZ-LDA | 54.3 |
| MIX_stream | 42.7 (= ER exactly) |
| MIX_resv | 53.6 |
| **MIX_half** | **57.9** |

- P2: MIX_half − best single = **+3.5 ± 0.4 \***. Either expert alone is right on 69.1% of items, both on 37.6%; ER+bc falls from 62 (oldest task) to 38 (newest), RC-LDA is flat at 53–56.
- P1: RC-LDA − ER+bc = +1.4 (4/5), not significant.
- MIX_stream puts weight 1.0000 on ER on every seed.
- Bias correction over-corrects: the newest task gets 10.9% of predictions vs a true 20%; an oracle scale α ≈ 0.7 gives 55.0. Not buffer memorisation: b_buf ≈ b_test.
- ACE − ER+bc = −1.8 \*. FZ-LDA ≈ RC (null).

## What we inferred
- **The two paths add** on real images.
- **The Bayes mixture's guarantee targets the wrong risk**: stream log loss measures the current task, where the plastic net wins, so it discards the stable expert. Predicted in advance and confirmed.
- **No fixed formula sets the correction size**: on MNIST, prior adjustment under-corrected; here mean-logit correction over-corrects.

## Why we think it occurred
The two experts fail on different items: the plastic net is strong on recent and old-but-replayed classes, while the stable path has no order effect at all. Averaging probabilities lets each cover the other's errors. Mean-logit correction over-corrects because ER's newest-class bias is partly item-specific; subtracting the full mean removes too much.

## How reliable this test is
- **High for P2** (5/5, replicated in tests 24, 25, 29). Self-tests C1–C13 pass; one real bug (float32 underflow making the NLL infinite) was caught by review before any result.
- lr chosen on the seed-100 test set (mild test reuse). The α and weight sweeps are diagnostic only.
