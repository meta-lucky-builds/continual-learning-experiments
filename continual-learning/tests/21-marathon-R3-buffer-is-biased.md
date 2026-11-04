# 21. Marathon revision R3: is the replay buffer a biased source of statistics?

**Date:** 2026-10-29. **Source files:** `source/notes/marathon/results.md` §0b, `buffer_bias_check.py`.

## Why we ran it
Test 08 said readouts plateau because of sample size. The audit asked whether replayed items are themselves atypical, since the network has been trained on them.

## What we did
Same checkpoints, 5 items per class. Compared old-class LDA from the replayed buffer against LDA from 5 fresh, never-replayed items. Measured buffer-mean error. Design test at equal memory: keep r items per class stored but never replayed.

## What we got
- MNIST old-class LDA: **49.4 from the buffer vs 73.8 from 5 fresh items** (+24.3, 5/5). Toy A: 46.6 vs 53.8 (+7.2, 5/5).
- Buffer means are ~2.3× farther from the true mean (0.38–0.52 vs 0.20–0.23 relative error).
- Never-replayed items: +5.8 to +7.3 \* over buffer LDA on MNIST, but **lose to biascorr on the full buffer** (−5.2 to −6.8 \*), because less replay hurts the network. No gain on the toy.

## What we inferred
The effect is real: statistics from replayed items are biased. But a reserve split is not a win at 5 per class. Stable statistics should be accumulated **at arrival**, from un-replayed data, in fixed coordinates.

## Why we think it occurred
The network fits the replayed items, pulling their features toward atypical, over-fitted positions, so their mean misrepresents the class.

## How reliable this test is
- **Medium-high on MNIST.** Notably, on CIFAR-100 this bias was **not** seen: 20 trained-on buffer items gave about the same LDA as 20 fresh items (test 35, A1). So the size of this effect depends on the setting.
