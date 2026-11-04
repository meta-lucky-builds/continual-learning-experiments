# 08. Marathon F8/W1: what limits post-hoc readouts?

**Date:** 2026-10-28. **Source files:** as test 05; revision in results.md §0b (R3).

## Why we ran it
Every fixed readout in test 05 plateaued at the same ceiling, far below the oracle. We needed to know whether the limit was the features or the readout's sample size.

## What we did
- F8: varied the buffer size (5 vs 50 per class) and compared buffer readouts against the oracle.
- W1: mixed LDA ingredients: full-data means with buffer covariance, and buffer means with full-data covariance.

## What we got
- Buffer readouts plateau at about 71 (MNIST, 5/class) vs an oracle of 89. At 50/class, buffer refit reaches 89.5 vs the oracle's 92.3.
- Full means + buffer covariance: **78.3**. Buffer means + full covariance: **55.5**. The gap is in the class **means**.

## What we inferred
The post-hoc ceiling is set by how well a few stored items estimate each class mean, not by the features.

## Why we think it occurred
Five samples in 256 dimensions give an error-dominated mean: squared error scales as σ²d/n. **Revised (test 21):** part of the gap is also buffer memorisation. Replayed items are fitted by the network, so their features are atypical.

## How reliable this test is
- **Medium-high.** Clean decomposition, 5 seeds. The interpretation was narrowed once. The same "means carry the gap" finding reappeared on CIFAR-100 (test 39, X2: means carry 92%).
