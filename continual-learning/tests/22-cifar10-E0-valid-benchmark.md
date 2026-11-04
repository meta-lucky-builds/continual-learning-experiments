# 22. CIFAR-10 E0: is CIFAR a valid test of feature learning?

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{plan.md, results.md, cifar.py}`, logs `results/e0/`.

## Why we ran it
On MNIST, fixed random features matched or beat learned ones with full data (test 07), so MNIST could not show that a continual learner's feature learning matters. Before running continual tests on CIFAR, we checked that learned features clearly beat random ones there. Pre-declared criterion: a gap of at least 5 points.

## What we did
Full-data LDA on several feature maps, seeds 0/1/2, and at k = 5 and k = 100 per class: a jointly trained CNN, a random CNN (512-d and 1024-d), a random pixel projection (4096-d), raw pixels.

## What we got

| Feature map | k=5 | k=100 | all |
|---|---|---|---|
| joint-trained CNN | 49/48/46 | 69/69/69 | 75.7/75.5/75.4 |
| random CNN 512-d | 25/21/24 | 47/45/46 | 55.3/52.0/54.0 |
| random CNN 1024-d | 24/22/24 | 47/47/47 | 56.7/57.8/58.4 |
| random pixels 4096 | 22/20/21 | 32/32/32 | ~50 |
| raw | 20/18/19 | 25/25/25 | 41.7 |

## What we inferred
**Criterion passed on every seed, by 17–19 points.** CIFAR is a meaningful test of continual feature learning. It also sets the headroom: joint training reaches ~75.5 with LDA.

## Why we think it occurred
Natural images need non-linear, learned features (edges, textures, parts) that random convolutions only partly provide; digits are almost linearly separable in pixels.

## How reliable this test is
- **High.** Large gap on all seeds; a cross-check later confirmed the streaming RC-LDA equals this full-data LDA within 0.1 point.
