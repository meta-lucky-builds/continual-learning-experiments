# 20. Marathon revision R2: are learned features more sample-efficient?

**Date:** 2026-10-29. **Source files:** `source/notes/marathon/results.md` §0b, `readout_checks.py`.

## Why we ran it
The audit argued test 07's "fixed features beat learned ones" was really about historical sample size: fixed features win when you have all data, but a continual learner has only a few stored items per class.

## What we did
Same frozen ER checkpoints (guard: they reproduce the earlier ER runs exactly). LDA from k random training items per class (k = 5, 20, 50, all), 10 draws per k, on trained features, the untrained net, raw pixels, and random ReLU at 256 and 1024 dims.

## What we got (MNIST)

| k | trained 256-d | untrained 256-d | raw | random ReLU 256 | random ReLU 1024 |
|---|---|---|---|---|---|
| 5 | **74.0** | 55.6 | 63.7 | 60.8 | 65.2 |
| 20 | **83.5** | 70.6 | 70.5 | 73.8 | 79.0 |
| 50 | 85.4 | 80.0 | 77.5 | 82.3 | **85.9** |
| all | 86.7 | 87.2 | 87.5 | 89.1 | **94.0** |

- At 5 per class, trained features beat every fixed alternative by 9–18 points (5/5).
- Toy A shows the same pattern. Width matters: random ReLU gives 89.1 (256-d), 94.0 (1024-d), 96.4 (4096-d).

## What we inferred
The fixed-feature win came from exact full-history statistics, not better features. With few samples, learned features are much better. "These benchmarks cannot test feature learning" was withdrawn.

## Why we think it occurred
Training concentrates class information into fewer directions, so a mean estimated from 5 items is much closer to the truth in learned coordinates. Random expansions spread information over many dimensions, which needs many samples but loses nothing given all data.

## How reliable this test is
- **High within MNIST/toy**: 10 draws × 5 seeds, guarded checkpoints, a clean crossover. It shaped the two-path design tested on CIFAR.
