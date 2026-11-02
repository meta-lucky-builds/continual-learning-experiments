# 07. Marathon W3 (and T3): floor control, learned vs fixed features

**Date:** 2026-10-28. **Source files:** as test 05; revision in results.md §0b (R2).

## Why we ran it
T3 found the features keep the information (a probe on all data gets 89% while the model gets 39% on old classes). Before crediting the *network* with that, we needed a floor: how well do fixed, untrained features do under the same oracle readout? If untrained features do as well, these benchmarks cannot test feature learning.

## What we did
Oracle LDA on all training data, on: ER's final features, the initial random network, raw pixels, features after task 1, and a random ReLU expansion of the input (1024-d).

## What we got

| Features | MNIST | toy A |
|---|---|---|
| ER features after the stream | 86.7 | 68.4 |
| Initial random network | 87.2 | 64.1 |
| Raw pixels | 87.5 | 78.4 |
| Features after task 1 | **75.2** (−11.5 \*) | — |
| Random ReLU 1024-d | **94.0** | 76.6 |

Trained methods on MNIST for comparison: ER 50.7, best post-hoc fix 71.2, ER-ACE 70.6.

## What we inferred (first), then what survived
- First: "on Split-MNIST learned features are no better than untrained ones; these benchmarks cannot test continual feature learning". **Revised by test 20 (R2):** with only 5 items per class, trained features beat every fixed alternative by 9–18 points. The fixed-feature win needs exact full-history statistics.
- **What survives:** training on task 1 alone *destroys* separability for future classes (87 → 75). Any claim that learned features are better *with full data* must be made on data where fixed features fail. That is why we moved to CIFAR (test 22).

## Why we think it occurred
MNIST digits are nearly linearly separable in raw pixel space, and a wide random ReLU expansion makes them more so. A network trained on "0 vs 1" collapses its features onto that one distinction, losing directions future classes need.

## How reliable this test is
- **Medium.** The numbers are clean (oracle readouts, 5 seeds). The trained-feature rows are diagnostics only: they need old inputs re-encoded, so they are not streaming-valid learners. The headline reading needed revision.
