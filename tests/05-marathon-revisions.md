# 05. Marathon revisions after the v1.3 deep audit

**Source:** ewt.md §5 (A6).

## Why we did it
Lucky's deep audit challenged three marathon conclusions. Each challenge was concrete enough to test, so we tested rather than argued.

## What we did and got
**R1: is the bias size a closed-form function of the class prior?**
- Under the spec's loss the induced prior ratio is 4, not 16. Logit adjustment by it under-corrects: −5.9 \* (toy), −12.5 \* (MNIST).
- The oracle-best τ is 2.7–3.4 (toy) and ≥ 4 (MNIST).
- The audit's global-temperature explanation was not supported (newest-class temperature ≈ 1).

**R2: are learned features worse than fixed ones?**
- At 5 per class, trained features beat every fixed alternative by 9–18 points (5/5).
- Fixed features win only with full-history statistics.
- Width matters for random ReLU: 89.1 (256-d), 94.0 (1024-d), 96.4 (4096-d).

**R3: is the replay buffer a fair source of statistics?**
- MNIST old-class LDA: 49.4 from the replayed buffer vs 73.8 from 5 fresh items (+24.3, 5/5). Buffer means sit ~2.3× farther from the truth.
- Holding memory out of replay helps LDA (+6–7 \*) but loses to bias correction at this budget.

## What we inferred
1. The bias correction must be **fitted on held-out data**. The "zero-parameter mechanism" claim was withdrawn.
2. Learned features are more sample-efficient. Fixed features need complete statistics.
3. Statistics must be accumulated **at arrival**, not from replayed items.

Together these produced the "two-path" design: a plastic network plus a stable statistics expert. The CIFAR tests then examined it.

## Why we think it happened
- The prior ratio is not the whole story: optimisation dynamics and feature drift add bias that a static formula cannot see.
- Replayed items are fitted by the network, so their features are pulled toward their class's decision region. The buffer then looks tighter and more separable than the population.

## Caveats
On CIFAR-100 the replay bias of R3 did **not** show up: 20 trained-on buffer items gave about the same LDA as 20 fresh items ([11](11-audit-corrections.md), and again in our own [18](18-atlas-v1-cifar100.md)). R3 is MNIST-specific, or depends on scale.
