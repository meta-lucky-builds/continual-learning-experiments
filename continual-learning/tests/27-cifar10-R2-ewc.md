# 27. CIFAR-10 round 2, protection: EWC on the two-path base

**Date:** 2026-10-29. **Source files:** as test 25.

## Why we ran it
The spec kept parameter protection as an option. With the two-path system as the base, does EWC add anything?

## What we did
EWC, λ = 10 (chosen on seed 100 from {1, 10, 100}; λ = 1,000 diverged). U1: EWC+bc vs ER+bc. U2: MIX_half(EWC, RC) vs MIX_half(ER, RC).

## What we got
- U1: 53.1 vs 52.4, +0.7 (3/5), not significant.
- U2: +0.1, null.

## What we inferred
No measurable benefit from EWC at this scale.

## Why we think it occurred
Forgetting here is mostly readout bias (test 33: width gives nothing raw, bias correction +9.8). A penalty on parameter changes does not target the readout offset.

## How reliable this test is
- **Medium.** λ from a 3-value grid on one seed; a diagonal Fisher; one architecture. A more recent paper (feed #33, Fisher-shaped EWC on a pretrained ViT) reports gains in another regime, so "EWC never helps" is not claimed.
