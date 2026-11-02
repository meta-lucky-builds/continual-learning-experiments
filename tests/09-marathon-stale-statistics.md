# 09. Marathon F7/W2: do stored class statistics go stale?

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
The spec planned to store class statistics (means, covariances) and read out from them later. That only works if they stay valid while the network keeps training. This test measured how fast they go stale, and whether a simple correction can repair them.

## What we did
- Stored class means at the time each class was learned, then scored with them at the end.
- Measured old-class mean drift relative to the mean's norm.
- Tried corrections: one shared affine map fitted leave-class-out (oracle, all classes); the same map fitted on the buffer alone; per-class translation (SDC).

## What we got
- Stale stored means: about **20%** (chance 10%) at every memory size, every setting.
- Drift is 1.0–1.9× the mean's own norm.
- Shared affine map, leave-class-out: toy 65.8 vs oracle 68.4 (works), **MNIST 68.3 vs 86.7** (−18 \*, fails).
- Affine fitted on the buffer alone: worse than plain buffer means at 5/class; marginally better at 50/class (90.5 vs ~89.6).
- Per-class translation (SDC): 25.9 on MNIST.

## What we inferred
Stored statistics go stale catastrophically, and on MNIST the drift is class-specific, not one shared transformation. Statistics stay valid only if the features are frozen, or if they are re-estimated or carried by a validated map.

## Why we think it occurred
A trained network reshapes the feature space differently for different classes as it learns new ones. An in-sample affine fit "explains" 100% only because of its parameter count, which is why leave-class-out arms were used.

## How reliable this test is
- **Medium.** Solid on MNIST and the toy. **Does not transfer to CIFAR-100:** in test 39 (X2) a shared affine map carried means as well as a class-local one. So "drift is class-specific" is a property of MNIST/this MLP, not a general law.
