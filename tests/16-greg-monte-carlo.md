# 16. Monte Carlo check of the stable-reference (GREG) estimator

**When:** 2026-10-30, while answering the ARCH-v1 audit. **Source:** `design/AUDIT-RESPONSE-v1.md` §3.

## Why we did it
The audit proposed rebuilding each class's mean in a new encoder from a stable random chart r, where the class's mean is known exactly. We recognised this as the **regression estimator (GREG / control variate)** from survey sampling:

μ̂ = ȳ_B + A(μ_r − r̄_B)

Here B is a small bridge sample of the class seen in both coordinates. Theory says its error variance shrinks by a factor (1 − R²), where R² is how well r predicts the new features *within the class*. If true, the value of the whole idea is one measurable number. We checked the formula numerically before building on it.

## What we did
Simulated classes with a chosen within-class R² between the new feature and r. Took a bridge of n_B = 20 items and estimated the class mean two ways: buffer-only (bridge mean) and stable-reference (GREG). 4,000 trials.

## What we got
| within-class R² | buffer-only error var × n_B | stable-reference error var × n_B |
|---|---|---|
| 0.0 | 0.98 | 0.98 |
| 0.5 | 0.98 | 0.50 |
| 0.9 | 0.96 | 0.10 |

Exactly (1 − R²), as theory predicts.

## What we inferred
- Whether the stable chart helps is decided by one number per class, the within-class R². It can be measured on held-out items before trusting the estimator.
- If R² of learned features on random ones is low, the stable chart gives nothing over buffer-only.
- So we proposed a **multi-auxiliary** version using [r, previous chart]: no compounding from r, and high R² from the previous chart.

## Why we think it happened
It is a textbook result. The correction term removes the part of the bridge mean's sampling error that r can explain, and r's population mean is known exactly.

## Caveats
The check is on synthetic Gaussians. Whether learned CIFAR features have a high R² on random features was the open empirical question that [18](18-atlas-v1-cifar100.md) measured: R² = 0.67, but only +0.5 accuracy once the previous chart is also used.
