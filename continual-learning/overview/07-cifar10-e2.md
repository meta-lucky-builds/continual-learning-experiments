# 07. CIFAR-10 E2: choosing the correction and the mixture weight online

**Source:** ewt.md §7.

## Why we did it
In E1 the best numbers used oracle or fixed settings (α = 0.7, w = 0.5). A continual learner cannot look at the test set. The question was whether the bias correction size α and the mixture weight w can be chosen **online, causally**, from a small held-out reserve.

## What we did
A reserve of 20 images per class (200 total), never used for training. At each task end, α̂ and ŵ were chosen by reserve log loss. 5 seeds, paired.

## What we got
| | Comparison | Result |
|---|---|---|
| Q1 | ER+bc(α̂) vs ER+bc(α = 1) | 54.7 vs 52.4: **+2.3 ± 0.4 \*** |
| Q2 | MIX(ŵ, α̂) vs MIX_half | 58.7 vs 57.3: **+1.4 ± 0.4 \*** |
| Q3 | Product of experts (POE_half) vs MIX_half | **+1.0 ± 0.2 \*** |
| Q4 | MIX(ŵ, α̂) vs the better single expert | **+4.0 ± 0.3 \*** |

The final α̂ values were 0.6–0.8, matching the oracle sweep. ŵ was 0.3–0.5.

## What we inferred
The two-path system works fully online and still wins. With 200 held-out items (but not 50), the scalars can be fitted causally.

## Why we think it happened
A class-balanced reserve measures the right objective: historical accuracy across all classes. It needs enough items to estimate a scalar, and 50 was too few. A product of experts beats an arithmetic mixture by a little because it sharpens agreement: when both experts lean the same way, the product is more confident than the average.

## Caveats
200 reserve items worked here; that is not a general threshold.
