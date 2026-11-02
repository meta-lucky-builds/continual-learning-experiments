# 06. CIFAR-10 round 1: E0 (is CIFAR a valid test?) and E1 (two-path test)

**Source:** ewt.md §6. **Code (not uploaded):** `cifar.py` with self-tests C1–C13, a pre-declared `plan.md`, and an independent review.

## Why we did it
Toy and MNIST results do not transfer automatically. On MNIST, random features beat learned ones, which would make "feature learning" meaningless as a test. So the first question was whether real images separate learned features from random ones (E0). Only then could the two-path design be tested: plastic network plus stable statistics expert (E1).

## What we did
**E0.** Full-data LDA on several fixed feature maps, seeds 0/1/2. Pre-declared criterion: learned must beat random by ≥ 5 points.

**E1.** Split CIFAR-10, 5 seeds, M = 500 buffer images, online single pass, lr 0.03 chosen on seed 100. Arms:
- ER;
- ER + bias correction;
- ACE;
- an LDA on a random conv net's features (RC-LDA);
- an LDA on a net frozen early (FZ-LDA);
- several ways of mixing the network and RC.

## What we got
| E0 feature map | LDA acc |
|---|---|
| joint-trained CNN | 75.7 / 75.5 / 75.4 |
| random CNN 1024-d | 56.7 / 57.8 / 58.4 |
| random CNN 512-d | 55.3 / 52.0 / 54.0 |
| random pixel projection | ~50 |
| raw pixels | 41.7 |

The criterion passed by 17–19 points on every seed.

| E1 arm | Final |
|---|---|
| ER | 42.7 |
| ER + bias correction | 52.6 |
| ACE | 50.8 |
| RC-LDA | 54.0 |
| FZ-LDA | 54.3 |
| MIX_stream (Bayes on stream loss) | 42.7 |
| MIX_resv (50 reserve items) | 53.6 |
| **MIX_half** | **57.9** |

- The two paths add: **+3.5 ± 0.4 \***. Their errors are complementary.
- The exact Bayes mixture on the stream log loss reduces to ER, with weight 1.0000 on every seed. This was predicted in advance.
- Bias correction **over-corrects** on CIFAR: the newest task gets 10.9% of predictions vs a true 20%. An oracle scale α ≈ 0.7 is best.
- RC-LDA vs ER+bc: +1.4 (4/5), not significant.

## What we inferred
- CIFAR is a valid testbed: learned features matter here.
- A stable statistics expert and the plastic network make different mistakes, so mixing them helps.
- A mixture weighted by current-stream loss throws the stable expert away, because it optimises the current task.

## Why we think it happened
- The Bayes stream mixture optimises log loss on the data arriving now. On current data the plastic net always wins, so the weight goes to 1. Balanced historical accuracy is a different objective and needs a class-balanced held-out set.
- Over-correction on CIFAR, against under-correction on the toy, shows the size of the bias depends on the setting. That is why it must be fitted.

## Caveats
- lr was chosen on the seed-100 test set (mild test reuse).
- Bugs caught before results: float32 probabilities underflowed (fixed with float64 log-probs), and vacuous checks C8/C9/C11 were strengthened.
