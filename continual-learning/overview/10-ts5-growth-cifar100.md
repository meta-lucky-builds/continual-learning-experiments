# 10. T-S5: growth budget on Split CIFAR-100

**Source:** ewt.md §10. Budget fixed in advance in `budget.json`.

## Why we did it
CIFAR-10 has only 5 tasks. A longer stream (20 tasks, 100 classes) tests whether the conclusions hold, and asks a growth question: is it better to keep frozen versions of the network (a "bank"), or to spend the same memory on a bigger buffer? Which versions should be kept?

## What we did
- 20 tasks of 5 classes, M = 2,000, lr 0.03 (not retuned), 5 seeds.
- Gates: self-tests still pass, CIFAR-10 streams bit-identical, CIFAR-100 stream checks 11/11.

## What we got
| Arm | Final (100 classes) |
|---|---|
| ER raw / +bc | 17.4 / 22.7 |
| RC-LDA | 28.2 |
| MIX_half(ER, RC) | 29.7 |
| BANK_all (+19 versions, +21 MB) | 31.0 |
| BANK_4spread (+4.4 MB) | **31.2** |
| BANK_4last (+4.4 MB) | 28.0 |
| MIX_half(ER at matched memory, RC) | 30.1 |
| Sleep+bc | 14.8 |

- G2: keeping the 4 *last* versions loses to all: **−3.1 \***. Our prediction was **wrong**.
- G3: 4 spread versions ≈ all versions (null).
- G5: Sleep −7.9 \*.

## What we inferred (at the time, later corrected)
Each frozen version's own LDA accuracy, by task frozen: 26.0, 27.8, 29.4, 29.1, 26.3, 24.9. We read this as "online features peak mid-stream, then decay below random". **That reading was wrong and was withdrawn.** See [11](11-audit-corrections.md).

## Why we think it happened (corrected)
Each number mixed encoder quality with readout data. A version frozen at task 18 had its old classes only from the buffer (~21 items each). A version frozen at task 0 had 480 each. The "decay" was readout coverage, not features. Spread versions win over the last ones because late versions are near each other and have poor coverage of old classes. Spread versions cover different eras of the stream.

## Caveats
"Spread" selection uses knowledge of the stream length. A streaming selection rule is untested.
