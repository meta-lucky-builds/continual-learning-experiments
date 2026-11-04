# 03. Toy test of M2c, the spec's lead protection mechanism

**Date:** 2026-10-28. **Spec section:** v1.3 A3.
**Source files:** `source/notes/v13-checks/toy_cl3.py` (earlier `toy_cl.py`, `toy_cl2.py`), logs `toy_run*.log`.

## Why we ran it
M2c was the spec's main mechanism against forgetting: at each step, project the update so that no stored item's margin against its top rivals decreases to first order. It had never been run. This was the first learning test of the spec.

## What we did
- 10 Gaussian classes in 20 dimensions, separation 0.6, split into 5 tasks of 2 classes.
- MLP 20-64-64-10, Adam lr 1e-3, 300 steps per task, batch 16 new + 4 replay, 5 memory items per class, 5 seeds.
- ER and M2c share identical streams and replay draws (paired).
- Baselines: fine-tune, GEM, ER, compute-matched ER (36 replay items, matching M2c's extra compute), joint training.

## What we got

| Method | Final acc | vs ER |
|---|---|---|
| fine-tune | 18.2 | |
| GEM | 39.1 | |
| ER | 41.5 | |
| M2c as specified (R_pre = 1 or 25) | 41.2 | −0.3 ± 0.5 |
| compute-matched ER | 44.9 | +3.4 ± 0.9 |
| joint (upper bound) | 73.5 | |

M2c raised accuracy on the buffer items (94.8 → 96.8%) but not on the test set. Spending its compute on more replay won by 3.7 points.

## What we inferred
M2c as specified does not beat ER. The spec's lead mechanism had no measurable benefit in its own toy.

## Why we think it occurred
Diagnosed in test 04: M2c's slack let well-fitted buffer items lose a lot of margin before any constraint bound, and most of ER's forgetting was a readout bias that margin protection does not target.

## How reliable this test is
- **Harness problems caught by review, all fixed before this result:** the default separation 1.6 saturated (ER 99.4%); the first version had unpaired streams; there was no compute matching; a docstring falsely claimed an assert that test items are never trained on.
- **Medium.** Paired seeds and a reviewed harness, but a toy: 20-d Gaussians, 5 seeds. It cannot speak to real images.
