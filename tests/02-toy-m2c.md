# 02. Toy test of M2c (the spec's protection mechanism)

**Source:** ewt.md §2 (v1.3 A3). **Code (not uploaded):** `toy_cl3.py`.

## Why we did it
M2c was the spec's lead anti-forgetting mechanism. It is a per-step projection that forbids first-order decreases of stored items' margins against their top rivals. It was the first claim about *learning* that could be tested. If protecting margins prevents forgetting, M2c should beat plain experience replay (ER).

## What we did
- Data: 10 Gaussian classes in 20 dimensions (separation 0.6), split into 5 tasks of 2 classes.
- Model: MLP 20-64-64-10, Adam lr 1e-3, 300 steps per task, batch 16 new + 4 replay, 5 memory items per class, 5 seeds.
- ER and M2c share identical streams and replay draws (paired).
- Added baselines: fine-tune, GEM, compute-matched ER (36 replay items), and joint training (upper bound).

## What we got
| Method | Final acc | vs ER |
|---|---|---|
| fine-tune | 18.2 | |
| GEM | 39.1 | |
| ER | 41.5 | |
| M2c as specified | 41.2 | −0.3 ± 0.5 |
| compute-matched ER | 44.9 | +3.4 ± 0.9 |
| joint | 73.5 | |

M2c raised accuracy on the buffer (94.8 → 96.8%) but **not on the test set**. Spending its compute on more replay won by 3.7.

## What we inferred
The headline mechanism of the spec did nothing measurable on the toy. Something other than margin damage on stored items was driving forgetting. That question became [03](03-m2c-diagnosis.md).

## Why we think it happened
Diagnosed in 03. In short, replay already keeps buffer items at large margins, so M2c's slack let them lose a lot before any constraint became active. The real loss was elsewhere, in the readout.

## Caveats and harness bugs caught by review
- The default separation 1.6 saturates (ER 99.4%), so 0.6 was used.
- The first version had unpaired streams and no compute matching.
- A docstring falsely claimed a test-never-trained assert.

All were fixed before the reported numbers.
