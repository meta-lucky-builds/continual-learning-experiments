# 03. Why M2c tied: the diagnosis

**Source:** [ewt.md](../source/notes/ewt.md) §3 (A4). **Code:** [`diag.py`](../source/notes/v13-checks/diag.py), [`why-m2c-failed.md`](../source/notes/v13-checks/why-m2c-failed.md).

## Why we did it
A null result is only useful if you know why it is null. Before redesigning anything, we wanted the mechanism. Was M2c weak, or was it protecting the wrong thing?

## What we did
Same toy as [02](02-toy-m2c.md). Variants of ER and M2c were run, and the share of old-class errors predicted as a newest class was measured. We also refit only the last layer on ER's frozen features.

| Method (buffer filled at task start) | Acc | Old-class errors → newest class |
|---|---|---|
| ER | 41.5 | 87% |
| M2c, zero slack | 44.6 (+3.1 ± 0.8, 5/5) | 84% |
| ER, replay loss ×9 | 47.8 | 79% |
| ER, replay loss ×20 | 50.0 | 76% |
| ER-ACE | **57.8** | 25% |

## What we got
- **Refitting only the last layer on ER's frozen features gives 70.6%, against 73.5% for joint training.** The knowledge survives in the features. The readout fails.
- Reweighting replay helps at no extra compute. Masking new items' loss to new logits (ER-ACE) helps most.
- M2c as specified removed only 19% of first-order damage on its pool (13% held out). Zero-slack M2c removed 86% of held-out damage but bought only +3.1 points.

## What we inferred
Forgetting in this setting is mostly **readout bias toward the newest classes**, not loss of what the features know. This is the root of almost everything that followed.

## Why we think it happened
The softmax gradient on a new item is p_j − 1[j = y]. It pushes every old logit down on every new item. With 16 new and 4 replay items per step, old classes face a ~16:1 pull. Both predictions of this account held: reweighting replay helped, and ER-ACE, which removes the pull, helped most. Compute-matched ER "won" in 02 because of loss balance, not compute.

## Caveats
- The ER-ACE 57.8 uses "start-insert": a task's memory items are added before they arrive. That is non-causal, so 57.8 is a diagnostic, not an online result (found in the audit, [11](11-audit-corrections.md)).
- An inference that "most accuracy loss is not first-order margin damage" was **withdrawn**. Removing 86% of *summed* damage can leave every boundary crossing in place.
- `diag.py`'s "finetune" silently replayed and was excluded. With end-of-task insertion, ER-ACE never replays the last task (protocol trap).
