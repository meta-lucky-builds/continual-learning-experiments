# 04. Why M2c tied ER: the diagnosis

**Date:** 2026-10-28. **Spec section:** v1.3 A4.
**Source files:** `source/notes/v13-checks/why-m2c-failed.md`, `diag.py`, `diag_*.jsonl`.

## Why we ran it
Test 03 showed M2c tying ER. Before discarding or fixing it, we needed to know *where* the accuracy was lost, and whether protection was even aimed at the right failure.

## What we did
On the same toy:
- measured where old-class errors go (to which classes);
- refitted only the last layer on ER's frozen features (does the knowledge survive in the features?);
- reweighted the replay loss (×9, ×20) at the same compute;
- ran ER-ACE (new items' loss only over new logits);
- ran M2c with zero slack and measured first-order margin damage on its pool and on held-out items.

## What we got

| Method (buffer filled at task start) | Acc | Old-class errors predicted as a newest class |
|---|---|---|
| ER | 41.5 | 87% |
| M2c, zero slack | 44.6 (+3.1 ± 0.8, 5/5) | 84% |
| ER, replay loss ×9 (same items, 1× compute) | 47.8 | 79% |
| ER, replay loss ×20 | 50.0 | 76% |
| ER-ACE | **57.8** (vs ER 39.9 in the same protocol) | 25% |

- Refitting only the last layer on ER's frozen features: **70.6%**, against 73.5% for joint training.
- M2c as specified removed only 19% of first-order damage on its pool and 13% held out. Zero-slack M2c removed 86% of held-out first-order damage but gained only +3.1 points.

## What we inferred
- **The knowledge survives in the features; the readout fails.** Most toy forgetting is an output-layer bias toward the newest classes.
- Compute-matched ER "won" in test 03 because of **loss balance, not compute**: the same share of loss obtained by weighting does better at 1× compute.
- M2c as specified does nothing because replay keeps buffer items at large margins, so its slack never binds.
- Zero-slack protection does transfer to held-out items, which **contradicted** my "constraints memorise" slogan (test 02).

## Why we think it occurred
The softmax gradient p_j − 1[j = y] pushes every old logit down on every new item. With 16 new and 4 replay items per step, that is a ~16:1 pull against each old class. Two predictions of this account held: reweighting replay helps at no extra compute, and masking the new items' loss to new logits (ER-ACE) helps most.

## How reliable this test is
- **Withdrawn part:** I first wrote "most of the accuracy loss is not first-order margin damage". After Lucky's audit this was withdrawn: removing 86% of *summed* damage can leave every boundary crossing in place. It only shows summed damage is a poor stand-in for accuracy.
- **Non-causal protocol (found by Lucky's audit):** the "start-insert" methods put a task's memory items in at task start, before those items arrive. So the ER-ACE 57.8 is a diagnostic, not an online result.
- **Harness defects found:** diag.py's "finetune" silently replays (excluded); with end-of-task insertion, ER-ACE never replays the last task.
- **Medium for the mechanism, low for the exact ER-ACE number.** Toy only.
