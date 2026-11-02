# Why M2c tied with replay, and why "compute-matched replay" won

2026-10-28. Diagnostic study on the v1.3 nonlinear toy. Toy results only; not evidence about real networks.

**Setup.** Same toy, data, model, seeds and streams as toy_cl3.py:
- 10 classes, 5 tasks, SEP 0.6, 5 memory items per class, 16 new and 4 replay items per step;
- 5 seeds, with ER and every M2c variant on identical streams;
- harness: notes/v13-checks/diag.py; raw logs: diag_*.jsonl.

**Harness checks before trusting anything:**
- diag.py reproduces toy_cl3's per-seed numbers exactly for ER, compute-matched ER and M2c.
- The weighting code at w = 1 reproduces ER bit for bit.
- Every projected step asserts surrogate descent and feasibility.
- The feasibility assert caught L-BFGS-B leaving ~1e-6 violations at zero slack. The zero-slack variant therefore uses an active-set-polished solver, unit-tested on 200 near-degenerate problems: worst relative violation 8e-10, 0 descent failures.
- The held-out set (64 old-class training items never in the buffer) has its own RNG.
- **One harness defect found:** in diag.py, "finetune" silently replays, so it duplicates ER. It is excluded below.
- **One protocol trap found:** this toy fills the buffer only at the end of a task. Under that policy, ER-ACE never replays the final task and scores 0% on it. ER-ACE is therefore compared with ER under the same start-of-task buffer policy (rows marked "start-insert").

## Results (means over 5 seeds)

**Test accuracy.** Old = classes of tasks 1–4; last = classes of task 5.

| Method | Compute vs ER | Test acc | Old-class acc | Last-task acc | Old-class errors predicted as a last-task class |
|---|---|---|---|---|---|
| ER | 1× | 41.5 | 29.6 | 89.3 | 87% |
| M2c as specified (κ_s = 0.2) | far more (minutes vs seconds per seed) | 41.2 | 29.1 | — | 88% |
| compute-matched ER (36 replay items) | 2.6× items per step | 44.9 | 34.3 | 87.3 | 81% |
| M2c, zero slack (κ_s = 0) | far more | 44.6 | 33.5 | 89.0 | 84% |
| M2c on top of compute-matched ER | far more | 45.1 | 34.4 | — | 81% |
| **ER, replay loss ×9 (same 4 items)** | **1×** | **47.8** | 38.1 | 86.7 | 79% |
| **ER, replay loss ×20** | **1×** | **50.0** | 40.8 | 86.5 | 76% |
| ER, start-insert | 1× | 39.9 | 27.4 | 89.7 | 88% |
| ER ×9, start-insert | 1× | 47.0 | 37.0 | 86.8 | 80% |
| **ER-ACE, start-insert** | **1×** | **57.8** | **59.0** | 53.0 | 25% |

**Where the knowledge is.** The last layer is refit on frozen penultimate features:

| Method | Its own test acc | Refit on all data | Refit on its 50 buffer items |
|---|---|---|---|
| ER | 41.5 | 70.6 | 55.5 |
| ER ×9 | 47.8 | 71.9 | 56.0 |
| compute-matched ER | 44.9 | 71.2 | 53.1 |
| M2c, zero slack | 44.6 | 70.8 | — |
| ER-ACE, start-insert | 57.8 | 72.3 | 57.4 |
| joint training (upper bound, toy_cl2) | 73.5 | — | — |

**M2c internals.** First-order damage is the decrease of the top-2-rival margins of currently correct items, at sampled steps:

| | M2c (κ_s = 0.2) | M2c (κ_s = 0) | M2c on compute-matched ER |
|---|---|---|---|
| Steps the projection changed | 44% | 99.8% | 7.7% |
| Step change, ‖Δ − Δ0‖ / ‖Δ0‖ in B⁻¹ | 8% | 51% | 0.8% |
| Damage removed, on the constrained pool | 19% | 100% | 0.7% |
| Damage removed, on held-out old-class items | 13% | 86% | 0.5% |

Paired differences against ER: M2c κ_s = 0.2 is −0.3 ± 0.5; M2c κ_s = 0 is **+3.1 ± 0.8**, positive in all 5 seeds; compute-matched ER is +3.4 ± 0.9.

## What the data establishes

**1. The dominant failure is a readout bias toward the newest classes, not loss of the features.**
- Under ER, 87% of old-class test errors are predicted as one of the two newest classes.
- Refitting only the last layer on the *same* frozen features recovers 70.6%, close to joint training's 73.5%. The refit is 55.5% even using only the 50 buffer items.
- So most of the old-class information survives in the representation. What fails is the mapping from features to the right logit. This matches the "access, not storage" pattern (DEEP-PATTERNS; H3 in the spec).

**2. Mechanism, from the softmax gradient.**
- For a new item of class y, the cross-entropy gradient on logit j is p_j − 1[j = y]. So every new item pushes **every old class's logit down** on its input.
- Through the shared head weights and features, that push carries over to old-class inputs.
- Replay pushes back, in proportion to its share of the loss. With 16 new and 4 replay items per step, each of the 2 new classes gets about 8 items of pull per step. Each of up to 8 old classes gets about 0.5 items of push-back. That is roughly a **16:1 imbalance per class**.
- **Prediction:** reweighting replay should help at no extra compute, and removing the new items' push on old logits should help most.
- **Both predictions hold.**
  - Weighting the same 4 replay items ×9 gives 47.8, and ×20 gives 50.0. Both are at ER's compute, and both beat replaying 9× more items (44.9).
  - ER-ACE masks the new items' loss to the current task's logits, so they no longer push old logits down. It gives 57.8 vs 39.9 under the same buffer policy, and recency errors fall from 88% to 25%.
  - ER-ACE pays on the newest task (53% vs 90%). That is a stability–plasticity trade, not a free lunch.

**3. Why "compute-matched replay" won: balance, not compute.**
- Its gain is almost entirely the larger replay share of the loss. The same share obtained by weighting, at 1× compute, does better.
- **Correction to v1.3:** "compute-matched ER is the bar" is wrong as stated. The bar is **replay with a tuned loss balance at the same compute**, plus a recency-bias fix such as ER-ACE or a balanced head refresh. Plain or compute-matched ER is a weak baseline on this toy.

**4. Why M2c as specified tied: it hardly acted.**
- Replay keeps buffer items fitted with large margins (95–97% buffer accuracy). The slack κ_s(m − m_floor) then permits large first-order decreases, so the constraints rarely bind.
- It changed 44% of steps, by 8% of the step's size, and removed only 19% of the damage even on its own pool.
- On top of heavier replay it removed 0.7%: replay had already made the constraints slack.

**5. With zero slack, M2c does act, and its protection does transfer. But it still does not beat cheap fixes.**
- With κ_s = 0 it removes 100% of first-order damage on the pool and **86% on held-out old-class items**.
- **This corrects my earlier summary "constraints memorise, penalties generalise".** In this nonlinear toy, first-order margin protection transfers well to unseen items of the same classes. The held-out failure seen in the first-order random toy was a property of that toy's weakly shared Jacobians.
- Yet accuracy rises only +3.1 points, to 44.6. That is the same as compute-matched ER, below ER ×9 (47.8, free) and well below ER-ACE (57.8, free). It also roughly halves the step.

**6. Open, and not explained by this data.**
- Removing 86% of held-out first-order margin damage bought only about 4 points of old-class accuracy, while the readout fixes bought 8–30. So most of the accuracy loss is **not** captured by step-wise first-order damage to currently correct items' top-2 margins.
- Candidates, all untested:
  - curvature accumulating over many small steps;
  - errors on items already wrong or near zero margin, which the damage metric excludes;
  - the new-class logit rising for reasons not visible in the top-2 rival set at the time of measurement.
- Resolving this needs a run that logs actual (nonlinear) margin change against the first-order prediction, per item.

## What this means for the design

- **The spec's M1 must be implemented faithfully first.** It already contains the component aimed at this failure: a class-balanced cosine head refreshed from stored statistics. The toy's plain linear head with plain cross-entropy left it out. M2c was therefore tested on a baseline missing the part designed for the dominant failure.
- **Baselines must include a recency-bias fix** (ER-ACE, or loss-balanced replay; also BiC/WA-type head correction) **at the same compute**. Every protection mechanism must beat those, not plain ER.
- **M2c stays a gated variant.** Zero slack makes it effective at its own object, at high cost and modest gain. Whether it adds anything **on top of** a debiased M1 is the real question, and it is untested.
- **The spec's own priority order is supported.** Fix readout and access (M1) before investing in parameter protection (M2).

Files: diag.py, diag_*.jsonl (this folder).
