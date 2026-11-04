# T-S5: growth budget on Split CIFAR-100, 20 tasks (pre-declared 2026-10-29, before any CIFAR-100 run)

**Question.** Sleep grows parameters with every consolidation. On CIFAR-10 our one growth mechanism that helped was the promotion bank: frozen network versions, each with its own streaming statistics (+2.7 \* in round 2, S3). Two things are tested here:
- Does growth keep helping over 20 tasks?
- Does the gain survive a **fixed parameter budget**, and does it beat simply spending the same memory on a bigger buffer?

**Data.** CIFAR-100 from the fast.ai mirror (cifar100.tgz). It was converted to data100/{train,test}.npz by a script that decodes each PNG, checks the image shape, and checks 500 train and 100 test images per fine class. Labels are the index of the fine class in the sorted list of folder names, saved in data100/classes.txt.

**Stream.**
- 20 tasks of 5 classes, a per-seed class permutation and the same RESV = 20 rule (2,000 reserve items).
- Online, single pass, batch 10 plus 10 replay.
- lr 0.03, carried over from CIFAR-10 and **not retuned** (declared).
- Buffer M = 2,000 (20 per class).
- Code: cifar.py, made configurable through the env variables CL_K = 100 and CL_NTASK = 20, plus CIFAR_DIR and MEM. With the defaults unchanged, behaviour is identical: self-tests C1–C13 are rerun at the defaults, and a CIFAR-100 stream check is added.

**Arms** (seeds 0–4, results/ts5):

| Arm | What |
|---|---|
| erp | ER at M = 2,000 with promote = 1. Versions are taken at the ends of tasks 1–19, each with 512-d streaming LDA from promotion on. Self-test K2 shows promotion does not change the learner, so this run also gives the ER path. |
| rc | RC-LDA (random-conv statistics expert) |
| er_mmatch | ER at M = 8,836, i.e. 2,000 + B_img with B_img = 6,836: BANK_all's extra memory as images, 19 versions × (93,696 encoder-body parameters + 182,628 statistic floats) × 4 bytes / 3,072 (results/ts5/budget.json) |
| slp | slp_r16_buf at M = 2,000, with the CIFAR-10 hyperparameters (dlr 0.03, 30 epochs), not retuned |

B_img is computed and fixed before running, from the actual StreamLDA statistic sizes; it is logged in results/ts5/budget.json.

**Combinations** (from the saved per-task-end outputs, no extra training):
- MIX_half = mean(ER+bc, RC).
- BANK_all = mean(ER+bc, RC, all 19 versions).
- BANK_4last = mean(ER+bc, RC, the 4 most recent versions).
- BANK_4spread = mean(ER+bc, RC, the versions from the ends of tasks 4, 9, 14 and 19).

A budget of 4 versions matches CIFAR-10's bank.

**Pre-declared comparisons** (final accuracy over 100 classes, paired, same significance rule):

| | Comparison | Question |
|---|---|---|
| G1 | BANK_all vs MIX_half | Does unbounded growth help over 20 tasks? |
| G2 | BANK_4last vs BANK_all | Does the gain survive a fixed budget, keeping the newest versions? |
| G3 | BANK_4spread vs BANK_all | The same, keeping spread-out versions |
| G4 | BANK_all vs MIX_half(er_mmatch, RC) | Does growth beat spending the same memory on a bigger buffer? |
| G5 | slp+bc vs ER+bc | Does Sleep-style consolidation help over 20 tasks? |

The accuracy-versus-memory curve is also reported, with the number of versions k = 0, 1, …, 19 on the x-axis (most recent first, and spread).

**Predictions.** G1 positive. G2 and G3 lose at most a third of the G1 gain. G4: the bigger buffer wins or ties. G5 is negative.

**Gates.**
- C1–C13 still pass at the CIFAR-10 defaults after the configurability change.
- A new check confirms the CIFAR-100 stream: 20 disjoint tasks of 5 classes, 20 reserve items per class, and no reserve items in any batch.
- For each arm, a 30-step smoke run is checked, then a config check in the analysis.
