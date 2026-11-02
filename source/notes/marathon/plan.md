# Sharp-test marathon: plan

2026-10-28. These tests isolate mechanisms; they do not show that the full system wins.

**Settings.** Each test runs on:
- the v1.3 toy, which has ground truth and is cheap;
- Split-MNIST, real data: 5 tasks of 2 digits, an MLP 784-256-256-10, Adam, batch 16 + replay, 5 memory items per class.

**Statistics.** 5 seeds; paired differences on identical streams wherever the design allows.

**Code.** notes/marathon/mar.py, with `selftest` run first; the results are in notes/marathon/results/.

## Reliability rules for every test (Lucky's rule)

Before any result counts, the harness must pass `python mar.py selftest`:

| # | Check | Kind |
|---|---|---|
| S1 | toy ER reproduces toy_cl3's per-seed numbers exactly | reproduction |
| S2 | the same config run twice gives bit-identical results | determinism |
| S3 | every "off" switch reproduces ER exactly: replay weight w = 1, warm-up 0, default init, EWC λ = 0 | null controls |
| S4 | ER-ACE with start-insert reproduces diag.py exactly | reproduction |
| S5 | nearest-class-mean head: 100% on separable synthetic features; equals linear-probe ranking on a planted case | unit + planted effect |
| S6 | weight aligning leaves predictions unchanged when all row norms are equal | unit |
| S7 | snapshot restore is bit-identical to the saved tensors; a wrong snapshot is really from the other time | unit |
| S8 | MNIST data: official train/test split, 10 classes, pixel range [0, 1]; joint upper bound > 95%; fine-tune ≈ last task only | sanity bounds |
| S9 | detection simulator reproduces the closed-form L_min boundary (two states) | closed-form check |

An independent reviewer reads the harness before results are reported. Any test that fails a check is reported as **invalid**, not dropped silently.

## The tests

Each test lists: the hypothesis, the comparison that isolates it, and what would kill the hypothesis.

**T1. Task-boundary shock (where the collapse comes from).** Old-class accuracy collapses 20–40 steps into each task.
- H: the collapse is caused by the newly trained class rows (initialisation and the first large updates), not by gradual drift.
- Arms, at equal compute:
  - (a) default;
  - (b) new rows initialised at the class-mean feature of the first batch, with norm matched to existing rows;
  - (c) head-only warm-up for the first W = 30 steps of each task;
  - (d) (b) + (c).
- Kill: none of (b)–(d) reduces the step-40 dip or improves final old-class accuracy by more than the seed noise.

**T2. Which part of the readout bias matters.** All arms at equal compute.
- Arms: ER; ER-ACE; replay weight ×9; post-hoc weight aligning (WA: rescale new rows' norms to the old mean); nearest-class-mean head from buffer features (the spec's M1-style statistics head); logit-bias-only correction (subtract the per-class mean logit on buffer items).
- H: the bias is mostly **row norm or bias** (WA or bias correction fixes it), versus **direction** (only NCM or refit fixes it).
- Kill: WA/bias ≈ NCM ⇒ norm and bias suffice; NCM ≫ WA ⇒ direction matters.

**T3. Features vs readout over time (H3, access vs storage).**
- Measures: after each task, the frozen-feature linear-probe accuracy (refit on all data, and on the buffer) next to the model's own accuracy.
- H: features keep most old-class information; the readout loses it.
- Kill: probe accuracy on old classes falls as fast as the model's own.

**T4. Fresh vs stale protection estimates (H1, anchor).**
- Arms: diagonal-Fisher quadratic penalty (EWC-style) with the Fisher estimated
  - **stale:** at each task end and frozen;
  - **fresh:** recomputed every 25 steps on the buffer at the current weights.
  Same anchor, λ ∈ {1, 10, 100} × normaliser, and ER underneath.
- H1 predicts fresh > stale at matched λ.
- Kill: fresh ≤ stale at every λ.

**T5. Snapshot savings on recurrence (M4, the spec's most distinctive piece).**
- Stream: tasks 1–5, then task 1 returns for 300 steps.
- Arms:
  - continue (ER);
  - restore the full task-1-end snapshot;
  - restore only the task-1 head rows from the snapshot;
  - the same but from a **wrong** snapshot (task-3-end);
  - a fresh model trained on task 1 alone.
- Metric: censored time to reach 90% of task 1's reference accuracy, and accuracy on the other tasks after recovery.
- H: the right snapshot cuts time-to-threshold vs both continuing and the wrong-snapshot control, without destroying the other tasks.
- Kill: right ≈ wrong snapshot, or the savings come only with a large loss on other tasks.

**T6. Rival coverage in M2c (running now).**
- Already measured: 97–100% of flips go to a newest-task class, which was in the item's start-of-task top-2 rivals only 0–17% of the time.
- Arms: zero-slack M2c with vs without the spec's "labels born since the last refresh" rival rule.
- Kill: the rule removes most of the flips, yet accuracy does not improve.

**T7. Context detection with real evidence (M3).**
- Simulator: the sticky-HMM filter with per-batch log-likelihood ratios computed from fitted Gaussian class-conditional models on toy and MNIST features, under a conditional-shift stream.
- H: detection succeeds iff the realised per-batch evidence exceeds the simulated multi-state threshold; delays match the simulation.
- Kill: systematic mismatch (an implementation or theory error).

**T8. Robustness of the headline findings.**
- Re-run T2 and T3 on a second toy setting (SEP 0.8, 20 memory items per class, 8 replay) and on MNIST.
- A finding counts only if its sign holds in all three settings.

## Explicitly excluded (and why)

- **Coverage via augmented neighbours.** Not fair on the toy, whose noise *is* the data distribution. Deferred to image data with real augmentations.
- **Usage and release dynamics.** These need long recurring horizons; they come after T5.
- **Reactivation acceptance power.** Already exact by simulation in v1.3.
