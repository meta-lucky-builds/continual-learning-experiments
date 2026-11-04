# 10. Marathon T5 + FU-d: do head-row snapshots give savings on recurrence?

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
The spec's M4 component kept snapshots of the head rows for old tasks, to restore when a task recurs ("savings": faster relearning). An earlier run showed strong apparent savings. This test asked whether that was real or a readout artefact.

## What we did
On recurrence of task 1, compared: restoring the task-1 head rows; keeping the current rows; restoring the initial rows; restoring a full snapshot. Scored under the raw readout and under a bias-corrected readout.

## What we got
- **Raw readout:** restoring rows looks like strong savings: start accuracy 67 vs 17 (toy), 87 vs 51 (MNIST).
- **Bias-corrected readout:** restored rows are **worse** than current ones: 17.0 vs 42.3 (toy, −25.3 \*), 20.0 vs 76.7 (MNIST, −56.7 \*). Only slightly better than restoring the initial rows (+9.4 \* toy, +1.8 \* MNIST).
- A full snapshot restores task 1 perfectly but wipes everything else (all-seen accuracy 21%).

## What we inferred
"Snapshot restore saves time" was a readout artefact, and is withdrawn. Head-row snapshots are invalid under feature drift unless paired with the features they were taken in.

## Why we think it occurred
Old rows no longer fit drifted features (test 09). Their apparent benefit came from their large "newest class" logits at the moment of restore, which is the readout bias of test 05 again. ER's fast raw recovery (~4 steps on MNIST) is also mostly the bias flipping back toward the current task.

## How reliable this test is
- **High for the withdrawal:** large effects, 5/5 seeds, and a clear mechanism. It shows how a raw-readout metric can produce false savings.
