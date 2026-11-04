# Decision 02: Compute, pretrained checkpoints, and skipping further reproduction

**Dates:** 2026-10-30 (compute and checkpoints); 2026-10-31 (move to Kaggle 2×T4); 2026-11-02 (local GPU for the robot phase)
**Details:**
- [`analysis/hjepa-on-colab-t4.md`](../analysis/hjepa-on-colab-t4.md)
- [`analysis/hjepa-reported-results.md`](../analysis/hjepa-reported-results.md)

## Compute

- **Colab T4 was enough to start.** The models are small: 21.7M parameters for three-level FourRoom and 34.8M for three-level AntMaze (measured). The limits are session length and storage, not memory.
- **The T4 has no bf16.** Training would need fp16 (`trainer.precision=16-mixed`). Evaluation runs in fp32.
- **The benchmark moved to Kaggle 2×T4** for longer sessions and two GPUs. The shift benchmark runs one process per GPU.
- **The SO-101 robot phase will use your own local GPU**, because the planner has to run next to the arm.

## Pretrained checkpoints instead of training

Everything is public:
- the code (MIT);
- 84 simulation checkpoints on Hugging Face: LeWM, H-JEPA with 2, 3 or 4 levels, and HWM, for FourRoom, AntMaze, Push-T and Cube, each at 3 seeds;
- the datasets and the 50 fixed eval tasks per environment.

**Decision: use the released checkpoints and never retrain H-JEPA ourselves.**

**Why:** JEPA-TTT freezes the encoder and adapts on top of a trained model, the looped planner only reads the model, and the shift test needs a model trained on normal data. That is exactly what a released checkpoint is. Training would cost hours to days per environment on a T4 and add a source of mismatch with the paper.

## Skipping further reproduction

After [test 03](../tests/03-fourroom-l3-seed42-reproduction.md) matched the paper (96.0%), the suggested next runs were seeds 43 and 44, the FourRoom LeWM baseline, and AntMaze three-level.

**Your decision (2026-10-30): don't reproduce more; go to the actual work.**

**Why this is reasonable:**
- One exact match already shows the pipeline works.
- More seeds would tighten a number we aren't trying to beat.
- Every reproduction run costs about 1 GPU-hour.

**The one exception we kept:** an unshifted control is still needed inside every shift experiment. It isn't reproduction. Without it, we can't say how much a shift hurts. The shift benchmark runs `clean` alongside every shift for exactly this reason.

## What the paper's results told us about where to work

| Environment | Flat LeWM | Best H-JEPA |
|---|---|---|
| AntMaze | 18.0% | 73.3% (3 levels) |
| FourRoom | 40.7% | 96.0% (3 levels) |
| Cube | 32.0% | 62.7% (4 levels) |
| Push-T | 40.0% | 45.3% (2 levels) |

- FourRoom is at its ceiling, so it serves as a cheap testbed *under shift*.
- AntMaze and Cube still have room to improve.
- Push-T gets worse with depth (45 → 17 → 0.7%), so it isn't worth our time.
