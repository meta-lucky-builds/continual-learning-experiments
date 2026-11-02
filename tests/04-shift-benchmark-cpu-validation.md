# Test 04: Validating the shift benchmark on CPU

**Date:** 2026-10-31
**Where:** Claude's cloud container (CPU only, no Hugging Face access)
**Artifacts tested:**
- [`code/hjepa_shift_bench.py`](../code/hjepa_shift_bench.py)
- [`code/hjepa_shift_before_kaggle.ipynb`](../code/hjepa_shift_before_kaggle.ipynb)

**Status:** passed end to end (exit 0), with random weights only

## Why we ran it

The shift benchmark is the "before" half of a before/after experiment. Its numbers are only useful if three things hold:

1. **Each shift does exactly what it claims.** A 90° rotation must rotate the action by 90° and change nothing else.
2. **"Clean" is really clean.** The clean condition must match the stock environment step for step, or the whole baseline is off.
3. **Every metric is computed correctly**, and splitting tasks across two GPUs doesn't change any result.

All three can be checked without a GPU or real weights, so we checked them before you spend Kaggle hours.

## What we did

1. **Shift correctness check (`CHECK_SHIFT`).** For each condition, we placed the agent at 3 positions, applied unit actions, and compared the measured displacement with the expected `speed · R(θ) · action`. Then we ran the clean shift environment and the stock `FourRoomDistractors` environment side by side for 50 random steps from the same seed.
2. **Action-transform trace.** We printed the raw action, the transformed action and the position for a few steps under `rot90`.
3. **Data collection.** We collected a few episodes per condition with the repo's own collection policy and HDF5 writer: clean, speed ×2, rot45, rot90, wave75.
4. **Model-error pipeline.** We ran the per-level metrics on the collected data:
   - teacher-forced loss;
   - k-step rollout error;
   - copy-last-latent baseline;
   - shuffled-action gap;
   - the xy linear probe.
5. **Planning.** We ran a few tasks per condition, sharded across 2 processes (standing in for the 2 GPUs), with per-task seeds.
6. **Summarize and report.** We checked the Wilson CIs, the McNemar pairing against clean, the planner stats, the figures and `report.md`.
7. **Full notebook execution** with only the Kaggle and GPU parts substituted.

## What we got

**Shift checks: all OK**

```
clean      speed=2.5  shift={'type': 'none'}                          OK
speed_x2   speed=5.0  shift={'type': 'none'}                          OK
rot45      speed=2.5  shift={'type': 'rotate', 'deg': 45}             OK
rot90      speed=2.5  shift={'type': 'rotate', 'deg': 90}             OK
wave75     speed=2.5  shift={'type': 'wave', 'amp_deg': 75, 'period_px': 112}  OK
clean shift env == stock env over 50 random steps: OK
```

**Action-transform trace under rot90** (clip to [-1, 1], then rotate by +90°, so (x, y) → (−y, x)):

```
ACT [ 1.138 -1.861] -> [1. 1.]          (clipped to [1, -1], rotated to [1, 1])
ACT [ 0.917 -0.992] -> [0.992 0.917]
ACT [-0.283 -1.361] -> [ 1.    -0.283]
```

Each step moved the position by 2.5 × the transformed action, as expected.

**Collection:** 5 conditions, 13 MB each, about 4 minutes on this CPU-limited container. We fitted a least-squares map from action to displacement on the collected rot90 data, and it came out as the 90° rotation matrix.

**Pipeline:** collection, model error, smoke, planning on both shards, summarize and report all finished, and the notebook exited with code 0.

## A finding the test produced: the first random model ignored its actions

The first model-error run gave:

```
L1: tf_mse=0.0673  mse_k8=0.0680  copy_k8=0.0000  action_gap_k8=0.0000
...
rot90: identical numbers to clean
```

An action gap of exactly 0, and clean equal to rot90, means **the predictor's output did not depend on the action at all.**

- **Cause:** H-JEPA's predictor injects actions through AdaLN layers that are zero-initialised. A freshly built model therefore has the action pathway switched off: every action predicts the same next latent, and the planner outputs zero actions.
- **Real checkpoints are fine:** training turns those weights on. But it meant our random test model couldn't exercise the rotation logic.
- **Fix, for testing only:** we re-initialised all 168 zero tensors with small random values (std 0.05). After that, actions changed the predictions and the planner produced non-zero actions, so the shift code path was really exercised.

**Copy baseline = 0.0000:** a random encoder maps consecutive FourRoom frames (mostly static background) to almost the same latent, so "copy the last latent" is trivially perfect. This is an artifact of random weights. With the real encoder the copy baseline is a meaningful bar the predictor must beat.

**xy probe R² was negative** (about −0.3 to −0.9). A random encoder carries no position information, so this is expected. It shows the probe code runs; it says nothing about the real encoder.

**Planning success was 0** on every condition, with final distances around 156–177 px, as expected from random weights.

## What we inferred

- The shift implementation is correct: measured motion matches the definition, and clean is identical to stock.
- The before/after harness works end to end, including GPU sharding with per-task seeds, so splitting across 2 GPUs gives the same result as 1.
- **The action-gap metric works as a detector.** It caught a model that ignores actions. On real checkpoints, that metric tells us how much each level actually uses its actions, which matters for where JEPA-TTT should adapt.
- **Zero-init AdaLN is a useful design pattern for us.** A new module that starts as a no-op lets the pretrained model be reproduced exactly at step 0. That is the same principle we planned for TTT's LoRA and the looped planner's correction term.

## What it did *not* test

- Real checkpoints
- The GPU install on Kaggle
- The Hugging Face download
- Real timings

The Kaggle run ([test 05](05-kaggle-before-benchmark.md)) is the first real run.

## Bugs found and fixed during this test

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: data` | `sys.path.insert(0, os.getcwd())` in the script |
| `env max_episode_steps must be greater than eval_budget` with a 20-step smoke budget | The smoke budget is now 40, since `max_episode_steps = 2·budget` must exceed `goal_offset = 60` |
| The smoke run would have run all 20 headroom tasks, because a condition's own `num_eval` overrode the flag | Added `--limit` to `plan` |
| Report label read "seed seed42" | Fixed |
| `nbformat` not installed for validation | Validated the notebook JSON and compiled every code cell instead |
