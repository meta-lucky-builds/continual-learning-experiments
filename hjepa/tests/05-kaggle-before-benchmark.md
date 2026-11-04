# Test 05: The "before" shift benchmark on Kaggle 2×T4

**Date built:** 2026-10-31
**Where:** Kaggle notebook, GPU T4 ×2, Internet on
**Artifact:** [`code/hjepa_shift_before_kaggle.ipynb`](../code/hjepa_shift_before_kaggle.ipynb), which embeds [`code/hjepa_shift_bench.py`](../code/hjepa_shift_bench.py)
**Status:** built and CPU-validated ([test 04](04-shift-benchmark-cpu-validation.md)). **Not yet run on real checkpoints, so there are no results yet.**

## Why we are running it

H-JEPA was only ever evaluated in the same dynamics it trained on. JEPA-TTT and a looped planner can only show value where the frozen model fails, and unshifted FourRoom is already at 96% ([test 03](03-fourroom-l3-seed42-reproduction.md)).

So this benchmark is the **"before"** measurement: frozen H-JEPA L3 under dynamics shifts, with no TTT and no looping. The later **"after"** run reuses the same folder, tasks, shifted data and script, and only swaps in the adapted model or the new planner. Any difference is then attributable to the method alone.

It is designed to say not just *whether* the model fails under shift but **why**, because that decides which fix to build first.

## Conditions

| Condition | Change | What it probes |
|---|---|---|
| `clean` | none | Control; must reproduce about 96% |
| `speed_x2` | agent speed 2.5 → 5.0 (the env's only native dynamics knob) | Wrong step size; changes which states are reachable per upper-level step |
| `rot45` | every action rotated 45° | Mild action remapping |
| `rot90` | every action rotated 90° | Severe action remapping |
| `wave75` | rotation angle 75°·sin(2πx/112), depending on position | JEPA-TTT's Two-Room wave field, ported to FourRoom |
| `clean_gd3x`, `rot45_gd3x` | 3× planner iterations (270, min 90), 20 tasks, paired with the base condition | Planner headroom |

Visuals never change, so the frozen encoder sees the same kind of images.

## What it measures, and why each one

| Metric | Why |
|---|---|
| Success, Wilson CI | Headline number |
| Paired comparison against clean (tasks lost or gained, exact McNemar p) | With 50 tasks, pairing by task ID is far more sensitive than comparing two rates |
| Final and minimum distance to goal, progress | Continuous; detects degradation before success flips |
| Steps to success | The budget is generous (test 03), so shifts may push successes into timeouts |
| Per-level teacher-forced loss on shifted data | This is the JEPA-TTT loss; shows which level the shift breaks |
| Per-level k-step rollout MSE, cosine, norm ratio | Error that compounds over the planning horizon |
| Copy-last-latent baseline | The predictor must beat it, or it isn't really predicting (from the probing paper) |
| Shuffled-action gap | How much each level uses its actions; a shift should shrink it, TTT should restore it |
| Linear xy probe, fit on clean, tested on shifted | Whether the encoder still knows where the agent is. Expected to hold, so failures would be in the predictor or planner, which TTT can fix |
| Planner iterations, cap-hit fraction, final cost | Whether the solver is running out of iterations |
| 3× iterations headroom | If more iterations recover tasks, a looped planner has something to gain |

Standard errors for model error are clustered by episode, so correlated windows from one episode don't inflate confidence.

## How it runs on 2×T4

- One process per GPU (`CUDA_VISIBLE_DEVICES`), with tasks sharded by `task % 2`.
- Each task has its own seed (`seed + task`), so the split doesn't change results.
- Each task writes its own JSON, so a re-run skips finished work. A Kaggle restart doesn't lose progress.
- Everything goes to `/kaggle/working/shift_bench`. Keep that folder for the "after" run.
- The smoke test runs 1 task per GPU with a 40-step budget and prints a time estimate before the long run.

## Predictions, written before the run so they can be checked

These are our hypotheses, not results.

1. **rot90 hurts most and speed ×2 least.** A 90° rotation makes every gradient step from the planner push the agent sideways. Speed ×2 still moves toward the goal, only overshooting.
2. **Most of the damage is at level 1.** Action rotations change how a raw action maps to motion, which is level 1's job. Upper levels pool level-1 action embeddings, so they may be less affected. If level 3's error barely moves under rot45 or rot90, that supports adapting level 1 only.
3. **Speed ×2 hurts the upper levels more than rotation does**, because it changes how far one 10- or 20-step macro-step goes.
4. **The xy probe holds across all shifts**, because the images are unchanged.
5. **Extra planner iterations won't recover rotated tasks**, because the model itself is wrong there (model error, not optimization error). They might recover some clean failures.

If prediction 5 fails (3× iterations do recover rotated tasks), the looped planner moves ahead of TTT in priority.

## Rough cost

About 65 s per task on one T4 (test 03), times 50 tasks × 5 conditions, plus 40 headroom tasks, plus collection and model error. Our estimate is 4–6 h on 2×T4. Failed tasks use the full budget, so shifted conditions take longer than clean.

## Results

*Pending your Kaggle run. When the results come in, this section gets `summary.json`, the tables and the figures, plus a check of each prediction above.*
