# Test 03: Reproducing three-level H-JEPA on FourRoom (seed 42)

**Date:** 2026-10-30
**Where:** Google Colab, T4 GPU
**Model:** released checkpoint `fourroom_hjepa_l3`, seed 42, from `jepa-world-models/h-jepa` on Hugging Face
**Eval:** the repo's own `eval.py` with config `fourroom_l3`, 50 fixed tasks, unchanged paper planner settings, fp32
**Status:** passed, matching the paper exactly

## Why we ran it

Every later experiment (shift test, JEPA-TTT, looped planner) is a comparison against this model. Before trusting any comparison, we had to show that our setup reproduces the paper:

- the Colab install,
- the downloaded checkpoint,
- the eval settings,
- the T4 instead of the authors' H200s.

If we couldn't get the paper's number, any later "improvement" or "drop" could just be a broken pipeline.

FourRoom three-level was chosen because it's the cheapest environment with a released three-level checkpoint, and three levels is the depth we want for JEPA-TTT and looping.

## What we got

```
env       model     seed  ours %   95% CI        paper %     consistent  hours
fourroom  hjepa_l3  42    96.0     86.5-98.9     96.0 ± 1.1  yes         0.99
```

| Metric | Value |
|---|---|
| Success | **48 / 50 = 96.0%** |
| Failed tasks | #24 and #29, both ran the full 210-step budget without reaching the goal |
| Steps to success (48 successes) | mean 175.3, median 180.5, min 135, max 202, SD 17.7 |
| Successes finishing at step ≥ 190 | 14 of 48 |
| Eval trajectory length | 76 frames for every task (a 75-step reference route) |
| Wall clock per task | about 54–75 s, around 65 s on average |
| Total evaluation time | 3,554 s (about 1 h) |

Fraction of the 50 tasks already solved by a given step (from the steps-to-success list):

| By step | 150 | 160 | 170 | 180 | 190 | 200 | 210 |
|---|---|---|---|---|---|---|---|
| Solved | 14% | 22% | 32% | 48% | 74% | 94% | 96% |

## What we inferred

1. **The pipeline is correct.** Matching 96.0% means the checkpoint, planner settings, Colab environment and fp32 eval on a T4 behave like the authors' setup.
2. **The exact match is partly luck.** The paper's 96.0 ± 1.1 is a mean over 3 seeds. With 50 tasks, our 95% Wilson interval is 86.5–98.9%, so this is a sanity check, not a precise measurement. GPU nondeterminism alone could flip a task or two.
3. **Unshifted FourRoom has no headroom.** At 96%, neither JEPA-TTT nor a looped planner can show a success gain here. This is why the next step became a *shift* test: we need conditions where the frozen model fails.
4. **The budget is generous, and that hides weakness.** The planner takes about 2.3× the 75-step reference route, and reading the cumulative curve as a budget sweep, a 180-step budget would give only about 48% success. A tighter budget would expose differences that success rate at 210 steps can't. This assumes the planner's choices don't depend on the budget, which we haven't checked.
5. **Steps-to-success and final distance are needed as metrics.** A better planner could show up as shorter paths even with success near 100%. Both went into the shift benchmark.
6. **Cost baseline: about 65 s per task on a T4.** This set the time estimates for every later run.

## Why we think it came out that way

- **96% rather than 100%:** FourRoom has doors and distractors. The two failures ran out of budget rather than diverging, which looks like slow progress (local minima in gradient planning, or detours through the wrong door) rather than a broken model. That is our inference; we didn't log the per-step trace for those two tasks in this run. The shift benchmark now records traces.
- **Paths about 2.3× longer than the reference:** the planner replans every step from a goal image, with hierarchical subgoals 10 and 20 env steps apart. Gradient planning in latent space doesn't optimize path length directly, it reduces latent distance to the goal, so detours are cheap for it. This is our reading; the paper doesn't discuss path efficiency.

## What came next

You decided not to spend more GPU time on reproduction (other seeds, the LeWM baseline). One reproduced run is enough to trust the setup, and the real work starts with the shift test. See [decisions/02](../decisions/02-skip-reproduction.md).
