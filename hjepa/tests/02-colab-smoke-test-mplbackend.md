# Test 02: First Colab smoke test (2 tasks) and the matplotlib crash

**Date:** 2026-10-30
**Where:** Google Colab, T4 GPU
**Command:**

```
/content/hjepa-venv/bin/python eval.py --config-name fourroom_l3 seed=42 \
  policy=/content/drive/MyDrive/hjepa_home/ckpts/fourroom/fourroom_hjepa_l3/seed42/fourroom_hjepa_l3_object.ckpt \
  output.dir=colab_smoke2_l3 eval.num_eval=2
```

**Status:** failed on the first try, fixed, then passed

## Why we ran it

The notebook runs a 2-task smoke test before the full 50-task evaluation. The smoke test has two jobs:

- catch any crash in a minute instead of an hour in;
- time one task, so we know how long the full run will take.

## What we got

The process crashed at import time, before any planning:

```
File "/content/H-JEPA/h_jepa/eval.py", line 8
  from droid_eval import run_clip_eval
  ...
File ".../lightning/fabric/utilities/imports.py", line 41
  _TORCHMETRICS_GREATER_EQUAL_1_0_0 = compare_version("torchmetrics", ...)
  ...
File ".../torchmetrics/utilities/checks.py", line 26
  ...  (matplotlib import fails)
```

Lightning imports torchmetrics, which imports matplotlib. Matplotlib then failed to load its backend.

## Why it occurred

- Colab sets the environment variable `MPLBACKEND` to its own inline plotting backend (`module://matplotlib_inline.backend_inline`).
- The notebook launches `eval.py` as a subprocess in a separate Python 3.11 venv, and the subprocess inherits `MPLBACKEND`.
- That inline backend package only exists in Colab's own Python, not in our venv. So matplotlib tries to import a module that isn't there and the whole import chain fails.

This could not show up in the CPU dry run ([test 01](01-colab-eval-cpu-dry-run.md)), because our container doesn't set `MPLBACKEND`. Claude reproduced the exact error locally by setting the variable to Colab's value.

## What we did

Set `MPLBACKEND='Agg'` (a non-interactive backend that ships with matplotlib) in the `ENV` dictionary passed to every repo command. The notebook's environment-check cell now also imports matplotlib and lightning, so this class of problem fails in the checks rather than in the eval.

You replaced cell 2 with the fixed block and re-ran the smoke test, which then passed and led to [test 03](03-fourroom-l3-seed42-reproduction.md).

## What we inferred

- Lesson for every later notebook (Colab and Kaggle): **always pass an explicit, clean environment to subprocesses**, including `MPLBACKEND=Agg`, `SDL_VIDEODRIVER=dummy` and `MUJOCO_GL=egl`. The Kaggle shift benchmark has this built in.
- Nothing was wrong with the repo, the checkpoint or the venv. It was purely the host environment leaking into the subprocess.
