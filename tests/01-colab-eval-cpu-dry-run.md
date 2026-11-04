# Test 01: CPU dry run of the Colab evaluation notebook

**Date:** 2026-10-30
**Where:** Claude's cloud container (CPU only, no GPU, no Hugging Face access)
**Artifact tested:** [`code/hjepa_colab_eval.ipynb`](../code/hjepa_colab_eval.ipynb)
**Status:** passed, with fixes applied before you ran it

## Why we ran it

You have no local GPU, so the first real run was always going to be on a Colab T4. A failed Colab run costs a session and GPU quota. The goal was to catch every bug we *could* catch before Colab, by running the real H-JEPA repo code end to end on a machine without a GPU.

The question this test answers: **does the notebook's pipeline (install → checkpoint checks → planning eval → resume → results table) work with the repo exactly as it is, at the pinned commit `97054ec`?**

It does *not* answer whether the model plans well. The weights were random.

## What we did

1. Cloned `kevinghst/H-JEPA` at commit `97054ec` and built the Python 3.11 venv with `uv pip install -e ".[train,env]"`, the same command the notebook uses.
2. Built a checkpoint with the real three-level FourRoom architecture but **random weights**, because Hugging Face downloads were blocked from the container.
3. Generated a tiny eval-task file locally with the repo's own task generator (3 tasks instead of 50).
4. Wrote a small harness (`cpu_eval.py`) that patches `nn.Module.to("cuda")` to CPU, then calls the repo's unmodified `eval.py`.
5. Executed the notebook's code cells in order (`nbtest.py`), substituting only the Colab-specific parts (Drive mount, GPU check, download).

## What we got

| Step | Result |
|---|---|
| Install of the repo with `train` and `env` extras | OK (exit 0) |
| SHA-256 check of the task file | `OK` |
| Checkpoint check | `HJEPA, levels=3, params=21.66M, finite=True`; level-2 and level-3 latent-action queues of shape (100, 4) present |
| Planning eval, 2 tasks | Ran to completion: `success_rate 0.0`, about 2–3 s per task on CPU |
| Smoke-test timing cell | `SMOKE OK ... 2 s per task` |
| Resume (re-run skips finished tasks) | OK |
| Results table cell | First failed with `NameError: name 'results_dir' is not defined`, fixed, then OK |

Problems hit along the way, and what each one told us:

| Error | Cause | What we did |
|---|---|---|
| `ModuleNotFoundError: droid_eval` | `eval.py` imports sibling modules, so it must run with `h_jepa/` as the working directory | The notebook always runs `eval.py` from `h_jepa/` |
| `UnsupportedInterpolationType: eval` | The configs use a custom `${eval:...}` resolver, registered only when `eval.py` is the entry point | Always go through `eval.py`, never compose the config by hand |
| `AssertionError: Higher-level GD requires latent action priors from the model` | Upper-level planners seed candidates from a stored latent-action queue. A freshly built model has an empty queue | Our random test model got a filled queue. Real checkpoints carry it, and the notebook's checkpoint check now verifies it is present |
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | No GPU EGL in the container (MuJoCo headless rendering) | Expected on CPU. It only matters for AntMaze and Cube, and the notebook sets `MUJOCO_GL=egl` for the GPU |
| `NameError: results_dir` in the results cell | A real notebook bug | Fixed |

## What we inferred

- The pipeline itself was sound: install, checkpoint loading, the repo's planner and the results code all worked with the unmodified repo.
- 0% success with random weights is the expected result, not a failure. A random world model gives the planner a meaningless cost.
- The latent-action queue is a hidden requirement of the upper-level planners. Anything we build later (TTT especially) must not overwrite it. This turned into a design rule in [decisions/03](../decisions/03-ttt-implementation-rules.md).

## Why it came out that way

The repo was written for GPU clusters, but nothing except the hard-coded `.to("cuda")` actually requires a GPU, so a one-line patch was enough to run the true code path. The errors we hit were entry-point and state assumptions (working directory, config resolver, stored queue), which are exactly the kind of thing a dry run finds.

## What it did *not* test

- The Hugging Face download (blocked from the container)
- CUDA itself
- Real checkpoint quality
- T4 speed
