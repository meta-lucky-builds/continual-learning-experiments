# H-JEPA + JEPA-TTT + looping: experiment log

This repo records every test we've run so far, and why. For each test it covers what we did, what we got, what we inferred and why we think it happened. It also records the design decisions those tests fed into.

**Goal of the project:** add JEPA-TTT (test-time training of the world model) and looped transformers or planners to three-level H-JEPA, and show the result keeps planning well when the environment's dynamics change. After that, the plan is a real SO-101 arm.

## Tests, in order

| # | Test | Date | Result | Main inference |
|---|---|---|---|---|
| 01 | [CPU dry run of the Colab eval notebook](tests/01-colab-eval-cpu-dry-run.md) | 2026-10-30 | Pipeline ran end to end; 0% success with random weights, as expected | The notebook works with the unmodified repo. Upper planners need the stored latent-action queue |
| 02 | [First Colab smoke test](tests/02-colab-smoke-test-mplbackend.md) | 2026-10-30 | Crashed: Colab's `MPLBACKEND` leaked into the venv; fixed with `Agg` | Always pass a clean environment to subprocesses |
| 03 | [FourRoom three-level H-JEPA, seed 42](tests/03-fourroom-l3-seed42-reproduction.md) | 2026-10-30 | **96.0% (48/50)**, matching the paper's 96.0 ± 1.1; about 65 s per task on a T4 | Setup is trustworthy. No headroom unshifted, so we need a shift test |
| 04 | [Shift benchmark, CPU validation](tests/04-shift-benchmark-cpu-validation.md) | 2026-10-31 | All shift checks OK; clean equals stock; full harness exit 0 | Shifts are correct. The action-gap metric caught a model that ignores actions |
| 05 | [**"Before" shift benchmark on Kaggle 2×T4**](tests/05-kaggle-before-benchmark.md) | built 2026-10-31 | **Pending your run** | Predictions written down in advance |

## Decisions

| # | Decision |
|---|---|
| 01 | [How JEPA-TTT goes into H-JEPA](decisions/01-jepa-ttt-into-hjepa.md): freeze all encoders and action encoders, adapt predictors only |
| 02 | [Compute, checkpoints, skipping reproduction](decisions/02-skip-reproduction.md) |
| 03 | [TTT implementation rules found in the code](decisions/03-ttt-implementation-rules.md) |
| 04 | [Where looping goes](decisions/04-where-looping-goes.md): the planner first, with a StateLoop-style learned Adam state |
| 05 | [The research claim and order of work](decisions/05-research-plan-and-order.md) |
| 06 | [SO-101 real-robot plan and training data](decisions/06-so101-real-robot.md) |

## Full analyses and code

- [`analysis/`](analysis/): full paper reads and design notes. They link to each other by their original project paths.
- [`code/`](code/):
  - `hjepa_colab_eval.ipynb`: Colab reproduction notebook (tests 01–03)
  - `hjepa_shift_bench.py`: the shift benchmark script
  - `hjepa_shift_before_kaggle.ipynb`: the Kaggle "before" notebook (tests 04–05)

## Pinned resources

- H-JEPA code: https://github.com/kevinghst/H-JEPA at commit `97054ec` (MIT)
- Checkpoints: https://huggingface.co/jepa-world-models/h-jepa
- Data and eval tasks: https://huggingface.co/datasets/jepa-world-models/h-jepa
- Papers:
  - H-JEPA (arXiv 2610.06805)
  - JEPA-TTT (arXiv 2610.00722)
  - How to Loop Your Transformer (2610.40325)
  - StateLoop (2610.35544)
  - Probing Spatial State in VFMs (2610.40636)
