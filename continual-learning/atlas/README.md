# Atlas: carrying class statistics across encoder versions

Atlas tests whether a learner can keep class statistics usable as its encoder changes, instead of storing images or old encoders.

| Path | What it is |
|---|---|
| [`RESULTS-v1.md`](RESULTS-v1.md) | Results and reading of the first CIFAR-100 run (Colab T4, 2 configs × 5 seeds) |
| [`notebooks/atlas_v2_tests.ipynb`](notebooks/atlas_v2_tests.ipynb) | The notebook behind those results, built from `src/v1/` |
| [`notebooks/atlas_v2b_tests.ipynb`](notebooks/atlas_v2b_tests.ipynb) | The follow-up, built from `src/v2b/`. **Designed, not yet run** |
| `src/v1/`, `src/v2b/` | Library, analysis and self-tests. Both are kept because each one is the exact source of its notebook |

The write-ups are [overview/17](../overview/17-atlas-notebook-unit-tests.md) (self-tests), [18](../overview/18-atlas-v1-cifar100.md) (v1 results) and [19](../overview/19-atlas-v2b-notebook.md) (v2b design).

The notebook names are historical: `src/v1/` builds `atlas_v2_tests.ipynb` because it tests architecture v2.

## Running

Self-tests use synthetic data with known answers, so they need no GPU or torch. From the repository root:

```bash
python scripts/check.py
```

To rebuild a notebook, run this from its source folder. It writes to that folder, and the content matches the copy in `notebooks/` (on Windows the line endings differ):

```bash
PYTHONUTF8=1 python build.py
```

The notebooks themselves run on Colab with a GPU. They download CIFAR-100 and stop if any self-test fails.

## Raw results

No raw results file from the v1 run was committed. Its numbers exist only in the appendix of `RESULTS-v1.md`, so they can't be recomputed here.
