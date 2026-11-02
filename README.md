# cl: every continual-learning test we have run, and what each one means

This repository records every test in the project so far:
- why we ran it;
- what we did;
- what we got;
- what we inferred;
- why we think it happened.

Each test has its own file in `tests/`. Negative results, withdrawn claims and our own mistakes get the same weight as positive results.

## The story in one paragraph
- We started with a spec for a continual learner whose centrepiece was protecting parameters (M2c). Its math was repaired (01), and then it failed to help (02).
- Diagnosis showed forgetting was mostly a **readout bias** toward new classes, not lost knowledge (03–05).
- On CIFAR, a **two-path** system won consistently: a plastic network plus a stable statistics expert, mixed with weights chosen on held-out data (06–08). Every "protect or compress the learner" idea lost (08, 09).
- On a 20-task stream, an outside audit showed the features never got worse. What decays is **class statistics stranded in old encoder coordinates** (10, 11).
- Carrying those statistics into new encoders worked offline (X2, 13).
- That led to the **Atlas** architecture: rebuild old classes' statistics in each new encoder and measure what can't be rebuilt. Its first version had seven false claims, shown by counterexamples (15). The fix used a survey-sampling estimator, checked by simulation (16).
- Our first fresh CIFAR-100 runs (17, 18):
  - confirmed that **per-class regression beats a global map**;
  - refuted "use a held-out reserve as the bridge";
  - failed to replicate X2's gain;
  - revealed that our learner was too weak: random features beat it.
- The next notebook (19) is built to fix that. It has not been run yet.

## Tests
| # | Test | One-line outcome |
|---|---|---|
| 01 | [Spec math checks](tests/01-spec-math-checks.md) | Critiques correct; math repaired; says nothing about learning |
| 02 | [Toy M2c](tests/02-toy-m2c.md) | M2c ties ER (−0.3); more replay wins |
| 03 | [Why M2c tied](tests/03-m2c-diagnosis.md) | Features keep the knowledge; the readout is biased (~16:1 pull) |
| 04 | [Marathon (toy, MNIST)](tests/04-marathon.md) | Bias correction +16 to +20; stored means go stale |
| 05 | [Marathon revisions](tests/05-marathon-revisions.md) | Correction must be fitted; statistics accumulated at arrival |
| 06 | [CIFAR-10 E0/E1](tests/06-cifar10-e0-e1.md) | CIFAR is valid; two paths add +3.5 \* |
| 07 | [CIFAR-10 E2](tests/07-cifar10-e2.md) | Fully online two-path +4.0 \* |
| 08 | [CIFAR-10 round 2](tests/08-cifar10-round2.md) | Statistics beat images at small memory; M2c −12.8 \*; EWC null |
| 09 | [Sleep tests](tests/09-sleep-tests.md) | Sleep distillation −8.4 \*; loss is all from approximate transfer |
| 10 | [T-S5 CIFAR-100 growth](tests/10-ts5-growth-cifar100.md) | Spread version bank 31.2; "feature decay" reading later withdrawn |
| 11 | [Audit corrections A1–A3](tests/11-audit-corrections.md) | Features improve all 20 tasks (26 → 35.3); statistics are stranded |
| 12 | [X1 weight scale](tests/12-x1-readout-bias-scale.md) | Equal head norms −8.3 \*; LDA readout best |
| 13 | [X2 statistics transport](tests/13-x2-statistics-transport.md) | +5.6 \* offline; did not replicate in our runs |
| 14 | [X3 dormant wake](tests/14-x3-dormant-wake.md) | 84–98% recovery from keys; only k = 1 is cheaper than an encoder |
| 15 | [ARCH-v1 counterexamples](tests/15-arch-v1-audit-counterexamples.md) | Seven design claims false |
| 16 | [GREG Monte Carlo](tests/16-greg-monte-carlo.md) | Error shrinks by exactly (1 − R²) |
| 17 | [Notebook correctness checks](tests/17-atlas-notebook-unit-tests.md) | All pass; four design decisions changed before any real run |
| 18 | [Atlas CIFAR-100 v1 results](tests/18-atlas-v1-cifar100.md) | Per-class GREG +4.3 over global map; reserve bridge lost; random 35.1 > learned 25.1 |
| 19 | [Atlas v2b design](tests/19-atlas-v2b-notebook.md) | Built and checked; **not yet run** |

Across the tests: [insights](tests/20-cross-test-insights.md), [every bug and mistake](tests/21-bugs-and-errors.md), [how far to trust the tests](tests/22-test-reliability.md), [open questions](tests/23-open-questions.md).

## Conventions
- **\*** means the pre-registered rule: |mean difference| > 2·SE **and** all 5 seeds share the sign. It means "consistent", not "proven". Not significant ≠ tie.
- All results are small-scale: toy MLPs, MNIST MLPs, and a ~100k-parameter CNN on CIFAR with an online single pass. Nothing here is ResNet multi-epoch scale.
- Tests 01–14 come from the project's experiment log (ewt.md). Their code and raw results were not uploaded, so only the write-ups are here. Tests 15–19 are from the Atlas design work. Their notebooks and source code are in `notebooks/` and `src/`.

## Layout
```
tests/       one .md per test, plus cross-cutting files 20-23
notebooks/   atlas_v2_tests.ipynb (v1, results in test 18), atlas_v2b_tests.ipynb (v2b, not yet run)
src/v1/      sources embedded in the v1 notebook (numpy analysis, PyTorch learner, checks)
src/v2b/     sources embedded in the v2b notebook
results/     RESULTS-v1.md: full analysis of the v1 CIFAR-100 run
```
