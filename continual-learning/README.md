# Continual learning: why networks forget, and what fixes it

This folder is the record of the continual-learning project up to 2026-11-02. It holds two write-ups of the same work, plus the raw evidence.

| Path | What it is |
|---|---|
| [`overview/`](overview/) | Short record, numbered 01-23. Includes the Atlas tests and the synthesis files: [cross-test insights](overview/20-cross-test-insights.md), [bugs and errors](overview/21-bugs-and-errors.md), [test reliability](overview/22-test-reliability.md), [open questions](overview/23-open-questions.md). **Start here.** |
| [`tests/`](tests/) | Detailed record, numbered 01-40, one file per sub-test. Each has a "How reliable this test is" section. Ends before Atlas. |
| [`atlas/`](atlas/README.md) | Code, notebooks and results for the Atlas tests (overview 15-19). |
| [`source/notes/`](source/notes/) | Original code, plans, logs and small result files. Left as they were, so scripts still name the original `/mnt/...` paths. |
| [`RELIABILITY.md`](RELIABILITY.md), [`BUGS.md`](BUGS.md) | Trust assessment and bug list as of the end of the detailed record. `overview/21` and `overview/22` are the later versions (37 rows against 26). |

The two numberings are separate, so "test 23" means different things in each:

| Overview | Detailed tests | Topic |
|---|---|---|
| 01 | 01-02 | Spec math checks |
| 02-03 | 03-04 | Toy M2c and diagnosis |
| 04-05 | 05-21 | Marathon and its revisions |
| 06-07 | 22-24 | CIFAR-10 E0, E1, E2 |
| 08 | 25-29 | CIFAR-10 round 2 |
| 09 | 30-33 | Sleep tests |
| 10-11 | 34-37 | T-S5 and audit A1-A3 |
| 12-14 | 38-40 | X1, X2, X3 |
| 15-19 | none | ARCH-v1 audit, GREG, Atlas |

**Later correction.** The fresh Atlas runs did not replicate X2's +5.6 (they gave -2.07, 0/5), and random features beat learned ones with that weaker learner. The detailed record below predates this, so read item 6 of "What we know now" together with [overview/18](overview/18-atlas-v1-cifar100.md).

**Reproducing.** Needs numpy, scipy and torch. The data is not included (CIFAR expects `train.npz` and `test.npz` with keys `X` and `y`), and the run scripts name the original paths. X1, X2 and X3 need the T-S5 encoders regenerated first. The headline numbers were traced to saved results in `source/notes/`, but the training runs have not been re-executed in this consolidation. T-S5 outputs were regenerated once earlier, after files vanished, and reproduced earlier numbers.

---

Each detailed test records:
- why each test was run;
- what was done;
- what came out;
- what we inferred;
- why we think it happened;
- how reliable the test itself is.

**Goal of the project (Lucky).** Continual learning means a system that keeps learning and **getting better**, **consolidates** what it learned, and keeps old knowledge in **low-activation dormant states that reactivate when the situation calls for it**, with a policy for what to forget. It does not mean remembering everything.

## Read these first

- [RELIABILITY.md](RELIABILITY.md) explains how far to trust the tests themselves, which is a separate question from whether the numbers are right.
  - The numbers are reliable: exact reproduction gates, self-tests and independent recomputation all agree.
  - The test designs are only moderately reliable. Every independently reviewed experiment had a design flaw, and some conclusions changed after outside checking.
- [BUGS.md](BUGS.md) lists every bug, mistake and process error, mine included.
- `source/notes/ewt.md` is the running write-up the test files are drawn from.

## Conventions

- **Star (\*)** means |mean| > 2·SE across 5 seeds **and** all 5 seeds share the sign. That is sign-test p = 1/32 one-sided. It means "consistent", not "proven". A non-significant result is **not** a tie.
- **Withdrawn** claims are kept and struck through or marked, not deleted.
- **Scale.** Everything ran on CPU only:
  - toy: 20-d Gaussian classes, MLP;
  - Split-MNIST: MLP 784-256-256-10;
  - Split CIFAR-10 and CIFAR-100: a 3-block, ~98k-parameter GroupNorm CNN, online single pass.
  
  Nothing here is ResNet-scale, multi-epoch or language-model scale.
- **ER** is experience replay. **bc** is bias correction (subtract each class's mean logit, estimated on held-out or buffer data). **RC-LDA** is a fixed random-conv encoder with streaming class statistics, read out by linear discriminant analysis. **M** is the replay buffer size.

## What we know now (short)

1. **Forgetting in class-incremental streams is largely an output-layer bias toward the newest classes, not lost knowledge.** Bias correction recovers +5 to +20 points on the same weights: tests 04, 05, 23 and 33.
2. **The size of that bias must be fitted on held-out data.** No formula we tried sets it: tests 06, 19 and 24.
3. **A stable statistics path mixed with the plastic network beats either alone.** CIFAR-10: +4.0 \* fully online. Tests 23–25 and 29.
4. **Protecting parameters lost or did nothing at our scale.** This covers M2c, EWC, head snapshots, Sleep-style distillation, and late frozen copies: tests 03, 10, 14, 27, 28 and 30.
5. **On CIFAR-100 the learned features keep improving.** What fails is class statistics stranded in old encoders' coordinates: test 35.
6. **Stranded statistics can be carried forward online by a simple ridge map.** This closes 54% of the gap, and in a mixture it beats a version bank with fewer bytes: test 39.
7. **Dormant statistics can be woken by a few key items.** 84–98% of the old readout is recovered, but this is not yet cheaper than keeping the encoder: test 40.
8. **Anything fitted on a set looks better on that set.** This held in every instance (tests 02, 21, 31, 38 and others), so every claim is judged on held-out data.

**Main open problem.** On CIFAR-10, joint training's features reach 75.5. The best online system reaches about 59–61.

## Index of tests

| # | Test | Phase | One-line result |
|---|---|---|---|
| [01](tests/01-spec-v12-math-checks.md) | Spec v1.2 math checks | spec | All 16 critique points correct; geometry rebuilt |
| [02](tests/02-spec-v13-math-checks.md) | Spec v1.3 checks and N1–N6 | spec | Audit counterexamples reproduced; my "constraints dominate" was in-sample |
| [03](tests/03-toy-m2c.md) | Toy test of M2c | toy | M2c ties ER (−0.3); compute-matched replay +3.4 |
| [04](tests/04-m2c-diagnosis.md) | Why M2c tied | toy | Features survive (refit 70.6 vs joint 73.5); the readout fails |
| [05](tests/05-marathon-readout-bias.md) | Readout bias (T2/F4) | marathon | Bias correction +16 toy, +20 MNIST; a common-mode offset |
| [06](tests/06-marathon-logit-adjustment.md) | Logit adjustment by the prior | marathon | Direction right, size wrong (revised by 19) |
| [07](tests/07-marathon-floor-control.md) | Floor control W3 | marathon | Random ReLU 94.0 beats trained ≤ 72 with full data (revised by 20) |
| [08](tests/08-marathon-readout-sample-size.md) | Readout sample size F8/W1 | marathon | The gap is in the class means |
| [09](tests/09-marathon-stale-statistics.md) | Stale statistics F7/W2 | marathon | Stale means ≈ 20%; drift class-specific on MNIST |
| [10](tests/10-marathon-snapshot-restore.md) | Head snapshots T5 | marathon | "Savings" were a readout artefact |
| [11](tests/11-marathon-repetition.md) | Repetition W6 | marathon | Small real savings on recurrence |
| [12](tests/12-marathon-boundary-shock.md) | Boundary shock T1 | marathon | Not what drives forgetting; killed |
| [13](tests/13-marathon-fresh-vs-stale-fisher.md) | Fresh vs stale Fisher T4 | marathon | Fresh better only at small λ |
| [14](tests/14-marathon-m2c-rival-coverage.md) | M2c rival rule T6 | marathon | +12.5 over ER but below free readout fixes at ~30× compute |
| [15](tests/15-marathon-context-detection.md) | Context detection T7 | marathon | L_min not necessary under noisy evidence |
| [16](tests/16-marathon-interference-prediction.md) | Predicting interference W7 | marathon | Raw inputs predict as well as features |
| [17](tests/17-marathon-end-sleep.md) | End-of-stream sleep R13 | marathon | ≈ bias correction |
| [18](tests/18-marathon-centring-head.md) | Centring the head | marathon | Failed |
| [19](tests/19-marathon-R1-bias-size-not-closed-form.md) | R1: bias size | marathon revision | Not closed form; must be fitted |
| [20](tests/20-marathon-R2-sample-efficiency.md) | R2: sample efficiency | marathon revision | Learned features win by 9–18 at 5/class |
| [21](tests/21-marathon-R3-buffer-is-biased.md) | R3: buffer bias | marathon revision | Buffer LDA 49 vs fresh 74 on MNIST |
| [22](tests/22-cifar10-E0-valid-benchmark.md) | CIFAR-10 E0 | CIFAR-10 | Learned beats random by 17–19; valid benchmark |
| [23](tests/23-cifar10-E1-two-paths.md) | CIFAR-10 E1: two paths | CIFAR-10 | MIX +3.5 \*; Bayes stream mixture = ER |
| [24](tests/24-cifar10-E2-streaming-choices.md) | CIFAR-10 E2: online choices | CIFAR-10 | Fully online two-path +4.0 \* |
| [25](tests/25-cifar10-R2-memory.md) | Round 2: memory | CIFAR-10 | Statistics beat images at small budgets only |
| [26](tests/26-cifar10-R2-promotion.md) | Round 2: promotion | CIFAR-10 | Version bank +2.7 at more memory |
| [27](tests/27-cifar10-R2-ewc.md) | Round 2: EWC | CIFAR-10 | Null |
| [28](tests/28-cifar10-R2-m2c.md) | Round 2: M2c | CIFAR-10 | −12.8 \* (faithful); removed |
| [29](tests/29-cifar10-R2-epochs.md) | Round 2: 3 epochs | CIFAR-10 | Two-path gain survives |
| [30](tests/30-sleep-T1-consolidation.md) | Sleep T1 | sleep | −8.4 \* |
| [31](tests/31-sleep-T2-heldout-gap.md) | Sleep T2: held-out gap | sleep | 6–9 points lost off the dream set |
| [32](tests/32-sleep-T3-dream-content.md) | Sleep T3: dream content | sleep | Noise −30.6; mixup null |
| [33](tests/33-sleep-T4-width.md) | Sleep T4: width | sleep | Bias correction +10.1 over 4× width |
| [34](tests/34-ts5-growth-budget.md) | T-S5 growth budget | CIFAR-100 | 4 spread versions ≈ all 19; "feature decay" withdrawn |
| [35](tests/35-audit-A1-matched-probes.md) | A1: matched probes | audit | Features improve monotonically; statistics are stranded |
| [36](tests/36-audit-A2-bank-vs-buffer.md) | A2: bank vs buffer | audit | Bank +1.3 \* at its own budget |
| [37](tests/37-audit-A3-exact-transfer-sleep.md) | A3: exact-transfer sleep | audit | All of Sleep's loss is approximate transfer |
| [38](tests/38-X1-is-the-bias-weight-scale.md) | X1: weight scale | X-series | Equal norms −8.3 \*; scale is not the bias |
| [39](tests/39-X2-carrying-statistics-forward.md) | X2: transport statistics | X-series | +5.6 \*, 54% of the gap; beats the bank with fewer bytes |
| [40](tests/40-X3-waking-dormant-statistics.md) | X3: dormant reactivation | X-series | 84–98% recovered; not cheaper except at 1 key |

## What is not in this repository

- Datasets (CIFAR, MNIST) and large binary outputs (`.npz` run files, `.pt` encoder checkpoints, files over 1 MB). The scripts regenerate them.
- The literature feeds (geometry, memory-and-drift) and the paper library. These are reading, not tests, and live in the project's `feed/` folder.
- Superseded and invalid runs are included under `source/notes/marathon/results/*superseded*` so the record is complete. They are not used by any test file.
