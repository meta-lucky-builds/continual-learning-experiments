# Continual learning experiments: testing ideas about learning and memory until they break

A research record of five related projects on one question: **how can a learning system keep improving without losing what it knows?** It covers continual learning, memory, world models and robot learning, and holds the tests, the raw results where they exist, and the failures.

Most ideas tested here lost. The strongest results explain why, and each finding links to the files its numbers come from.

> **Scale.** Everything is small: toys, Split-MNIST on an MLP and Split CIFAR on small CNNs, mostly on CPU. Nothing has run on a robot. Treat these as findings at this scale, not general results.

## Projects

| Folder | Question | Record | Status |
|---|---|---|---|
| [`continual-learning/`](continual-learning/README.md) | Why do networks forget in class-incremental streams, and what fixes it? | 40 detailed tests, plus Atlas tests, code and raw results | Findings hold at small-CNN scale. Atlas v2b is designed but not run |
| [`lifelong-robot/`](lifelong-robot/README.md) | Can new mechanisms beat existing stacks for a lifelong robot learner? | 10 tests, 3 of them pilots | No clean win against the strongest competitor |
| [`experiments/`](experiments/README.md) | Three focused questions on world-model geometry and memory | 3 tests, with code and per-seed results | One mechanism killed, one idea mostly killed, one confounded win |
| [`fristonx/`](fristonx/README.md) | Which approval and mastery rules can a robot learner trust when real episodes are scarce? | 25 simulations, 24 rule runs, 30 reviews | Design record. All rates are assumed, and the Stage 0 code is parked |
| [`hjepa/`](hjepa/README.md) | Does test-time training keep H-JEPA planning well when dynamics change? | 5 tests, 6 decisions, literature reads | Baseline reproduced on one seed. The key benchmark has not been run |

## Top findings

Each finding is tagged with how strong its evidence is:
- **Measured**: repeated seeds, paired comparisons.
- **Pre-registered**: rules fixed before the treatment ran.
- **Pilot**: few seeds, untuned; good for direction, not verdicts.
- **Exploratory**: noticed after the fact.
- **Simulated**: Monte Carlo with assumed rates.

A star (\*) means |mean| > 2·SE across 5 seeds and all 5 seeds share the sign. It means "consistent", not "proven".

### 1. Forgetting here is mostly lost access, not lost knowledge
*Measured, strong at this scale.*

- Refitting only the last layer on replay's frozen features gives **70.6%**, against 41.5% for the network as trained and 73.5% for joint training ([toy](continual-learning/overview/03-m2c-diagnosis.md)).
- On CIFAR-100, the encoder frozen at the first task scores 26.0 and the final encoder 35.3 when every encoder gets the same 480 items per class. Matched accuracy rose at every probed version, though the gain on task-0 classes alone was not significant ([A1](continual-learning/overview/11-audit-corrections.md)).
- An earlier reading that features "decay" was wrong and was withdrawn. Readouts had been fitted on 21 to 480 items per class. An outside audit caught it.
- On CIFAR-10, bias correction is worth +10.1\* more than making the network 4× wider ([T4](continual-learning/overview/09-sleep-tests.md)).

### 2. Reading around drift beat constraining the learner
*Measured. Each row has its own baseline; see the linked file.*

| Approach | Result | Source |
|---|---|---|
| M2c margin protection (faithful version) | **−12.8\*** | [Round 2](continual-learning/overview/08-cifar10-round2.md) |
| EWC, λ = 10 | +0.7 and +0.1, null | [Round 2](continual-learning/overview/08-cifar10-round2.md) |
| Sleep-style self-distillation | **−8.4\*** | [Sleep](continual-learning/overview/09-sleep-tests.md) |
| Equalising head norms after training (τ-norm 1, CIFAR-100, no bias correction) | **−8.3\*** against the plain head | [X1](continual-learning/overview/12-x1-readout-bias-scale.md) |
| Plastic network + stable statistics, fully online | **+4.0\*** over the better single expert | [E2](continual-learning/overview/07-cifar10-e2.md) |

A control isolates why Sleep lost. An exact teacher copy plus the same reset scores 52.2, against 52.4 for replay ([A3](continual-learning/overview/11-audit-corrections.md)). So all of the loss comes from approximate transfer, not from sleeping.

### 3. A fresh re-implementation flipped two conclusions
*Measured, single replication.*

- **Transport did not replicate.** Carrying class statistics into new encoders gained +5.63\* in the original runs ([X2](continual-learning/overview/13-x2-statistics-transport.md)) and −2.07 (0/5) in fresh runs with a weaker learner ([Atlas](continual-learning/overview/18-atlas-v1-cifar100.md)).
- **Learned against random features reversed.** The original runs gave learned 35.3, random 28.2. The fresh runs gave learned 25.1, random 35.1. Two things changed: a weaker learner (head 8.1 vs 17.4) and a stronger random chart.
- **What the fresh runs showed:** per-class regression beat a global map (+2.76, 5/5) on the same bridge items. Encoder drift is close to linear between neighbouring tasks (R² 0.955) but not across 19 tasks (0.675).

The weaker learner is the likely cause of the failed replication, but that is not yet tested: the [v2b notebook](continual-learning/overview/19-atlas-v2b-notebook.md) is built to test it and has not been run.

### 4. A recurrent memory wall broke when the state got a reader, wherever the credit arrived
*Pre-registered verdict, with an exploratory explanation. One toy setting.*

On the source paper's exact code ([experiment 03](experiments/03-idea-b-write-credit/RESULTS.md)), lock-in counts over 10 seeds were:

| Arm | Lock-in |
|---|---|
| Dense supervision | 1/10 |
| Dense + gap curriculum | 6/10 |
| Dense + state probe at every step | **10/10** |
| Dense + state probe at the last step only | **9/10** |

The hypothesis was that the wall is a delayed-credit problem. The every-step and last-step probes can't be separated on lock-in (p = 0.50), so **the pre-registered verdict killed that mechanism**. The probe adds no new information, only a path. The working explanation is read-write bootstrapping: writes can't be shaped until a reader exists, and the reader can't learn until writes are clean. That explanation has not been tested on its own.

### 5. Fast and slow memories do nothing without selection
*Measured, 5 seeds, toy.*

With shared addresses and an additive read, copying every fast slot at sleep is **mathematically one memory**, and the code difference is 0.0 ([experiment 02](experiments/02-sleep-consolidation/RESULTS.md)). A plain counter on a single memory ties the oracle (0.935 at 40 distractors), against 0.546 for the single memory without it. Soft slot copy earns a real but narrow niche: on noisy repeats, with no item identity, it matches methods given identity (0.827 vs 0.824).

### 6. No lifelong-robot idea beat the strongest competitor
*Pre-registered main runs and pilots. CPU toys only.*

| Test | Result | Bar | Outcome |
|---|---|---|---|
| [T-COMPOSE](lifelong-robot/03-T-COMPOSE.md) | 1.298× COIN's error | ≤ 0.25× | Failed |
| [T-P14](lifelong-robot/05-T-P14-practice-prior.md) | 1.302× the frames with the learned prior on | Faster with it on | Failed |
| [T-DORMANT](lifelong-robot/10-T-DORMANT.md) | 12 wins, 11 losses, 1 tie | Beat every comparator, pass controls | Killed |
| [T-CAUSE](lifelong-robot/07-T-CAUSE.md) (pilot) | C_pub 2.3–3.7× worse than cxl at 200 frames, in the decisive worlds | Bars not applicable to a pilot; would fail both | No verdict |

One caveat: Test P's prior passed its rule against the competitor named for that test, but that competitor was slower than an isotropic learner with no practice ([Test P](lifelong-robot/02-test-P-practice.md)). The principles often held while the machinery lost. Never forgetting on a clock is sound, for example, but plain methods get it for free ([lessons](lifelong-robot/12-lessons-across-tests.md)).

### 7. Designer-built toys flatter the designer
*Checked by audit, at toy level.*

All four cluster designers' toys overstated their ideas ([file 06](lifelong-robot/06-cluster-designer-self-tests.md)). One advantage of −0.19 shrank to −0.106, −0.089 or −0.023 under audit, and a pooled learner with a fast offset tied that design overall. Another design was slower than an audit-built no-practice learner with the same structure in 20 of 30 cells (equal in 7, faster in 3). In lifelong-robot, decisive tests are now built by someone other than the designer, and competitors get the same knowledge.

### 8. A false-write guarantee does not stop writes under a wrong model
*Pre-registered main run, toy.*

When the reference equals the truth, evidence-gated writes stay within their bound (22 crossings against a bound of 582.54). Under misfit, commits in the mixed world kept growing almost linearly, with exponent **1.06 [1.00, 1.13]**. The false-write level λ is not a control knob for this: a hundredfold change in λ only moved commits from 517 to 1100 per run ([Test M](lifelong-robot/01-test-M-misfit-writes.md)).

### 9. Structured dynamics won a world built to suit them
*Measured, 3 seeds, confounded.*

Phase-space dynamics beat a free GRU on planning (0.87 vs 0.34) and on 20-step rollout error (0.44 vs 8.18, with one GRU seed diverging to 21.6). A quasimetric head only helped on top of good dynamics: on the GRU model, plain L1 did better (0.53 vs 0.34) ([experiment 01](experiments/01-world-model-geometry/RESULTS.md)). The toy world is a damped point mass, which matches the winner's prior exactly, and the two models differ in three ways at once.

### 10. Small statistical choices decide whether approval rules can be trusted
*Simulated, assumed rates.*

- **Compare against a fixed reference, not the parent.** With 15 candidates each losing 0.02, mean final loss was 0.070 against the parent (10.3% of runs past the margin) and 0.036 against a fixed reference (0.0%) ([006](fristonx/simulations/006-margin-creep.md)).
- **Draw pairs independently.** False approval was 0.960 with an ordered stream and 0.008 with independent draws ([004](fristonx/simulations/004-sampling-and-family-effects.md)).
- **Review (not simulated).** Fifteen hostile reads of the spec logged 599 findings (our sum across [the records](fristonx/README.md)), falling from about 50 per read to 12–15, but later reads looked only at changes, and a later check still said ["no freeze yet"](fristonx/reviews/071-short-check-round-12.md). Separately, a partial check of the Stage 0 code found 6 Critical issues, with 175 of 202 mutants killed ([079](fristonx/reviews/079-stage0-independent-check.md)). Those findings were not reproduced.

### 11. H-JEPA baseline reproduced on one seed, and it shows why a shift test is needed
*Measured once.*

FourRoom three-level H-JEPA reached **48/50 = 96.0%** (Wilson 86.5–98.9) against the paper's 96.0 ± 1.1 ([test 03](hjepa/tests/03-fourroom-l3-seed42-reproduction.md)). At 96% there is no headroom to show an improvement. Only 48% of tasks were solved by step 180, so the step budget hides weakness. The "before" benchmark under shifted dynamics is built and validated on CPU but [has not been run](hjepa/tests/05-kaggle-before-benchmark.md), so **this project has no result on its main question yet**.

## What the projects say together

These are readings across projects, not results of any single test.

1. **Stored is not the same as reachable.** Class-incremental forgetting is a readout failure (finding 1). The memory wall yields to a reader (finding 4). Dormant statistics come back from 1–5 key items per class, recovering 84–95% of the old readout on a task-aware measure ([X3](continual-learning/overview/14-x3-dormant-wake.md)). Three separate lines point at access rather than storage.
2. **The headline mechanism is rarely what matters.** Selection beat hierarchy. A reader beat locality. Readout fixes beat protection. Simple methods keep tying clever ones once they get the same knowledge.
3. **Anything judged on what it was fitted to looks better than it is.** This covers buffer accuracy, dream agreement ([Sleep](continual-learning/overview/09-sleep-tests.md)), in-sample constraints, and designers' own toys. It recurs throughout the record ([insights](continual-learning/overview/20-cross-test-insights.md)).
4. **Self-tests check code against design, not design against the question.** Arithmetic held up well. In continual learning, both experiments that got full independent review still had design-level flaws, and one conclusion changed ([reliability](continual-learning/overview/22-test-reliability.md)).

## What has not been shown

- Nothing has run on a robot, on RoboJEPA, at ResNet multi-epoch scale, or on a public lifelong benchmark ([never tested](lifelong-robot/11-never-tested.md)).
- The continual-learning harness has not reproduced any published number. Experiment 03's controls are consistent with its source paper, and H-JEPA's baseline matches on one seed.
- The star rule is weak and applied many times, so a few starred results are expected by chance.
- Atlas v1 numbers exist only in its write-up. No raw results file was committed.
- Lifelong-robot and fristonx cite files outside this repository, so their numbers can only be checked against these records here.

## Most informative next tests

| Test | What it would settle | State |
|---|---|---|
| "Machine Sperling" linear probe on frozen walled models | Whether the memory wall is a write failure or an access failure | Proposed |
| [Atlas v2b](continual-learning/overview/19-atlas-v2b-notebook.md) | Whether the learner explains finding 3, and whether buffer bias is real on CIFAR-100 | Notebook ready, needs a Colab GPU |
| [H-JEPA "before" benchmark](hjepa/tests/05-kaggle-before-benchmark.md) | Whether shifted dynamics break the frozen model, with predictions written down in advance | Notebook ready, needs Kaggle |
| [T-CAUSE and T-P14D main runs](lifelong-robot/11-never-tested.md) | Whether pilot directions hold with tuned arms and full seeds | Built, needs a rented machine (about $10) |
| [T-DORMANT under capacity pressure](lifelong-robot/11-never-tested.md) | The only real chance left for evidence-driven forgetting | Proposed |

## Repository map

```text
continual-learning/   overview/ (short record), tests/ (detailed), atlas/, source/notes/ (raw evidence)
lifelong-robot/       10 test write-ups, lessons, and everything never tested
experiments/          01 world-model geometry, 02 sleep consolidation, 03 write credit
fristonx/             simulations/, rule-runs/, reviews/
hjepa/                tests/, decisions/, analysis/, code/
scripts/check.py      link check, Atlas self-tests, experiment 03 statistics
```

## Check it yourself

```bash
pip install -r requirements.txt
python scripts/check.py
```

`check.py` needs no GPU, data or torch. It checks every relative markdown link, runs the Atlas self-tests on data with known answers, and recomputes experiment 03's lock-in counts and p-values from its per-seed results.

Training runs need torch, and some need data that isn't stored here. Each project's README has the details.

## Audit of this record (2026-11-04)

When this record was consolidated, the claims above were checked against the repository.

**Rechecked:**
- Atlas self-tests pass in both `v1` and `v2b`, and both notebooks rebuild from `build.py` with identical content (with `PYTHONUTF8=1`; on Windows the only difference is line endings).
- Experiment 03's lock-ins, validity checks and p-values recompute from its JSON files. The tables for experiments 01 and 02 match their JSON files.
- Continual-learning headline numbers trace to saved results in `continual-learning/source/notes/`.

**Corrected in the files:**
- Experiment 03's layer-1 probe count was 9/10, not 8/10.
- M2c logit ranges now match their source.
- Summary wording in `lifelong-robot/` that rounded in the project's favour is fixed. For example, "tuned existing stack" described untuned pilot arms.

**Not rechecked:**
- Training runs were not re-executed, apart from the rerun noted in [`experiments/README.md`](experiments/README.md).
- Claims about external papers were not checked against those papers.

## License

[MIT](LICENSE). `experiments/03-idea-b-write-credit/hay.py` copies MIT-licensed code from [JIBSIL/dualgoose](https://github.com/JIBSIL/dualgoose) (© 2026 Julian Boesch and Andrew Wee), as stated in its header.
