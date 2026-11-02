# 11 What was designed or planned but never tested

This file lists everything that was built, designed or planned as a test but **never produced a result**. It is here so nobody reads the absence of a failure as a pass.

## 1. The v2 test suite (tests 0 and A–I), planned 2026-10-15

**What it was.** Ten tests with about 27 pass/fail checks for the v2 architecture. Source: research/architecture/tests-explained.md and research/architecture/defense/MERGED-v2.md.

| Test | What it would have checked |
|---|---|
| 0 | exactness |
| A | attribution (changed vs new vs decayed vs slip) |
| B | strongest simple baseline |
| C | two-sided stability |
| D | consolidation |
| E | learning faster |
| F | meta level |
| G | acting |
| H | composition |
| I | perception |

**Why it never ran.** The v2 design was replaced, first by v3.x and then by the v3.6 core, before any code was written. The next tests to actually run were Test M and Test P (files 01 and 02). [checked: memory notes first-principles-architecture-draft.md and -v31.md]

**What we lost.** Test B ("does it beat one predictor with replay and test-time training?") was the right first question. It was asked again in every later round, and every later answer was "no".

## 2. T-SPAN: the only test planned on real RoboJEPA with a GPU

**What it was.** Proposed in the core-v4 audit (research/architecture/core-v4/AUDIT/TESTS.md). It would check whether changes in the real world live in a small code space of a frozen RoboJEPA. The estimate was about 2 GPU-days on RoboJEPA 1B plus real condition data.

**Why it never ran.** Lucky chose CPU tests first on 2026-10-18. Since then, no GPU has been rented and no RoboJEPA checkpoint has been run. [checked: memory]

**Consequence.** Every result in this repository comes from small numpy or PyTorch toys on 4 CPUs. Nothing has been measured on the actual robot model.

## 3. Main (tuned, full-seed) runs of T-CAUSE and T-P14D

**What it was.** Both tests were built, pre-registered and piloted on 2026-10-19 to 21:
- T-CAUSE: 143 core-hours at 24 seeds;
- T-P14D: about 90 core-hours at 48 seeds;
- an optional cxl ablation block: +170 core-hours.

The plan was about 1.5 hours on a rented 192-vCPU machine, costing about $10. [checked: memory tests-v6-status.md; research/tests-v6/cause/estimate.md; research/tests-v6/p14d/estimate.md]

**Why it never ran.** How to reach the machine (Remote Control, or Lucky running one command) was never decided, and the machine was never rented.

**Consequence.** Files 07, 08 and 09 report **pilot** numbers only: 1–8 seeds, untuned default configurations. Pilots are for debugging and timing, not for verdicts.

## 4. ARCH-v8 and ARCH-v8.1 (the adopted-stack architecture)

**What it was.** research/architecture/arch-v8/ARCH.md and ARCH-v8.1.md:
- a frozen RoboJEPA;
- add-on records that are never overwritten (context edits, exact body statistics, an episode vault, verified habit shortcuts, an offline replay adapter);
- a fixed-share mixture that always includes a fresh learner;
- one forecast auditor;
- M10, invented "motor forms" for P14;
- M11, within-episode fast weights.

**Why it was never tested.** It was written after the tests in files 07–09. It was meant to be checked by the T-CAUSE and T-P14D main runs, which never happened.

The architect's own odds [checked: memory MEMORY.md]:

| Bet | Chance it works |
|---|---|
| Fresh-learner blend keeps its gain | 50% |
| The auditor works | 45% |
| Real P14 from M10 | 15% |
| Beating the LIBERO-Long leader | 15% |

**Known untested flaw.** v8.1's exact running statistics stay valid only while the features are frozen. Its slow replay adapter moves the features, so the statistics may go stale. This was flagged during the CL sweep v2 reading and never checked.

## 5. A capacity-pressure re-run of T-DORMANT

**What it was.** The T-DORMANT world (file 10) had no memory pressure: 12 active keys fit easily in about 4200 floats. Under those conditions, "never forget" cannot lose. The re-run would use a tight budget, or unrelated skills competing for the same state.

**Status.** Offered to Lucky on 2026-10-23. Not answered, not run.

## 6. Re-runs proposed after the core-v4 audit

research/toy-tests-v4/VERIFY.md proposed two re-runs. Neither was run.

- **A valid false-alarm control.** World d with ν=4 added to the model's grid, 20 seeds. A one-seed check showed 0 false writes.
- **The interacting arm with the missing ψ scratch component.**

## 7. Public benchmarks

**What it was.** research/benchmarks-v1/SOTA-MAP.md mapped the candidates: LIBERO-Lifelong, Continual World, RoboLab, RoboCasa365, RoboMME and others. RoboLab was recommended as the first standard target.

**Status.** Choosing between RoboLab and RoboCasa365 was left to Lucky. No benchmark run exists.

## 8. The flywheel "ΔW-code context" experiment (proposed 2026-10-28)

**What it was.** An outside proposal, fact-checked in research/literature-v6/FLYWHEEL-FACTCHECK-2026-10-28.md, to identify contexts from how the weights change. It is the one piece not already covered by T-CAUSE and T-DORMANT.

**Status.** Not built.

---

**Summary.** The project's largest untested items are:
- anything on RoboJEPA or a GPU;
- tuned main runs of the two independent tests;
- the v8.1 architecture;
- any public benchmark.

All results in files 01–10 are CPU toys, and three of them are pilots.
