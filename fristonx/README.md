# fristonx: test record of a robot-learner design

fristonx is a design specification for an extreme sample-efficiency robot learner. Its main objective is to minimise real interactions per newly mastered task while keeping old tasks working. The specification itself says nothing in it has been built or run on hardware.

The specification and Stage 0 code live in a separate private repository, [`meta-lucky-builds/fristonx`](https://github.com/meta-lucky-builds/fristonx/tree/stage-0/foundation) (branch `stage-0/foundation`). This folder holds only the test record, one file per test: why it ran, what was done, the result, what we inferred, and why we think it happened.

Read these with care:
- **Citations point to the other repository.** Paths like `docs/spec/SPEC.md`, `audit/` and `scripts/` are there, and `tests/` in a record means that project's tests, not a folder here.
- **Coverage has gaps.** Spec sections 21.13 and 21.21 have no record here.
- **Many rule runs are pointers.** 029, 034, 036, 039, 041, 044, 045, 046, 047, 048 and 049 point to spec sections without giving result numbers.
- **Several simulation tables are stale.** Records 003, 016, 017 and 018 say so in place, because the rules changed after those runs.
- **Reviews 065-079 don't record their method.** They say "See the source named above".
- **Stage 0 is not a clean pass.** [077](reviews/077-stage0-test-suite.md) and [078](reviews/078-builder-mutation-run.md) report passing checks, but the later independent check [079](reviews/079-stage0-independent-check.md) found 6 Critical issues, and no record says they were fixed. The code stayed parked when the plan moved to LeRobot.

Simulation results are copied from the spec by line range. All rates are assumed, and nothing here ran on a robot. "Why we think this happened" is an explanation, not a measurement.

| # | Test | Group |
|---|---|---|
| 001 | [Raw approval cost](simulations/001-raw-approval-cost.md) | simulations |
| 002 | [Simulation-guided bets](simulations/002-sim-guided-bets.md) | simulations |
| 003 | [Approval rule with no retained stratum](simulations/003-approval-rule-no-stratum.md) | simulations |
| 004 | [Sampling and family effects](simulations/004-sampling-and-family-effects.md) | simulations |
| 005 | [Probation against the pinned stack](simulations/005-probation-pinned-stack.md) | simulations |
| 006 | [Margin creep](simulations/006-margin-creep.md) | simulations |
| 007 | [Cost of proving superiority](simulations/007-superiority-cost.md) | simulations |
| 008 | [Mastery certification](simulations/008-mastery-certification.md) | simulations |
| 009 | [Measuring information](simulations/009-measuring-information.md) | simulations |
| 010 | [Replication and promotion](simulations/010-replication-promotion.md) | simulations |
| 011 | [Evidence overlap](simulations/011-evidence-overlap.md) | simulations |
| 012 | [Verifier alarm](simulations/012-verifier-alarm.md) | simulations |
| 013 | [Final approval rule](simulations/013-final-approval-rule.md) | simulations |
| 014 | [Reference decay over 5 years](simulations/014-reference-decay.md) | simulations |
| 015 | [Continuous floors](simulations/015-continuous-lines.md) | simulations |
| 016 | [Working mastery, the earlier rule](simulations/016-working-mastery-old-rule.md) | simulations |
| 017 | [Task loop and strategy table](simulations/017-task-loop.md) | simulations |
| 018 | [Selective labels](simulations/018-selective-labels.md) | simulations |
| 019 | [Noise detector and C2 gate](simulations/019-noise-detector-c2-gate.md) | simulations |
| 020 | [Combined simulation: can it show old failures?](simulations/020-combined-model-validity.md) | simulations |
| 021 | [Combined simulation: task outcomes](simulations/021-combined-task-outcomes.md) | simulations |
| 022 | [Combined simulation: commits and rollbacks](simulations/022-combined-commits.md) | simulations |
| 023 | [Combined simulation: shared causes](simulations/023-combined-shared-causes.md) | simulations |
| 024 | [Mastery block and revocation choices](simulations/024-mastery-block-choice.md) | simulations |
| 025 | [Below P_cal 0.65](simulations/025-low-start-band.md) | simulations |
| 026 | [Mastery with a flat prior](rule-runs/026-x1-x5-mastery-flat-prior.md) | rule-runs |
| 027 | [Revocation boundary](rule-runs/027-x2b-revocation.md) | rule-runs |
| 028 | [Stuck rule](rule-runs/028-x3-stuck-rule.md) | rule-runs |
| 029 | [Routine, family and severe event rules](rule-runs/029-x4-x6-event-processes.md) | rule-runs |
| 030 | [Family map form](rule-runs/030-x5-family-map-form.md) | rule-runs |
| 031 | [Staged sensor failures](rule-runs/031-x4-staged-sensor-failures.md) | rule-runs |
| 032 | [Test event process](rule-runs/032-x8-test-event-process.md) | rule-runs |
| 033 | [Probation wealth over a void](rule-runs/033-x8-probation-void.md) | rule-runs |
| 034 | [Row weight](rule-runs/034-x8-row-weight.md) | rule-runs |
| 035 | [Two-hypothesis closed form](rule-runs/035-x7-two-hypothesis-closed-form.md) | rule-runs |
| 036 | [Failure-streak margin](rule-runs/036-x9-failure-streak.md) | rule-runs |
| 037 | [Label readings and relabelling](rule-runs/037-d13-label-readings.md) | rule-runs |
| 038 | [Reference health as one stream](rule-runs/038-d12-reference-health.md) | rule-runs |
| 039 | [Block in a new context](rule-runs/039-d12-block-new-context.md) | rule-runs |
| 040 | [Power of real model controls](rule-runs/040-d12-model-controls-power.md) | rule-runs |
| 041 | [Gold reading error](rule-runs/041-d12-gold-reading.md) | rule-runs |
| 042 | [C2 and family monitors](rule-runs/042-d4-monitors.md) | rule-runs |
| 043 | [Drift check](rule-runs/043-d11-drift.md) | rule-runs |
| 044 | [Arm-wise reading and doubling](rule-runs/044-d8-d9-arm-reading.md) | rule-runs |
| 045 | [Rebuild of the combined simulation](rule-runs/045-d10-repro.md) | rule-runs |
| 046 | [Hypotheses in pairs of draws](rule-runs/046-d18-paired-draws.md) | rule-runs |
| 047 | [Verifier alarm period](rule-runs/047-d15-verifier-period.md) | rule-runs |
| 048 | [Perception alarm](rule-runs/048-d16-perception-alarm.md) | rule-runs |
| 049 | [Lapse skip](rule-runs/049-d17-lapse-skip.md) | rule-runs |
| 050 | [Hostile read 1](reviews/050-hostile-read-1.md) | reviews |
| 051 | [Hostile read 2](reviews/051-hostile-read-2.md) | reviews |
| 052 | [Hostile read 3](reviews/052-hostile-read-3.md) | reviews |
| 053 | [Hostile read 4](reviews/053-hostile-read-4.md) | reviews |
| 054 | [Hostile read 5](reviews/054-hostile-read-5.md) | reviews |
| 055 | [Hostile read 6](reviews/055-hostile-read-6.md) | reviews |
| 056 | [Hostile read 7](reviews/056-hostile-read-7.md) | reviews |
| 057 | [Hostile read 8](reviews/057-hostile-read-8.md) | reviews |
| 058 | [Hostile read 9](reviews/058-hostile-read-9.md) | reviews |
| 059 | [Hostile read 10](reviews/059-hostile-read-10.md) | reviews |
| 060 | [Hostile read 11](reviews/060-hostile-read-11.md) | reviews |
| 061 | [Hostile read 12](reviews/061-hostile-read-12.md) | reviews |
| 062 | [Hostile read 13](reviews/062-hostile-read-13.md) | reviews |
| 063 | [Hostile read 14](reviews/063-hostile-read-14.md) | reviews |
| 064 | [Hostile read 15](reviews/064-hostile-read-15.md) | reviews |
| 065 | [Audit rounds 1 to 3](reviews/065-audit-rounds-1-to-3.md) | reviews |
| 066 | [Audit round 5](reviews/066-audit-round-5.md) | reviews |
| 067 | [Audit round 7](reviews/067-audit-round-7.md) | reviews |
| 068 | [Audit round 9](reviews/068-audit-round-9.md) | reviews |
| 069 | [Recheck of all findings](reviews/069-recheck-all-findings.md) | reviews |
| 070 | [Fresh full review](reviews/070-fresh-review-round-11.md) | reviews |
| 071 | [Short check of the fix pass](reviews/071-short-check-round-12.md) | reviews |
| 072 | [Check of the sweep decisions](reviews/072-sweep-decisions-check.md) | reviews |
| 073 | [Math source checks](reviews/073-math-source-checks.md) | reviews |
| 074 | [Math coverage check](reviews/074-math-coverage.md) | reviews |
| 075 | [Stage 0 packet reproductions](reviews/075-packet-reproduction.md) | reviews |
| 076 | [Task card coverage](reviews/076-task-card-coverage.md) | reviews |
| 077 | [Stage 0 test suite](reviews/077-stage0-test-suite.md) | reviews |
| 078 | [Builder mutation run](reviews/078-builder-mutation-run.md) | reviews |
| 079 | [Independent check of Stage 0](reviews/079-stage0-independent-check.md) | reviews |
