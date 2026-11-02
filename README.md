# Test record

Every test from the fristonx design work, one file each: why it ran, what was done, the result, what we inferred, and why we think it happened.

Simulation results are copied from the spec by line range. All rates are assumed, and nothing here ran on a robot. "Why we think this happened" is an explanation, not a measurement.

| # | Test | Group |
|---|---|---|
| 001 | [Raw approval cost](tests/simulations/001-raw-approval-cost.md) | simulations |
| 002 | [Simulation-guided bets](tests/simulations/002-sim-guided-bets.md) | simulations |
| 003 | [Approval rule with no retained stratum](tests/simulations/003-approval-rule-no-stratum.md) | simulations |
| 004 | [Sampling and family effects](tests/simulations/004-sampling-and-family-effects.md) | simulations |
| 005 | [Probation against the pinned stack](tests/simulations/005-probation-pinned-stack.md) | simulations |
| 006 | [Margin creep](tests/simulations/006-margin-creep.md) | simulations |
| 007 | [Cost of proving superiority](tests/simulations/007-superiority-cost.md) | simulations |
| 008 | [Mastery certification](tests/simulations/008-mastery-certification.md) | simulations |
| 009 | [Measuring information](tests/simulations/009-measuring-information.md) | simulations |
| 010 | [Replication and promotion](tests/simulations/010-replication-promotion.md) | simulations |
| 011 | [Evidence overlap](tests/simulations/011-evidence-overlap.md) | simulations |
| 012 | [Verifier alarm](tests/simulations/012-verifier-alarm.md) | simulations |
| 013 | [Final approval rule](tests/simulations/013-final-approval-rule.md) | simulations |
| 014 | [Reference decay over 5 years](tests/simulations/014-reference-decay.md) | simulations |
| 015 | [Continuous floors](tests/simulations/015-continuous-lines.md) | simulations |
| 016 | [Working mastery, the earlier rule](tests/simulations/016-working-mastery-old-rule.md) | simulations |
| 017 | [Task loop and strategy table](tests/simulations/017-task-loop.md) | simulations |
| 018 | [Selective labels](tests/simulations/018-selective-labels.md) | simulations |
| 019 | [Noise detector and C2 gate](tests/simulations/019-noise-detector-c2-gate.md) | simulations |
| 020 | [Combined simulation: can it show old failures?](tests/simulations/020-combined-model-validity.md) | simulations |
| 021 | [Combined simulation: task outcomes](tests/simulations/021-combined-task-outcomes.md) | simulations |
| 022 | [Combined simulation: commits and rollbacks](tests/simulations/022-combined-commits.md) | simulations |
| 023 | [Combined simulation: shared causes](tests/simulations/023-combined-shared-causes.md) | simulations |
| 024 | [Mastery block and revocation choices](tests/simulations/024-mastery-block-choice.md) | simulations |
| 025 | [Below P_cal 0.65](tests/simulations/025-low-start-band.md) | simulations |
| 026 | [Mastery with a flat prior](tests/rule-runs/026-x1-x5-mastery-flat-prior.md) | rule-runs |
| 027 | [Revocation boundary](tests/rule-runs/027-x2b-revocation.md) | rule-runs |
| 028 | [Stuck rule](tests/rule-runs/028-x3-stuck-rule.md) | rule-runs |
| 029 | [Routine, family and severe event rules](tests/rule-runs/029-x4-x6-event-processes.md) | rule-runs |
| 030 | [Family map form](tests/rule-runs/030-x5-family-map-form.md) | rule-runs |
| 031 | [Staged sensor failures](tests/rule-runs/031-x4-staged-sensor-failures.md) | rule-runs |
| 032 | [Test event process](tests/rule-runs/032-x8-test-event-process.md) | rule-runs |
| 033 | [Probation wealth over a void](tests/rule-runs/033-x8-probation-void.md) | rule-runs |
| 034 | [Row weight](tests/rule-runs/034-x8-row-weight.md) | rule-runs |
| 035 | [Two-hypothesis closed form](tests/rule-runs/035-x7-two-hypothesis-closed-form.md) | rule-runs |
| 036 | [Failure-streak margin](tests/rule-runs/036-x9-failure-streak.md) | rule-runs |
| 037 | [Label readings and relabelling](tests/rule-runs/037-d13-label-readings.md) | rule-runs |
| 038 | [Reference health as one stream](tests/rule-runs/038-d12-reference-health.md) | rule-runs |
| 039 | [Block in a new context](tests/rule-runs/039-d12-block-new-context.md) | rule-runs |
| 040 | [Power of real model controls](tests/rule-runs/040-d12-model-controls-power.md) | rule-runs |
| 041 | [Gold reading error](tests/rule-runs/041-d12-gold-reading.md) | rule-runs |
| 042 | [C2 and family monitors](tests/rule-runs/042-d4-monitors.md) | rule-runs |
| 043 | [Drift check](tests/rule-runs/043-d11-drift.md) | rule-runs |
| 044 | [Arm-wise reading and doubling](tests/rule-runs/044-d8-d9-arm-reading.md) | rule-runs |
| 045 | [Rebuild of the combined simulation](tests/rule-runs/045-d10-repro.md) | rule-runs |
| 046 | [Hypotheses in pairs of draws](tests/rule-runs/046-d18-paired-draws.md) | rule-runs |
| 047 | [Verifier alarm period](tests/rule-runs/047-d15-verifier-period.md) | rule-runs |
| 048 | [Perception alarm](tests/rule-runs/048-d16-perception-alarm.md) | rule-runs |
| 049 | [Lapse skip](tests/rule-runs/049-d17-lapse-skip.md) | rule-runs |
| 050 | [Hostile read 1](tests/reviews/050-hostile-read-1.md) | reviews |
| 051 | [Hostile read 2](tests/reviews/051-hostile-read-2.md) | reviews |
| 052 | [Hostile read 3](tests/reviews/052-hostile-read-3.md) | reviews |
| 053 | [Hostile read 4](tests/reviews/053-hostile-read-4.md) | reviews |
| 054 | [Hostile read 5](tests/reviews/054-hostile-read-5.md) | reviews |
| 055 | [Hostile read 6](tests/reviews/055-hostile-read-6.md) | reviews |
| 056 | [Hostile read 7](tests/reviews/056-hostile-read-7.md) | reviews |
| 057 | [Hostile read 8](tests/reviews/057-hostile-read-8.md) | reviews |
| 058 | [Hostile read 9](tests/reviews/058-hostile-read-9.md) | reviews |
| 059 | [Hostile read 10](tests/reviews/059-hostile-read-10.md) | reviews |
| 060 | [Hostile read 11](tests/reviews/060-hostile-read-11.md) | reviews |
| 061 | [Hostile read 12](tests/reviews/061-hostile-read-12.md) | reviews |
| 062 | [Hostile read 13](tests/reviews/062-hostile-read-13.md) | reviews |
| 063 | [Hostile read 14](tests/reviews/063-hostile-read-14.md) | reviews |
| 064 | [Hostile read 15](tests/reviews/064-hostile-read-15.md) | reviews |
| 065 | [Audit rounds 1 to 3](tests/reviews/065-audit-rounds-1-to-3.md) | reviews |
| 066 | [Audit round 5](tests/reviews/066-audit-round-5.md) | reviews |
| 067 | [Audit round 7](tests/reviews/067-audit-round-7.md) | reviews |
| 068 | [Audit round 9](tests/reviews/068-audit-round-9.md) | reviews |
| 069 | [Recheck of all findings](tests/reviews/069-recheck-all-findings.md) | reviews |
| 070 | [Fresh full review](tests/reviews/070-fresh-review-round-11.md) | reviews |
| 071 | [Short check of the fix pass](tests/reviews/071-short-check-round-12.md) | reviews |
| 072 | [Check of the sweep decisions](tests/reviews/072-sweep-decisions-check.md) | reviews |
| 073 | [Math source checks](tests/reviews/073-math-source-checks.md) | reviews |
| 074 | [Math coverage check](tests/reviews/074-math-coverage.md) | reviews |
| 075 | [Stage 0 packet reproductions](tests/reviews/075-packet-reproduction.md) | reviews |
| 076 | [Task card coverage](tests/reviews/076-task-card-coverage.md) | reviews |
| 077 | [Stage 0 test suite](tests/reviews/077-stage0-test-suite.md) | reviews |
| 078 | [Builder mutation run](tests/reviews/078-builder-mutation-run.md) | reviews |
| 079 | [Independent check of Stage 0](tests/reviews/079-stage0-independent-check.md) | reviews |
