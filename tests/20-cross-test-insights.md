# 20. What the tests say together

Each insight is labelled with how strongly the data supports it. Numbers link back to the test files.

**I1. Forgetting here is mostly an access problem, not a storage problem.** [strong]
- Refitting only the last layer recovers most accuracy on the toy (70.6 vs joint 73.5, [03](03-m2c-diagnosis.md)).
- Bias correction gains +5 to +20 in every setting ([04](04-marathon.md), [06](06-cifar10-e0-e1.md), [10](10-ts5-growth-cifar100.md)).
- Matched probes show CIFAR-100 features improving for all 20 tasks (26.0 → 35.3, [11](11-audit-corrections.md)).
- What decays is class statistics stranded in old encoder coordinates.

**I2. Anything judged on the items it was fitted to looks better than it is.** [strong, many instances]
- In-sample "constraints dominate" ([01](01-spec-math-checks.md)).
- Buffer LDA 49 vs fresh 74 on MNIST ([05](05-marathon-revisions.md)).
- Head-snapshot savings that vanish under bias correction ([04](04-marathon.md)).
- The stream-loss Bayes mixture choosing the current-task expert ([06](06-cifar10-e0-e1.md)).
- Sleep distillation agreeing on its dreams but not held out ([09](09-sleep-tests.md)).

Rule: choices are fitted on held-out control items, and final claims use separate evaluation data.

**I3. Methods that constrain or compress the learner lost; methods that read around its drift won.** [strong at this scale]
- Lost or null: M2c (−12.8 to −26.1), EWC, Sleep (−8), head snapshots, late frozen versions.
- Won: bias correction, a fitted α̂, the statistics expert mixture, PoE, spread version banks, transport of statistics.

**I4. Correction sizes must be fitted, not derived.** [moderate] Prior-based adjustment under-corrects on the toy and MNIST; mean-logit subtraction over-corrects on CIFAR ([05](05-marathon-revisions.md), [06](06-cifar10-e0-e1.md)). Only data-chosen α̂ was right everywhere.

**I5. Memory economics depend on the budget.** [moderate] At small budgets statistics beat images per byte (+3.2 \*); at larger budgets a bigger buffer catches up ([08](08-cifar10-round2.md)). A small spread bank beat its own memory-matched buffer on CIFAR-100 (+1.3 \*, [11](11-audit-corrections.md)).

**I6. Per-class corrections beat global ones.** [moderate, newest]
- Per-class GREG beat X2's global map by ~4 points in our fresh runs ([18](18-atlas-v1-cifar100.md)).
- A per-class mean shift lost to a full map in X2 ([13](13-x2-statistics-transport.md)).

Each class needs its own anchor, and the map between encoders must be more than a shift.

**I7. Replay bias is real on MNIST but invisible on CIFAR-100.** [moderate] MNIST buffer means were ~2.3× farther from the truth ([05](05-marathon-revisions.md)). On CIFAR-100, 20 trained-on items matched 20 fresh ones ([11](11-audit-corrections.md)), and the buffer bridge beat the reserve on sample count ([18](18-atlas-v1-cifar100.md)). The matched-size test in [19](19-atlas-v2b-notebook.md) is designed to settle it.

**I8. Absolute numbers depend on the learner far more than we assumed.** [new, single result] The same pipeline gave learned > random in ewt (35.3 vs 28.2) and random > learned in our re-implementation (25.1 vs 35.1, [18](18-atlas-v1-cifar100.md)). X2's +5.6 did not replicate. Conclusions about *estimators* must be re-checked under a learner at least as strong as ewt's.

**I9. Rigour paid for itself.** [fact] Self-tests and reviews caught dozens of bugs before they produced numbers ([21](21-bugs-and-errors.md)). Yet both independently reviewed experiments still had a design flaw that self-tests could not see ([22](22-test-reliability.md)).
