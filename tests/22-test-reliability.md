# 22. How far to trust these tests

Two layers, kept separate:
- **Computational reliability:** given the design, did the code compute what it claims?
- **Design reliability:** does the test measure the question, with fair comparisons?

## Computational reliability: strong
- **Exact reproduction gates.** ewt's experiments reload saved runs and must reproduce earlier numbers exactly first. All passed, including after result files vanished twice and were regenerated.
- **Planted self-tests** before each real run (C1–C13, K1–K10, sleep Z-tests, X1 P1–P5, X2 T1–T4, X3 S1–S3, and our notebook's checks in [17](17-atlas-notebook-unit-tests.md) and [19](19-atlas-v2b-notebook.md)).
- **Independent reviewers** recomputed X2 and X3 numbers by hand.
- **Cross-checks against scikit-learn** for our notebook's covariance, ridge and LDA.

A reported number is very likely the number its code computes. The withdrawals were wrong *readings* or *designs*, not wrong arithmetic.

## Design reliability: moderate (the real weakness)
- Both experiments that got full independent review (X2, X3) had a design-level flaw; one changed a conclusion.
- The T-S5 version curve was mis-designed (unmatched readouts) and corrected only by Lucky's audit.
- Our own v1 notebook had three design problems found from its results: a weak learner, a reserve-vs-buffer comparison that confounded item count with provenance, and a misleading error metric.

Roughly, a conclusion that passed gates but not independent review has perhaps a 1-in-3 chance that a design flaw changes part of its reading. This rests on very few cases.

## Specific weaknesses
1. ewt's recent tests (A1, X1–X3) all reuse the same 5 training runs.
2. No published number has been reproduced on any of these harnesses.
3. One learner type and one scale throughout: online ER, small CNN, CIFAR, 5 seeds.
4. The significance rule (2·SE + 5/5 signs) is weak and applied many times, so a few starred results are expected by chance.
5. Pre-declaration is self-reported. Our notebooks embed hypotheses in the file before any run, which is better but still not third-party.
6. ewt's X2 and X3 were offline replays; our notebooks run fresh but use a re-implemented learner.
7. Hyperparameters: lr chosen on a test seed (ewt). v2b moves selection to a separate seed and control items.
8. Measures differ (X3 is task-aware 5-way; the others are 100-class).

## What would raise trust
- Independent review of a design before its results are read.
- Matched comparisons by construction.
- Reproducing one published number.
- Replicating key findings on fresh runs with a strong learner. v2b does this partly.
