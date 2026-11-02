# 40. X3: can dormant class statistics be woken with a few key items?

**Date:** 2026-10-29. **Source files:** `source/notes/cifar/{x3_plan.md, x3.py, analyze_x3.py, results_x3.md, x3_review_check.py}`, `results/x3/` (including `review.md` and `review_check/`).

## Why we ran it
Lucky's goal for continual learning includes knowledge kept in low-activation, dormant states that can be reactivated by situation. We defined dormancy operationally as **recoverability**: if the old encoder is gone and only its class statistics remain, can a map from today's features back into the old coordinates, fitted on k stored "key" items per class, reawaken the old readout?

## What we did
For every task u ≥ 1 (CIFAR-100, 5 seeds), with test items of task u scored 5-way within the task:
- **OLD:** the old encoder v_{u−1} with its own stream statistics;
- **REACT_k:** old statistics scored on final features mapped back by an affine ridge fitted on k keys per class (k = 1, 5, all) from the final buffer;
- **BUF_k:** the same keys' means in final features, no old statistics;
- **TRANS:** X2's transported means.
Declared readings: R1 REACT − BUF; R2 REACT/OLD by age; R3 REACT − TRANS. Gates gR, gS, gO; self-test S1–S3.

## What we got

| Arm | Acc |
|---|---|
| OLD | 78.94 |
| REACT_all / 5 / 1 | 77.25 / 75.34 / 66.60 |
| TRANS | 74.49 |
| BUF_all / 5 / 1 | 67.80 / 57.76 / 35.35 |

- R1: +31.2 \* (k = 1), +17.6 \* (k = 5), +9.5 \* (all); about 1.8 points of each is a covariance difference.
- R2: recovery 98% (all keys), 95% (5), 84% (1); oldest tasks least (95% with all keys).
- **R3, corrected after review:** as run, +2.8 \*, but OLD/REACT used a 5-class covariance and TRANS a 100-class one. Matched: +0.9 \* overall, and transport wins by 3.1 \* on the oldest tasks.
- **"OLD is the ceiling" was false:** the final encoder with full statistics beats it by +3.1 \*.
- **Bytes, corrected after review:** keys' old-coordinate features cost 0.11 / 0.56 / 2.24 MB per version on average (k = 1 / 5 / all) vs the 0.375 MB encoder body they replace. Only k = 1 is cheaper.

## What we inferred
Dormancy as recoverability works and is measurable: a few key items wake old statistics to 84–98% of the old readout's within-task accuracy. It is not better than forward transport once covariances are matched, and as memory it is only cheaper than keeping the encoder at 1 key per class.

## Why we think it occurred
The map back needs enough paired items to pin down a 512→512 linear map; ridge makes it work with few, but 1 key per class leaves it under-determined, hence 84%. The oldest tasks recover least because the final encoder has moved furthest from theirs.

## How reliable this test is
- **Independently reviewed:** no bug that changes a number, no leakage, but three of my readings were wrong (R3, the ceiling, the bytes) and are corrected above.
- Within-task (task-aware) measure, not 100-class. Keys chosen from the final buffer use knowledge of which items survive. The plan file was saved only 19 s before the code, so pre-declaration cannot be proved from files.
