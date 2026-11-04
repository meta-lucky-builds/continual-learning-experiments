# X3 results: dormant class statistics reactivated through a few key items (2026-10-29, as declared in x3_plan.md)

**Status: independent code review done (results/x3/review.md; checks in x3_review_check.py, outputs results/x3/review_check/).** It found no bug that changes a number and no leakage, but three of my readings were wrong; they are corrected below and marked [corrected after review]. Gates gR, gS, gO passed on all 5 seeds; self-test S1–S3 passed. Outputs: results/x3/x3_s.json and results/x3/summary.txt (analyze_x3.py).

**Measure.** 5-way accuracy within task u (argmax over task u's own 5 classes), test items of task u, averaged over u = 1..19, mean ± SE over 5 seeds. This is a task-aware measure: it asks whether the old statistics still *separate* their classes, not the cross-task recency bias.

## Arms (%)

| Arm | Acc |
|---|---|
| OLD: the old encoder v_{u−1} with its own stream statistics (needs the encoder kept). *Not a ceiling [corrected after review]: the final encoder with task-u stream statistics in its own coordinates scores +3.07 ± 0.21 \* higher.* | 78.94 ± 0.26 |
| REACT_all: old statistics, final features mapped back with all buffer keys (6–39 per class) | 77.25 ± 0.29 |
| REACT_5 | 75.34 ± 0.32 |
| REACT_1 | 66.60 ± 0.38 |
| TRANS: X2's online chained cv means, final buffer covariance | 74.49 ± 0.27 |
| BUF_all: buffer keys' means in final features, final buffer covariance, no old statistics | 67.80 ± 0.44 |
| BUF_5 | 57.76 ± 0.44 |
| BUF_1 | 35.35 ± 0.34 |

## Declared readings (star = |mean| > 2·SE and 5/5 signs)

- **R1. REACT_k − BUF_k:** k = 1 **+31.2 \***, k = 5 **+17.6 \***, k = all **+9.5 \***. The dormant statistics add a lot over the keys alone, most when keys are few. *Prediction (positive, largest at k = 1): right.*
- **R2. Recovered fraction REACT_all / OLD:** 97.9% overall (−1.7 points \*). By age: oldest tercile (u 1–6) 94.8% (−4.1 \*); middle 98.8% (−0.9, not starred); newest 99.6% (−0.3, not starred). With 5 keys: −3.6 \* overall; with 1 key: −12.4 \*. *Prediction (falls with age, 80–90% overall): direction right, level wrong; recovery is higher than I predicted.*
- **R3. REACT_all − TRANS: +2.8 \* as run, but mostly a covariance effect [corrected after review].** OLD and REACT use a covariance fitted on task u's 5 classes; TRANS uses the covariance pooled over all 100 classes. With matched covariance:
  - both pooled (streaming-valid): **+0.92 ± 0.23 \***;
  - both 5-class (TRANS's version is an oracle): −0.25 ± 0.38 (not starred);
  - oldest tercile (u 1–6), matched: **TRANS beats REACT by 3.1 \***, so the sign reverses there.
  
  So mapping back and transporting forward are about tied overall, and transport is better for the oldest tasks. *My prediction (about a tie) was closer than the as-run number suggested.* REACT_1 − TRANS −7.9 \* stands.
  - The 5-class covariance is also worth +2.4 \* to OLD and +1.8 \* to REACT, so part of R1 is covariance as well; R1 stays large and positive. R2 is unaffected (REACT and OLD share the covariance; under pooled covariance the ratio is 0.986).

Exploratory (not declared): TRANS − OLD is −2.7 \* in the oldest tercile but −5.4 \* and −5.2 \* in the middle and newest. Transport loses less on the oldest tasks, the opposite of REACT. I have no explanation yet.

## Caveats that change how to read this

1. **Bytes: REACT costs more than I first wrote [corrected after review].** The map back needs each key's features in the old encoder's coordinates, computed while v_{u−1} still exists and stored per version (2,048 B per key). Per version, on average over u: k = 1 0.113 MB, k = 5 0.56 MB, all 2.24 MB; at u = 19: 0.205, 1.02 and 4.1 MB. REACT keeps the old statistics anyway, so what it replaces is only the encoder **body, 0.375 MB** (the 1.105 MB "version" figure includes 0.73 MB of statistics). So **k = 1 is about 0.3× the body (cheaper), k = 5 about 1.5× on average and more than the body from about u = 7, all about 6×.** Streaming validity was checked: every earlier-task item in the final buffer was already in the buffer at the end of task u−1, so storing those features was possible.
2. **Key choice uses the future.** Keys are "the first k items of each class in the final buffer". A streaming learner does not know which items survive; it would have to pick keys when they arrive and keep them permanently (raw images, for the final-feature side). This should not change the numbers much, but it is not what was run.
3. **Covariance asymmetry in R3:** quantified above; it explains most of R3.
4. **Plan record.** x3_plan.md was last modified 19 s before x3.py (17:38:23 vs 17:38:42 UTC), and its "Map back" item appears twice, so the files alone cannot prove the plan predates the code. The self-test S1 tolerance in code is 2 points vs the plan's 1; the observed S1 error is 0.4, within both.
5. One learner, CIFAR-100, 20 tasks, offline replay of saved runs; ridge λ fixed in advance.

## What this means for Lucky's dormancy idea

- **Old knowledge stored as statistics is recoverable.** Mapping today's features back into the old coordinates with a handful of paired keys recovers 95–98% of the old readout's within-task accuracy, with no old encoder at test time. That is a working, measured form of "dormant but reactivatable".
- **The cue size matters sharply.** 1 key per class recovers 84%; 5 keys 95%; all 98%. Oldest tasks recover least.
- **Mapping back is not better than transporting forward** once the covariance is matched (+0.9 overall, and transport wins by 3.1 on the oldest tasks).
- **As a memory design it is only cheaper at k = 1** (0.3× the encoder body, 84% recovery). At k = 5 the keys cost about 1.5× the body they replace. The next useful test is a cheaper map (low-rank, or shared across versions) so the cue costs well under an encoder, and a test on the cross-task (all-class) readout rather than within task.
