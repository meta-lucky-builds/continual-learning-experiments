# X3: dormant statistics, reactivated by a small key set (declared 2026-10-29, file time = declaration time; written before x3.py)

**Origin.** Lucky's goal: knowledge kept in low-activation states that reactivate by situation (lucky-cl-goal), and SYNTHESIS.md D9, "define dormancy as recoverability".

**The question.** Suppose an old encoder is gone and only its class statistics remain. Can a map fitted on k stored "key" items per class, from today's features back into the old coordinates, reawaken the old readout? How does that recovery depend on k and on age?

**Material.** Same as X2: T-S5 erp runs, seeds 0–4, with the gates gR, gS and gO re-run.

**Setup.** For every task u ≥ 1 we have:
- **Source.** v_{u−1}'s statistics for task-u classes, from about 480 stream items each, plus the pooled within-class covariance of those stream items. This is the "dormant" store.
- **Keys.** k items per class of task u, taken from the final buffer (first k by buffer position), with k ∈ {1, 5, all}. "All" is 6 to 39 per class.
- **Map back.** Ridge from final features to v_{u−1} features on the keys of *all* classes of tasks ≤ u present in the final buffer, using the same k cap per class. λ = 1e-3·tr/d, as in X2. Fitting a 512→512 map on a few items relies on the ridge.
- **Score.** For test items of task-u classes: 5-way within-task accuracy. Every arm uses argmax over task u's 5 classes.
- **Map back.** An affine ridge (with an intercept column), with Y = v_{u−1} features ≈ map(final features).

**Arms (5-way within task u, test items of task u):**

| Arm | Description |
|---|---|
| OLD | the old encoder itself with its own statistics. The ceiling, which needs the encoder kept |
| REACT_k | old statistics scored on mapped-back final features |
| BUF_k | LDA on the final features from the same k keys per class (means) plus the final buffer pooled covariance. No old statistics |
| TRANS | X2's online chained-cv transported means, 5-way restricted. For reference |

**Declared readings** (star rule as before; age = 19 − u):
- **R1.** REACT_k − BUF_k at k = 1, 5 and all, averaged over u. Do dormant statistics add over the keys alone?
- **R2.** REACT_all / OLD: the recovered fraction, overall and in age terciles (u 1–6, 7–12, 13–19).
- **R3.** REACT_all − TRANS: is mapping back at need any better or worse than transporting forward every task?

**Predictions (mine):**
- R1 is positive and largest at k = 1 (keys alone are weak there).
- R2 falls with age, with overall recovery around 80–90% of OLD.
- R3 is about a tie.

**Planted self-test.** Synthetic data, old = Q·new + noise with Q orthogonal and class means known:
- S1: with enough keys, REACT equals OLD to within 1 point.
- S2: with Q = I and no noise, REACT equals OLD exactly.
- S3: the 5-way accuracy function equals C.acc_seen restricted to the same classes.

**Output.** results/x3/x3_s.json and summary.txt (analyze_x3.py). Results go into results_x3.md.
