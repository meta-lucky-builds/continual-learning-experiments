# X3 independent code review (2026-10-29)

Scope: x3_plan.md, x3.py, analyze_x3.py, results/x3/{summary.txt, x3_0..4.json}, results/x3_0..4.log. I checked x2.py and cifar.py only where x3 calls them. I edited no existing file. My check script is `/tmp/claude-0/-home-user/da7a0050-f410-5db2-8360-29a4b9e95c98/scratchpad/chk3.py` (outside the project). It ran on all 5 seeds in about 2 min per seed. It reproduces OLD, REACT_all, REACT_1 and TRANS from the json exactly (max diff 0.0 on every u and seed) and adds the variant arms listed below.

## Bottom line

- The code does what the plan says, with no correctness bugs in the arms, keys, map, scoring or gates that change a number. The analysis numbers are correct.
- No test or reserve item enters any fit.
- **R3 is mostly a covariance artefact.** REACT_all − TRANS is +2.76 \* as run. With the covariance matched it is +0.92 \* (both pooled all-class covariances, streaming-valid) or −0.25 (both 5-class stream covariances, not starred). On the oldest tercile, TRANS *beats* REACT by 3.1 \* with matched covariances.
- **"OLD is the ceiling" is false.** The final encoder with task-u statistics in its own coordinates beats OLD by +3.07 \*.
- **The byte comparison in results_x3.md uses the wrong encoder size.** 1.105 MB is a whole *version*: the encoder body plus its statistics. The body alone is 0.375 MB, and that body is all REACT saves. Under that comparison k = 5 costs more than keeping the encoder for every u ≥ 7.

## 1. Plan vs implementation: deviations

| # | Where | Deviation | Effect |
|---|---|---|---|
| D1 | x3.py:38 | Self-test S1 tolerance is `0.02` (2 points); the plan says "within 1 point". | None in practice. I re-ran the self-test and S1 gives 0.720 vs 0.724 (0.4 points), so it also passes the plan's tolerance. All of S1–S3 PASS. |
| D2 | x3.py:11 via x2.py:22-25 | The affine ridge penalises the intercept column too, and λ = 1e-3·tr(XᵀX)/d is computed with the ones column included (d = 513). The plan says "affine ridge (with an intercept column)" and "as in X2", and X2's `aff` does the same, so this is a literal reading. | Checked: REACT_1 with an unpenalised intercept (centre X and Y, ridge, add means back) − REACT_1 = −0.00 ± 0.11 over 5 seeds. Immaterial. |
| D3 | x3.py:47-48, 76-80 | x3 drops three of x2's asserts: `len(E['versions']) == T`, `v['t'] == t` (x2.py:87-89) and `not torch.isnan(mu).any()` for the chained means (x2.py:133). | Checked: TRANS means have no NaN on any seed; gR would catch misordered versions anyway. Hygiene only. |
| D4 | plan "Material/Keys" | Plan: "All" is 6 to 39 per class. | Confirmed: per-class final-buffer counts are min 6 (seed 3) and max 39 (seed 4). |
| D5 | plan, "OLD … The ceiling" | This is a claim, not code, but it does not hold. FIN_oracle (final encoder, task-u stream means and 5-class covariance in final coordinates) − OLD = **+3.07 ± 0.21 \***. Even with pooled covariances (final buffer for FIN, the version's own for OLD) the gap is +1.90 \*. So the final features are better for these classes than v_{u−1}'s. OLD is the "dormant store as it was" reference, not an upper bound on recoverable accuracy. |
| D6 | provenance | Plan mtime 17:38:23, x3.py mtime 17:38:42, first log ~17:40. "Written before x3.py" holds by mtime, but only by 19 s. Also "Map back" appears twice in the plan, which looks like an edit made after x3.py was drafted. mtime is last-modified, so pre-declaration cannot be confirmed from files. |

Everything else matches the plan:
- Source: v_{u−1} stream means and 5-class pooled covariance over ~2,400 stream items (x3.py:62-67).
- Keys: first k by final-buffer position, with `k=0` meaning all (x3.py:18-22).
- Map: fitted on keys of all classes of tasks ≤ u with the same k cap; Y is v_{u−1} features (x3.py:87-90).
- BUF_k: keys of task-u classes only, plus the final-buffer pooled covariance (x3.py:92-93).
- TRANS: the same chained-cv loop as x2 CH_cv with `cov_buf` (x3.py:76-80, 86).
- Scoring: 5-way argmax over `cs`, with column order consistent with `cs` everywhere (x3.py:14-16).
- age = 19 − u.
- Gates: gR is identical to x2; gS and gO are as in x2, and all three PASS in all 5 logs.

## 2. Leakage

- Test items: `Ft` / `Fte_o` are used only as scoring inputs (x3.py:72, 85-86, 91, 93). They are never used in a mean, covariance or map fit.
- Reserve items: `C.batches` excludes reserve, so the stream statistics, the buffer and the keys contain none. Checked: final buffer ∩ reserve = 0 on all seeds. `order[~resv]` is used only for gO (x3.py:71-72).
- Information the plan disallows: none within the plan's terms. REACT never uses the old encoder on test items. It does use v_{u−1} on final-buffer items (`Fb_old`, x3.py:87), which the analysis computes by running the "gone" encoder at the end. That is the plan's design, but it is only valid if those features were stored earlier (see 3).
- Key selection ("first k in the *final* buffer") uses knowledge of which items survive. Keys are chosen without looking at features, so I expect little bias. But a learner with exactly k stored keys could not reproduce this selection (see 3).
- One more design point: the task-u keys are among the stream items that produced `src_mu[u]`. This is a reuse of training data, not leakage.

## 3. Is REACT streaming-valid, and what does it cost?

**What must be stored.** For each version v_{u−1} (u = 1..19), the 512-d v_{u−1} feature of every key item. The key items are the final-buffer items of classes in tasks ≤ u.
- Items of tasks < u that survive to the end were already in the buffer at the end of task u−1, because the reservoir never readmits an item. Checked: 0 violations on all seeds.
- Items of task u pass through while v_{u−1} is live, since v_{u−1} computes the task-u stream statistics (gS).

So these features can all be computed in time. The learner does not know which items will survive, so to reproduce the run exactly it must attach a v_{u−1} feature to every buffer slot holding a class ≤ u, and drop it on eviction.
- For k = all: at the end this equals the counts below; during the run it is up to 2,000 × 2,048 B = 4.1 MB per live version.
- For k = 1 or 5 with only k stored per class: keys must be fixed on arrival and kept. That means pinning them in the buffer (which changes the buffer), or storing their raw images separately (3,072 B each, shared across versions: 0.31 MB for k = 1, 1.54 MB for k = 5, once).

REACT also needs the dormant statistics, as OLD does: the 5 class means plus a 512×512 covariance per task, about 0.53 MB float32 symmetric.

**Bytes per version, key features only.** 512 × 4 = 2,048 B per key. n keys = k·5(u+1), or for all, the final-buffer count of classes ≤ u (seed 0 shown; other seeds within ±3%).

| | u = 1 | u = 19 | mean over u | total, 19 versions |
|---|---|---|---|---|
| k = 1 | 10 keys, 20.5 KB | 100 keys, 0.205 MB | 0.113 MB | 2.14 MB |
| k = 5 | 50 keys, 0.102 MB | 500 keys, 1.024 MB | 0.563 MB | 10.7 MB |
| all | 191 keys, 0.391 MB | 2,000 keys, 4.096 MB | 2.24 MB (1,092 keys) | 42.5 MB |

**Which encoder figure to compare against.** results/ts5/budget.json gives bytes_per_version = 1,105,296. That is the encoder **body** (93,696 params × 4 = 374,784 B) plus float32 statistics (730,512 B: symmetric second moment, class sums, counts). The full state_dict with head is 579,984 B. REACT keeps statistics too, so what it replaces is only the body, 0.375 MB:
- k = 1: 0.30× the body on average, 0.55× at u = 19. Cheaper.
- k = 5: 1.5× the body on average, 2.7× at u = 19. Breaks even at 183 keys, i.e. u ≈ 6.
- all: about 6× the body on average, 11× at u = 19.

Against the full 1.105 MB version figure, k = 5 is 0.09–0.93× (mean 0.51×). results_x3.md caveat 1 compares against 1.1 MB. That understates REACT's relative cost by about 3×, since the statistics it counts in the encoder are kept by REACT as well. That caveat should be corrected.

## 4. Are OLD, REACT and TRANS comparable?

They are not. OLD and REACT score with the **5-class** within-class covariance of the task-u stream (~2,400 items, old coordinates). TRANS and BUF use the **100-class pooled** final-buffer covariance. A covariance fitted to just the 5 competing classes is a large advantage in a 5-way test.

I added variant arms, with 5 seeds, mean ± SE over seeds of the u-average, and the same star rule:

| Check | Value |
|---|---|
| REACT_all − TRANS, as run | +2.76 ± 0.22 \* |
| REACT_all_vcov − TRANS: REACT with v_{u−1}'s own pooled all-class covariance (from its saved S1/n/XX, streaming-valid) vs TRANS with the final-buffer pooled covariance | **+0.92 ± 0.23 \*** |
| REACT_all − TRANS_fin5cov: both with 5-class task-u stream covariance; TRANS's is in final coordinates, an oracle that is not streaming-valid | **−0.25 ± 0.38** |
| OLD − OLD_vcov (the 5-class covariance's worth to OLD) | +2.43 ± 0.14 \* |
| REACT_all − REACT_all_vcov | +1.84 ± 0.09 \* |
| TRANS_fin5cov − TRANS | +3.01 ± 0.24 \* |
| TRANS with 5-class *final-buffer* covariance (only ~100 items) − TRANS | −3.47 ± 0.21 \* |
| REACT_all_vcov / OLD_vcov − 1 (R2 under pooled covariance) | −0.014 ± 0.001 (as run: −0.021) |

By tercile, matched pooled covariance (REACT_all_vcov − TRANS):
- u 1–6: **−3.11 ± 0.40 \***
- u 7–12: +2.36 ± 0.57 \*
- u 13–19: +3.13 ± 0.49 \*

The matched 5-class version gives −4.62 \*, +1.51 and +1.98 \*.

Reading:
- About two-thirds of the as-run R3 gap (1.8 of 2.8 points) comes from the covariance asymmetry.
- With 5-class covariance on both sides the gap is gone.
- Transport beats map-back on the oldest tasks, which reverses the sign of R3 there.
- R1 has the same asymmetry: BUF uses the pooled buffer covariance while REACT uses the 5-class stream covariance. Part of REACT_k − BUF_k (about 1.8 points by the row above) is covariance, not means. R1 stays strongly positive regardless.
- R2 (REACT/OLD) is within-pair consistent, since both use the same covariance, so it is fine.

## 5. analyze_x3.py

- The star rule (|mean| > 2·SE over 5 seeds, and 5/5 signs) matches analyze_x2.py's `st()`. Correct.
- R2 uses the ratio of per-seed means (u-averaged REACT_all over u-averaged OLD), minus 1, then the same rule. That is a reasonable reading of "recovered fraction", and the label says "ratio−1". The mean of per-u ratios gives −0.0533 vs −0.0523 for u 1–6, which is immaterial.
- Every json has exactly u = 1..19, and every accuracy is a multiple of 1/500.

Recomputed independently from the json in plain Python (no analyze code):
- **R1, k = 1:** per-seed u-means of REACT_1 − BUF_1 are 29.947, 33.000, 31.495, 31.253, 30.526 → mean +31.24, SE 0.52, all positive → \*. This matches summary.txt "+31.24 ± 0.52 \*". One cell by hand: seed 0, u = 1: 0.492 − 0.412 = 0.080.
- **R2, u 1–6, ratio − 1:** per seed −0.0492, −0.0427, −0.0533, −0.0612, −0.0550 → mean −0.0523, SE 0.0031 → matches "−0.052 ± 0.003 \*".
- Also: pooled REACT_all/OLD = 0.979, REACT_1/OLD = 0.844 and REACT_5/OLD = 0.954. These match results_x3.md's 97.9%, 84% and 95%.

## Smaller notes

- The docstring says each seed "takes many minutes"; the logs show about 1.5–2 min per seed.
- results_x3.md caveat 3 already flags the covariance asymmetry. The numbers above size it: it accounts for most of R3. The R3 verdict and the "mapping back at need beats transporting forward" sentence should be revised.
