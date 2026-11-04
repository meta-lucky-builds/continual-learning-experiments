# X2 results: carrying stranded class statistics into new encoders (2026-10-29, as declared in x2_plan.md)

**Status: independent code review done (17:35 UTC). It found no result-changing bug.** It independently checked:
- streaming validity;
- the buffer reconstruction, including the final buffer: B_only and ER+bc_buf equal the original run's logged buflda and bc_buf exactly, on all 5 seeds;
- the maps, the LDA and the mixtures.

Its minor findings are corrected below and in x2_plan.md: wrong plan timestamps, the MIX3-best-arm deviation, and the byte figures.

**Gates, all passed:**
- **gR.** The reconstructed buffer reproduces every version's statistics on its old classes.
- **gS.** The recomputed stream statistics match the saved version statistics.
- **gO.** O_full equals the saved matched probe PF to 1e-6.
- **gB.** The recomputed ER+bc_buf is 22.70 (want 22.7), BANK_4spread 31.21 (want 31.2), RC 28.17 (want 28.2).
- **Self-test.** T1–T4 pass. Two test thresholds were amended before any real run, both logged in the plan.

5 seeds. Outputs: results/x2/x2_s.json and results/x2/summary.txt.

## Accuracy (%, mean ± SE; LDA readout over all 100 classes, final coordinates)

| Arm | Acc |
|---|---|
| O_full: oracle, full-stream statistics in final features | 35.30 ± 0.36 |
| O_mean: oracle means, buffer covariance | 34.44 ± 0.33 |
| B_only: final buffer only (about 20 per class) | 24.93 ± 0.21 |
| **CH_cv: online chained control-variate transport** | **30.56 ± 0.17** |
| CH_cv_tr: as CH_cv, plus chained covariance | 30.79 ± 0.15 |
| CH_aff: online chained shared affine | 30.50 ± 0.33 |
| CH_shift: online chained class-local shift (SDC-like) | 26.87 ± 0.16 |
| DI_cv / DI_aff / DI_shift: direct, non-online | 30.24 / 31.61 / 24.27 |
| BANK_4spread (ER+bc, RC, and 4 versions; 4.4 MB of versions) | 31.21 ± 0.27 |
| **MIX3_CH_cv (ER+bc, RC, and CH_cv; about 0.73 MB of statistics)** | **32.08 ± 0.21** |
| MIX3_B_only | 28.12 ± 0.31 |

## Declared comparisons (star = |mean| > 2·SE and 5/5 signs)

- **Q1.** CH_cv − B_only = **+5.63 ± 0.10 \***. Online transport works.
  - My prediction was +2 to +5, so it came in above the range.
- **Q2.** CH_aff − CH_cv = −0.06 (not significant). CH_shift − CH_cv = −3.68 \*.
  - **My prediction that aff would be below cv was wrong.** On CIFAR a shared affine map carries the means about as well as cv. The MNIST E3 failure does not replicate here.
  - The SDC-style shift is the weak one.
- **Q3.** Gap closed (X − B_only) / (O_full − B_only): CH_cv 0.54, CH_cv_tr 0.57, CH_aff 0.54, DI_aff 0.64, CH_shift 0.19.
  - My prediction was 20–50%, so this is slightly above it.
- **Q4.** MIX3_CH_cv − BANK_4spread = **+0.86 ± 0.29 \***.
  - It beats the version bank while storing statistics only.
  - **Corrected bytes:**
    - MIX3_CH_cv uses only the transported means: 100×512 float32, about 0.2 MB.
    - It also holds a transient frozen encoder body, about 0.37 MB.
    - With the chained covariance as well (CH_cv_tr), the statistics total about 0.73 MB.
    - The bank's 4 versions are 4.42 MB.
  - So it uses about 4–8× fewer bytes, depending on what is counted.
  - **Fragility:** the per-seed gains are +0.74, +0.04, +1.83, +0.72 and +0.99. Seed 1 is only about 4 test images.
  - My prediction was a tie or a loss, so this was **wrong in the good direction**.
  - CH_cv alone vs the bank: −0.65 ± 0.20 (not 5/5).
- **Q5.** Means carry about 92% of the gap: O_mean − B_only = +9.50 \*, against O_full − O_mean = +0.87 \*.
  - This matches the prediction.
- **Q6.** CH_cv_tr − CH_cv = +0.23 \*. Transporting the covariance helps a little.

Exploratory comparisons (not declared): DI_aff − CH_cv = +1.05 \*; MIX3_CH_cv − MIX3_B_only = +3.95 \*; CH_cv − BANK = −0.65 (not significant). The analysis script is analyze_x2.py.

## What this means

1. **The stranded-statistics problem is about half solvable online, cheaply.** The method:
   - At each task end, regress the new encoder's buffer features on the old encoder's features, after centring within each class.
   - Move every stored class mean with that map.
   - Discard the old encoder.

   This recovers 54–57% of the gap to oracle statistics. It stores only means and a covariance.
2. **Most of the remaining loss is in the means, not the covariance.** The best non-online map (direct affine) gets 64%, so chaining costs about 10 points of gap fraction.
3. **E3 needs restating.** "A shared affine map failed" was an MNIST result. On CIFAR a global linear map transports means about as well as the class-local control variate. The SDC-style mean shift, which is the published method family, is clearly worse.
4. **This is the first component here that beats the frozen-version bank at fewer bytes.** It is a candidate for the ground-up design's memory: statistics, transported, plus a validated map.

## Limits

- One learner (online ER, M = 2000, GroupNorm CNN), CIFAR-100, 20 tasks.
- Transport is replayed offline on saved runs. It has not yet been run live inside the learner, though every input is streaming-valid by construction; the reviewer is checking that.
- Ridge λ = 1e-3 relative was fixed in advance, not tuned.
- The mixture weights are equal (1/3), as in the bank.
