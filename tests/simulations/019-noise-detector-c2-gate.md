# Noise detector and C2 gate

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_physics.py

## Why we ran it

To check that alarms survive real noise, and that a physics update can't pass by widening.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5031 to 5041 in the fristonx repo.

## What we got

### 21.20 Noise detector and C2 gate

`audit/round6_work/r6_physics.py`. With `S_w` and `μ_w` refit on 400 real windows, at real-to-simulated noise ratios of 1–3 and biases up to 1 simulated standard deviation, the CUSUM with `h_c = 8` gave a false alarm about every 550–1,200 windows, against 3.7 windows before the refit at a ratio of 3, and `γ_noise` stayed at 0.99–1.00, against 0.30. Before dividing by `ŝ_nom`, `ŝ` read 0.86–0.91, which is why that term exists. A 0.5 standard deviation step was caught about every 23 windows.

| C2 case, 3 groups of 3 | Log-score gate passes | Final gate passes |
|---|---|---|
| Widening only, old predictive 2× too narrow | 0.787 | 0.000 |
| Widening, with mean error 0.2 worse | 0.778 | 0.032 |
| Widening, with mean error 0.4 worse | 0.746 | 0.023 |
| Real bias removed, 1 noise std | 0.207 | 0.207 |
| Real bias removed, 2 noise std | 0.473 | 0.473 |

## What we inferred

Adopted the three-part C2 gate and refitting on real windows.

## Why we think this happened

Log score rewards vagueness when a forecast was too narrow, so widening alone looked like a gain.

## Limits

Assumed noise ratios. Real noise is unknown.
