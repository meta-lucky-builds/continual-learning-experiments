# Combined simulation: commits and rollbacks

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/c3_combined.py, colab_f6.txt

## Why we ran it

To check whether rollbacks fire on healthy releases.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5128 to 5142 in the fristonx repo.

## What we got

**Commits**, means per run over 36 months:

| Scenario | Tested | Activated | Rollbacks per activation | Neutral releases rolled back | Days in REVIEW_HOLD |
|---|---|---|---|---|---|
| Baseline | 8.1 | 4.35 | 0.029 | none | 0 |
| All candidates neutral | 8.6 | 2.20 | 0.045 | 0.045 | 0 |
| All candidates good | 7.2 | 6.67 | 0.026 | none | 0 |
| All candidates bad | 10.9 | 0.00 | none | none | 0 |
| Severe events 0.002 | 8.2 | 4.33 | 0.069 | 0.066 | 0 |
| Severe events 0.005, `q_a` 0.1, `q_u` 0.3, `P_R` 0.3 | 8.2 | 4.15 | 0.193 | 0.192 | 0 |
| Routine events 0.08 | 8.2 | 4.28 | 0.029 | none | 0 |
| Reference drift 0.008 a month | 7.9 | 4.80 | 0.036 | 0.047 | 0 |
| Drift, and a Reviewer who often doesn't answer | 7.9 | 4.80 | 0.036 | 0.047 | 0 |

No floor alarm came in the 40 runs of the drift rows, so a Reviewer's absence cost nothing there. In the build before this one, a single alarm in 40 runs cost 188 days in all. Pooled over the 25 unelevated scenarios named in the summary, 1,000 runs, neutral releases rolled back 2.7% of the time, over 1,376 activations, and good releases 2.3%, over 2,778, against 27–100% for the single trigger this replaced. So K7's 30% isn't reached by base rates at these settings, by the paths modelled. It's approached only at a severe rate of 0.005 per episode, where the composed figure of 0.19 is close to the 0.23 that Section 13.7's formula gives. Timing and floor rollbacks aren't in that figure. Demotions ran at about 0.1 a year on the baseline and 0.25 at a severe rate of 0.002. Of 160 unsafe releases activated, 60 rolled back within their Probation or tail. That's what Section 13.7's honest limit predicts, and a small harm that an approval test passes stays for the floors.

## What we inferred

Neutral rollbacks were 2.7% and good releases 2.3%, against 27-100% for the single trigger replaced.

## Why we think this happened

Requiring a cause and evidence stops one random event from rolling back a healthy release.

## Limits

Unsafe releases that pass a test often stay: 60 of 160 rolled back in Probation.
