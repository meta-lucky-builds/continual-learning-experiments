# Simulation-guided bets

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit*.py

## Why we ran it

To see if a simulator's prediction can safely speed up a real test.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4794 to 4800 in the fristonx repo.

## What we got

### 21.2 Sim-guided bets

| Prior | Median pairs (margin 0.05, level 0.10, true +0.05) |
|---|---|
| None | 81 |
| Good | 58 |
| Bad | 92 |

## What we inferred

Simulation may set the starting bet, but never the verdict. A bad prior costs little and a good one saves pairs.

## Why we think this happened

The bet only changes how fast evidence builds. Validity comes from the real outcomes, so a wrong prior slows the test without breaking it.

## Limits

Assumed rates. Real simulator quality is unknown.
