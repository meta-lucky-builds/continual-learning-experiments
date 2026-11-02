# Combined simulation: task outcomes

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/c3_combined.py, colab_f6.txt

## Why we ran it

To see learning cost across 14 scenarios.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5107 to 5126 in the fristonx repo.

## What we got

**Task outcomes**, means per run:

| Scenario | Tasks | Mastered | Parked, of those good tasks | False at mastery | Opportunities per mastered task | Help requests per task | Blocks |
|---|---|---|---|---|---|---|---|
| Baseline | 198 | 180 | 11, 10 | 2.5% | 32.6 | 0.25 | 0.1 |
| Tasks start alike | 204 | 193 | 5, 5 | 2.3% | 32.5 | 0.18 | 0.0 |
| Tasks differ, spread 0.20 | 196 | 167 | 22, 16 | 2.9% | 31.0 | 0.37 | 0.3 |
| Families differ by 0.10 | 186 | 162 | 18, 14 | 2.7% | 33.4 | 0.35 | 0.0 |
| Human labels, 2 days | 150 | 136 | 8, 7 | 2.9% | 33.4 | 0.26 | 0.0 |
| Human labels, 7 days | 97 | 82 | 8, 7 | 2.8% | 38.9 | 0.39 | 0.0 |
| 5% of labels lost for good | 115 | 101 | 8, 7 | 1.8% | 44.2 | 0.32 | 0.0 |
| `P_cal` low, 0.65 | 225 | 207 | 12, 11 | 2.9% | 29.1 | 0.20 | 3.2 |
| Map off by 0.20 | 207 | 192 | 9, 9 | 2.7% | 31.2 | 0.21 | 0.0 |
| Help poor | 211 | 178 | 26, 25 | 2.6% | 30.6 | 0.25 | 0.0 |
| Gate false stops 10% | 183 | 167 | 10, 10 | 1.9% | 35.2 | 0.28 | 0.0 |
| Gate false stops 25% | 161 | 138 | 16, 15 | 1.5% | 41.1 | 0.51 | 0.0 |
| All candidates neutral | 134 | 103 | 23, 12 | 3.4% | 47.6 | 0.70 | 0.1 |
| Hard mix | 96 | 72 | 18, 12 | 3.0% | 35.0 | 0.60 | 6.2 |

Hard mix is spread 0.20, `P_cal` 0.65, human labels, map off by 0.10, 20% wrong-goal tasks, 10% false stops, 50% person availability and 40% helpful corrections. In the baseline, good commits lift every task, so "all candidates neutral" is the fairer view of the task loop alone: about 48 opportunities per mastered task. The 7-day rows are upper figures, because the pair cap isn't modelled, and it slows tests at that delay. Across all scenarios under these rules, no task spent over 120 settled opportunities between resets. The failure streak's cost to improving tasks shows only as help requests per task, because this model doesn't mark which tasks were improving when it fired. K6's line fired in about 1 of 8 first spans on the baseline, about 2 of 3 with every candidate neutral, and always with every candidate bad.

## What we inferred

About 33 opportunities per mastered task, or about 48 with neutral commits, which is the fairer view of the task loop.

## Why we think this happened

Good commits lift all tasks, so removing them shows what the task loop does alone.

## Limits

All rates assumed. Not robot rates.
