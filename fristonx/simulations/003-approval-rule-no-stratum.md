# Approval rule with no retained stratum

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4.py, budget_audit4b.py

## Why we ran it

To measure the approval rule before stored tasks were added, across family counts.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4802 to 4815 in the fristonx repo.

## What we got

### 21.3 Approval rule with no retained stratum

Margin 0.10, level 0.08, at least 8 pairs in every family, family screen at −0.25, 150 maximum, independent draws, matched pairs. This applies to a pool with no stratum, without the futility stop. Section 21.14 gives the final rule.

| True change | 6 families: approved | Mean episodes | 8 families: approved | Mean episodes |
|---|---|---|---|---|
| −0.10 | 0.009 | 299 | 0.003 | 300 |
| −0.05 | 0.088 | 289 | 0.063 | 295 |
| 0.00 | 0.470 | 248 | 0.434 | 266 |
| +0.05 | 0.924 | 177 | 0.896 | 219 |
| +0.10 | 0.998 | 145 | 0.987 | 200 |
| +0.15 | 1.000 | 142 | 0.992 | 199 |

Pairs until every family has 8: mean 71 (p10–p90 58–88) for 6 families, 100 (82–122) for 8, and 128 (106–154) for 10.

## What we inferred

Fixed the rule's shape, and showed that the per-family minimum sets much of the cost.

## Why we think this happened

Filling every family's minimum takes pairs even when the answer is already clear, so more families cost more episodes.

## Limits

Superseded by the final rule in test 13.
