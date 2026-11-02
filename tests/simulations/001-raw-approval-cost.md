# Raw approval cost

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit*.py

## Why we ran it

To learn how many real episodes it takes to approve a model change, before any rule was fixed. Real episodes are the scarce resource.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4781 to 4792 in the fristonx repo.

## What we got

### 21.1 Raw approval cost, before final rules (median pairs)

| Level | Margin | True change | Independent pairs | Matched pairs (15% disagreement) |
|---|---|---|---|---|
| 0.05 | 0.05 | 0.00 | 1,463 | 687 |
| 0.05 | 0.05 | +0.05 | 291 | 109 |
| 0.10 | 0.05 | 0.00 | 1,183 | 516 |
| 0.10 | 0.10 | 0.00 | 263 | 103 |
| 0.10 | 0.10 | +0.05 | 88 | 31 |
| 0.10 | 0.10 | +0.10 | 33 | 18 |

Each pair is 2 real episodes.

## What we inferred

Neutral changes are too costly to approve one at a time, so the design added a predicted-gain filter, change classes and bundling. Matched pairs became the standard.

## Why we think this happened

Proving "no worse than before" gets very expensive as the true change nears the margin. Matching reduces noise, because both arms share the same conditions, so far fewer pairs are needed.

## Limits

Statistical model with assumed rates. Rules at that time, later replaced by the final rule.
