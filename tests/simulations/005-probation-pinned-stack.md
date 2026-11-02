# Probation against the pinned stack

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To learn how fast a harmful release is caught after going live.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4828 to 4837 in the fristonx repo.

## What we got

### 21.5 Probation against the pinned stack

| True drop | Rollback within 40 pairs | 80 pairs | 120 pairs |
|---|---|---|---|
| 0.00 | 0.002 | 0.003 | 0.003 |
| 0.05 | 0.013 | 0.024 | 0.026 |
| 0.10 | 0.059 | 0.132 | 0.184 |
| 0.20 | 0.462 | 0.782 | 0.923 |
| 0.30 | 0.820 | 0.983 | 1.000 |
| 0.40 | 0.953 | 0.999 | 1.000 |

## What we inferred

Probation catches large harm quickly but small harm slowly, so it needs longer-term floors behind it.

## Why we think this happened

Detection delay grows sharply as the effect shrinks, because each pair carries little evidence about a small drop.

## Limits

Later runs show failed resets and disputes raise false rollback.
