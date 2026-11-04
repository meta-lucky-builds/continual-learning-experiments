# Verifier alarm

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To see how fast a failing outcome verifier gets noticed.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4898 to 4905 in the fristonx repo.

## What we got

### 21.12 Verifier alarm

| True disagreement | Alarm within 400 audits | Median audits |
|---|---|---|
| 3% | 0.000 | None |
| 5% | 0.002 | None |
| 10% | 0.728 | 275 |
| 20% | 1.000 | 43 |

## What we inferred

The alarm was adopted, because it stays quiet at normal disagreement and fires at real faults.

## Why we think this happened

Audit disagreement is a sparse signal, so evidence builds slowly near the alarm line.

## Limits

Later math review notes K5 at exactly 10% detects only 73% within 400 audits.
