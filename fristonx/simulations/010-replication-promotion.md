# Replication and promotion

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py; audit/round6_work/r6_promo.py

## Why we ran it

To check when a physics fact seen in several contexts can be treated as global.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4877 to 4887 in the fristonx repo.

## What we got

### 21.10 Replication and promotion

Six contexts, 30 episodes each, between-context spread 0.1, testing one coordinate once.

| Truth | Pooling episodes (rejects null) | Context-level test (rejects null) |
|---|---|---|
| No global shift | 25.3% | 5.1% |
| No global shift, one outlier context | 92.9% | 0.1% |
| True global shift 0.2 | 99.8% | 88.1% |

Final promotion rule, with 20 coordinates, 4 looks at 6, 8, 10 and 12 contexts, the sign rule and the size rule: 0.2% false promotion per coordinate, about 3% across 20 if independent, and power of 12%, 78% and 99.5% for shifts of 0.1, 0.2 and 0.3. `audit/round6_work/r6_promo.py` reran the false-promotion figure.

## What we inferred

Promotion must replicate across contexts, not pool episodes (invariant I10).

## Why we think this happened

Episodes in one context are correlated, so pooling treats them as more evidence than they are.

## Limits

A later math check found that 0.2% across 20 coordinates is 3.9%, not about 3%.
