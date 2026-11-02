# Sampling and family effects

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/budget_audit4*.py

## Why we ran it

To check whether the order of pairs or a harmful family can fool the test.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4817 to 4826 in the fristonx repo.

## What we got

### 21.4 Sampling and family effects

Family effects were +0.10, +0.10, +0.10, −0.30, −0.30 and −0.30, for a mean at the −0.10 boundary.

| Pair order | False approval |
|---|---|
| Independent draws from pool | 0.008 |
| Ordered stream (forbidden) | 0.960 |

Those two figures used a fixed minimum of 48 pairs. Under the final rule with 8 families, half at +0.10 and half at −0.30, false approval was 0.001. With one family at −0.50 and seven at +0.13, the screen rejected 76% of runs, and 6% were approved.

## What we inferred

Pairs must be independent draws from a frozen pool. Ordered streams are forbidden, and family screens were kept.

## Why we think this happened

If good families come first, an anytime test can stop on them before the bad ones appear. Random draws remove that ordering.

## Limits

Assumed family effects. The 0.008 and 0.960 figures used a fixed minimum of 48 pairs.
