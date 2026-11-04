# Continuous floors

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_floor.py

## Why we ran it

To set the alarm threshold for long-term floors over 5 years.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4977 to 4987 in the fristonx repo.

## What we got

### 21.16 Continuous lines

`audit/round6_work/r6_floor.py`, 4,000 runs, 16 pairs a month over 5 years, bet window 48. "Alarm" is the share of runs alarming within 5 years.

| `A` | True `D` +0.05 | 0.00 | −0.05 | −0.10 |
|---|---|---|---|---|
| 20 | 0.000 | 0.000 | 0.002 | 0.106 |
| 50 | 0.000 | 0.000 | 0.001 | 0.043 |
| 100 | 0.000 | 0.000 | 0.000 | 0.018 |

After a step from 0 at month 15, with `A = 50`: −0.15 alarms in 71% of runs within the remaining 45 months, with a median of 373 pairs when it does. −0.20 alarms at a median of 163 pairs, −0.30 at 71, and −0.40 at 49. The absolute line behaved alike: at a true rate of 0.70 against a line at 0.70, about 6% alarmed in 5 years, and a true 0.60 was detected at a median of about 124–250 pairs across `A` from 20 to 200.

## What we inferred

Chose A = 50 and accepted slow detection of small drops, as a stated limit.

## Why we think this happened

The quarterly stream gives 16 pairs a month, so evidence about a small drop arrives slowly.

## Limits

A bounds average run length, not the chance of a false alarm over a finite horizon.
