# Combined simulation: shared causes

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/c3_combined.py, colab_f6.txt

## Why we ran it

To test the rule for a cause hitting many tasks at once.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5144 to 5144 in the fristonx repo.

## What we got

**Shared causes.** With no lens events, the line opened 0.26 times a year on the baseline and 1.0 on the hard mix, all of them false, each costing an operator's look and about 2 days of practice in that context. With 9 lens events a run, it caught about 17% of 14-day events and about 59% of 45-day events. So the line pays on long causes, and it does little for short ones, which end before enough tasks fire.

## What we inferred

The line pays on long causes and does little for short ones.

## Why we think this happened

Short events end before enough tasks have fired.

## Limits

Assumed event rates.
