# Mastery block and revocation choices

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/c3_combined.py, x1 runs

## Why we ran it

To compare 4-of-5 with 5-of-5, and revocation settings.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5146 to 5166 in the fristonx repo.

## What we got

**Choices 1 and 3 together.** With neutral commits, so nothing lifts tasks, a spread of 0.20, and a stable work pool:

| Block, revocation | Mastered | False at mastery | Still false at the end | Opportunities per mastered task |
|---|---|---|---|---|
| 4 of 5, H0 0.75, `A_r` 20: this text | 91.1 | 5.6% | 5.8% | 44.7 |
| 5 of 5, H0 0.75, `A_r` 20 | 78.8 | 3.6% | 3.8% | 49.7 |
| 4 of 5, H0 0.70, `A_r` 10 | 90.6 | 5.5% | 5.8% | 44.6 |
| 5 of 5, H0 0.70, `A_r` 10 | 78.8 | 3.6% | 3.8% | 49.7 |

Revocation doesn't change these columns, because most mastered tasks here get too few opportunities for it to read: it can only act on tasks that keep running. On tasks that do, with commits neutral and one work-pool task in about two months falling by 0.25 in true rate:

| Block, revocation | Dropped tasks revoked within the run | Mean days to revocation | Healthy tasks revoked per run |
|---|---|---|---|
| 4 of 5, H0 0.75, `A_r` 20: this text | 22% | 273 | 0.05 |
| 5 of 5, H0 0.75, `A_r` 20 | 22% | 301 | 0.00 |
| 4 of 5, H0 0.70, `A_r` 10 | 17% | 311 | 0.03 |
| 5 of 5, H0 0.70, `A_r` 10 | 16% | 287 | 0.05 |

So the choice of block sets false mastery, and revocation doesn't undo it for tasks that rarely run. 4 of 5 trades about 2 percentage points of false mastery for about 16% more tasks mastered and about 5 fewer opportunities each. The boundary in this text finds about a third more dropped tasks than the looser one, at no clear cost in healthy revocations, which stay near 0.05 a run or below. Most dropped tasks weren't revoked within the run, which is what Section 26 item 39 records.

**Stated false-mastery rates.** Under these assumptions, 1.5–7.3% of tasks at the moment of mastery have a true rate below 0.80. On a task that holds a flat rate, with no stuck rule and no improvement, the rule masters a true 0.65 task 5% of the time within 60 opportunities and 9% within 120, a true 0.70 task 13% and 23%, a true 0.75 task 28% and 52% (`x1`). Those flat rates are the plainer statement of its error, and the 1.5–7.3% depends on how few tasks sit near the target in this model.

## What we inferred

Kept 4 of 5: about 16% more mastered, for about 2 points more false mastery.

## Why we think this happened

Revocation only reads tasks that keep running, so it can't undo false mastery on rare tasks.

## Limits

A later check found a flat true-0.75 task masters 52% within 120 opportunities.
