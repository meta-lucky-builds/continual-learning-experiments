# Working mastery, the earlier rule

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_mastery.py

## Why we ran it

To compare mastery rules using a family prior.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4989 to 5002 in the fristonx repo.

## What we got

### 21.17 Working mastery, the earlier rule

**These figures no longer apply to the rule in Section 12.8.** They used a family prior and constant task rates, and that's the rule that failed when tasks improve. They're kept as the record of why it was first adopted. Section 21.22 gives the current rule's behaviour.

`audit/round6_work/r6_mastery.py`, 120 repetitions of 60 tasks, Beta-distributed task rates, no task improvement, family prior from each task's first 8 opportunities.

| Family mean | Old rule: mastered, truly below 0.80, mean episodes | Final rule: mastered, truly below 0.80, mean episodes |
|---|---|---|
| 0.65 | 3%, 80%, 21 | 1%, 38%, 24 |
| 0.75 | 19%, 50%, 18 | 10%, 27%, 23 |
| 0.85 | 92%, 21%, 11 | 76%, 15%, 17 |
| 0.92 | 100%, 2%, 9 | 98%, 2%, 14 |

"Old rule" here is the last-10-outcome rule with this same unbiased prior. Abstention no longer helps: with hard instances at 0.3 and easy ones at 0.95, mastery within 60 opportunities fell from 6.5% with no aborts to 1.0% at 80% aborts. Before the sixth pass it rose from 37% to 91% (round 5).

## What we inferred

Adopted at the time, then replaced. The spec keeps it only as the record.

## Why we think this happened

The family prior was fitted on tasks with constant rates. When tasks improve, early failures poison it.

## Limits

No longer applies to the rule in Section 12.8.
