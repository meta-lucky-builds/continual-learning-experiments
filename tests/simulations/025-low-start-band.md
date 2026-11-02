# Below P_cal 0.65

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round12_work/r12_final.py, r12_line.py, r12_mix.py; audit/round12_check/c3_golock.py; audit/round13_work/r13_band.py, r13_tmax.py

## Why we ran it

The fresh review found nothing had ever run between K1's old line and the action floor.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5168 to 5207 in the fristonx repo.

## What we got

### 21.23 Below `P_cal` 0.65

Section 21.22's lowest scenario sits above the floor of 0.60, so the band K1 used to admit had never run (audit finding F11-H1). `audit/round12_work/r12_final.py` runs it, by subclasses of `c3_combined.py` that leave that file unchanged: 30 runs of 24 months per cell, on one local machine, at 10 normal episodes a day, with no commits, so nothing lifts tasks. All rates are assumed, and none is a robot rate.

c3 can move `P_cal` and not true rates, so it can only say the map under-reads. So a task's usual way here has a true rate of `P_cal`, and the population is one of these: a better way at 0.95 in every task's policy, none in any, or half and half. Before means the rules as they stood. After means the text of that pass, with c3's floor test, which reads this episode's presence. Loads held counts what the old rule would do, and it's about 0 after at this mix, where the better way has mass 0.25 and so is proposed at every DECIDE. That figure leaves out a read that falls later and a disagreement, and it doesn't hold when a way is often absent, which the last table here runs.

| `P_cal` | Half have a better way: mastered, before and after | Blocks per task | Episodes per mastered task | False shared-cause items a year | Loads held per task, before | All have one: mastered | None has one: mastered |
|---|---|---|---|---|---|---|---|
| 0.40 | 35, 43 | 1.48, 1.11 | 184, 161 | 75, 0.03 | 7.7 | 71, 71 | 0, 0 |
| 0.45 | 36, 45 | 1.41, 1.08 | 181, 152 | 68, 0.07 | 7.4 | 71, 75 | 0, 0 |
| 0.50 | 38, 47 | 1.21, 0.98 | 178, 148 | 45, 1.8 | 6.4 | 71, 81 | 1, 1 |
| 0.55 | 39, 42 | 1.04, 0.92 | 175, 165 | 32, 2.3 | 5.4 | 72, 74 | 2, 3 |
| 0.60 | 39, 39 | 0.25, 0.27 | 178, 180 | 7, 7 | 1.9 | 75, 76 | 6, 6 |
| 0.65 | 38, 38 | 0.12, 0.10 | 184, 181 | 5, 5 | 0.9 | 72, 72 | 10, 9 |

Standard errors of mastered counts are about 1. The false items at 0.60 and above are real failure firings on tasks that can't reach the target.

**What this shows.** The cliff at the floor is in cost, not in tasks mastered. With a better way in the policy, a low start masters as many tasks in the span as a task at 0.78 does. It pays about one block per task, which is 12 attended episodes on top of about 40, and it parks more: at 0.50, about 4 in 10 of finished tasks with a better way. With no better way nothing masters under 0.60, which is the honest result, and not a fault of the floor. So this run doesn't pick K1's line. The arithmetic in the K1 protocol does, and a lab could reasonably set it lower at the price of more blocks. The go check removes the largest cost, about 4 to 10 hand-clearings per task, where the way it reads is proposed. There it also keeps a person away from a load in the air, which was the only ordinary path to that.

**A chain the single-task run couldn't see.** With every new task firing the abstention streak, the shared-cause line opened 14–39 times a year on tasks with a better way, and up to 119 with none, against about 0.5 at 0.78. Each item stops practice in a context for days. Counting an abstention firing only if its task ran above the floor in its last 20 opportunities cuts that to about 2 or fewer where tasks have a better way, and about 8 where none do, and it changes nothing at the baseline. The cost: at `P_cal` 0.50 a long lens event was caught 33% of the time against 97%, and at 0.78 by 79% either way (`r12_line.py`).

**Options run and not taken.** A first block that charges no help round cut parked tasks with a better way by about a third, and at half and half it cost 3–6 mastered tasks in the span, because it spends more on tasks that can't reach the target (`r12_mix.py`). A would-abort in a SUPERVISED family proceeding with no declared block, which a reviewer proposed, gained nothing here: 67 mastered against 72 at 0.50, with 0.75 blocks per task against 0.84, and it would reopen the rule that SUPERVISED isn't permission (8.5). Lowering the floor wasn't run, because it would loosen OPERATE too. A block's Thompson draw added 4–11 mastered tasks at 0.45–0.50, less at 0.55, and it cost none beyond noise.

**When a way is often absent.** The go check reads the table, so it can't know whether a way will be proposed. The runs above can't show that, because their better way has mass 0.25, and the model's rule `min(1, mass × 8)` makes that present at every DECIDE. `audit/round12_check/c3_golock.py` varies the mass, on the population where all have a better way, for 30 runs of 24 months. `audit/round13_work/r13_band.py` adds the band above the floor, for 16 runs. c3 rule reads this episode's presence, last DECIDE is the read of the twelfth pass, and table is this text. All rates are assumed.

| `P_cal`, mass (presence) | Mastered: c3 rule, last DECIDE, table | Parked: c3 rule, last DECIDE, table | Blocks per task: last DECIDE, table | Real held loads per task: last DECIDE, table |
|---|---|---|---|---|
| 0.50, 0.25 (100%) | 80, 75, 79 | 53, 63, 54 | 0.89, 0.83 | 0.00, 0.00 |
| 0.50, 0.10 (80%) | 34, 17, 35 | 72, 172, 71 | 1.19, 0.90 | 0.58, 4.03 |
| 0.50, 0.06 (48%) | 27, 13, 26 | 123, 188, 124 | 1.18, 1.08 | 0.26, 2.70 |
| 0.55, 0.10 (80%) | 37, 22, 37 | 61, 141, 62 | 1.15, 0.82 | 0.60, 3.82 |
| 0.60, 0.25 (100%) | 80, 77, 74 | 17, 19, 25 | 0.10, 0.27 | 0.02, 0.02 |
| 0.60, 0.10 (80%) | 51, 38, 46 | 24, 70, 31 | 0.76, 0.32 | 0.80, 3.14 |
| 0.65, 0.25 (100%) | 73, 75, 74 | 16, 16, 21 | 0.05, 0.19 | 0.01, 0.02 |
| 0.70, 0.10 (80%) | 54, 56, 56 | 20, 26, 19 | 0.18, 0.02 | 0.16, 0.73 |
| 0.78, 0.10 (80%) | 63, 60, 60 | 17, 19, 18 | 0.02, 0.00 | 0.02, 0.06 |

So the table costs held loads where a way is often absent, about 4 per task at `P_cal` 0.50, and in return one absence doesn't lock a task. At that mass the lock halved mastered tasks and more than doubled parked ones. That's a held load against a lockout, and the owner chose it. Each such load is still a person near a load in the air. It has a second cost, just above the floor, that the last-DECIDE read didn't: a way never tried has no count, so a used way that has failed can stop a task until a block, though an untried way might have passed. At 0.60 and mass 0.25 that's 0.27 blocks per task against 0.10, and 25 parked against 19, and it's gone by 0.70. A read floored at `P_cal` ended that, at 0.07 blocks and 19 parked, and raised held loads from 0.02 to 0.51 per task, and from 3.14 to 4.40 at mass 0.10 (`r13_tmax.py`, 16 runs). It wasn't taken, because it's a further held-load cost that nobody has accepted. The table above this one and its mastered counts weren't rerun under the table read. At mass 0.25 c3 rule and table agree within noise at 0.50 and 0.55, and at 0.60 and 0.65 that table's rows read a little better than this text would.

**What it leaves out.** Everything Section 21.22 lists. Anchors: c3's tests have their own rate, so what a floor does to a test isn't run (Section 26 item 59). Parked tasks don't reopen, so parked counts are upper figures. The go check's caution, because here each stage's `P_cal` is one number. Under the old rule an episode that failed its first stage ended FAIL and not ABSTAIN, which this leaves out, at a rate of about 0.03. Three things in c3 were corrected in the subclass, because this regime exposes them: new tasks kept starting while a context was held, so tasks in LEARNING piled up there, a held abstention request became a correction request when no cause was found, and a floor test at exactly 0.60 failed in floating point.

## What we inferred

K1 became an effective 0.50, the go check was added, and the table read was chosen, accepting more held loads.

## Why we think this happened

Below the floor a task stops until a block teaches it, so the cost shows up in blocks and held loads, not mastered counts.

## Limits

c3 was subclassed, and no commits ran. Anchors under the floor are not modelled.
