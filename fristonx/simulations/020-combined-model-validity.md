# Combined simulation: can it show old failures?

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/c3_combined.py, colab_f6.txt, d10_repro.py, colab_reg.txt

## Why we ran it

To run the rules together for the first time, and prove the model can show known failures.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5079 to 5105 in the fristonx repo.

## What we got

### 21.22 The rules run together

`audit/round10_work/c3_combined.py`, build `114e1a6a`, 40 runs of 36 months per scenario, 2,840 runs in all, 10 normal episodes a day, which at the 50% share of Section 13.12 is 20 in all, on one remote machine (Python 3.13, NumPy 2.1.3). Summaries are in `audit/round10_work/colab_f6.txt`. It's the eighth pass's simulation, rebuilt by `mk_c2.py` and `mk_c3.py` through asserted replacements, so each old rule stays behind a switch. Mastery, the stuck rule and help, the strategy table, Task Update, calibration and abstention, Supervised Attempts, the Goal Gate, Probation-lite, tests, Probation, safety events and the floors share one set of tasks and one clock. It asserts the rules' invariants as it runs, for example that no NEEDS_HELP task gets an ordinary episode.

**Added in the tenth pass.** Setup events in tests and practice, the ratchet, the bar with `ĝ` and moving futility, the label check, Probation's close against label lag and late filings, the failure-streak trigger and its extra pairs, doubt scale across retraining, shared causes with the line and the maintenance item, withdrawn periods and replay, contexts, single-strategy and small families, doubling only for fragile or unknown tasks, label absence with the automatic outage and the 20% lapse rule, Probation-lite's Beta-predictive baseline, the absolute floor and reference health, the lapse skip on the last 20 results, a confirming reading on a hold, and the wait for a test's filing deadline before an activation.

**Assumptions.** All of these are assumed, and none is a robot rate. Tasks have two contact stages, with the strategy rates of 21.18 and a task offset on the place stage, so improvement comes from the table itself and not from an assumed curve. Baseline: 6 families, between-task spread 0.10, `P_cal` 0.78, 10% fragile, 6 active tasks, 30% of episodes on a work pool of 8 mastered tasks, sensor labels, help within about 3 days with 10% never arriving, 60% of corrections helpful, 80% person availability, 10% wrong-goal tasks with 90% caught by the goal check, Gate false stops 2% per episode before rebinding, routine events 0.03 and severe events 0.001 per episode, causes `q_a` 0.05, `q_u` 0.30, `P_R` 0.20, a candidate every 75 days: 40% neutral, 40% good at +0.08, 15% bad at −0.10, 5% unsafe with a fivefold event rise overall or a tenfold rise in one family.

**When it ran.** On the rules as they stand, except for what's listed below. Edits after this run changed nothing it models, except the twelfth pass's three, which Section 21.23 runs. Its floor test reads both stages before any stage runs, as the go check does, though at this episode's presence, which no go check can know. At its masses every way is present at every DECIDE, so the two read alike here, and Section 21.23 runs the case where they don't. A block's draws and the shared-cause clause act only under the floor, where these scenarios spend almost nothing.

**What it leaves out.** Robots, and all of physics, perception and policy, hypothesis formation with them, so false disagreement is read only in simulation (6.8) and on the dashboard (20.2). C2 commits, C2 monitors and family monitors (`d4_monitors.py` gives their rates). Lane R. Lane N and Lane F as such: "bad lanes" stands for an activation with no approval test, and the lane limit isn't modelled. Recertification beyond resetting the reference, eras, bridges, pool refresh, anchor retirement and pacing. Object keys, confirmed matches, substitutes and open clauses. The goal check and Gate as anything but rates. Reopening parked tasks, so parked counts are upper figures. Skills. Teacher reliability. Verifier audits, verifier taint and failure-mode notes. Disputes and failed resets, which appear only through the old setup scoring. The family-map monitor. The retained stratum's own effect, because pairs carry one effect. The timing trigger. Family cap and target choice. The C3 hold's pool-eligibility route. Block expiry. Suspended segments. The session overlay beyond one day. Audit's rate rises. Budget lines and resets. State per context, because each task runs in one. The label gap, the pair cap, gold readings and relabelling, because labels here are true. The controller record, the model controls and the set flag. The shared-cause line over the robot, its revocation clause, and pairs pausing under an item. Reference health's noise per anchor, which `d12_fix.py` reads. The evidence-use check, so every candidate reaches its test. A label that arrives after its deadline, so no replay removes a firing or a revocation. Articulated parts, tools and the release rules. The 60, 120 and 180-day limits and the small-scale line, because the baseline runs at 20 a day. `ĝ` per family, and the reserved rerun. Perception and verifier alarms. Step 7, the offset and era loss. The device-fault process, history breaks and waivers. A ratchet's wait for the filing deadline. Each is a place where these results could be wrong.

**Is the model able to show the old failures?** Yes. With each old rule restored in the same code:

| Check | Old rule | Rule in this text |
|---|---|---|
| Family prior, tasks starting alike | 58 mastered, 49 parked, of 189 tasks. 0.8 healthy tasks revoked per run | 193 mastered, 5 parked, of 204 |
| Fixed stuck count | 0.75 help requests per task, 27 parked | 0.25, 11 parked |
| Goal Gate removed | 834 wrong attempts per run | 14 with the Gate, and 0.7 with the goal check as well |
| Setup events scored against the candidate, at 5% | 2.27 activations of 8.9 tested | 4.5 of 8.2 |
| Band and discordance, near the ceiling | 4.1 activations of 9.0 tested | 5.8 of 7.8 |
| Doubt rollback, scale spread 0.20 | 0.33 neutral rollbacks per run | 0.00 |
| Probation closing on pairs, bad lanes, labels at 7 days | 11 mastered of 339, 65 days in REVIEW_HOLD | 39 of 112, 22 days |
| Doubling on every task, one strategy, `P_cal` 0.65 | 244 mastered, 71 blocks | 302, 12 |
| No shared-cause line, a 45-day lens event every 120 days | 140 mastered, 29 parked, 99 help requests | 143, 17, 82 |

The eighth pass's rules, restored whole in the build before this one, gave 219 tasks and 204 mastered. 60 fresh seeds gave 222 and 206 against 215 and 199 for the original script, with a standard error of about 5, so that rebuild didn't change its behaviour (`d10_repro.py`). With this build's three new rules switched off, 120 runs over three scenarios matched that build run for run on every shared metric (`colab_reg.txt`), so `mk_c3.py` changed nothing else.

## What we inferred

Each old failure reproduced with its old rule, so the model has real power to catch these faults.

## Why we think this happened

Rules that are fine alone can fail together. Keeping each old rule behind a switch lets each failure be replayed.

## Limits

Run on Colab, and raw run files were lost. It leaves out robots, physics, perception and policy, as listed.
