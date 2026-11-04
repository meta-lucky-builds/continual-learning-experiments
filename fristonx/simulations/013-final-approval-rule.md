# Final approval rule

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_seq.py

## Why we ran it

To measure the rule actually adopted, including stored-task harm the anchors can't see.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 4926 to 4960 in the fristonx repo.

## What we got

### 21.14 Final approval rule, sixth pass

`audit/round6_work/r6_seq.py`, 20,000 runs per row. Pool of 6 anchor families and the retained stratum at weight 0.25, at least 8 pairs per anchor family and 24 in the stratum, screens at −0.25 and −0.10, futility stop, otherwise as Section 21.3. The same script with the old rule reproduced Section 21.3 to within simulation error (0.419 against 0.434, 0.891 against 0.896, 0.989 against 0.987).

| True change | Approved | Mean episodes |
|---|---|---|
| −0.20 | 0.000 | 63 |
| −0.10 | 0.002 | 191 |
| −0.05 | 0.058 | 265 |
| 0.00 | 0.407 | 265 |
| +0.05 | 0.879 | 229 |
| +0.10 | 0.987 | 213 |
| +0.15 | 0.991 | 213 |

Harm on stored tasks that anchors can't see:

| Anchors | Stored tasks | Final rule approves | Old rule approves |
|---|---|---|---|
| +0.06 | 0.00 | 0.82 | 0.94 |
| +0.06 | −0.10 | 0.43 | 0.94 |
| +0.06 | −0.15 | 0.21 | 0.94 |
| +0.06 | −0.20 | 0.09 | 0.94 |
| +0.10 | −0.15 | 0.26 | 0.99 |

So the stratum costs some real approvals, at +0.06 and 0.00, in exchange. With the effective margin:

| `ε_eff` | True −0.05 | 0.00 | +0.05 | +0.10 | +0.15 |
|---|---|---|---|---|---|
| 0.10 | 0.057 | 0.400 | 0.881 | 0.987 | 0.991 |
| 0.07 | 0.012 | 0.184 | 0.709 | 0.983 | 0.990 |
| 0.05 | 0.004 | 0.085 | 0.533 | 0.961 | 0.991 |
| 0.03 | 0.001 | 0.036 | 0.346 | 0.904 | 0.991 |
| 0.00 | 0.000 | 0.007 | 0.129 | 0.701 | 0.982 |

Ratchet at the final pool: approved 0.004, 0.102, 0.644 and 0.977 at true gains of 0, 0.05, 0.10 and 0.15, at about 216–300 episodes.

## What we inferred

Adopted the retained stratum, screens, futility stop and effective margin. A 20,000-run rerun of the old rule reproduced earlier results.

## Why we think this happened

Stored tasks add a direct read of harm that anchors miss, and that costs some approvals of real gains.

## Limits

Assumes independent pairs at 15% discordance. Label faults are not in this run.
