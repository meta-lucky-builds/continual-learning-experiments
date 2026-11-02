# Selective labels

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round6_work/r6_ipw.py; audit/round8_work/x8_changed_rules.py

## Why we ran it

To see if calibration can be fitted when the robot only attempts high-score cases.

## What we did

Statistical model of outcomes, as described below. Results are copied from `docs/spec/SPEC.md` lines 5017 to 5029 in the fristonx repo.

## What we got

### 21.19 Selective labels

**The `π` rows below no longer apply.** Supervised Attempts now always proceed inside a block (8.5). `audit/round8_work/x8_changed_rules.py` reran the question on the same model, for a share `e` of would-aborts proceeding, and it found spread at scores 0.10–0.45 of 0.06, 0.045, 0.03 and 0.02 at `e` of 0.1, 0.25, 0.5 and 1. A weight on those rows made it slightly worse, so Section 12.7 applies the weight only to earlier decisions' rows.

`audit/round6_work/r6_ipw.py`. A two-parameter logistic map on 600 rows, with attempts only above a score of 0.5. Spread across runs of the fitted map's error below that score:

| Data | Spread |
|---|---|
| Attempts only | 0.11 |
| Plus Supervised Attempts at `π` = 0.1, 0.25, 0.5 | 0.078, 0.050, 0.033 |
| All rows | 0.025 |

Mean error was near zero in all cases, because this map is correctly specified. A non-parametric map with no low-score rows has nothing to fit there at all.

## What we inferred

Added Supervised Attempts. The pi rows were later replaced, because attempts now always proceed in a block.

## Why we think this happened

With no low-score rows, the map has nothing to fit there.

## Limits

The model is correctly specified, which real maps may not be.
