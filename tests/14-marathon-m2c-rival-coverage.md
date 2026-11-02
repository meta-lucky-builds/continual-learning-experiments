# 14. Marathon T6: M2c with the spec's rival rule

**Date:** 2026-10-28. **Source files:** as test 05; invalid runs archived in `v13/invalid_sep16/` (not in this repo).

## Why we ran it
Zero-slack M2c removed most first-order damage yet gained little (test 04). One candidate cause: the constraint set missed the classes that cause damage. The spec says rivals should include "labels born since the last refresh". This test checked whether that rule makes M2c work.

## What we did
Zero-slack M2c on toy A with (a) top-2 rivals and (b) top-2 plus born-since-refresh rivals. Measured test accuracy, held-out old-class accuracy at task end, and prediction flips per task.

## What we got

| Arm | Test acc | Held-out old acc | Flips/task |
|---|---|---|---|
| ER | 41.5 | 33.1 | 73 |
| M2c, top-2 rivals | 44.6 (+3.1 \*) | 38.5 | 67 |
| **M2c + born-label rivals** | **54.1** (+9.5 \* vs top-2, +12.5 \* vs ER) | 52.5 | 54 |

## What we inferred
The spec's rival rule is necessary for M2c. But M2c + new rivals costs ~30× ER compute and is still **below** the free readout fixes on the same toy (biascorr 57.7, logit adjustment 57.2, ER-ACE 57.8). M2c is not justified.

## Why we think it occurred
New classes are top-2 rivals 90% of the time at refresh, but the collapse starts 20–40 steps into a task, before the first refresh (every 25 steps). In that window only the born-label rule constrains the new rivals.

## How reliable this test is
- **Medium**, after a process error. After a container restart I relaunched T6 **without the toy-A settings**, so those runs used the saturated toy. They were caught, archived as invalid and re-run. A guard now checks ER reproduces the known seeds.
- Whether M2c stacks with logit adjustment was never tested.
