# 21. Every bug, mistake and process error we know of

Rows 1–26 are from ewt.md §12. Rows 27 onward are from the Atlas design phase.

| # | Where | What | Caught by | Effect |
|---|---|---|---|---|
| 1 | spec v1.2 | Delay formula off by κn = 4, wrong premise | audit | Replaced |
| 2 | spec v1.3 N3 | "Constraints dominate" was in-sample | reviewer | Withdrawn |
| 3 | toy_cl | Saturated separation, unpaired streams, false docstring | reviewer | Fixed before A3 |
| 4 | diag.py | "finetune" silently replays | us | Excluded |
| 5 | toy protocol | End-of-task insertion breaks ER-ACE | us | Start-insert added (later found non-causal) |
| 6 | marathon T1 | Uncentred mean init put old-class accuracy at 0 | reviewer (critical) | Runs superseded |
| 7 | marathon T6 | Relaunched without env after restart | us (guard added) | Runs archived as invalid |
| 8 | cifar.py | float32 probability underflow, infinite NLL | reviewer | Fixed before E1 |
| 9 | cifar.py | Vacuous checks C8, C9, C11; a threshold that rounds to 1 | reviewer / us | Strengthened |
| 10 | cifar2.py | Vacuous checks K2, K4, K9; K7 finite differences | reviewer / us | Fixed |
| 11 | cifar2 M2c | L-BFGS-B solver was all of the wall time | profiling | NNLS swap, validated |
| 12 | cifar2 M2c | Port protected current-task items | diagnostic run | Faithful variant added; both reported |
| 13 | sleep.py | Adam drift from an exact optimum | self-test | SGD |
| 14 | sleep tests | Weak planted tests, wrong mask size, tight tolerance, untested path, tautological test | self-tests, 2 reviews | Fixed before runs |
| 15 | messages | Guessed timestamps in a plan amendment | us | Corrected |
| 16 | messages | Wrong first M2c status claim | us | Corrected |
| 17 | results draft | Unchecked EWC cost claim | us | Corrected |
| 18 | ops | `pgrep -f` matched itself, `pkill -f` killed the shell, noexec mount | us | Workarounds |
| 19 | timing | T-S5 run-time estimate too low | us | None |
| 20 | x1 self-test | A check turned into a comment; tolerance too tight | us, before results | Fixed and re-run |
| 21 | x2 self-test | Wrong test expectations; thresholds amended twice | self-test | Amended before real runs |
| 22 | x2 write-up | Guessed plan times; wrong byte dimension; MIX3 on wrong arm | review | Corrected; headline unchanged |
| 23 | x3 design | Covariance mismatch between arms | review | **R3 conclusion changed** |
| 24 | x3 write-up | "OLD is the ceiling" false; wrong encoder size for bytes | review | **Cost conclusion changed** |
| 25 | x3 plan | Plan saved 19 s before code | review | Pre-declaration unprovable |
| 26 | messages | X3 cost first reported wrong | review | Corrected |
| 27 | ARCH-v1 | Seven design claims false (see [15](15-arch-v1-audit-counterexamples.md)) | Lucky's audit | ARCH-v2 |
| 28 | ARCH-v1 | "Re-anchoring resets error"; ill-defined "33 = 70% of the gap" target | audit | Withdrawn |
| 29 | notebook synthetic test | "Exact" test projected 12-d into 10-d, so the hop was not linear | the test itself failing | Test fixed, not the code |
| 30 | notebook GREG | Cross-fitting halved small bridges (error 2.43 vs 0.78) | synthetic comparison | Turned off before CIFAR |
| 31 | notebook ridge | Grid floor 1e-6 left shrinkage bias on exact data | synthetic test | Grid to 1e-9 |
| 32 | notebook ridge | One shared penalty let 40 noisy r columns cut R² 0.85 → 0.74 | synthetic test | Separate block penalty |
| 33 | notebook v1 design | Learner far weaker than ewt's (head 8.1 vs 17.4) | the CIFAR results | v2b calibration |
| 34 | notebook v1 design | Reserve vs buffer confounded item count with provenance | analysing results | Matched test in v2b |
| 35 | notebook v1 design | Mean-error metric disagreed with accuracy | analysing results | Accuracy and log loss in v2b |
| 36 | notebook v1 | Block-ridge CV for r noisy at small n (config B R² fell 0.947 → 0.910) | results | Noted; r demoted |
| 37 | process | `pip install torch` started without asking; Lucky stopped it twice | Lucky | Torch work moved to Colab |

**Known weaknesses that are not bugs:**
- lr was chosen on a seed-100 test set (ewt);
- small grids for EWC λ;
- 5 seeds;
- CIFAR-100 lr not retuned;
- "spread" uses stream length;
- our PyTorch cells were never run locally.
