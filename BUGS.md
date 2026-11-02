# Every bug, mistake and process error

Copied from section 12 of `source/notes/ewt.md` (updated 2026-10-30). Test numbers refer to `tests/`.


| # | Where | What | Caught by | Effect on results |
|---|---|---|---|---|
| 1 | spec v1.2 | Delay formula off by a factor of κn = 4, with a wrong premise | audit | Formula replaced |
| 2 | spec v1.3 N3 | "Constraints dominate" was in-sample | reviewer | Withdrawn |
| 3 | toy_cl | Saturated default separation; unpaired streams; false docstring | reviewer | Fixed before A3 |
| 4 | diag.py | "finetune" silently replays | me | Excluded |
| 5 | toy protocol | End-of-task insertion breaks ER-ACE | me | Start-insert comparison added |
| 6 | marathon T1 | Uncentred mean init put old-class accuracy at 0 | reviewer (CRITICAL) | Runs superseded |
| 7 | marathon T6 | Relaunched without the env after a restart, so ran the saturated toy | me (guard added) | Runs archived as invalid |
| 8 | cifar.py | float32 probability underflow made the NLL infinite | reviewer | Fixed before E1 |
| 9 | cifar.py | Vacuous checks C8, C9, C11; a C8 threshold that rounds to 1 | reviewer / me | Strengthened |
| 10 | cifar2.py | Vacuous checks K2, K4 (EWC part) and K9; K7 finite differences dominated by curvature | reviewer / me | X1–X5 added; K7 fixed |
| 11 | cifar2 M2c | L-BFGS-B solver was 100% of the wall time | profiling | NNLS swap, validated |
| 12 | cifar2 M2c | Port protected current-task items (differs from the toy) | diagnostic run | Faithful variant added; both reported |
| 13 | sleep.py | Adam drift from an exact optimum | self-test Z5 | SGD |
| 14 | sleep tests | Weak planted tests; a test with the wrong mask size; a too-tight rank tolerance; an untested save path; a tautological mixup test | self-tests and 2 reviews | All fixed before runs |
| 15 | my messages | Guessed timestamps (07:50/07:52 instead of 07:41) written into a plan amendment and a chat message | me | Corrected, with an edit note |
| 16 | my messages | First M2c status claim (before the solver profile) was wrong | me | Corrected by edit |
| 17 | results_r2.md draft | Claimed EWC cost "1.1× ER" without checking. Wall times were measured under different loads. | me | Corrected |
| 18 | ops | `pgrep -f` matched itself; `pkill -f` killed my shell; the noexec mount | me | Workarounds noted in memory |
| 19 | timing | My T-S5 estimate of 2–3 h was too low. A later estimate was corrected. | me | None on results |
| 20 | x1.py self-test | An edit turned check P4 into a comment, so it was neither printed nor counted; also a float32 tolerance too tight | me, before reading results | Fixed and re-run before any real result |
| 21 | x2.py self-test | T1 expectations wrong (shift recovers under rotation; ridge biases aff); thresholds amended twice | self-test failure | Amended before any real run, logged |
| 22 | x2 write-up | Plan timestamps were my guesses; bytes used d = 256 instead of 512; MIX3 computed for the worst arm, not the best | independent review | Corrected; headline unchanged |
| 23 | x3 design | Covariance mismatch between REACT/OLD (5-class) and TRANS/BUF (100-class) | independent review | **R3 conclusion changed** |
| 24 | x3 write-up | "OLD is the ceiling" false; bytes compared against a whole version (1.1 MB) instead of the encoder body (0.375 MB) | independent review | **Cost conclusion changed** |
| 25 | x3 plan record | Plan saved 19 s before code; one item duplicated, so pre-declaration cannot be proved from files | independent review | Noted |
| 26 | my messages | X3 cost first reported to Lucky as "about the size of the encoder" | review | Corrected by edit with a note |

**Known weaknesses that are not bugs:**
- lr was chosen on the seed-100 test set (CIFAR);
- EWC λ came from a 3-value grid on one seed;
- the distillation hyperparameters came from one seed's 200 reserve items;
- many comparisons were run with only 5 seeds;
- CIFAR-100 used an lr that wasn't retuned;
- "spread" version selection uses knowledge of the stream length.

---

