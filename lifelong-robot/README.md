# Lifelong robot project (RoboJEPA): every test, explained

This folder records every experiment run so far (2026-10-17 to 2026-10-23) in the project to build a lifelong-learning robot architecture around RoboJEPA. The project targets 14 problems: adaptation, protecting old skills, relearning, habits, memory, consolidation, self-knowledge, recovery, sample efficiency, and transfer of motor ability (P1–P14).

Each file explains one test:
- why we ran it;
- what we did;
- what we got;
- what we inferred;
- why we think it happened;
- caveats;
- what it changed afterwards.

Numbers cite the result files in the project's shared folder (paths like `research/...` are relative to `/mnt/project-files/`; that folder is not part of this repository). Explanations are tagged **[checked]** (shown by a result or audit file), **[inferred]** (our reasoning) or **[guess]**.

## The short answer
**No idea of ours has yet beaten the strongest existing method under a pre-registered bar.**

Our ideas beat *classic* baselines several times, at toy or pilot level. Each time a tuned existing stack was in the race, the stack won.

All tests are small CPU toys, and three of them are pilots (1–8 seeds, untuned). Nothing has run on RoboJEPA or a GPU.

## Files

| # | File | Test | Verdict |
|---|---|---|---|
| 01 | [01-test-M-misfit-writes.md](01-test-M-misfit-writes.md) | Do evidence-gated writes stop when the base model is wrong? | FAILED: bad writes grew about linearly |
| 02 | [02-test-P-practice.md](02-test-P-practice.md) | Does a practice prior cover new task families? (first P14 test) | UNCLEAR; condition (L) failed in 0/20 seeds |
| 03 | [03-T-COMPOSE.md](03-T-COMPOSE.md) | Predict never-seen mixes of known causes by inference | FAILED: 1.30× COIN's error (bar ≤0.25×) |
| 04 | [04-T-CHURN.md](04-T-CHURN.md) | Do writes level off now, and is accuracy as good as LoRA + replay? | FAILED on accuracy; writes grow sub-linearly |
| 05 | [05-T-P14-practice-prior.md](05-T-P14-practice-prior.md) | Does a learned practice prior make learning faster? | FAILED: slower than no prior and than Reptile |
| 06 | [06-cluster-designer-self-tests.md](06-cluster-designer-self-tests.md) | Four cluster architects' own toys (L, C, M, S), plus audits | All flattered by their own toys; L fatal |
| 07 | [07-T-CAUSE.md](07-T-CAUSE.md) | Independent test of "store causes" vs existing methods | Pilot: existing stack cxl beat every C variant |
| 08 | [08-T-P14D-body-discovery.md](08-T-P14D-body-discovery.md) | Can a robot discover its body and transfer motor ability? | Pilot: no arm showed P14; ours below generic adapters |
| 09 | [09-p14v7-memory-of-errors.md](09-p14v7-memory-of-errors.md) | Carry *how fast* to learn ("memory of errors") | Pilot: own kill test fired |
| 10 | [10-T-DORMANT.md](10-T-DORMANT.md) | Forget only on contradicting evidence, never by the clock | KILLED: no-forgetting beat evidence-driven forgetting |
| 11 | [11-never-tested.md](11-never-tested.md) | Everything designed or planned but never tested | |
| 12 | [12-lessons-across-tests.md](12-lessons-across-tests.md) | What the tests say together and why it keeps happening | |

## How these were written
- Files 01–10 were written by separate agents. Each read the protocol, result tables and audit files for its tests, and none of them designed the ideas under test.
- Files 11, 12 and this README were written by the coordinating session.
- Where an agent found the project's own summaries rounded in our favour, the file says so. For example, merge_exact is about 0.49× COIN, not 0.40–0.43×; the T-CHURN writes did not fully meet their bar; and T-CAUSE "ties" at 1–3 seeds mean "undecided".
