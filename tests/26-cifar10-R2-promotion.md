# 26. CIFAR-10 round 2, promotion: frozen copies of the learner as stable experts

**Date:** 2026-10-29. **Source files:** as test 25.

## Why we ran it
Random features are a weak stable path. An obvious upgrade: at each task end, freeze a copy of the learner ("promote" it) and accumulate streaming statistics on its features from then on.

## What we did
- S1: last frozen version (end of task 4) with its statistics vs RC-LDA.
- S2: MIX3 = mean(ER+bc, RC, P_last) vs MIX_half.
- S3: BANK = mean(ER+bc, RC, all 4 versions) vs MIX_half.

## What we got
- S1: 50.2 vs 54.0, **−3.8 \*** (a late copy is worse than random features).
- Individual versions: P0 55.4, P1 56.6, P2 55.1, P3 50.2.
- S2: +0.6 (4/5), not significant.
- S3: 60.0 vs 57.3, **+2.7 \***, at ~3.7 MB extra (more than R2's 1,200-image buffer, so not memory-matched).

## What we inferred
Frozen copies are not free stable experts; a bank helps but costs more memory than an equal buffer.

## Why we think it occurred (and a later correction)
At the time I read S1 as late features being over-specialised to the last task. **Later shown to be confounded (test 35, A1):** a version frozen at task t gets statistics for *earlier* classes only from the buffer (~20 items each), so late versions are judged with tiny samples. On CIFAR-100, matched probes showed the features improve monotonically. The same confound applies here, so S1 should not be read as feature decay.

## How reliable this test is
- **Numbers reliable; the reading of S1 was wrong** (readouts not matched).
