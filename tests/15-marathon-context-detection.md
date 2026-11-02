# 15. Marathon T7: context detection and the L_min threshold

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
The spec's M3 component detects context switches with a sticky filter. Test 02 (N5) derived an exact detectability threshold, log L_min = 0.166, and the spec treated it as necessary. This tested detection and the threshold.

## What we did
Ran the detector with real LDA label evidence, a null control (q = 0), and a noisy-evidence scan below L_min with 2 and 3 states.

## What we got
- Real evidence: detection in 1–4 batches, no false switches. The null control never switches.
- Below the two-state L_min: detection succeeds 11/20 (mean log L 0.126, 2 states) and 17/20 (3 states).

## What we inferred
L_min is a constant-evidence boundary, **not a necessary condition** under noisy evidence. The spec wording was changed. Multi-state helps.

## Why we think it occurred
The threshold was derived for evidence that is the same every batch. Noisy evidence sometimes delivers strong batches, which can carry the filter across even when the average is below threshold.

## How reliable this test is
- **Medium.** Simulated evidence; the switch detection was never tested on a real stream with recurring contexts.
