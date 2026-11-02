# 17. Marathon R13: an end-of-stream "sleep" on the buffer

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
A cheap consolidation idea: after the stream, train for 200 steps on the buffer alone. Does it add anything beyond bias correction?

## What we did
200 buffer-only steps, full network vs head only, after ER and after ER-ACE.

## What we got
- After ER: ≈ biascorr (toy A +1.1, MNIST +1.1, both not significant).
- After ER-ACE: beats biascorr (+2.8 \* MNIST); toy B +1.4 \*.
- Full network beats head only (+4.6 \* toy A, +11.3 \* MNIST).

## What we inferred
A modest win, mostly the same debiasing as bias correction.

## Why we think it occurred
Training on the class-balanced buffer removes the prior offset (test 05), which is what bias correction does in one step.

## How reliable this test is
- **Medium.** Small effects; MLP scale. The CIFAR sleep tests (tests 30–33) are the stronger evidence on consolidation.
