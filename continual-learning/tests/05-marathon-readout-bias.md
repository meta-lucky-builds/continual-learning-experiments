# 05. Marathon T2/F4: is forgetting a readout bias?

**Date:** 2026-10-28. **Source files:** `source/notes/marathon/{plan.md, plan2.md, results.md, mar.py, mar2.py, analyze*.py}`, runs in `source/notes/marathon/results/`.

**Shared marathon settings (tests 05–21):** toy A (SEP 0.6, 5 items/class, 4 replay), toy B (SEP 0.8, 20 items/class, 8 replay), Split-MNIST (MLP 784-256-256-10, 5 items/class, 4 replay). 5 seeds, paired. Harnesses pass self-tests S1–S11 and N1–N16 and had two independent code reviews.

## Why we ran it
Test 04 suggested that most toy forgetting is a bias toward the newest classes in the output layer. If true, the spec should fix the readout rather than protect parameters. This test checked it on two toys and on Split-MNIST, and asked *which* part of the head carries the bias (row norms, direction, or a common offset).

## What we did
Took the same trained ER weights and changed only the readout:
- per-class bias correction (subtract each class's mean logit, estimated on held-out data);
- weight aligning (WA, equalise row norms);
- nearest-class-mean (NCM) and buffer refit;
- F4: decompose ER's error gap into a common-mode term and an item-specific term.

## What we got
- Bias correction: **41.5 → 57.7** (toy A), **79.9 → 83.4** (toy B), **50.7 → 71.2** (MNIST), all \*.
- WA does nothing (gamma ≈ 1).
- Bias correction, NCM and buffer refit all fix most of it, and every method plus its best buffer readout reaches the same ceiling.
- F4: on MNIST, the common term is 4.3 of ER's 4.56 gap.

## What we inferred
Forgetting in these settings is mostly a common-mode logit offset toward the newest classes, not lost features and not a norm problem.

## Why we think it occurred
Softmax cross-entropy fits the log-odds of the *training* distribution. In the last task, new classes are 16/20 of every batch and old classes share 4/20, so the model learns logits offset by about log(0.4/0.025) ≈ 2.8 nats on a shared feature direction. At test time classes are balanced, so the offset is pure bias.

## How reliable this test is
- **High within this setting:** large effects, 5/5 seeds, reviewed harness, replicated later on CIFAR-10 (+9.8) and CIFAR-100 (+5.3).
- The common ceiling is a readout sample-size limit (test 08), and part of it is buffer memorisation (test 21).
- Known in the literature (logit adjustment, Menon et al. 2021; online CL variants). What we add is direct confirmation on our harness.
