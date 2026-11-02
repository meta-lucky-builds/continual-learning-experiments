# 09. Sleep-paper tests T-S1 to T-S4 (CIFAR-10)

**Source:** ewt.md §9. Paper: Behrouz et al., "Language Models Need Sleep" (arXiv 2606.03979). **Code (not uploaded):** `sleep.py`, 20 self-tests, two review rounds.

## Why we did it
The paper proposes periodic "sleep": compress what the network learned into a new expert by distillation on self-generated "dreams", then reset fast weights. If that consolidation works, it is an alternative to the readout fixes. It also directly challenges our finding that forgetting is a readout problem.

## What we did
An image-classifier **analogue**, not a reproduction: no language model, no self-generated inputs, no RL. At each task end, distil the network's outputs on "dreams" (buffer images, mixup or noise) into a rank-r or full update, reset, continue. 5 seeds.

## What we got
| | Comparison | Result |
|---|---|---|
| T1 | Sleep (rank 16)+bc vs ER+bc | 44.0 vs 52.4, **−8.4 \*** |
| T1b | Sleep (full rank)+bc vs ER+bc | **−8.4 \*** (compression is not the cause) |
| T1c | MIX(Sleep, RC) vs BANK | **−6.0 \*** |
| T2 | Buffer agreement − reserve agreement | 6.3 to 8.5 points, all 5/5: **lossy** |
| T2b | Cost per sleep | −3.6 to −5.7 points each, every arm |
| T3 | Mixup vs buffer dreams | Null |
| T3 | Noise vs buffer dreams | **−30.6 \*** |
| T4 | ER+bc vs 4× wider ER (raw) | **+10.1 \*** |
| T4b | Wide ER+bc vs ER+bc | +1.4 \* at ~4× FLOPs |

Later control (A3, in [11](11-audit-corrections.md)): an exact copy of the teacher plus the same reset gives 52.2 vs ER's 52.4. **All** of Sleep's loss comes from approximate transfer through distillation on 500 images.

## What we inferred
- Sleep-style self-distillation is not a forgetting remedy here; it loses accuracy.
- Forgetting on CIFAR-10 is a readout problem, not a capacity problem. Bias correction is worth +10; four times the width is worth almost nothing raw.

## Why we think it happened
Distillation on 500 buffer images matches the teacher on those images (85–93% agreement) but not elsewhere (78–85% held out). Each sleep throws away the part of the function the dreams did not cover, and the losses compound over tasks. Noise dreams are far out of distribution, so the transfer is worse still.

## Caveats
- The analogue does not test the paper's upward distillation or RL dreaming, so it does not refute the paper's main claim.
- Bugs caught by self-tests: Adam drifted away from an exact optimum (KL 0 → 0.033), so we switched to SGD; weak planted tests; one test with a wrong mask size; a too-tight rank tolerance.
