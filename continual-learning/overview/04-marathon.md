# 04. The sharp-test marathon (toy + Split-MNIST)

**Source:** [ewt.md](../source/notes/ewt.md) §4 (A5). **Code:** [`mar.py`](../source/notes/marathon/mar.py), [`mar2.py`](../source/notes/marathon/mar2.py), self-tests S1–S11 and N1–N16, two independent reviews.

## Why we did it
After 03 pointed at the readout, many separate claims of the spec and the literature could be tested cheaply, each with a sharp prediction:
- readout bias;
- logit adjustment;
- fixed vs learned features;
- stale statistics;
- snapshots;
- repetition;
- EWC;
- context detection.

Small, fast settings let us kill or keep each idea before moving to real images.

## What we did
- Toy A: separation 0.6, 5 items per class.
- Toy B: separation 0.8, 20 per class.
- Split-MNIST: MLP, 5 per class.

5 seeds each, paired.

## What we got
| Test | Result | Status |
|---|---|---|
| Readout bias: per-class bias correction on the same ER weights | 41.5 → 57.7 (toy A), 79.9 → 83.4 (toy B), 50.7 → 71.2 (MNIST), all \* | Holds |
| Logit adjustment by stream prior (no fitted parameter) | Recovered 66–160% of the correction gain; correctly predicted no bias under ER-ACE | Partly withdrawn in [05](05-marathon-revisions.md): right direction, wrong size |
| Fixed vs learned features, oracle LDA (MNIST) | ER features 86.7, random ReLU 1024-d **94.0**; every trained method ≤ 72.2; task 1 alone destroys separability (87 → 75) | Revised in 05: only with full-history statistics |
| Readout sample size | Buffer readouts plateau ~71 vs oracle 89; the gap is in the class means | Revised in 05: partly buffer memorisation |
| Stale statistics | Stored class means go stale catastrophically (~20%); shared-affine correction on MNIST 68.3 vs 86.7 | Holds (but the affine failure did not replicate on CIFAR, [13](13-x2-statistics-transport.md)) |
| Head-row snapshots | "Restore saves time" was a readout artefact; restored rows are worse once bias-corrected (−25.3 \*, −56.7 \*) | Claim withdrawn |
| Repetition | Small savings on recurrence: +3.1 \* toy, +8.2 \* MNIST | Holds |
| Boundary shock | Centred mean init raises the dip, never improves final accuracy | Killed |
| EWC fresh vs stale Fisher | Fresh wins at small λ; much worse at λ ≥ 10 on MNIST | Holds only at small λ |
| M2c rival coverage | Adding newly born labels as rivals: 44.6 → 54.1, still below free fixes, ~30× ER compute | M2c still not justified |
| Context detection | Detected in 1–4 batches, no false switches; worked below L_min | L_min rewritten |
| Predicting interference | Feature similarity 63–72%, raw input as good | Not an insight |
| End-of-stream "sleep" | ≈ bias correction | Same debiasing |
| Centring the head | Bias moves to another term | Failed |

## What we inferred
- Bias correction is the single biggest, most reliable gain.
- Class statistics go stale when the encoder moves.
- Fixed random features can carry a lot of information if you have full statistics.

## Why we think it happened
- Bias correction directly cancels the 16:1 pull described in 03.
- Stored means go stale because they live in the coordinates of an encoder that no longer exists.
- Random wide ReLU features help on MNIST because MNIST is nearly linearly separable in a wide random basis. That did not transfer to real images ([06](06-cifar10-e0-e1.md)).

## Caveats
After a container restart, T6 was relaunched without the toy-A settings, so it ran the saturated toy. The runs were caught, archived as invalid and re-run. A guard now checks that ER reproduces S1.
