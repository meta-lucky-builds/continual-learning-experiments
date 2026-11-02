# 12. X1: is the readout bias just head weight scale? (CIFAR-100)

**Source:** ewt.md §15. Plan `x1_plan.md`; results `results_x1.md`.

## Why we did it
The geometry literature feed claimed much of recency bias is a profile of head weight norms: new classes get bigger weights. If true, equalising norms (τ-norm) would be a parameter-free fix, which would undercut our need for a fitted correction.

## What we did
Post-hoc readouts on the final network of the T-S5 ER learner, 5 seeds. Fitting used the 20-per-class reserve (never trained on); scoring used the full test set. Gate: the loaded network reproduces its saved probe exactly.

## What we got
| Arm | Plain | + held-out bias correction |
|---|---|---|
| Linear head | 17.37 | 22.70 |
| τ-norm 0.5 | 12.87 | **24.60** |
| τ-norm 1 (equal norms) | 9.08 | 23.94 |
| cosine NMC, reserve means | 21.41 | 18.00 |
| **LDA on reserve (20 per class)** | **25.43** | – |

Equalising norms made accuracy **worse**: −8.3 \*. Our prediction (scale explains 40–80%) was **wrong**.

## What we inferred
The norm story does not hold for our learner. A fitted held-out correction stays necessary, and the best readout is again a class-statistics (LDA) readout.

## Why we think it happened
Under replay, the oldest classes have the *largest* norms (1.99 for task 0 down to 1.14 for task 19, Spearman −0.91), because replay keeps training them. The bias lies in how weights align with features, not in their lengths. Equalising norms shrinks the old classes and hurts them.

## Caveats
X1 was not independently reviewed. A trained cosine head was not tested, only post-hoc readouts.
