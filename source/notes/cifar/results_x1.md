# X1 results: is the readout bias weight scale? (2026-10-29, as declared in x1_plan.md)

**Setup.**
- The final ER-trained network of the T-S5 erp learner: CIFAR-100, 20 tasks, M = 2000, online, linear head.
- 5 seeds. The reserve (20 per class, never trained on) is used for fitting. Scoring is on the full test set.
- Gates: the self-test passes (P1–P5), and g1 passes on every seed (the loaded network reproduces its saved matched probe exactly).
- Outputs: results/x1/x1_s.json and results/x1/summary.txt.
- One self-test bug was found and fixed before trusting P4. A sed-style edit had turned P4's check into a comment, so P4 was neither printed nor counted. The fix was made and the self-test re-run before any real result was read. The real runs use only P4's accuracy function, which the re-run verified.

## Accuracy (%, mean ± SE over 5 seeds)

| Arm | Plain | + held-out bias correction |
|---|---|---|
| A linear (W, b) | 17.37 ± 0.43 | 22.70 ± 0.18 |
| B linear, no bias | 17.59 ± 0.40 | 22.70 ± 0.18 |
| C½ τ-norm 0.5 | 12.87 ± 0.51 | **24.60 ± 0.14** |
| C1 τ-norm 1 (cosine to head) | 9.08 ± 0.38 | 23.94 ± 0.33 |
| D cosine-NMC, reserve means | 21.41 ± 0.39 | 18.00 ± 0.28 |
| E Euclid-NMC, reserve means | 21.39 ± 0.40 | – |
| R LDA on reserve (20 per class) | **25.43 ± 0.36** | – |

A+bc equals the T-S5 "ER+bc" figure (22.7) and uses the same estimator. That is a consistency check.

## Declared readings

**Primary.** The scale share S = (C1 − A) / (A+bc − A) is **−1.61**. Every seed is between −1.1 and −2.0.
- Equalising the head's weight norms makes accuracy **worse** by 8.3 \*. It does not remove the bias.
- By the declared rule, scale is not the source of the bias. It points the opposite way.
- **My prediction (S between 0.4 and 0.8) was wrong.**

**Why.** The norm profile is reversed compared with Sezer's long-tail setting.
- In our replay learner the **oldest** classes have the **largest** weight norms: 1.99 for task 0, falling steadily to 1.14 for task 19.
- The Spearman correlation between norm and task index is −0.91.
- Old classes keep getting gradient through replay, so their weights keep growing.
- The recency bias therefore lives in the weight-feature alignment, not in the norms. Removing norms takes away the old classes' one advantage.
- The bias term b is tiny: B − A = +0.22 \*.

**Secondary.**
- **A fitted correction is still essential after the scale fix.** C1+bc − C1 = +14.9 \*.
- **Combined, it is a small real gain.** τ-norm 0.5 + bc beats A+bc by **+1.90 ± 0.06 \***. τ = 1 + bc gives +1.24 \*. So scale and offset are separate defects, and fixing both helps a little.
- **NMC on 20 reserve items per class** gives 21.4. That is below A+bc (−1.3 \*). Normalising features changes nothing (D − E = +0.02). Adding bc to NMC hurts (−3.4 \*): NMC has no recency offset to remove.
- **The best readout is a statistics readout.** LDA on the same 20 per class beats A+bc by +2.7 \* and beats C½+bc by +0.8 \*.

## What this changes

1. SYNTHESIS.md pattern D2 ("much of the bias is a scale profile") **does not hold for our replay learner**.
   - It holds in frozen-feature, long-tailed settings (Sezer) and for NMC vs linear on the same features (Łapacz).
   - It does not hold when replay keeps training old heads. There the norms favour old classes.
   - "Use a scale-free head instead of a fitted correction" is ruled out for this learner.
2. A fitted held-out correction stays necessary (E1 stands). Correcting norm as well adds +1.9 \*.
3. The strongest readout again uses class statistics: LDA on 20 fresh items per class. This supports the plan to make X2, statistic transport, the main target.

**Limits.**
- One learner type: online ER, M = 2000, GroupNorm CNN, CIFAR-100, 20 tasks.
- Post-hoc arms only. No trained cosine head; a trained one may behave differently.
- τ was not tuned beyond {0.5, 1}. Both values were declared in advance.
