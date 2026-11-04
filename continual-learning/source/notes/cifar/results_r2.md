# CIFAR-10 CPU tests, E2 and round 2: results (2026-10-29)

Plan and amendments: plan.md (sections "E2" and "Round 2"). Code: cifar.py (self-tests C1–C13), cifar2.py (K1–K10), cifar2_checks.py (X1–X5), qp_check.py (Q1–Q2), m2c2_check.py (M1–M4); all pass. Analysis: analyze_e2.py, analyze_r2.py, saved output results/r2/analysis.txt. Every arm uses RESV = 20, seeds 0–4, the same five streams, lr 0.03 and M = 500 unless stated.

The significance rule is the same as before: a paired mean ± SE that excludes 0, with all 5 seeds sharing the sign. A star (\*) marks a pre-declared comparison that meets it. Accuracies are final accuracy over all 10 classes (%).

## E2: choosing the weight and the correction size online, from the reserve

| Arm | Final |
|---|---|
| ER+bc_buf (α = 1) | 52.4 ± 0.8 |
| ER+bc(α̂) | 54.7 |
| RC-LDA | 54.0 ± 0.5 |
| MIX_half | 57.3 ± 0.6 |
| POE_half | 58.3 ± 0.5 |
| **MIX(ŵ, α̂)** | **58.7 ± 0.4** |

- **Q1.** A correction size chosen from data beats the formula: +2.3 \*. The chosen α̂ is mostly 0.7–0.8, which matches the oracle sweep in round 1 (about 0.7).
- **Q3.** A product of experts beats the probability average: +1.0 \*.
- **Q4, the headline.** The fully streaming-valid two-path system beats the better single expert by +4.0 \*. This closes the round-1 open problem: 200 reserve items are enough to set both the weight and α causally.

## Step 2: matched memory and a wide random expert

| | Comparison | Result |
|---|---|---|
| R1 | MIX_half(ER500, RC) vs ER+bc at M = 680 (same total memory) | 57.3 vs 54.2: **+3.2 ± 0.5 \*** |
| R2 | MIX_half(ER500, RCW) vs ER+bc at M = 1,200 | 59.4 vs 58.2: +1.2 ± 0.7 (4/5), not significant |
| R2' | MIX(ŵ, α̂)(ER500, RCW) vs ER+bc at M = 1,200 | 60.6 vs 58.2: +2.4 ± 0.8 (4/5), not significant |
| R3 | RCW-LDA vs RC-LDA | 57.8 vs 54.0: **+3.8 ± 0.7 \*** |

**Reading.** At the 512-d expert's memory cost, two paths clearly beat spending that memory on more buffer. At the 1024-d expert's cost (2.1 MB, the same as 700 extra images), a bigger buffer catches up, and the advantage is no longer significant. ER+bc gains 5.8 points from M = 500 to M = 1,200. So "statistics beat buffer images per byte" holds at the small size and is unproven at the larger one.

## Step 3: promotion (frozen copies of the learner plus streaming statistics)

| | Comparison | Result |
|---|---|---|
| S1 | P_last (version from the end of task 4) vs RC-LDA | 50.2 vs 54.0: **−3.8 ± 1.1 \*** (worse) |
| S2 | MIX3 = mean(ER+bc, RC, P_last) vs MIX_half | +0.6 ± 0.4 (4/5), not significant |
| S3 | BANK = mean(ER+bc, RC, all 4 versions) vs MIX_half | 60.0 vs 57.3: **+2.7 ± 0.5 \*** |

The individual versions score P0 55.4, P1 56.6, P2 55.1 and P3 50.2.

**Reading.** A single late frozen copy is *worse* than random features. Early versions are better: their features were learned on fewer classes, but they are not yet specialised to the last task. The whole bank helps (+2.7), at about 3.7 MB extra: four network copies plus four sets of 512-d statistics. That is more memory than R2's 1,200-image buffer, so S3 is not a memory-matched win.

## Step 4: protection on the two-path base

**EWC.** λ = 10 was chosen on seed 100 from {1, 10, 100}. λ = 1,000 diverged on seed 100 (the run died with a Cholesky failure), and it is excluded.

| | Comparison | Result |
|---|---|---|
| U1 | EWC+bc vs ER+bc | 53.1 vs 52.4: +0.7 ± 0.4 (3/5), not significant |
| U2 | MIX_half(EWC, RC) vs MIX_half(ER, RC) | +0.1 ± 0.2, null |

**M2c, as first ported (m2c=1).**

| | Comparison | Result |
|---|---|---|
| U3 | M2c+bc vs ER+bc | 26.3 vs 52.4: −26.1 ± 1.4 \* |
| U4 | MIX_half(M2c, RC) vs MIX_half(ER, RC) | −30.7 ± 1.5 \* |

This is a collapse, and I checked it before reporting it:
- The projection is feasible throughout: the worst relative violation is below 3e-10 on every seed, and 99% of steps are projected.
- The NNLS solver swap is not the cause. The NNLS solver matched the original to 2e-15 on 310 instances (qp_check.py). A direct comparison on seed 0 (1,596 steps) gives the same logit-scale trajectory under both solvers to the printed precision, for example 13,018.2 at step 1,200 with each. If anything, the old solver was the worse one: its worst relative violation was 0.99, while NNLS's was 1e-10.
- The test logits blow up, reaching 1.7e3 to 5.6e5 at the end (ER's stay around 16). A trajectory diagnostic on seed 0 shows the scale climbing steadily from the start of task 1: 5, then 365 by step 600, then 4e4 by step 1,550.
- **Cause: a porting difference.** The toy's buffer was filled only at task ends, so M2c there protected *past-task* items only. On CIFAR the reservoir holds current-task items from the first batch, so m2c=1 forbids any first-order margin decrease on the items currently being learned, including during task 1. Satisfying that from step to step drives the logit scale up without bound. U3 and U4 are therefore results about that variant, not about M2c as the toy tested it.
- In the mixtures, the inflated logits also make float32 softmax underflow (a divide warning in the analysis), so MIX_half and POE_half for m2c=1 are unreliable. MIX(ŵ, α̂) puts almost all its weight on RC (53.9).

**M2c, faithful to the toy (m2c=2: pool drawn from past-task items only).** This variant was added at 07:41 UTC, before any m2c=2 result, and its checks M1–M4 pass.

| | Comparison | Result |
|---|---|---|
| U3' | M2c(faithful)+bc vs ER+bc | 39.7 vs 52.4: **−12.8 ± 1.1 \*** |
| U4' | MIX_half(M2c faithful, RC) vs MIX_half(ER, RC) | 40.8 vs 57.3: **−16.5 ± 0.8 \*** |

- The projection is feasible: the worst relative violation is at most 6e-11.
- 80% of steps are projected (none in task 1, as required).
- The test logits still inflate after task 1: from about 12 at the end of task 1 to 270–1,360 at the end of task 5, against about 16 for ER.
- Under the streaming-valid weight choice, the mixture again moves almost all its weight to RC (54.3).

**Verdict.** Zero-slack margin protection loses badly on CIFAR in both variants. Forbidding any first-order decrease of old items' margins still forces the logit scale up while new classes are learned, and accuracy falls.
- Caveats: the toy-best settings were used without tuning (as declared), and the plan used plain SGD at lr 0.03, while the toy used Adam at 1e-3.
- By the audit's stopping rule, M2c is **removed at this scale**.

**Compute.** EWC, M2c and ER all pass the same 99,590 images through forward and backward passes. Their extra costs are not counted in that figure: EWC's Fisher passes, and M2c's Jacobian columns and QP. Wall times were measured under different machine loads, so they compare only roughly:
- ER: 175 s, and EWC: 167 s, both in the same batch of runs.
- m2c=1: 740–1,350 s, and m2c=2: 740–2,610 s, each with four or more jobs sharing four cores. Under similar load, an ER-sized run (ER at M = 680) took 300 s, so M2c costs roughly 2.5–4.5× ER.

## Step 5: three passes per task (lr 0.03, not retuned)

| | Comparison | Result |
|---|---|---|
| V1 | ER+bc(3 epochs) vs RC-LDA | 53.2 vs 54.0: −0.9 ± 1.1, not significant |
| V2 | MIX_half(3 epochs) vs the better single expert | 58.0: **+3.3 ± 0.1 \*** |
| V3 | ER+bc, 3 epochs vs 1 | +0.8 ± 0.4 (4/5), not significant |

**Reading.** The two-path gain survives three passes per task almost unchanged. Extra passes barely help ER here, which may be because lr was not retuned.

## What this changes

1. **The two-path system is now fully streaming-valid and still wins.** MIX(ŵ, α̂) gives +4.0 \* over the best single expert, and R1 gives +3.2 \* at matched memory. This is the strongest result so far.
2. **At larger memory budgets the case is open.** R2 is not significant. A bigger buffer is a strong competitor, and the spec should say so.
3. **Frozen copies are not free stable experts.** A late copy is worse than random features (S1). A bank of versions helps, but costs more memory than an equivalent buffer.
4. **Protection adds nothing measurable on top of two paths.** EWC is null, and M2c is harmful in both variants (−12.8 \* faithful, −26.1 \* as first ported). It is removed at this scale.
5. **Bias correction should use α̂, not 1.**

## Caveats

- One small CNN, an online setting, five seeds. Several comparisons are made, so treat single borderline results cautiously.
- EWC λ was chosen on one seed with a three-value grid. λ = 1,000 diverged.
- The NNLS solver replaced L-BFGS-B mid-round. It was validated before any M2c result existed, and that was declared in plan.md.
- The m2c=1 runs were the second attempt: the first, with the slower solver, was stopped at about 40%, also declared.
- m2c=2 is a post-hoc correction of a porting difference, added after seeing m2c=1. It was declared before it was run, and both variants are reported.
- The container restarted once (04:33 UTC). Every queue is resumable, and no result came from a partial run.
