# Sleep-inspired tests T-S1 to T-S4 on Split CIFAR-10: results (2026-10-29)

Plan, amendments and predictions: sleep_plan.md, all written before the runs they govern. Code: sleep.py. Self-tests: sleep_selftest.py Z1–Z14, all 20 checks pass (results/sleep_selftest_v4.log), plus sleep_z4_control.py, which shows the rank test can fail. An independent review gave the verdict "safe to run as is" after two rounds of fixes. Analysis: analyze_sleep.py, output in results/sleep/analysis.txt.

**Gates passed.** Every run's config matched the declaration. Every sleep arm is bit-identical to the E2 ER run through the end of task 2, in its stream predictions, its task-1 logits and the teacher at the first sleep. So any difference comes from sleeping, not from the harness. There was no machine restart during the runs: the boot time is still 04:33 UTC.

**Setting.** As in round 2: RESV = 20, seeds 0–4, online, lr 0.03, M = 500. Distillation is SGD with momentum, dlr 0.03 and 30 epochs over the dreams, chosen on seed 100 by reserve accuracy. The first grid's best point was at its edge, so the grid was extended before choosing; see the plan amendment.

## Final accuracy (%, 10 classes, mean ± SE)

| Arm | raw | +bc_buf | MIX_half(·, RC) | MIX(ŵ, α̂) |
|---|---|---|---|---|
| ER (E2) | 42.6 | **52.4** | 57.3 | 58.7 |
| Sleep, rank 16, buffer dreams | 40.0 | 44.0 | 54.0 | 54.9 |
| Sleep, rank 4, buffer dreams | 39.3 | 44.6 | 54.1 | 54.7 |
| Sleep, full rank, buffer dreams | 41.1 | 44.0 | 53.5 | 55.5 |
| Sleep, full rank, mixup dreams | 39.2 | 44.0 | 53.8 | 54.9 |
| Sleep, full rank, noise dreams | 11.7 | 13.3 | 53.7 | 54.0 |
| ER at width 64 (no sleep) | 42.3 | 53.8 | 58.3 | 59.4 |

## Pre-declared comparisons

| | Comparison | Result | Prediction |
|---|---|---|---|
| T1 | Sleep r16+bc vs ER+bc | **−8.4 ± 2.8 \*** | No gain: confirmed, and worse than predicted |
| T1b | Sleep full rank+bc vs ER+bc | **−8.4 ± 0.5 \*** | No gain: confirmed |
| T1c | MIX_half(Sleep r16, RC) vs BANK (round 2) | **−6.0 ± 1.1 \*** | BANK higher: confirmed |
| T2 | Held-out gap (buffer agreement − reserve agreement), r16 | **6.6 ± 1.7 points, 5/5: LOSSY** | Lossy: confirmed |
| T2b | Accuracy cost of each sleep (student − teacher) | r16 −5.7, r4 −5.7, full −3.6, mix −3.8: every arm negative on 5/5 seeds | Negative: confirmed. The ordering r4 < r16 < full is **not** confirmed: r4 = r16 (+0.0 ± 0.4) |
| T3 | Mixup vs buffer dreams | reserve agreement +0.7 ± 0.7, final +0.0, both null | mix ≥ buf: tie |
| T3 | Noise vs buffer dreams | reserve agreement −46 \*, final −30.6 \* | buf > noise: confirmed |
| T4 | ER+bc vs wide ER, raw | **+10.1 ± 2.0 \*** (more than 5: yes) | Confirmed |
| T4b | Wide ER+bc vs ER+bc | +1.4 ± 0.4 \*, at roughly 4× the FLOPs (420 s vs 175 s wall time, under different machine loads) | Small gain: confirmed |

## What it means

1. **Sleep-style consolidation by self-distillation hurts here. It is not a forgetting remedy.** Each sleep compresses a task's learning into a new expert by matching the network's own outputs on 500 buffer images, then resets the fast weights. Every sleep loses 3.6–5.7 points (5/5 seeds), and the losses add up to −8.4 at the end.
   - Full rank does no better than rank 16 at the end (−8.4 both), so **the loss is not compression**. It comes from distilling only on what the dreams cover.
2. **The loss the paper never measures is real and large.**
   - The student matches the teacher on 85–93% of the buffer images it trained on, but only 78–85% of held-out reserve images. That is a gap of 6–9 points, 5/5 seeds.
   - Test-set agreement equals reserve agreement, so the reserve is a fair proxy.
   - In the language-model setting, the analogue is that a model distilled on its own samples copies what those samples cover and loses the rest.
3. **Dream content matters, but more diversity didn't help.**
   - Mixup closes the held-out gap (1.5 points), but only by fitting the buffer worse; final accuracy is unchanged.
   - Noise dreams are catastrophic (13.3%). Consolidation needs inputs from the data distribution.
4. **Forgetting here is mostly readout bias, not missing capacity (T4).**
   - Four times the channels gives no raw gain (42.3 vs 42.6).
   - Bias correction alone gives +9.8 at the original width.
   - With correction, extra width adds only +1.4.

   This contradicts the paper's "forgetting = insufficient capacity" framing on this benchmark.
5. **The stable path absorbs the damage.** With RC mixed in, the sleep arms recover to 53.5–55.5, but they stay below MIX with plain ER (57.3–58.7).

## Caveats (honest)

- **This is an analogue, not a reproduction.**
  - A CNN classifier can't generate its own inputs, so the dreams are buffer images, mixup or noise.
  - The paper's RL imitation reward and SEAL-style dreaming are not included.
  - The merged expert adds no inference-time capacity (no routing). Capacity is tested separately (T4, and the round-2 bank).
- **Distillation hyperparameters** were chosen on one seed's 200 reserve items, from a 7-point grid. A much larger dream set, or many more epochs with a better regulariser, might narrow the held-out gap.
  - Against that, at 100 epochs the student already fits the buffer almost perfectly while the reserve gap grows, so more fitting alone overfits.
- **One small CNN, online, five seeds.** The width-64 arm uses the same lr (0.03), not retuned.
- **Compute.** Each sleep arm adds 60,000 student forward-and-backward passes and 60,000 teacher forward passes to ER's 99,590 images, about 2.2× ER's image passes. The wide net does the same number of image passes as ER at about 4× the FLOPs per image.
