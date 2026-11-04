# Sleep-inspired tests T-S1 to T-S4 on Split CIFAR-10 (pre-declared 2026-10-29, before any sleep result)

Source of the ideas: Behrouz et al., "Language Models Need Sleep" (arXiv 2606.03979), reviewed in the Claude Doc "Sleep framework (Behrouz et al.): a rigorous read". The code is sleep.py; its self-tests are sleep_selftest.py.

## What is being tested, and the honest limits of the analogue

The paper's consolidation step does three things: it adds a new low-rank expert to a slower block, distils the current model (the teacher) into "slow weights plus new expert" using the model's own samples, and resets the fast block. A CNN classifier has no frequency hierarchy, and it cannot generate its own inputs, so this is an **analogue**, not a reproduction:

- **Slow weights S.** The network at the end of task 1, plus every earlier consolidated expert.
- **Fast update F.** What plain ER has learned since the last sleep. The current net is S + F.
- **Sleep,** at the end of tasks 2 to 5:
  1. The teacher is the current net.
  2. The student is S plus a new expert: a rank-r additive update B·A on every conv and linear weight, plus full deltas on biases and GroupNorm affines. B starts at 0, so the student starts exactly at S, which is the reset of F.
  3. Only the expert is trained, by KL(teacher ‖ student) on "dreams", using Adam.
  4. The net becomes the merged student, and ER continues from it.
- **Dreams.** An image classifier cannot sample its own inputs. The analogues are:
  - buffer images;
  - mixup of buffer images with different labels, our analogue of the paper's random-expert mixing;
  - iid Gaussian noise in input space.
- **No routing.** The merged expert is additive, so it adds no parameters at inference. This test therefore isolates *consolidation by self-distillation into a constrained update*, not capacity growth. Capacity is tested separately in T-S4 (a wider net) and was tested in round 2 (the promotion bank).

Not reproduced: the RL imitation reward, SEAL-style dreaming with task-metric selection, and the language-model setting.

## Setup

- Same as round 2: RESV = 20, seeds 0–4, the same streams, online ER at lr 0.03, M = 500, plain SGD (no optimiser state to go stale).
- The sleep code uses its own RNGs, so the replay sampling sequence is identical to ER's. Arms differ only through the weights.
- ER and RC-LDA are reused from results/e2. Self-test Z1 shows that sleep.py at sleep=0 equals cifar.py's ER exactly. The analysis also checks that every sleep arm's predictions are bit-identical to E2's ER through the end of task 2.

**Arms** (results/sleep):

| Arm | rank | dreams |
|---|---|---|
| slp_r4_buf | 4 | buffer |
| slp_r16_buf (main) | 16 | buffer |
| slp_full_buf | full | buffer |
| slp_full_mix | full | mixup |
| slp_full_noise | full | noise |
| erw | none (no sleep) | width 64: 1024-d features, about 4× the parameters and compute; lr 0.03, not retuned (declared) |

**Distillation hyperparameters.** SGD with momentum 0.9, dlr ∈ {0.01, 0.03} × epochs ∈ {10, 30}, with each epoch one pass over 500 dreams in batches of 64. (This originally said Adam with dlr ∈ {1e-3, 3e-3}; see the amendment below.)
- They are chosen on seed 100 only, for slp_r16_buf, by final ER+bc_buf accuracy on the **reserve** (held out of all training). The test set is not used.
- The chosen pair is used for every sleep arm.

## Metrics

- **Final accuracy** over all 10 classes: raw and bc_buf. Bias correction is computed by the student on the buffer, exactly as for ER.
- **At every sleep:**
  - teacher–student argmax agreement over seen classes, and KL(teacher ‖ student), on three sets: the buffer, the reserve items of seen classes (held out), and the test items of seen classes;
  - the accuracy cost of the sleep, i.e. student minus teacher test accuracy (bc_buf, seen classes).
- **Compute:** images through forward and backward passes, plus wall time.

## Pre-declared comparisons

Each comparison is paired over 5 seeds. A significant result (\*) needs a mean ± SE that excludes 0 with all 5 seeds sharing the sign.

| | Comparison | Question | My prediction |
|---|---|---|---|
| T1 | slp_r16_buf+bc vs ER+bc | Does Sleep-style consolidation by itself reduce forgetting? | No gain: Δ ≤ +0.5 or negative |
| T1b | slp_full_buf+bc vs ER+bc | The same without compression | No gain |
| T1c | MIX_half(slp_r16_buf, RC) vs BANK (60.0, round 2) | Does consolidating into new capacity beat stored frozen copies plus statistics? Memory is reported. | BANK higher |
| T2 | Held-out gap = agree_buf − agree_resv, per sleep | Is self-distillation lossy off the dream set (the gap the paper never measures)? "Lossy" means a mean gap above 5 points, 5/5 seeds. | Lossy |
| T2b | Sleep cost (student − teacher accuracy) | Does each sleep lose accuracy? | Negative, and larger at lower rank: r4 < r16 < full |
| T3 | agree_resv and final accuracy: mix vs buf, noise vs buf (full rank) | Do more diverse synthetic inputs transfer more than replayed items? | mix ≥ buf > noise |
| T4 | erw raw vs ER+bc | Capacity without correction vs correction without capacity | ER+bc higher by more than 5 |
| T4b | erw+bc vs ER+bc | Does capacity add on top of the correction? | Small gain, at about 4× compute |

**Interpretation rules, fixed in advance:**
- If T1 is significantly positive, Sleep-style consolidation is a forgetting remedy here, and the spec gains a candidate rule.
- If T4 holds (ER+bc > erw raw by more than 5), the paper's "forgetting = insufficient capacity" framing is contradicted on this benchmark. Most forgetting here is then readout bias.

T-S5 (growth budget, Split CIFAR-100, 20 tasks) is declared separately after T-S1 to T-S4, because it needs a new dataset and a promotion-bank budget design.

## Reliability gates

All of these must pass before any result is trusted:
1. sleep_selftest.py Z1–Z13 all pass:
   - equivalence to ER, and the timing of the first sleep;
   - determinism, and causality across a sleep;
   - the reserve is disjoint from the buffer;
   - dep = 0 resets the net to S exactly;
   - the teacher is untouched, and the rank of each consolidated change is at most r;
   - a planted null (nothing to consolidate gives KL 0);
   - a planted effect (full-rank distillation fits the teacher on its dreams);
   - a leak detector (reserve dreams raise reserve agreement);
   - mixup exactness, noise independent of the data, the agreement metric on a hand example, and the width.
2. An independent code review of sleep.py and sleep_selftest.py before the 5-seed runs.
3. Seeds and configs are logged in each run's JSON line.
4. The analysis checks that each arm is identical to ER through task 2.

## Amendment, 2026-10-29 07:53 UTC (self-test first run; no sleep result exists)

The first self-test run failed three checks, and all three were real problems:
- **Z5.** With nothing to consolidate, Adam still moved the weights: KL went 0 → 0.033. Adam normalises near-zero gradients into steps of size lr, so it drifts away from an exact optimum.
- **Z6a.** Distillation made the planted student *worse*: agreement 0.89 → 0.80. The planted case was also too weak, since the teacher was S plus small noise.
- **Z11.** The test itself was wrong: it used a 3-class hand example against a 10-class mask.

Fixes:
- The optimiser is now SGD with momentum 0.9, which makes no step at a zero gradient. The hyperparameter grid changed to match.
- The Z6 planted teacher is now S plus 300 real SGD steps on task-1 items.
- Z11 now uses 10-class logits.

All self-tests are rerun on the fixed code.

## Amendment, 2026-10-29 07:55 UTC (independent review; no sleep result exists)

The review verdict was "safe after fixes". Changes:
- **Z11 crash and the Z5 failure:** fixed as in the amendment above.
- **New test Z14.** It runs the task-end path: task 1 equals ER; at the first sleep, the saved teacher logits and b_buf are bit-equal to ER's; the student's saved logits differ; and the npz holds the e1_T_* keys.
- **Z9b.** Causality is now tested *across* a sleep with label-using mix dreams, with a flip inside task 3. The first sleep and the predictions before the flip must be identical, and the second sleep must differ.
- **Z7** is no longer tautological. Synthetic images with channels (v, v², v³) make every dream's pair and λ recoverable from the image alone.
- **Z13** now goes through run(width=64).
- **Declared fidelity note (review item 4).** The rank cap is min(r, out, fan_in) per layer. The head is 10 × 512, so at r = 16 the head update is full rank (10), while at r = 4 it has rank 4. **So r16 compresses the body but not the readout.** The r4 vs r16 contrast therefore partly measures head compression. Per-layer ranks are logged in every sleep's stats.
- **Compute (review item 9).** The ER images, the student forward+backward images, the teacher forward-only images and the sleep wall time are now logged separately. Evaluation time is not counted as sleep time.
- **Leak guard.** The 'resv' dream is refused from the command line.
- **Selection script (review item 12).** pick_sleep.py takes final reserve accuracy = argmax(e4_resv_logits − e4_b_buf) against resv_y, over 200 items on seed 100. That is noisy, as declared, and ties are broken toward fewer epochs, then the lower dlr.
- **Analysis gate.** The "identical to ER through task 2" check compares e1_T_test_logits (the teacher) to ER's e1_test_logits.

## Amendment, 2026-10-29 08:33 UTC (after the seed-100 tuning runs, before any seed 0–4 sleep run)

Seed-100 reserve accuracy (ER+bc_buf, final, 200 items) for the four grid points:

| dlr | epochs | Reserve accuracy |
|---|---|---|
| 0.01 | 10 | 0.245 |
| 0.01 | 30 | 0.405 |
| 0.03 | 10 | 0.235 |
| 0.03 | 30 | **0.445** |

The best point sits at the **corner of the grid**, with the most epochs and the highest dlr. The logs show distillation still improving at the last epoch. Each sleep starts from a large KL, about 4–5 at the first batch, because resetting the fast update erases a whole task's learning. Reaching KL 0.04–0.09 then takes all 30 epochs.

Choosing from this grid would therefore handicap the sleep arms through under-distillation, not through the mechanism. So the grid is **extended on seed 100 only**, with the same selection rule:
- dep ∈ {30, 100} × dlr ∈ {0.03, 0.1};
- if the best point is again at the edge, one more step: dep = 300 or dlr = 0.3.

This only gives the method more room. Seeds 0–4 have not been run with any setting.

**Result of the extended grid (seed 100, reserve accuracy):** (0.03, 30) 0.445, (0.03, 100) 0.440, (0.1, 100) 0.390, (0.1, 30) 0.345. The choice stays **dlr 0.03, 30 epochs**, now an interior point. At 100 epochs the student fits the buffer almost perfectly (agreement 0.98–1.00) but not the reserve (0.80–0.91), so more epochs overfit the dream set. The 5-seed runs started at 08:45 UTC with this setting.
