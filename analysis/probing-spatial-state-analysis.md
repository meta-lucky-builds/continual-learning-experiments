# "Probing Spatial State Representations in Visual Foundation Models" (arXiv 2610.40636): read and relevance to H-JEPA

Paper: Yin, Cai, Wang, Liang, Yang, Liu (NYU, NTU, HKUST, CUHK), 29 Oct 2026. https://www.alphaxiv.org/abs/2610.40636

**Coverage.** I read the full text: the main body, all appendix prose and the tables. I skipped the reference list. Statements marked "inference" are my reasoning and were not tested by the paper.

**What kind of paper it is.** This is a probing/benchmark study, not a method paper. It does not loop anything, does no test-time training, and does no planning in closed loop.

## 1. Setup

- **Ten frozen representations:**
  - image: DINOv2-L;
  - video: V-JEPA2-L, VideoMAE-L;
  - 3D: Fast3R;
  - video generators: Wan2.1-1.3B, CogVideoX-5B, MiniMax-H3, LingBot-video;
  - world model: Aether;
  - multimodal: Qwen2.5-VL-7B.
- **Data:** posed ScanNet++ egocentric trajectories through *static* indoor scenes. Training and validation scenes are disjoint (69 validation scenes, 276 windows).
- **Inputs:** causal prefixes of 8, 12, 16 or 24 frames, each encoded independently. So this measures what a representation of the history affords, not an online state update.
- **Probes:** shallow probes trained on frozen features. There are four levels:

| Level | Test | Metric |
|---|---|---|
| L1 Spatial structure | Depth, camera pose, instance identity, view consistency | Loss, AUC@30, balanced accuracy |
| L2 Object persistence | Bearing of an object that has left the view | Angular error |
| L3 Predictive update | Predict the future feature given the camera motion (direct, h ∈ {1,2,4}); recurrent path integration | Global R@1 over 828 candidates |
| L4 Actionability | Pick or predict a 9-D camera motion toward a goal frame | Accuracy, positive progress |

## 2. Findings

1. **No model wins everything.**
   - Fast3R is best on camera pose (AUC@30 0.387, against 0.041 for DINOv2) and on persistence (32.8°).
   - DINOv2 is best on direct prediction (R@1 0.237).
   - V-JEPA2 is best on path integration (0.159).
2. **Persistence tracks camera-pose readout, not visible-scene quality.**
   - Rank correlation with pose is ρ = 0.76, against ≤ 0.43 for depth, instance and view.
   - Within a model, persistence peaks at the same layer as metric geometry.
3. **Prediction tracks visible-scene readout** (ρ = 0.52 to 0.63). It is unrelated to persistence (|ρ| < 0.1).
   - Fast3R, best at geometry, is near the bottom on prediction.
4. **Layer and history length matter, and longer history usually hurts.**
   - 45 of 56 model–metric pairs favor the 8-frame prefix over 24 frames.
   - All 8 persistence pairs get worse. VideoMAE goes from 29.6° to 54.8°.
   - Caveat from the paper: a longer prefix also moves the endpoint, so this does not isolate history.
5. **Controls separate "decodable" from "used".**
   - **For V-JEPA2, copying the last feature beats its learned one-step prediction** in retrieval (19.9% against 13.0%).
   - Correct motion does beat matched wrong motion (+18.8 pp for V-JEPA2).
   - VideoMAE localizes well but gains nothing measurable from history beyond the current frame.
   - Action sensitivity is weak in absolute terms. In within-trajectory endpoint matching, only DINOv2 (0.029) and V-JEPA2 (0.006) have cosine margins clearly above zero.
6. **Predicted features are off the real-feature interface.**
   - A depth reader trained on real V-JEPA2 future features, applied to V-JEPA2's own native predictions, gets MAE 0.246 against 0.182 on real features. That is worse than a 0.241 layout prior.
   - Retraining the reader on the predictions gives 0.223.
   - So structure is still recoverable, but the original readout no longer applies.
7. **Goal sensitivity is not goal utility.**
   - Candidate selection works: 76 to 80% against a 67.8% action-only baseline, and shuffling the goal costs 12 to 21 pp.
   - Continuous action prediction never beats a current-state-only control.
   - Swapping the goal rotates the predicted action by 30 to 46°, yet progress is unchanged (correct minus shuffled: −2.3 to +1.2 pp, all intervals contain 0).
8. **Two "ingredient chains":**
   - pose → persistence → action;
   - visible-scene readout → feature prediction.
   - Only 4 models were tested at L4, so the action link rests on very few points.

## 3. How much to trust it

**Strengths:**
- Careful matched controls: copy, no-action, shuffled action, shuffled goal, history removal.
- Paired scene-bootstrap intervals.
- Explicit caveats throughout. The authors are unusually honest about what each test does not show.

**Weaknesses:**
- **The main table uses one probe-training seed.** The intervals cover evaluation sampling only, not seed variance.
- **Static indoor scenes with camera motion only.** No object dynamics, contact, or agent embodiment.
- **No closed loop.** All action results are offline supervised diagnostics.
- **Fast3R's training data includes ScanNet++,** and some validation scenes are in it. Its lead on pose, persistence and action may be partly familiarity.
- **Retrieval scores live in each model's own feature space,** so cross-model prediction numbers are not on a physical scale.
- **Cross-model correlations are over 9 or 10 models** that differ in architecture, data and objective, so the "chains" are suggestive, not causal.
- **V-JEPA2 is the only JEPA model,** and it is probed at one designated layer. Nothing here is a small, action-conditioned, simulation-trained JEPA like H-JEPA.

## 4. What it means for our H-JEPA work (inference)

It doesn't change the plan, but it gives us a diagnostics checklist we should adopt.

1. **Always report copy and shuffled-action baselines for H-JEPA predictions.**
   - V-JEPA2's predictor lost to "copy the last frame".
   - Before claiming that JEPA-TTT or a looped predictor "predicts better", show:
     - multi-step latent error against copying z_t;
     - error with true actions against shuffled actions;
     - both before and after adaptation.
   - This costs nothing extra on Colab.
2. **Check action sensitivity explicitly.**
   - Weak action margins were common here.
   - JEPA-TTT adapts the action encoder and predictor, so the true-versus-shuffled action gap is the cleanest signal of whether TTT fixed the dynamics or just fit the mean.
   - This is especially relevant for the AntMaze shift test (friction, mass, gear).
3. **Predicted latents drift off the encoder's distribution.**
   - This is the same phenomenon as the RMS mismatch in the looping paper.
   - In H-JEPA the planner compares *predicted* latents to an *encoded* goal, so this drift directly biases the cost.
   - Worth measuring: the distance between predicted and encoded latents (and their norm ratio) per level and per rollout step, with and without TTT. It also motivates re-injecting the encoded z_t when looping.
4. **Closed-loop success is the only real test of actionability.**
   - Offline goal-sensitivity looked good here and turned out useless. Our success-rate evals are the right primary metric; offline probes are secondary.
5. **A cheap probe for the shift test.**
   - AntMaze episodes record agent position and orientation.
   - A linear probe from each H-JEPA level's latent to (x, y, heading) tells us whether a dynamics shift breaks the *representation* or only the *predictor*.
   - The encoder is frozen and a shift doesn't change visuals much, so I expect only the predictor to break. That is the case JEPA-TTT is designed for.
6. **Short history is fine.** Longer prefixes hurt persistence and prediction here. H-JEPA's history_size of 3 is not obviously a bottleneck, so I wouldn't spend effort extending it.

## 5. Bottom line

- **What it is useful for:** a methodology paper, telling us how to evaluate our latents honestly (copy, shuffled-action and shuffled-goal controls; readout on predicted versus real features; closed loop over offline).
- **What it offers our TTT-plus-looping method:** nothing directly.
- **What I'd add to our experiments:**
  - copy and shuffled-action baselines;
  - the predicted-versus-encoded latent gap;
  - a position probe in the AntMaze shift test.
