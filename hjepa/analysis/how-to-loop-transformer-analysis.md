# "How to Loop Your Transformer" (arXiv 2610.40325): deep read and what it means for H-JEPA

Paper: *How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference* (FAIR/Meta and MIT, Oct 2026). https://www.alphaxiv.org/abs/2610.40325

I read the whole main body and the appendix prose: design sweeps, specialization (F), PLT normalization and gates (G), PLT scaling (H), PLT plus specialization (I) and limitations (J). I skimmed the hundreds of per-dataset figure captions. Where I say "inference", that is my reasoning, not something the paper tested.

## 1. What they did

- **Models.** Dense decoder-only language models trained from scratch: 70M, 100M, 200M, 300M, 500M and 1B parameters, with 4 to 16 stored layers.
- **Training scale.** 10 to 160 tokens per parameter (TPP), up to 155B tokens and 1.07e21 FLOPs. Loop counts L go up to 32.
- **Metric.** Mostly bits-per-byte (BPB) on DCLM. Also five other corpora, math and code likelihood, and 8 multiple-choice tasks (MMLU stays at chance).
- **Three questions:**
  1. Where and how many times to loop.
  2. Whether cheap loop-specific weights help.
  3. Whether loops can run in parallel during decoding (PLT).

## 2. Findings

### 2.1 Where and how much to loop

| Design | What loops | Result |
|---|---|---|
| Full-Stack | The whole block stack, L times | Best at fixed params and data, especially at larger scale and higher TPP |
| Middle-Stack | First and last ⌊B/3⌋ blocks run once; the middle loops | Best return when FLOPs are the constraint |
| Per-Layer | Each layer repeated in place | Weaker |
| Attention-only, MLP-only or independent attention/MLP | One sublayer type | Degrades sharply as L grows |

- **More loops stop paying off.** For L ≤ 16, BPB improves with diminishing returns. At L = 32 it saturates, reverses or goes unstable. This is worst for small models (70M/100M) and low TPP.
- **Placement matters more than the specialization method.**
- **Trade-off against plain depth.** Looping is parameter- and data-efficient. But if you can afford more stored parameters, adding real layers beats looping at matched training FLOPs.
  - The headline FLOP-efficiency result: L = 2 at TPP 80 beats no looping at TPP 160, at comparable FLOPs.
  - It holds only at some operating points, and the authors explicitly decline to call PLT "uniformly more data-efficient".
  - L = 4 reaches lower BPB but is worse per FLOP.

### 2.2 Loop-specific specialization

- Each loop ℓ gets W^(ℓ) = W + (α/r)·ΔW^(ℓ) at rank ≤ 8, trained jointly. There are three variants:
  - **Loop LoRA:** each loop has its own BA.
  - **Loop Matrix:** each loop mixes a shared bank of LoRAs through an unconstrained L×L matrix M.
  - **Loop Softmax:** M is row-softmaxed.
- **Loop Matrix is best.** Gains are marginal at L = 2 and clear at L = 4 and 8. It costs +1.1 to 4.3% parameters, versus +51% for doubling depth.
- **It is not compute-efficient.** A truly deeper model wins at matched FLOPs.
- **Where it helps:** Full-Stack most. It still helps at 500M and 1B, with smaller gains.

### 2.3 Parallel Loop Transformer (PLT)

- **The idea.** Loop ℓ for token n takes e_n + h_{n−1}^(ℓ−1), the previous token's state from the previous loop. This lets all loops of a new token run at once during decoding.
- **Unfixed, it fails.** Raw shifted PLT degrades with more loops and can be worse than no looping. KV-sharing and G-SWA recover only part of the loss.
- **Diagnosis: a scale mismatch.** The hidden-state RMS is much larger than the embedding RMS, and the ratio drifts with TPP and loop depth. The embedding gets drowned out.
- **The fix.** Normalize the two inputs separately, then combine with a gate: b = g⊙RMSNorm(e) + (1−g)⊙RMSNorm(h).
  - A fixed 50/50 gate already works. Rank-8 linear is marginally best, at 51K parameters for 1B.
  - Normalizing first and then adding without a gate is worse.
  - **Normalization is the important part; the gate barely matters.**
- **The anchor is essential.** Without the current-token embedding, shifted PLT is worse than no looping and several runs diverge.
  - Their hypothesis: the previous-token state is a *prediction* of the current position, and looping on predictions without the observed anchor drifts.
  - They say plainly that this explanation is unproven.
- **The unshifted control is the best design overall.** This is e_n + h_n^(ℓ−1) with the same normalization and gate, i.e. Full-Stack plus gated, normalized input re-injection every loop. It beats plain Full-Stack everywhere.
  - Its loops are still sequential, though.
- **Shifted PLT against that control.**
  - At L = 2 and 1B, shifted PLT tracks the control across all six corpora and all TPP values.
  - At L = 4 a gap remains at every size. Their explanation is that cross-token error accumulates.
- **Specialization combines cleanly.** Loop Matrix on top of PLT keeps the same ordering (unshifted > shifted > Full-Stack). It again helps clearly only at L = 4.

## 3. How much to trust it

**Strong:**
- A very broad and systematic sweep: 6 sizes, 5 budgets, many designs, 6 corpora.
- Rankings are consistent across all of them.
- The RMS diagnosis is backed by direct measurements (G.2) and a clean ablation of what feeds each loop.

**Weak:**
- **Mostly a single seed per setting.** Many differences are a few hundredths of a BPB, and no error bars are shown.
- **Language modeling and likelihood only.** No free-form generation, no reasoning or extrapolation-in-depth tests. Downstream gains are small, and MMLU is at chance.
- **No wall-clock measurements.** "Parallel decoding" is argued from data dependencies, not timed.
- **Everything is trained from scratch.** Nothing tests looping a model that was trained without loops, which is our situation.
- **Scale tops out at 1B dense.** The paper does not establish that shifted PLT holds at larger L.

## 4. What transfers to H-JEPA (all of this is my inference)

**Key distinction.** The paper loops *depth*: the same blocks run repeatedly on one token. In H-JEPA, rollout is already a loop over *time*, where the predicted latent is fed back as input. A "looped H-JEPA predictor" means depth looping inside each predictor call. Rollout is the autoregressive axis.

1. **Use few loops.** Our predictors are 4 to 6 layers and about 3 to 5M parameters, more than ten times smaller than their smallest model. The paper's instability at large L is worst for small models, so plan for L = 2 to 4 and expect saturation early.
2. **Use Full-Stack looping of the predictor.** Middle-Stack on 4 layers would loop just 2 blocks, which is barely different. Never loop attention or MLP sublayers independently.
3. **Re-inject the anchors every loop, normalized separately and then gated.**
   - Our anchors are the current latent z_t and the action embedding (the adaLN input).
   - This is the paper's single most transferable result.
   - It is cheap: a fixed 50/50 gate already works.
   - Without it, the hidden state's scale will drift away from the latent scale that SIGReg shaped.
4. **The anchor result explains rollout drift.** "Looping on a prediction without the observed anchor diverges" is the same failure as open-loop latent rollout. That supports keeping the real action and goal signals re-injected at every step and every loop.
5. **Loop-specific low-rank deltas are a natural TTT target.**
   - Rank 8 is about 1 to 4% of the predictor.
   - Adapting only those at test time fits JEPA-TTT's small update budget (about 240 updates) and limits forgetting.
   - The paper only shows they help when trained jointly, and only clearly at L ≥ 4.
6. **The cross-step shift (PLT) could pipeline rollouts.**
   - Loop ℓ at rollout step t+1 could start from loop ℓ−1 at step t. That would let the depth loops of a rollout run concurrently.
   - With models this small, a T4 is not latency-bound per step, and the paper shows the shift costs quality at L = 4.
   - Low priority.
7. **Compare at matched planner compute.**
   - L = 2 roughly doubles predictor FLOPs inside 90 GD iterations. FourRoom would go from about 65 s to about 130 s per task.
   - H-JEPA's own results are reported at ≤ 100 planner TFLOPs per episode, so a looped model must win at matched compute, not just raw.
   - The paper's own finding, that real depth beats loops per FLOP, is a warning here.
8. **Pretrained checkpoints are the big open risk.**
   - Looping a predictor that was trained unlooped is untested. The RMS-mismatch finding suggests that naively feeding outputs back as inputs will be off-distribution.
   - Expect to need a short fine-tune with the normalized gated re-injection, starting from the checkpoint, before evaluating.

## 5. Bottom line

- **What it tells us how to do:** how to loop well (Full-Stack, small L, normalized gated input re-injection, optional low-rank per-loop deltas). The evidence on these points is solid.
- **What it does not tell us:**
  - whether looping helps a world model predict better or plan better;
  - whether it works on top of pretrained checkpoints.
- **Cheapest test of our own (inference):**
  1. Fine-tune one H-JEPA predictor (FourRoom or AntMaze level 1, or an upper level) with L = 2 and the normalized 50/50 re-injection, starting from the checkpoint.
  2. Compare 1-step and multi-step latent prediction error against the original.
  3. Only if prediction improves, run planning at matched planner TFLOPs.
